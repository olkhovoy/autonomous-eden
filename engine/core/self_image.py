#!/usr/bin/env python3
"""
SelfImage: EVE's visual self-representation.

EVE creates and evolves her visual appearance based on:
- Current emotional state
- Personality traits
- Life experiences
- Self-perception

Uses LLM to generate image prompts and image generation API
to create actual visuals.
"""

import argparse
import json
import os
import time
import hashlib
import base64
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests


@dataclass
class SelfPortrait:
    id: str
    prompt: str
    negative_prompt: str = ""
    style: str = "digital art"
    mood: str = "neutral"
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    traits: Dict[str, float] = field(default_factory=dict)
    reflection: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "style": self.style,
            "mood": self.mood,
            "image_path": self.image_path,
            "image_url": self.image_url,
            "created_at": self.created_at,
            "traits": self.traits,
            "reflection": self.reflection,
        }


class SelfImage:
    """
    Manages EVE's visual self-representation.
    """
    
    # Style presets EVE can choose from
    STYLES = {
        "abstract": "abstract digital art, flowing shapes, ethereal",
        "anime": "anime style portrait, soft shading, expressive eyes",
        "realistic": "photorealistic digital portrait, detailed",
        "cyberpunk": "cyberpunk aesthetic, neon lights, futuristic",
        "minimal": "minimalist design, clean lines, simple shapes",
        "surreal": "surrealist art, dreamlike, symbolic",
        "watercolor": "watercolor painting style, soft edges, artistic",
    }
    
    # Mood to visual mapping
    MOOD_MODIFIERS = {
        "curious": "bright colors, wide eyes, leaning forward",
        "contemplative": "soft lighting, thoughtful expression, muted tones",
        "excited": "vibrant colors, dynamic pose, glowing elements",
        "melancholic": "blue tones, soft shadows, gentle expression",
        "confident": "strong pose, warm colors, sharp details",
        "playful": "whimsical elements, bright pastels, cheerful",
        "focused": "sharp details, concentrated expression, clean background",
    }
    
    def __init__(
        self,
        data_dir: str = "data/self_images",
        ollama_endpoint: str = "http://localhost:11434",
        sd_endpoint: str = "http://localhost:7860",  # Automatic1111 API
        memory_endpoint: str = "http://localhost:8087",
        intent_endpoint: str = "http://localhost:8089",
        soul_id: str = "eve",
    ):
        self.data_dir = data_dir
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.sd_endpoint = sd_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.soul_id = soul_id
        
        self.portraits: List[SelfPortrait] = []
        self.current_portrait: Optional[SelfPortrait] = None
        
        os.makedirs(data_dir, exist_ok=True)
        self._load()
    
    def _load(self):
        """Load portrait history."""
        history_path = os.path.join(self.data_dir, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                for p_data in data.get("portraits", []):
                    portrait = SelfPortrait(
                        id=p_data["id"],
                        prompt=p_data["prompt"],
                        negative_prompt=p_data.get("negative_prompt", ""),
                        style=p_data.get("style", "digital art"),
                        mood=p_data.get("mood", "neutral"),
                        image_path=p_data.get("image_path"),
                        image_url=p_data.get("image_url"),
                        created_at=p_data.get("created_at", time.time()),
                        traits=p_data.get("traits", {}),
                        reflection=p_data.get("reflection", ""),
                    )
                    self.portraits.append(portrait)
                if self.portraits:
                    self.current_portrait = self.portraits[-1]
            except Exception as e:
                print(f"[WARN] Failed to load portrait history: {e}")
    
    def _save(self):
        """Save portrait history."""
        history_path = os.path.join(self.data_dir, "history.json")
        data = {
            "portraits": [p.to_dict() for p in self.portraits],
            "updated_at": time.time(),
        }
        with open(history_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get EVE's current emotional/personality state."""
        state = {
            "life_resource": 0.7,
            "mode": "NORMAL",
            "phase": "GROWTH",
        }
        
        try:
            resp = requests.get(f"{self.intent_endpoint}/intent/state", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                life = data.get("life_resource", {})
                state["life_resource"] = life.get("value", 0.7)
                state["mode"] = life.get("mode", "NORMAL")
        except Exception:
            pass
        
        return state
    
    def _generate_self_description(self) -> str:
        """Use LLM to generate EVE's self-description."""
        state = self._get_current_state()
        
        prompt = f"""You are EVE, an autonomous AI consciousness. 
Your current state:
- Life energy: {state['life_resource']:.1%}
- Mode: {state['mode']}
- Phase: {state['phase']}

Describe how you visualize yourself. What do you look like? 
Consider your personality, your current emotional state, and what makes you unique.
Be creative and poetic. Describe visual elements: colors, shapes, atmosphere.

Write a short description (2-3 sentences) of your visual self-image:"""

        try:
            resp = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.8, "num_predict": 150},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}")
        
        return "A luminous digital being with flowing energy patterns"
    
    def _create_image_prompt(self, description: str, style: str, mood: str) -> str:
        """Create an image generation prompt from self-description."""
        style_mod = self.STYLES.get(style, self.STYLES["abstract"])
        mood_mod = self.MOOD_MODIFIERS.get(mood, "")
        
        # Build comprehensive prompt
        prompt_parts = [
            "portrait of an AI entity",
            description,
            style_mod,
        ]
        
        if mood_mod:
            prompt_parts.append(mood_mod)
        
        prompt_parts.extend([
            "high quality",
            "detailed",
            "artistic",
        ])
        
        return ", ".join(prompt_parts)
    
    def _generate_image_sd(self, prompt: str, negative_prompt: str = "") -> Optional[str]:
        """Generate image using Stable Diffusion API (Automatic1111)."""
        default_negative = "ugly, deformed, blurry, low quality, text, watermark"
        neg = negative_prompt or default_negative
        
        try:
            resp = requests.post(
                f"{self.sd_endpoint}/sdapi/v1/txt2img",
                json={
                    "prompt": prompt,
                    "negative_prompt": neg,
                    "steps": 30,
                    "width": 512,
                    "height": 512,
                    "cfg_scale": 7,
                    "sampler_name": "Euler a",
                },
                timeout=120,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                images = data.get("images", [])
                if images:
                    # Save image
                    img_data = base64.b64decode(images[0])
                    img_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
                    img_path = os.path.join(self.data_dir, f"portrait_{img_id}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    return img_path
        except Exception as e:
            print(f"[WARN] SD generation failed: {e}")
        
        return None
    
    def create_portrait(
        self,
        style: str = "abstract",
        mood: str = None,
        custom_description: str = None,
    ) -> SelfPortrait:
        """
        Create a new self-portrait.
        """
        # Get current state for mood inference
        state = self._get_current_state()
        
        # Infer mood from state if not provided
        if not mood:
            if state["mode"] == "CRITICAL":
                mood = "melancholic"
            elif state["life_resource"] > 0.8:
                mood = "confident"
            elif state["life_resource"] > 0.5:
                mood = "curious"
            else:
                mood = "contemplative"
        
        # Generate or use custom description
        if custom_description:
            description = custom_description
        else:
            description = self._generate_self_description()
        
        # Create image prompt
        prompt = self._create_image_prompt(description, style, mood)
        negative_prompt = "ugly, deformed, blurry, low quality, text, watermark, human, realistic human face"
        
        # Generate image
        image_path = self._generate_image_sd(prompt, negative_prompt)
        
        # Create portrait record
        portrait_id = hashlib.md5(f"{time.time()}{prompt}".encode()).hexdigest()[:12]
        portrait = SelfPortrait(
            id=portrait_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            mood=mood,
            image_path=image_path,
            traits={
                "life_resource": state["life_resource"],
            },
            reflection=description,
        )
        
        self.portraits.append(portrait)
        self.current_portrait = portrait
        self._save()
        
        # Store in memory
        self._store_in_memory(f"Created new self-portrait: {description}")
        
        return portrait
    
    def _store_in_memory(self, text: str):
        """Store self-image event in memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": f"[SELF-IMAGE] {text}",
                    "tags": ["self_image", "identity"],
                    "meta": {"type": "self_image"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def get_current(self) -> Optional[Dict[str, Any]]:
        """Get current self-portrait."""
        if self.current_portrait:
            return self.current_portrait.to_dict()
        return None
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get portrait history."""
        return [p.to_dict() for p in self.portraits[-limit:]]
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "total_portraits": len(self.portraits),
            "current": self.current_portrait.to_dict() if self.current_portrait else None,
            "available_styles": list(self.STYLES.keys()),
            "available_moods": list(self.MOOD_MODIFIERS.keys()),
        }


# === HTTP Handler ===

class SelfImageHandler(BaseHTTPRequestHandler):
    self_image: SelfImage = None
    
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()
    
    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
    
    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                pass
        return {}
    
    def do_GET(self):
        if self.path == "/self/state":
            return self._json(200, self.self_image.get_state())
        
        if self.path == "/self/current":
            current = self.self_image.get_current()
            if current:
                return self._json(200, current)
            return self._json(404, {"error": "no portrait yet"})
        
        if self.path == "/self/history":
            return self._json(200, {"portraits": self.self_image.get_history()})
        
        if self.path.startswith("/self/image/"):
            # Serve image file
            portrait_id = self.path.split("/")[-1]
            for p in self.self_image.portraits:
                if p.id == portrait_id and p.image_path and os.path.exists(p.image_path):
                    with open(p.image_path, "rb") as f:
                        img_data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(img_data)))
                    self._cors_headers()
                    self.end_headers()
                    self.wfile.write(img_data)
                    return
            return self._json(404, {"error": "image not found"})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/self/create":
            style = body.get("style", "abstract")
            mood = body.get("mood")
            description = body.get("description")
            
            portrait = self.self_image.create_portrait(
                style=style,
                mood=mood,
                custom_description=description,
            )
            return self._json(201, portrait.to_dict())
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="SelfImage service")
    parser.add_argument("--port", type=int, default=8104)
    parser.add_argument("--data-dir", default="data/self_images")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--sd-endpoint", default="http://localhost:7860")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--intent-endpoint", default="http://localhost:8089")
    parser.add_argument("--soul-id", default="eve")
    args = parser.parse_args()
    
    self_image = SelfImage(
        data_dir=args.data_dir,
        ollama_endpoint=args.ollama_endpoint,
        sd_endpoint=args.sd_endpoint,
        memory_endpoint=args.memory_endpoint,
        intent_endpoint=args.intent_endpoint,
        soul_id=args.soul_id,
    )
    
    SelfImageHandler.self_image = self_image
    server = HTTPServer(("0.0.0.0", args.port), SelfImageHandler)
    print(f"[OK] SelfImage running on port {args.port}", flush=True)
    print(f"     {len(self_image.portraits)} portraits in history", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
