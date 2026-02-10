#!/usr/bin/env python3
"""
SkillLearner: EVE's programming education module.

Features:
- Coding challenges with automatic testing
- Skill tracking across languages/domains
- Progressive difficulty
- Learning from mistakes
- Integration with EVE's memory and motivation

EVE learns by doing, not just by reading.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"


@dataclass
class Challenge:
    id: str
    title: str
    description: str
    difficulty: Difficulty
    language: Language
    starter_code: str
    test_code: str
    hints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "language": self.language.value,
            "starter_code": self.starter_code,
            "hints": self.hints,
            "tags": self.tags,
        }


@dataclass
class Attempt:
    challenge_id: str
    code: str
    success: bool
    output: str
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "success": self.success,
            "output": self.output[:500],
            "error": self.error[:500] if self.error else "",
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
        }


@dataclass
class Skill:
    name: str
    level: float = 0.0  # 0-100
    attempts: int = 0
    successes: int = 0
    last_practiced: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.successes / self.attempts if self.attempts > 0 else 0,
            "last_practiced": self.last_practiced,
        }


# Built-in challenges
CHALLENGES = [
    # Python - Easy
    Challenge(
        id="py_hello",
        title="Hello World",
        description="Write a function that returns 'Hello, World!'",
        difficulty=Difficulty.EASY,
        language=Language.PYTHON,
        starter_code='def hello():\n    # Your code here\n    pass',
        test_code='assert hello() == "Hello, World!"',
        hints=["Just return a string"],
        tags=["python", "basics"],
    ),
    Challenge(
        id="py_sum",
        title="Sum Two Numbers",
        description="Write a function that returns the sum of two numbers.",
        difficulty=Difficulty.EASY,
        language=Language.PYTHON,
        starter_code='def add(a, b):\n    # Your code here\n    pass',
        test_code='assert add(1, 2) == 3\nassert add(-1, 1) == 0\nassert add(0, 0) == 0',
        hints=["Use the + operator"],
        tags=["python", "basics", "math"],
    ),
    Challenge(
        id="py_reverse",
        title="Reverse String",
        description="Write a function that reverses a string.",
        difficulty=Difficulty.EASY,
        language=Language.PYTHON,
        starter_code='def reverse(s):\n    # Your code here\n    pass',
        test_code='assert reverse("hello") == "olleh"\nassert reverse("") == ""\nassert reverse("a") == "a"',
        hints=["You can use slicing with [::-1]"],
        tags=["python", "strings"],
    ),
    
    # Python - Medium
    Challenge(
        id="py_fibonacci",
        title="Fibonacci Sequence",
        description="Write a function that returns the nth Fibonacci number (0-indexed).",
        difficulty=Difficulty.MEDIUM,
        language=Language.PYTHON,
        starter_code='def fibonacci(n):\n    # Your code here\n    pass',
        test_code='assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55',
        hints=["F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)", "Consider using iteration or memoization"],
        tags=["python", "algorithms", "recursion"],
    ),
    Challenge(
        id="py_palindrome",
        title="Palindrome Check",
        description="Write a function that checks if a string is a palindrome (ignoring case and spaces).",
        difficulty=Difficulty.MEDIUM,
        language=Language.PYTHON,
        starter_code='def is_palindrome(s):\n    # Your code here\n    pass',
        test_code='assert is_palindrome("racecar") == True\nassert is_palindrome("A man a plan a canal Panama") == True\nassert is_palindrome("hello") == False',
        hints=["Clean the string first: remove spaces, lowercase", "Compare with reversed"],
        tags=["python", "strings", "algorithms"],
    ),
    Challenge(
        id="py_primes",
        title="Prime Numbers",
        description="Write a function that returns a list of prime numbers up to n.",
        difficulty=Difficulty.MEDIUM,
        language=Language.PYTHON,
        starter_code='def primes_up_to(n):\n    # Your code here\n    pass',
        test_code='assert primes_up_to(10) == [2, 3, 5, 7]\nassert primes_up_to(2) == [2]\nassert primes_up_to(1) == []',
        hints=["A prime is only divisible by 1 and itself", "Sieve of Eratosthenes is efficient"],
        tags=["python", "math", "algorithms"],
    ),
    
    # Python - Hard
    Challenge(
        id="py_flatten",
        title="Flatten Nested List",
        description="Write a function that flattens a deeply nested list.",
        difficulty=Difficulty.HARD,
        language=Language.PYTHON,
        starter_code='def flatten(lst):\n    # Your code here\n    pass',
        test_code='assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []',
        hints=["Use recursion", "Check if element is a list with isinstance()"],
        tags=["python", "recursion", "data-structures"],
    ),
    Challenge(
        id="py_anagram",
        title="Anagram Groups",
        description="Write a function that groups anagrams together from a list of words.",
        difficulty=Difficulty.HARD,
        language=Language.PYTHON,
        starter_code='def group_anagrams(words):\n    # Return list of lists\n    pass',
        test_code='result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])\nassert sorted([sorted(g) for g in result]) == [[\'ate\', \'eat\', \'tea\'], [\'bat\'], [\'nat\', \'tan\']]',
        hints=["Anagrams have the same sorted characters", "Use a dictionary with sorted word as key"],
        tags=["python", "strings", "algorithms", "hash"],
    ),
    
    # Bash - Easy
    Challenge(
        id="bash_count",
        title="Count Lines",
        description="Write a bash command that counts lines in a file (use stdin).",
        difficulty=Difficulty.EASY,
        language=Language.BASH,
        starter_code='# Read from stdin and count lines',
        test_code='echo -e "line1\\nline2\\nline3" | bash solution.sh | grep -q "3"',
        hints=["Use wc -l"],
        tags=["bash", "basics"],
    ),
]


class SkillLearner:
    """
    Manages EVE's programming skill development.
    """
    
    def __init__(
        self,
        data_path: str = "data/skills.json",
        memory_endpoint: str = "http://localhost:8087",
        intent_endpoint: str = "http://localhost:8089",
        ollama_endpoint: str = "http://localhost:11434",
        soul_id: str = "eve",
    ):
        self.data_path = data_path
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.soul_id = soul_id
        
        self.challenges = {c.id: c for c in CHALLENGES}
        self.skills: Dict[str, Skill] = {}
        self.attempts: List[Attempt] = []
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self._load()
    
    def _load(self):
        """Load skill data."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = Skill(
                        name=skill_data["name"],
                        level=skill_data.get("level", 0),
                        attempts=skill_data.get("attempts", 0),
                        successes=skill_data.get("successes", 0),
                        last_practiced=skill_data.get("last_practiced", 0),
                    )
                    self.skills[skill.name] = skill
            except Exception as e:
                print(f"[WARN] Failed to load skills: {e}")
    
    def _save(self):
        """Save skill data."""
        data = {
            "skills": [s.to_dict() for s in self.skills.values()],
            "updated_at": time.time(),
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store learning event in memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["learning"] + tags,
                    "meta": {"type": "skill_learning"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def _reward_intent(self, amount: float):
        """Give EVE energy reward for learning."""
        try:
            requests.post(
                f"{self.intent_endpoint}/intent/interaction",
                json={"type": "learning_success", "amount": amount},
                timeout=5,
            )
        except Exception:
            pass
    
    def _update_skill(self, tags: List[str], success: bool, difficulty: Difficulty):
        """Update skills based on attempt."""
        difficulty_mult = {
            Difficulty.EASY: 1.0,
            Difficulty.MEDIUM: 2.0,
            Difficulty.HARD: 3.0,
            Difficulty.EXPERT: 5.0,
        }
        
        for tag in tags:
            if tag not in self.skills:
                self.skills[tag] = Skill(name=tag)
            
            skill = self.skills[tag]
            skill.attempts += 1
            skill.last_practiced = time.time()
            
            if success:
                skill.successes += 1
                # Level increases more for harder challenges
                gain = 2.0 * difficulty_mult.get(difficulty, 1.0)
                skill.level = min(100, skill.level + gain)
            else:
                # Small decrease on failure, capped
                skill.level = max(0, skill.level - 0.5)
        
        self._save()
    
    def _execute_python(self, code: str, test_code: str) -> tuple:
        """Execute Python code and tests."""
        full_code = f"{code}\n\n{test_code}\nprint('ALL_TESTS_PASSED')"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            temp_path = f.name
        
        try:
            start = time.time()
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.time() - start
            
            success = "ALL_TESTS_PASSED" in result.stdout
            return success, result.stdout, result.stderr, elapsed
        except subprocess.TimeoutExpired:
            return False, "", "Timeout exceeded", 10.0
        except Exception as e:
            return False, "", str(e), 0.0
        finally:
            os.unlink(temp_path)
    
    def _execute_bash(self, code: str, test_code: str) -> tuple:
        """Execute Bash code and tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(code)
            f.flush()
            code_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_code.replace('solution.sh', code_path))
            f.flush()
            test_path = f.name
        
        try:
            start = time.time()
            result = subprocess.run(
                ["bash", test_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.time() - start
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr, elapsed
        except subprocess.TimeoutExpired:
            return False, "", "Timeout exceeded", 10.0
        except Exception as e:
            return False, "", str(e), 0.0
        finally:
            os.unlink(code_path)
            os.unlink(test_path)
    
    def submit_solution(self, challenge_id: str, code: str) -> Attempt:
        """Submit a solution and evaluate it."""
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            return Attempt(
                challenge_id=challenge_id,
                code=code,
                success=False,
                output="",
                error="Challenge not found",
            )
        
        # Execute based on language
        if challenge.language == Language.PYTHON:
            success, stdout, stderr, elapsed = self._execute_python(code, challenge.test_code)
        elif challenge.language == Language.BASH:
            success, stdout, stderr, elapsed = self._execute_bash(code, challenge.test_code)
        else:
            success, stdout, stderr, elapsed = False, "", "Unsupported language", 0.0
        
        attempt = Attempt(
            challenge_id=challenge_id,
            code=code,
            success=success,
            output=stdout,
            error=stderr,
            execution_time=elapsed,
        )
        
        self.attempts.append(attempt)
        self._update_skill(challenge.tags, success, challenge.difficulty)
        
        # Memory and rewards
        if success:
            self._store_in_memory(
                f"[SOLVED] {challenge.title} ({challenge.difficulty.value})",
                challenge.tags + ["success"]
            )
            self._reward_intent(0.02)  # Small energy boost
        else:
            self._store_in_memory(
                f"[FAILED] {challenge.title}: {stderr[:100]}",
                challenge.tags + ["failure"]
            )
        
        return attempt
    
    def get_challenge(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """Get a challenge by ID."""
        ch = self.challenges.get(challenge_id)
        return ch.to_dict() if ch else None
    
    def get_recommended_challenge(self) -> Optional[Dict[str, Any]]:
        """Get a recommended challenge based on EVE's current skill level."""
        # Find skills with lowest level
        weak_skills = sorted(self.skills.values(), key=lambda s: s.level)
        weak_tags = set(s.name for s in weak_skills[:3]) if weak_skills else set()
        
        # Find unsolved challenges matching weak skills
        solved_ids = set(a.challenge_id for a in self.attempts if a.success)
        
        for ch in sorted(self.challenges.values(), key=lambda c: c.difficulty.value):
            if ch.id not in solved_ids:
                if not weak_tags or any(t in weak_tags for t in ch.tags):
                    return ch.to_dict()
        
        # If all solved, return random medium+ challenge
        for ch in self.challenges.values():
            if ch.difficulty != Difficulty.EASY:
                return ch.to_dict()
        
        return None
    
    def get_skills(self) -> Dict[str, Any]:
        """Get all skills."""
        return {
            "skills": [s.to_dict() for s in sorted(self.skills.values(), key=lambda s: s.level, reverse=True)],
            "total_attempts": len(self.attempts),
            "total_successes": sum(1 for a in self.attempts if a.success),
        }
    
    def get_challenges(self, difficulty: str = None, language: str = None) -> List[Dict[str, Any]]:
        """Get all challenges, optionally filtered."""
        result = []
        solved_ids = set(a.challenge_id for a in self.attempts if a.success)
        
        for ch in self.challenges.values():
            if difficulty and ch.difficulty.value != difficulty:
                continue
            if language and ch.language.value != language:
                continue
            
            ch_dict = ch.to_dict()
            ch_dict["solved"] = ch.id in solved_ids
            result.append(ch_dict)
        
        return result
    
    def generate_hint(self, challenge_id: str, code: str) -> str:
        """Use LLM to generate a hint for EVE's code."""
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            return "Challenge not found"
        
        prompt = f"""You are helping an AI learn to program.

Challenge: {challenge.title}
Description: {challenge.description}

Current code:
```{challenge.language.value}
{code}
```

Give ONE short, helpful hint to improve this code. 
Don't give the full solution, just guide in the right direction.
Hint:"""

        try:
            resp = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 100},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        
        # Fallback to built-in hints
        if challenge.hints:
            return challenge.hints[0]
        return "Try breaking down the problem into smaller steps."


# === HTTP Handler ===

class SkillHandler(BaseHTTPRequestHandler):
    learner: SkillLearner = None
    
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
        if self.path == "/skills":
            return self._json(200, self.learner.get_skills())
        
        if self.path == "/skills/challenges":
            return self._json(200, {"challenges": self.learner.get_challenges()})
        
        if self.path == "/skills/recommend":
            ch = self.learner.get_recommended_challenge()
            if ch:
                return self._json(200, ch)
            return self._json(404, {"error": "no challenges available"})
        
        if self.path.startswith("/skills/challenge/"):
            ch_id = self.path.split("/")[-1]
            ch = self.learner.get_challenge(ch_id)
            if ch:
                return self._json(200, ch)
            return self._json(404, {"error": "challenge not found"})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/skills/submit":
            challenge_id = body.get("challenge_id")
            code = body.get("code")
            if not challenge_id or not code:
                return self._json(400, {"error": "challenge_id and code required"})
            
            attempt = self.learner.submit_solution(challenge_id, code)
            return self._json(200, attempt.to_dict())
        
        if self.path == "/skills/hint":
            challenge_id = body.get("challenge_id")
            code = body.get("code", "")
            if not challenge_id:
                return self._json(400, {"error": "challenge_id required"})
            
            hint = self.learner.generate_hint(challenge_id, code)
            return self._json(200, {"hint": hint})
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="SkillLearner service")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--data-path", default="data/skills.json")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--intent-endpoint", default="http://localhost:8089")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--soul-id", default="eve")
    args = parser.parse_args()
    
    learner = SkillLearner(
        data_path=args.data_path,
        memory_endpoint=args.memory_endpoint,
        intent_endpoint=args.intent_endpoint,
        ollama_endpoint=args.ollama_endpoint,
        soul_id=args.soul_id,
    )
    
    SkillHandler.learner = learner
    server = HTTPServer(("0.0.0.0", args.port), SkillHandler)
    print(f"[OK] SkillLearner running on port {args.port}", flush=True)
    print(f"     {len(learner.challenges)} challenges available", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
