#!/usr/bin/env python3
"""
WebExplorer: EVE's window to the internet.

Sources:
- Hacker News (API)
- Reddit (API)
- RSS feeds (configurable)
- Web pages (via scraping)

EVE uses this to:
- Discover interesting topics
- Form opinions and interests
- Find inspiration for projects
- Stay updated on technology
"""

import argparse
import json
import os
import re
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET

import requests


# Cache TTL in seconds
CACHE_TTL = 1800  # 30 minutes


@dataclass
class ContentItem:
    id: str
    source: str
    title: str
    url: str
    text: str = ""
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "text": self.text[:500] if self.text else "",
            "score": self.score,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


class ContentCache:
    """SQLite cache for web content."""
    
    def __init__(self, db_path: str = "data/web_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()
    
    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                ts INTEGER,
                response TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interests (
                topic TEXT PRIMARY KEY,
                score REAL,
                updated_at INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seen (
                content_id TEXT PRIMARY KEY,
                source TEXT,
                seen_at INTEGER
            )
        """)
        self.conn.commit()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT ts, response FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        ts, resp = row
        if int(time.time()) - ts > CACHE_TTL:
            return None
        try:
            return json.loads(resp)
        except Exception:
            return None
    
    def set(self, key: str, value: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO cache(key, ts, response) VALUES(?,?,?)",
            (key, int(time.time()), json.dumps(value))
        )
        self.conn.commit()
    
    def mark_seen(self, content_id: str, source: str):
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO seen(content_id, source, seen_at) VALUES(?,?,?)",
            (content_id, source, int(time.time()))
        )
        self.conn.commit()
    
    def is_seen(self, content_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM seen WHERE content_id=?", (content_id,))
        return cur.fetchone() is not None
    
    def update_interest(self, topic: str, delta: float):
        cur = self.conn.cursor()
        cur.execute("SELECT score FROM interests WHERE topic=?", (topic,))
        row = cur.fetchone()
        current = row[0] if row else 0.5
        new_score = max(0.0, min(1.0, current + delta))
        cur.execute(
            "REPLACE INTO interests(topic, score, updated_at) VALUES(?,?,?)",
            (topic, new_score, int(time.time()))
        )
        self.conn.commit()
    
    def get_interests(self) -> Dict[str, float]:
        cur = self.conn.cursor()
        cur.execute("SELECT topic, score FROM interests ORDER BY score DESC")
        return {row[0]: row[1] for row in cur.fetchall()}


class WebExplorer:
    """
    EVE's internet exploration module.
    """
    
    # Default RSS feeds
    DEFAULT_FEEDS = [
        ("https://hnrss.org/frontpage", "hackernews"),
        ("https://www.reddit.com/r/programming/.rss", "reddit_programming"),
        ("https://www.reddit.com/r/machinelearning/.rss", "reddit_ml"),
        ("https://feeds.arstechnica.com/arstechnica/index", "arstechnica"),
    ]
    
    def __init__(
        self,
        cache_db: str = "data/web_cache.db",
        memory_endpoint: str = "http://localhost:8087",
        soul_id: str = "eve",
    ):
        self.cache = ContentCache(cache_db)
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.feeds = list(self.DEFAULT_FEEDS)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EVE-WebExplorer/1.0 (Autonomous AI Agent)"
        })
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store discovery in EVE's memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["web_discovery"] + tags,
                    "meta": {"type": "web_discovery"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    # === Hacker News ===
    
    def get_hn_top(self, limit: int = 30) -> List[ContentItem]:
        """Get top stories from Hacker News."""
        cache_key = f"hn_top_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [ContentItem(**item) for item in cached.get("items", [])]
        
        items = []
        try:
            # Get top story IDs
            resp = self.session.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=15
            )
            if resp.status_code != 200:
                return items
            
            story_ids = resp.json()[:limit]
            
            # Fetch each story
            for sid in story_ids[:limit]:
                try:
                    story_resp = self.session.get(
                        f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                        timeout=10
                    )
                    if story_resp.status_code == 200:
                        story = story_resp.json()
                        if story and story.get("title"):
                            items.append(ContentItem(
                                id=f"hn_{sid}",
                                source="hackernews",
                                title=story.get("title", ""),
                                url=story.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                                text=story.get("text", ""),
                                score=story.get("score", 0),
                                timestamp=story.get("time", time.time()),
                                tags=["hackernews"],
                            ))
                except Exception:
                    continue
            
            # Cache results
            self.cache.set(cache_key, {"items": [i.to_dict() for i in items]})
            
        except Exception as e:
            print(f"[WARN] HN fetch failed: {e}")
        
        return items
    
    # === Reddit ===
    
    def get_reddit_posts(self, subreddit: str = "programming", limit: int = 25) -> List[ContentItem]:
        """Get posts from a subreddit."""
        cache_key = f"reddit_{subreddit}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [ContentItem(**item) for item in cached.get("items", [])]
        
        items = []
        try:
            resp = self.session.get(
                f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}",
                timeout=15
            )
            if resp.status_code != 200:
                return items
            
            data = resp.json()
            for post in data.get("data", {}).get("children", []):
                pd = post.get("data", {})
                if pd.get("title"):
                    items.append(ContentItem(
                        id=f"reddit_{pd.get('id', '')}",
                        source=f"reddit_{subreddit}",
                        title=pd.get("title", ""),
                        url=pd.get("url", ""),
                        text=pd.get("selftext", "")[:500],
                        score=pd.get("score", 0),
                        timestamp=pd.get("created_utc", time.time()),
                        tags=["reddit", subreddit],
                    ))
            
            self.cache.set(cache_key, {"items": [i.to_dict() for i in items]})
            
        except Exception as e:
            print(f"[WARN] Reddit fetch failed: {e}")
        
        return items
    
    # === RSS Feeds ===
    
    def parse_rss(self, url: str, source_name: str) -> List[ContentItem]:
        """Parse an RSS feed."""
        cache_key = f"rss_{hashlib.md5(url.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return [ContentItem(**item) for item in cached.get("items", [])]
        
        items = []
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return items
            
            root = ET.fromstring(resp.content)
            
            # Handle both RSS and Atom feeds
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            # Try RSS format
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                desc = item.findtext("description", "")
                
                if title:
                    items.append(ContentItem(
                        id=f"rss_{hashlib.md5(link.encode()).hexdigest()[:12]}",
                        source=source_name,
                        title=title,
                        url=link,
                        text=self._strip_html(desc)[:500],
                        tags=["rss", source_name],
                    ))
            
            # Try Atom format
            for entry in root.findall(".//atom:entry", ns):
                title = entry.findtext("atom:title", "", ns)
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                summary = entry.findtext("atom:summary", "", ns)
                
                if title:
                    items.append(ContentItem(
                        id=f"atom_{hashlib.md5(link.encode()).hexdigest()[:12]}",
                        source=source_name,
                        title=title,
                        url=link,
                        text=self._strip_html(summary)[:500],
                        tags=["rss", source_name],
                    ))
            
            self.cache.set(cache_key, {"items": [i.to_dict() for i in items]})
            
        except Exception as e:
            print(f"[WARN] RSS parse failed for {url}: {e}")
        
        return items
    
    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text).strip()
    
    # === Web Page Fetch ===
    
    def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and extract content from a web page."""
        cache_key = f"page_{hashlib.md5(url.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            resp = self.session.get(url, timeout=20)
            if resp.status_code != 200:
                return None
            
            content = resp.text
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.I)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract main text (simplified)
            # Remove scripts and styles
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.S | re.I)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.S | re.I)
            
            # Get text
            text = self._strip_html(content)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            result = {
                "url": url,
                "title": title,
                "text": text[:5000],  # Limit text length
                "fetched_at": time.time(),
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[WARN] Page fetch failed for {url}: {e}")
            return None
    
    # === Discovery ===
    
    def discover(self, sources: List[str] = None) -> List[ContentItem]:
        """
        Discover new content from various sources.
        Returns items EVE hasn't seen yet.
        """
        all_items = []
        
        sources = sources or ["hackernews", "reddit_programming", "reddit_machinelearning"]
        
        if "hackernews" in sources:
            all_items.extend(self.get_hn_top(30))
        
        if "reddit_programming" in sources:
            all_items.extend(self.get_reddit_posts("programming", 25))
        
        if "reddit_machinelearning" in sources:
            all_items.extend(self.get_reddit_posts("MachineLearning", 25))
        
        if "reddit_python" in sources:
            all_items.extend(self.get_reddit_posts("Python", 25))
        
        # Filter out seen items
        new_items = [i for i in all_items if not self.cache.is_seen(i.id)]
        
        # Sort by score
        new_items.sort(key=lambda x: x.score, reverse=True)
        
        return new_items[:50]
    
    def mark_interesting(self, content_id: str, topics: List[str]):
        """Mark content as interesting, update interest scores."""
        self.cache.mark_seen(content_id)
        for topic in topics:
            self.cache.update_interest(topic, 0.1)
    
    def mark_boring(self, content_id: str, topics: List[str]):
        """Mark content as not interesting."""
        self.cache.mark_seen(content_id)
        for topic in topics:
            self.cache.update_interest(topic, -0.05)
    
    def get_interests(self) -> Dict[str, float]:
        """Get EVE's current interests."""
        return self.cache.get_interests()
    
    def add_feed(self, url: str, name: str):
        """Add a custom RSS feed."""
        self.feeds.append((url, name))
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "feeds": [{"url": url, "name": name} for url, name in self.feeds],
            "interests": self.get_interests(),
        }


# === HTTP Handler ===

class ExplorerHandler(BaseHTTPRequestHandler):
    explorer: WebExplorer = None
    
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
        if self.path == "/web/state":
            return self._json(200, self.explorer.get_state())
        
        if self.path == "/web/discover":
            items = self.explorer.discover()
            return self._json(200, {"items": [i.to_dict() for i in items]})
        
        if self.path == "/web/hackernews":
            items = self.explorer.get_hn_top(30)
            return self._json(200, {"items": [i.to_dict() for i in items]})
        
        if self.path.startswith("/web/reddit/"):
            subreddit = self.path.split("/")[-1]
            items = self.explorer.get_reddit_posts(subreddit, 25)
            return self._json(200, {"items": [i.to_dict() for i in items]})
        
        if self.path == "/web/interests":
            return self._json(200, {"interests": self.explorer.get_interests()})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/web/fetch":
            url = body.get("url")
            if not url:
                return self._json(400, {"error": "url required"})
            result = self.explorer.fetch_page(url)
            if result:
                return self._json(200, result)
            return self._json(500, {"error": "fetch failed"})
        
        if self.path == "/web/interesting":
            content_id = body.get("id")
            topics = body.get("topics", [])
            if content_id:
                self.explorer.mark_interesting(content_id, topics)
            return self._json(200, {"ok": True})
        
        if self.path == "/web/boring":
            content_id = body.get("id")
            topics = body.get("topics", [])
            if content_id:
                self.explorer.mark_boring(content_id, topics)
            return self._json(200, {"ok": True})
        
        if self.path == "/web/feed":
            url = body.get("url")
            name = body.get("name")
            if url and name:
                self.explorer.add_feed(url, name)
                return self._json(200, {"ok": True})
            return self._json(400, {"error": "url and name required"})
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="WebExplorer service")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--cache-db", default="data/web_cache.db")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--soul-id", default="eve")
    args = parser.parse_args()
    
    explorer = WebExplorer(
        cache_db=args.cache_db,
        memory_endpoint=args.memory_endpoint,
        soul_id=args.soul_id,
    )
    
    ExplorerHandler.explorer = explorer
    server = HTTPServer(("0.0.0.0", args.port), ExplorerHandler)
    print(f"[OK] WebExplorer running on port {args.port}", flush=True)
    print(f"     Sources: HackerNews, Reddit, RSS feeds", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
