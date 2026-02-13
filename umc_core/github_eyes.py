#!/usr/bin/env python3
"""GitHubEyes: lightweight GitHub watcher with cache."""

import argparse
import os
import re
import json
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional

import requests


CACHE_TTL = 3600


class GitHubCache:
    def __init__(self, db_path: str = "data/github_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts INTEGER, response TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS watched (owner TEXT, repo TEXT, PRIMARY KEY(owner, repo))")
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
        cur.execute("REPLACE INTO cache(key, ts, response) VALUES(?,?,?)", (key, int(time.time()), json.dumps(value)))
        self.conn.commit()

    def watch(self, owner: str, repo: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO watched(owner, repo) VALUES(?,?)", (owner, repo))
        self.conn.commit()
        return True

    def watched(self) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT owner, repo FROM watched")
        return [{"owner": r[0], "repo": r[1]} for r in cur.fetchall()]


class GitHubEyes:
    def __init__(self, cache_db: str = "data/github_cache.db"):
        self.token = os.getenv("GITHUB_TOKEN", "")
        self.cache = GitHubCache(cache_db)

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get_trending(self, language: str = None, period: str = "daily") -> List[Dict[str, Any]]:
        lang = f"/{language}" if language else ""
        url = f"https://github.com/trending{lang}?since={period}"
        cache_key = f"trending:{language}:{period}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("items", [])
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        html = resp.text
        items = []
        for m in re.finditer(r"href=\"/([^/]+)/([^/]+)\"[^>]*>\s*<h2", html):
            owner, repo = m.group(1), m.group(2)
            items.append({"owner": owner, "repo": repo, "url": f"https://github.com/{owner}/{repo}"})
        self.cache.set(cache_key, {"items": items})
        return items

    def search_repos(self, query: str, sort: str = "stars") -> List[Dict[str, Any]]:
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": sort}
        cache_key = f"search:{query}:{sort}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("items", [])
        resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, data)
        return data.get("items", [])

    def get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        cache_key = f"repo:{owner}/{repo}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, data)
        return data

    def get_repo_commits(self, owner: str, repo: str, limit: int = 10) -> List[Dict[str, Any]]:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": limit}
        cache_key = f"commits:{owner}/{repo}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("items", [])
        resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, {"items": data})
        return data

    def watch_repo(self, owner: str, repo: str) -> bool:
        return self.cache.watch(owner, repo)

    def get_watched_updates(self) -> List[Dict[str, Any]]:
        updates = []
        for item in self.cache.watched():
            commits = self.get_repo_commits(item["owner"], item["repo"], limit=3)
            updates.append({"repo": f"{item['owner']}/{item['repo']}", "commits": commits})
        return updates


class GitHubEyesHandler(BaseHTTPRequestHandler):
    eyes: GitHubEyes = None

    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path.startswith("/github/trending"):
                query = self._parse_query()
                lang = query.get("lang")
                period = query.get("period", "daily")
                items = self.eyes.get_trending(lang, period)
                return self._json(200, {"items": items})
            if self.path.startswith("/github/search"):
                query = self._parse_query()
                q = query.get("q", "")
                items = self.eyes.search_repos(q)
                return self._json(200, {"items": items})
            if self.path.startswith("/github/repo/"):
                parts = self.path.split("/")
                if len(parts) >= 5:
                    owner, repo = parts[3], parts[4]
                    data = self.eyes.get_repo_info(owner, repo)
                    return self._json(200, data)
            if self.path.startswith("/github/watched"):
                return self._json(200, {"updates": self.eyes.get_watched_updates()})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        if self.path == "/github/watch":
            owner = data.get("owner")
            repo = data.get("repo")
            if not owner or not repo:
                return self._json(400, {"error": "owner and repo required"})
            ok = self.eyes.watch_repo(owner, repo)
            return self._json(200, {"ok": ok})
        return self._json(404, {"error": "not found"})

    def _parse_query(self) -> Dict[str, str]:
        if "?" not in self.path:
            return {}
        query = self.path.split("?", 1)[1]
        out = {}
        for part in query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = v.replace("+", " ")
        return out


def main():
    parser = argparse.ArgumentParser(description="GitHubEyes service")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--db-path", type=str, default="data/github_cache.db")
    args = parser.parse_args()

    eyes = GitHubEyes(cache_db=args.db_path)
    GitHubEyesHandler.eyes = eyes
    server = HTTPServer(("0.0.0.0", args.port), GitHubEyesHandler)
    print(f"GitHubEyes listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
