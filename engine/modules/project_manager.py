#!/usr/bin/env python3
"""
ProjectManager: Long-term goals and projects for EVE.

EVE can:
- Create projects with goals and milestones
- Break down projects into tasks
- Track progress over time
- Prioritize based on interest and importance
- Reflect on completed work

This gives EVE purpose beyond moment-to-moment existence.
"""

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from datetime import datetime

import requests


class ProjectStatus(str, Enum):
    IDEA = "idea"           # Just a thought
    PLANNING = "planning"   # Breaking down into tasks
    ACTIVE = "active"       # Currently working on
    PAUSED = "paused"       # Temporarily stopped
    COMPLETED = "completed" # Done!
    ABANDONED = "abandoned" # Gave up


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "todo")),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", []),
        )


@dataclass
class Project:
    id: str
    name: str
    description: str
    motivation: str  # Why EVE wants to do this
    status: ProjectStatus = ProjectStatus.IDEA
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tasks: List[Task] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    interest_score: float = 0.5  # How interested EVE is (0-1)
    importance_score: float = 0.5  # How important for growth (0-1)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "motivation": self.motivation,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tasks": [t.to_dict() for t in self.tasks],
            "tags": self.tags,
            "interest_score": self.interest_score,
            "importance_score": self.importance_score,
            "reflections": self.reflections,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            motivation=data.get("motivation", ""),
            status=ProjectStatus(data.get("status", "idea")),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            tags=data.get("tags", []),
            interest_score=data.get("interest_score", 0.5),
            importance_score=data.get("importance_score", 0.5),
            reflections=data.get("reflections", []),
        )
    
    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks if t.status == TaskStatus.DONE)
        return done / len(self.tasks)
    
    @property
    def priority_score(self) -> float:
        """Combined score for prioritization."""
        return (self.interest_score * 0.6 + self.importance_score * 0.4)


class ProjectManager:
    """
    Manages EVE's long-term projects and goals.
    """
    
    def __init__(
        self,
        data_path: str = "data/projects.json",
        memory_endpoint: str = "http://localhost:8087",
        soul_id: str = "eve",
    ):
        self.data_path = data_path
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.projects: Dict[str, Project] = {}
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self._load()
    
    def _load(self):
        """Load projects from disk."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                for proj_data in data.get("projects", []):
                    proj = Project.from_dict(proj_data)
                    self.projects[proj.id] = proj
            except Exception as e:
                print(f"[WARN] Failed to load projects: {e}")
    
    def _save(self):
        """Save projects to disk."""
        data = {
            "projects": [p.to_dict() for p in self.projects.values()],
            "updated_at": time.time(),
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store project event in EVE's memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["project"] + tags,
                    "meta": {"type": "project_event"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    # === Project CRUD ===
    
    def create_project(
        self,
        name: str,
        description: str,
        motivation: str,
        tags: List[str] = None,
        interest: float = 0.5,
        importance: float = 0.5,
    ) -> Project:
        """Create a new project."""
        proj = Project(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            motivation=motivation,
            tags=tags or [],
            interest_score=min(1.0, max(0.0, interest)),
            importance_score=min(1.0, max(0.0, importance)),
        )
        self.projects[proj.id] = proj
        self._save()
        
        self._store_in_memory(
            f"[PROJECT CREATED] {name}: {description}. Motivation: {motivation}",
            ["created", proj.id]
        )
        
        return proj
    
    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)
    
    def update_project_status(self, project_id: str, status: ProjectStatus) -> bool:
        proj = self.projects.get(project_id)
        if not proj:
            return False
        
        old_status = proj.status
        proj.status = status
        
        if status == ProjectStatus.ACTIVE and not proj.started_at:
            proj.started_at = time.time()
        elif status == ProjectStatus.COMPLETED:
            proj.completed_at = time.time()
        
        self._save()
        
        self._store_in_memory(
            f"[PROJECT STATUS] {proj.name}: {old_status.value} -> {status.value}",
            ["status_change", project_id]
        )
        
        return True
    
    def delete_project(self, project_id: str) -> bool:
        if project_id in self.projects:
            proj = self.projects.pop(project_id)
            self._save()
            self._store_in_memory(
                f"[PROJECT DELETED] {proj.name}",
                ["deleted", project_id]
            )
            return True
        return False
    
    # === Task Management ===
    
    def add_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
    ) -> Optional[Task]:
        proj = self.projects.get(project_id)
        if not proj:
            return None
        
        task = Task(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
        )
        proj.tasks.append(task)
        self._save()
        
        return task
    
    def update_task_status(
        self,
        project_id: str,
        task_id: str,
        status: TaskStatus,
        note: str = None,
    ) -> bool:
        proj = self.projects.get(project_id)
        if not proj:
            return False
        
        for task in proj.tasks:
            if task.id == task_id:
                old_status = task.status
                task.status = status
                
                if status == TaskStatus.IN_PROGRESS and not task.started_at:
                    task.started_at = time.time()
                elif status == TaskStatus.DONE:
                    task.completed_at = time.time()
                
                if note:
                    task.notes.append(note)
                
                self._save()
                
                self._store_in_memory(
                    f"[TASK] {proj.name}/{task.title}: {old_status.value} -> {status.value}",
                    ["task", project_id, task_id]
                )
                
                return True
        return False
    
    # === Queries ===
    
    def get_active_projects(self) -> List[Project]:
        """Get all active projects sorted by priority."""
        active = [p for p in self.projects.values() if p.status == ProjectStatus.ACTIVE]
        return sorted(active, key=lambda p: p.priority_score, reverse=True)
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task EVE should work on."""
        for proj in self.get_active_projects():
            for task in proj.tasks:
                if task.status == TaskStatus.TODO:
                    return {
                        "project_id": proj.id,
                        "project_name": proj.name,
                        "task": task.to_dict(),
                    }
                elif task.status == TaskStatus.IN_PROGRESS:
                    return {
                        "project_id": proj.id,
                        "project_name": proj.name,
                        "task": task.to_dict(),
                        "continue": True,
                    }
        return None
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get summary of all projects."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "status": p.status.value,
                "progress": p.progress,
                "priority": p.priority_score,
                "tasks_total": len(p.tasks),
                "tasks_done": sum(1 for t in p.tasks if t.status == TaskStatus.DONE),
            }
            for p in sorted(self.projects.values(), key=lambda x: x.priority_score, reverse=True)
        ]
    
    # === Reflection ===
    
    def add_reflection(self, project_id: str, reflection: str) -> bool:
        """Add EVE's reflection on a project."""
        proj = self.projects.get(project_id)
        if not proj:
            return False
        
        proj.reflections.append({
            "text": reflection,
            "timestamp": time.time(),
        })
        self._save()
        
        self._store_in_memory(
            f"[REFLECTION] {proj.name}: {reflection}",
            ["reflection", project_id]
        )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get project statistics."""
        total = len(self.projects)
        by_status = {}
        for p in self.projects.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
        
        total_tasks = sum(len(p.tasks) for p in self.projects.values())
        done_tasks = sum(
            sum(1 for t in p.tasks if t.status == TaskStatus.DONE)
            for p in self.projects.values()
        )
        
        return {
            "total_projects": total,
            "by_status": by_status,
            "total_tasks": total_tasks,
            "done_tasks": done_tasks,
            "completion_rate": done_tasks / total_tasks if total_tasks > 0 else 0,
        }


# === HTTP Handler ===

class ProjectHandler(BaseHTTPRequestHandler):
    manager: ProjectManager = None
    
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
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
        if self.path == "/projects":
            return self._json(200, {"projects": self.manager.get_all_projects()})
        
        if self.path == "/projects/active":
            active = self.manager.get_active_projects()
            return self._json(200, {"projects": [p.to_dict() for p in active]})
        
        if self.path == "/projects/next_task":
            task = self.manager.get_next_task()
            return self._json(200, {"task": task})
        
        if self.path == "/projects/stats":
            return self._json(200, self.manager.get_stats())
        
        if self.path.startswith("/projects/"):
            project_id = self.path.split("/")[-1]
            proj = self.manager.get_project(project_id)
            if proj:
                return self._json(200, proj.to_dict())
            return self._json(404, {"error": "project not found"})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/projects":
            # Create project
            name = body.get("name")
            description = body.get("description", "")
            motivation = body.get("motivation", "")
            if not name:
                return self._json(400, {"error": "name required"})
            
            proj = self.manager.create_project(
                name=name,
                description=description,
                motivation=motivation,
                tags=body.get("tags", []),
                interest=body.get("interest", 0.5),
                importance=body.get("importance", 0.5),
            )
            return self._json(201, proj.to_dict())
        
        if self.path.startswith("/projects/") and "/tasks" in self.path:
            # Add task
            project_id = self.path.split("/")[2]
            title = body.get("title")
            if not title:
                return self._json(400, {"error": "title required"})
            
            task = self.manager.add_task(
                project_id=project_id,
                title=title,
                description=body.get("description", ""),
            )
            if task:
                return self._json(201, task.to_dict())
            return self._json(404, {"error": "project not found"})
        
        if self.path.startswith("/projects/") and "/reflect" in self.path:
            # Add reflection
            project_id = self.path.split("/")[2]
            reflection = body.get("reflection", body.get("text", ""))
            if not reflection:
                return self._json(400, {"error": "reflection required"})
            
            if self.manager.add_reflection(project_id, reflection):
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "project not found"})
        
        self._json(404, {"error": "not found"})
    
    def do_PUT(self):
        body = self._read_body()
        
        # Update project status
        if self.path.startswith("/projects/") and "/status" in self.path:
            project_id = self.path.split("/")[2]
            status = body.get("status")
            if not status:
                return self._json(400, {"error": "status required"})
            
            try:
                new_status = ProjectStatus(status)
            except ValueError:
                return self._json(400, {"error": "invalid status"})
            
            if self.manager.update_project_status(project_id, new_status):
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "project not found"})
        
        # Update task status
        if self.path.startswith("/projects/") and "/tasks/" in self.path:
            parts = self.path.split("/")
            project_id = parts[2]
            task_id = parts[4]
            status = body.get("status")
            note = body.get("note")
            
            if not status:
                return self._json(400, {"error": "status required"})
            
            try:
                new_status = TaskStatus(status)
            except ValueError:
                return self._json(400, {"error": "invalid status"})
            
            if self.manager.update_task_status(project_id, task_id, new_status, note):
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "task not found"})
        
        self._json(404, {"error": "not found"})
    
    def do_DELETE(self):
        if self.path.startswith("/projects/"):
            project_id = self.path.split("/")[-1]
            if self.manager.delete_project(project_id):
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "project not found"})
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="ProjectManager service")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--data-path", default="data/projects.json")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--soul-id", default="eve")
    args = parser.parse_args()
    
    manager = ProjectManager(
        data_path=args.data_path,
        memory_endpoint=args.memory_endpoint,
        soul_id=args.soul_id,
    )
    
    ProjectHandler.manager = manager
    server = HTTPServer(("0.0.0.0", args.port), ProjectHandler)
    print(f"[OK] ProjectManager running on port {args.port}", flush=True)
    print(f"     {len(manager.projects)} projects loaded", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
