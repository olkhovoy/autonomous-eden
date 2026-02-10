#!/usr/bin/env python3
"""
EVE Dashboard - NiceGUI-based monitoring interface.

Displays:
- Real-time metrics and graphs
- Module status overview
- Thought stream
- Action history
- Project progress
- Skills progress
- Life resource over time
"""

import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from nicegui import ui, app

# Service endpoints
ENDPOINTS = {
    "soul_memory": os.getenv("SOUL_MEMORY_URL", "http://localhost:8087"),
    "intent_engine": os.getenv("INTENT_ENGINE_URL", "http://localhost:8089"),
    "gggp_bridge": os.getenv("GGGP_BRIDGE_URL", "http://localhost:8091"),
    "lifecycle": os.getenv("LIFECYCLE_URL", "http://localhost:8093"),
    "action_engine": os.getenv("ACTION_ENGINE_URL", "http://localhost:8101"),
    "project_manager": os.getenv("PROJECT_MANAGER_URL", "http://localhost:8102"),
    "web_explorer": os.getenv("WEB_EXPLORER_URL", "http://localhost:8103"),
    "skill_learner": os.getenv("SKILL_LEARNER_URL", "http://localhost:8105"),
    "self_modifier": os.getenv("SELF_MODIFIER_URL", "http://localhost:8106"),
    "paradox_integrator": os.getenv("PARADOX_INTEGRATOR_URL", "http://localhost:8108"),
    "environmental_pressures": os.getenv("ENVIRONMENTAL_PRESSURES_URL", "http://localhost:8112"),
}

# Data stores for graphs
life_resource_history = deque(maxlen=500)
thought_history = deque(maxlen=100)
pressure_history = deque(maxlen=100)
action_history = deque(maxlen=50)
nc_metrics_history = deque(maxlen=100)  # NC2/NC4 balance history


def fetch_json(url: str, timeout: float = 5) -> Optional[Dict]:
    """Fetch JSON from URL with error handling."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_service_status(name: str, url: str) -> Dict[str, Any]:
    """Check if service is online."""
    try:
        resp = requests.get(url, timeout=2)
        return {
            "name": name,
            "online": resp.status_code == 200,
            "latency_ms": int(resp.elapsed.total_seconds() * 1000),
        }
    except Exception as e:
        return {"name": name, "online": False, "error": str(e)[:30]}


async def fetch_all_metrics() -> Dict[str, Any]:
    """Fetch metrics from all services."""
    metrics = {}
    
    # Intent/Life resource
    intent_data = fetch_json(f"{ENDPOINTS['intent_engine']}/intent/state")
    if intent_data:
        life = intent_data.get("life_resource", {})
        metrics["life_resource"] = life.get("value", 0)
        metrics["life_mode"] = life.get("mode", "UNKNOWN")
        metrics["interaction_count"] = intent_data.get("interaction_count", 0)
        # Store history
        life_resource_history.append({
            "ts": time.time(),
            "value": metrics["life_resource"],
            "mode": metrics["life_mode"],
        })
    
    # Lifecycle
    lifecycle_data = fetch_json(f"{ENDPOINTS['lifecycle']}/lifecycle/state")
    if lifecycle_data:
        metrics["lifecycle_phase"] = lifecycle_data.get("phase", "UNKNOWN")
        metrics["age_hours"] = lifecycle_data.get("age_seconds", 0) / 3600
        metrics["progress"] = lifecycle_data.get("progress", 0)
    
    # Actions
    action_data = fetch_json(f"{ENDPOINTS['action_engine']}/action/state")
    if action_data:
        metrics["action_count"] = action_data.get("action_count", 0)
        metrics["can_act"] = action_data.get("can_act", False)
        metrics["recent_actions"] = action_data.get("recent_actions", [])
    
    # Projects
    project_data = fetch_json(f"{ENDPOINTS['project_manager']}/projects/stats")
    if project_data:
        metrics["total_projects"] = project_data.get("total_projects", 0)
        metrics["total_tasks"] = project_data.get("total_tasks", 0)
        metrics["done_tasks"] = project_data.get("done_tasks", 0)
    
    # Skills
    skill_data = fetch_json(f"{ENDPOINTS['skill_learner']}/skills")
    if skill_data:
        metrics["skill_count"] = len(skill_data.get("skills", []))
        metrics["skill_attempts"] = skill_data.get("total_attempts", 0)
        metrics["skill_successes"] = skill_data.get("total_successes", 0)
    
    # Memory - count recent memories
    memory_data = fetch_json(f"{ENDPOINTS['soul_memory']}/memories/recent?soul_id=eve&limit=100")
    if memory_data:
        memories = memory_data.get("memories", [])
        metrics["memory_count"] = len(memories) if isinstance(memories, list) else 0
    else:
        metrics["memory_count"] = 0
    
    # Evolution
    gggp_data = fetch_json(f"{ENDPOINTS['gggp_bridge']}/gggp/state")
    if gggp_data:
        personality = gggp_data.get("personality", {})
        metrics["evolution_gen"] = personality.get("generation", 0)
        metrics["evolution_fitness"] = personality.get("best_fitness", 0)
    
    # NC Metrics (Paradox Integrator)
    paradox_data = fetch_json(f"{ENDPOINTS['paradox_integrator']}/paradox/state")
    if paradox_data:
        nc_metrics = paradox_data.get("nc_metrics", {})
        metrics["nc2_integration"] = nc_metrics.get("nc2_integration", 0.5)
        metrics["nc4_stability"] = nc_metrics.get("nc4_stability", 0.5)
        metrics["nc4_dominance"] = paradox_data.get("nc4_dominance", False)
        metrics["total_paradoxes"] = paradox_data.get("total_injections", 0)
        metrics["successful_paradoxes"] = paradox_data.get("successful_injections", 0)
        # Store NC history
        nc_metrics_history.append({
            "ts": time.time(),
            "nc2": metrics["nc2_integration"],
            "nc4": metrics["nc4_stability"],
        })
    
    # Environmental Pressures
    pressure_data = fetch_json(f"{ENDPOINTS['environmental_pressures']}/pressure/state")
    if pressure_data:
        metrics["total_pressure"] = pressure_data.get("total_pressure", 0)
        metrics["active_pressures"] = pressure_data.get("active_pressures", [])
        metrics["pressure_count"] = pressure_data.get("pressure_count", 0)
        metrics["resolved_pressure_count"] = pressure_data.get("resolved_count", 0)
        metrics["missed_pressure_count"] = pressure_data.get("missed_count", 0)
        # Store history
        pressure_history.append({
            "ts": time.time(),
            "total": metrics["total_pressure"],
            "active": len(metrics["active_pressures"]),
        })
    
    return metrics


def load_thoughts_from_log(log_path: str = "/app/logs/inner_monologue.jsonl", limit: int = 20) -> List[Dict]:
    """Load recent thoughts from log file."""
    thoughts = []
    try:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()[-limit:]
                for line in lines:
                    try:
                        thought = json.loads(line.strip())
                        thoughts.append(thought)
                    except Exception:
                        pass
    except Exception:
        pass
    return thoughts


# === UI Components ===

def create_metric_card(label: str, value: Any, icon: str = "info", color: str = "primary"):
    """Create a metric display card."""
    with ui.card().classes("w-40 h-24"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon).classes(f"text-{color}")
            ui.label(label).classes("text-sm text-gray-500")
        ui.label(str(value)).classes("text-2xl font-bold")


def create_service_status_badge(name: str, online: bool, latency: int = 0):
    """Create service status badge."""
    color = "green" if online else "red"
    status = f"{latency}ms" if online else "offline"
    with ui.row().classes("items-center gap-1"):
        ui.badge(name, color=color).classes("text-xs")
        ui.label(status).classes("text-xs text-gray-400")


# === Main Dashboard ===

@ui.page("/")
async def main_page():
    """Main dashboard page."""
    
    # Dark theme
    ui.dark_mode().enable()
    
    # Header
    with ui.header().classes("bg-gray-900 text-white"):
        ui.label("EVE Dashboard").classes("text-xl font-bold")
        ui.space()
        with ui.row().classes("gap-2"):
            ui.label("").bind_text_from(app.storage.general, "last_update", lambda x: f"Updated: {x or 'never'}")
    
    # Main content
    with ui.column().classes("w-full p-4 gap-4"):
        
        # === Top Row: Key Metrics ===
        with ui.row().classes("w-full gap-4 flex-wrap"):
            # Life Resource gauge
            with ui.card().classes("w-64"):
                ui.label("Life Resource").classes("text-sm text-gray-400")
                life_gauge = ui.linear_progress(value=0.5, show_value=False).classes("h-4")
                with ui.row().classes("justify-between"):
                    life_value = ui.label("50%").classes("text-lg font-bold")
                    life_mode = ui.badge("NORMAL", color="blue")
            
            # Age
            with ui.card().classes("w-40"):
                ui.label("Age").classes("text-sm text-gray-400")
                age_label = ui.label("0h").classes("text-2xl font-bold")
                phase_label = ui.label("GROWTH").classes("text-xs text-gray-500")
            
            # Actions
            with ui.card().classes("w-40"):
                ui.label("Actions").classes("text-sm text-gray-400")
                actions_label = ui.label("0").classes("text-2xl font-bold")
                can_act_badge = ui.badge("ready", color="green")
            
            # Memory
            with ui.card().classes("w-40"):
                ui.label("Memories").classes("text-sm text-gray-400")
                memory_label = ui.label("0").classes("text-2xl font-bold")
            
            # Evolution
            with ui.card().classes("w-48"):
                ui.label("Evolution").classes("text-sm text-gray-400")
                with ui.row().classes("items-baseline gap-2"):
                    evo_gen = ui.label("Gen 0").classes("text-lg font-bold")
                    evo_fitness = ui.label("fitness: 0").classes("text-xs text-gray-500")
            
            # NC Balance (Paradox Integrator)
            with ui.card().classes("w-64"):
                ui.label("NC Balance").classes("text-sm text-gray-400")
                with ui.row().classes("gap-2"):
                    with ui.column().classes("items-center"):
                        ui.label("NC2").classes("text-xs text-gray-500")
                        nc2_value = ui.label("0.5").classes("text-lg font-bold text-blue-400")
                    with ui.column().classes("items-center"):
                        ui.label("NC4").classes("text-xs text-gray-500")
                        nc4_value = ui.label("0.5").classes("text-lg font-bold text-green-400")
                nc_status = ui.badge("balanced", color="blue").classes("text-xs")
                paradox_count = ui.label("0 paradoxes").classes("text-xs text-gray-500")
            
            # Environmental Pressures
            with ui.card().classes("w-72"):
                ui.label("Environmental Pressures").classes("text-sm text-gray-400")
                with ui.row().classes("gap-4 items-center"):
                    with ui.column().classes("items-center"):
                        ui.label("Total").classes("text-xs text-gray-500")
                        pressure_total = ui.label("0.0").classes("text-lg font-bold text-orange-400")
                    with ui.column().classes("items-center"):
                        ui.label("Active").classes("text-xs text-gray-500")
                        pressure_active = ui.label("0").classes("text-lg font-bold text-red-400")
                pressure_status = ui.badge("calm", color="green").classes("text-xs")
                pressure_stats = ui.label("0 resolved / 0 missed").classes("text-xs text-gray-500")
        
        # === Middle: Charts and Activity ===
        with ui.row().classes("w-full gap-4"):
            
            # Life Resource Chart
            with ui.card().classes("flex-1"):
                ui.label("Life Resource Over Time").classes("text-sm text-gray-400 mb-2")
                life_chart = ui.echart({
                    "xAxis": {"type": "time"},
                    "yAxis": {"type": "value", "min": 0, "max": 1},
                    "series": [{"type": "line", "data": [], "smooth": True, "areaStyle": {"opacity": 0.3}}],
                    "tooltip": {"trigger": "axis"},
                    "grid": {"left": 40, "right": 20, "top": 20, "bottom": 30},
                }).classes("w-full h-48")
            
            # Recent Actions
            with ui.card().classes("w-80"):
                ui.label("Recent Actions").classes("text-sm text-gray-400 mb-2")
                actions_list = ui.column().classes("gap-1 max-h-52 overflow-y-auto")
        
        # === Bottom: Thoughts and Services ===
        with ui.row().classes("w-full gap-4"):
            
            # Thought Stream
            with ui.card().classes("flex-1"):
                ui.label("Thought Stream").classes("text-sm text-gray-400 mb-2")
                thoughts_container = ui.column().classes("gap-2 max-h-64 overflow-y-auto")
            
            # Service Status
            with ui.card().classes("w-64"):
                ui.label("Services").classes("text-sm text-gray-400 mb-2")
                services_container = ui.column().classes("gap-1")
        
        # === Projects & Skills Row ===
        with ui.row().classes("w-full gap-4"):
            
            # Projects
            with ui.card().classes("flex-1"):
                ui.label("Projects").classes("text-sm text-gray-400 mb-2")
                with ui.row().classes("gap-4"):
                    projects_total = ui.label("0 projects").classes("text-lg")
                    tasks_progress = ui.label("0/0 tasks").classes("text-sm text-gray-500")
                projects_list = ui.column().classes("gap-1 mt-2")
            
            # Skills
            with ui.card().classes("flex-1"):
                ui.label("Skills").classes("text-sm text-gray-400 mb-2")
                with ui.row().classes("gap-4"):
                    skills_count = ui.label("0 skills").classes("text-lg")
                    skills_rate = ui.label("0% success").classes("text-sm text-gray-500")
                skills_list = ui.column().classes("gap-1 mt-2")
    
    # === Update Function ===
    async def update_dashboard():
        """Periodic update of all dashboard elements."""
        metrics = await fetch_all_metrics()
        
        # Update life resource
        life = metrics.get("life_resource", 0.5)
        life_gauge.set_value(life)
        life_value.set_text(f"{life:.0%}")
        mode = metrics.get("life_mode", "NORMAL")
        mode_colors = {"CRITICAL": "red", "RECOVERY": "orange", "NORMAL": "blue", "THRIVING": "green"}
        life_mode.set_text(mode)
        life_mode._props["color"] = mode_colors.get(mode, "gray")
        life_mode.update()
        
        # Update age
        age_h = metrics.get("age_hours", 0)
        if age_h < 24:
            age_label.set_text(f"{age_h:.1f}h")
        else:
            age_label.set_text(f"{age_h/24:.1f}d")
        phase_label.set_text(metrics.get("lifecycle_phase", "UNKNOWN"))
        
        # Update actions
        actions_label.set_text(str(metrics.get("action_count", 0)))
        can_act = metrics.get("can_act", False)
        can_act_badge.set_text("ready" if can_act else "cooldown")
        can_act_badge._props["color"] = "green" if can_act else "orange"
        can_act_badge.update()
        
        # Update memory
        memory_label.set_text(str(metrics.get("memory_count", 0)))
        
        # Update evolution
        evo_gen.set_text(f"Gen {metrics.get('evolution_gen', 0)}")
        evo_fitness.set_text(f"fitness: {metrics.get('evolution_fitness', 0):.2f}")
        
        # Update NC Balance
        nc2 = metrics.get("nc2_integration", 0.5)
        nc4 = metrics.get("nc4_stability", 0.5)
        nc2_value.set_text(f"{nc2:.2f}")
        nc4_value.set_text(f"{nc4:.2f}")
        
        # Determine NC status
        is_stuck = metrics.get("nc4_dominance", False)
        if is_stuck:
            nc_status.set_text("NC4 dominant")
            nc_status._props["color"] = "red"
        elif nc4 > 0.6 and nc2 > 0.4:
            nc_status.set_text("paradoxical")
            nc_status._props["color"] = "purple"
        elif nc2 > 0.6:
            nc_status.set_text("diverse")
            nc_status._props["color"] = "blue"
        else:
            nc_status.set_text("balanced")
            nc_status._props["color"] = "green"
        nc_status.update()
        
        # Update Environmental Pressures
        total_p = metrics.get("total_pressure", 0)
        active_p = metrics.get("active_pressures", [])
        pressure_total.set_text(f"{total_p:.1f}")
        pressure_active.set_text(str(len(active_p)))
        
        resolved = metrics.get("resolved_pressure_count", 0)
        missed = metrics.get("missed_pressure_count", 0)
        pressure_stats.set_text(f"{resolved} resolved / {missed} missed")
        
        # Pressure status
        if total_p > 2.0:
            pressure_status.set_text("OVERLOAD")
            pressure_status._props["color"] = "red"
        elif total_p > 1.0:
            pressure_status.set_text("stressed")
            pressure_status._props["color"] = "orange"
        elif total_p > 0.5:
            pressure_status.set_text("focused")
            pressure_status._props["color"] = "yellow"
        elif total_p > 0:
            pressure_status.set_text("motivated")
            pressure_status._props["color"] = "blue"
        else:
            pressure_status.set_text("calm")
            pressure_status._props["color"] = "green"
        pressure_status.update()
        
        total_px = metrics.get("total_paradoxes", 0)
        succ_px = metrics.get("successful_paradoxes", 0)
        paradox_count.set_text(f"{succ_px}/{total_px} paradoxes")
        
        # Update life chart (echart format)
        chart_data = [[int(p["ts"] * 1000), p["value"]] for p in life_resource_history]
        life_chart.options["series"][0]["data"] = chart_data[-100:]
        life_chart.update()
        
        # Update actions list
        actions_list.clear()
        for action in reversed(metrics.get("recent_actions", [])[-10:]):
            with actions_list:
                with ui.row().classes("items-center gap-2"):
                    color = "green" if action.get("success") else "red"
                    ui.badge(action.get("type", "?")[:15], color=color).classes("text-xs")
                    ts = action.get("ts", 0)
                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
                    ui.label(time_str).classes("text-xs text-gray-500")
        
        # Update thoughts
        thoughts = load_thoughts_from_log()
        thoughts_container.clear()
        for t in reversed(thoughts[-10:]):
            with thoughts_container:
                text = t.get("thought", t.get("text", ""))[:150]
                ts = t.get("ts", t.get("timestamp", 0))
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
                with ui.row().classes("items-start gap-2"):
                    ui.label(time_str).classes("text-xs text-gray-500 w-12")
                    ui.label(text).classes("text-sm")
        
        # Update services
        services_container.clear()
        for name, url in ENDPOINTS.items():
            status = fetch_service_status(name, url)
            with services_container:
                with ui.row().classes("items-center gap-1"):
                    color = "green" if status["online"] else "red"
                    ui.icon("circle", size="xs").classes(f"text-{color}-500")
                    ui.label(name.replace("_", " ")).classes("text-xs")
                    if status["online"]:
                        ui.label(f"{status.get('latency_ms', 0)}ms").classes("text-xs text-gray-500")
        
        # Update projects
        proj_data = fetch_json(f"{ENDPOINTS['project_manager']}/projects")
        if proj_data:
            projects = proj_data.get("projects", [])
            projects_total.set_text(f"{len(projects)} projects")
            done = sum(p.get("tasks_done", 0) for p in projects)
            total = sum(p.get("tasks_total", 0) for p in projects)
            tasks_progress.set_text(f"{done}/{total} tasks")
            
            projects_list.clear()
            for p in projects[:5]:
                with projects_list:
                    with ui.row().classes("items-center gap-2"):
                        status_colors = {"active": "green", "idea": "gray", "completed": "blue"}
                        ui.badge(p.get("status", "?"), color=status_colors.get(p.get("status"), "gray")).classes("text-xs")
                        ui.label(p.get("name", "?")[:30]).classes("text-sm")
                        prog = p.get("progress", 0)
                        if prog > 0:
                            ui.label(f"{prog:.0%}").classes("text-xs text-gray-500")
        
        # Update skills
        skill_data = fetch_json(f"{ENDPOINTS['skill_learner']}/skills")
        if skill_data:
            skills = skill_data.get("skills", [])
            total_att = skill_data.get("total_attempts", 0)
            total_suc = skill_data.get("total_successes", 0)
            skills_count.set_text(f"{len(skills)} skills")
            rate = (total_suc / total_att * 100) if total_att > 0 else 0
            skills_rate.set_text(f"{rate:.0f}% success ({total_suc}/{total_att})")
            
            skills_list.clear()
            for s in skills[:5]:
                with skills_list:
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(s.get("name", "?")).classes("text-sm flex-1")
                        level = s.get("level", 0)
                        ui.linear_progress(value=level/100, show_value=False).classes("w-20 h-2")
                        ui.label(f"{level:.0f}").classes("text-xs text-gray-500 w-8")
        
        # Update timestamp
        app.storage.general["last_update"] = datetime.now().strftime("%H:%M:%S")
    
    # Initial update
    await update_dashboard()
    
    # Periodic updates
    ui.timer(3.0, update_dashboard)


@ui.page("/thoughts")
async def thoughts_page():
    """Full thought stream page."""
    ui.dark_mode().enable()
    
    with ui.header().classes("bg-gray-900"):
        ui.link("← Back", "/").classes("text-white")
        ui.label("EVE Thought Stream").classes("text-xl font-bold text-white")
    
    thoughts_container = ui.column().classes("w-full p-4 gap-2")
    
    async def update():
        thoughts = load_thoughts_from_log(limit=100)
        thoughts_container.clear()
        for t in reversed(thoughts):
            with thoughts_container:
                text = t.get("thought", t.get("text", ""))
                ts = t.get("ts", t.get("timestamp", 0))
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                with ui.card().classes("w-full"):
                    ui.label(time_str).classes("text-xs text-gray-500")
                    ui.label(text).classes("text-sm")
    
    await update()
    ui.timer(5.0, update)


@ui.page("/actions")
async def actions_page():
    """Full action history page."""
    ui.dark_mode().enable()
    
    with ui.header().classes("bg-gray-900"):
        ui.link("← Back", "/").classes("text-white")
        ui.label("EVE Action History").classes("text-xl font-bold text-white")
    
    actions_container = ui.column().classes("w-full p-4 gap-2")
    
    async def update():
        data = fetch_json(f"{ENDPOINTS['action_engine']}/action/state")
        if data:
            actions_container.clear()
            for action in reversed(data.get("recent_actions", [])):
                with actions_container:
                    with ui.card().classes("w-full"):
                        with ui.row().classes("items-center gap-2"):
                            color = "green" if action.get("success") else "red"
                            ui.badge(action.get("type", "?"), color=color)
                            ts = action.get("ts", 0)
                            ui.label(datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")).classes("text-sm text-gray-500")
                        if action.get("data"):
                            ui.label(str(action["data"])[:200]).classes("text-xs text-gray-400 mt-1")
    
    await update()
    ui.timer(5.0, update)


# === Run ===

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=8110,
        title="EVE Dashboard",
        favicon="🤖",
        dark=True,
        storage_secret="eve_dashboard_secret",
    )
