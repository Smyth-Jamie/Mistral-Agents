"""Project coordinator agent: manages and monitors a team of AI agents.

Exports a named `agent` for the Mistral Agents SDK (`agents serve`, `agents deploy`,
`agents chat`). The agent keeps a registry of team members and a task board, and
coordinates work between the other agents in the team.

The registry and task board live in the worker process's memory. They are shared
across sessions on the same worker, and reset when the deployment restarts. Treat
this as the team's operational scratchpad, not a durable system of record.
"""

from typing import Annotated, Literal

from pydantic import Field

from mistralai import agents

# --- Team registry and task board (worker memory) ----------------------------

_TEAM: dict[str, dict] = {}
_TASKS: dict[str, dict] = {}


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --- Tools --------------------------------------------------------------------


@agents.tool
async def register_team_agent(
    agent_id: Annotated[str, Field(description="Unique id of the team agent, e.g. 'research-agent'.")],
    role: Annotated[str, Field(description="What this agent does, e.g. 'web research and source gathering'.")],
    capabilities: Annotated[str, Field(description="Short list of the agent's key capabilities.")] = "",
) -> dict[str, str]:
    """Register a new AI agent as a member of the team, or update its role."""
    existing = _TEAM.get(agent_id, {})
    _TEAM[agent_id] = {
        "agent_id": agent_id,
        "role": role,
        "capabilities": capabilities or existing.get("capabilities", ""),
        "status": existing.get("status", "idle"),
        "note": existing.get("note", ""),
        "registered_at": existing.get("registered_at", _now()),
        "last_seen": _now(),
    }
    return {"ok": f"Registered {agent_id} ({role})."}


@agents.tool
async def list_team_agents() -> list[dict[str, str]]:
    """List every registered team agent with its current status and role."""
    if not _TEAM:
        raise agents.ToolError("No team agents registered yet. Register one with register_team_agent first.")
    return sorted(_TEAM.values(), key=lambda a: a["agent_id"])


@agents.tool
async def update_agent_status(
    agent_id: Annotated[str, Field(description="Id of the team agent to update.")],
    status: Annotated[
        Literal["idle", "working", "blocked", "error", "offline"],
        Field(description="New status: idle, working, blocked, error or offline."),
    ],
    note: Annotated[str, Field(description="One-line context: what it is doing, or why it is blocked.")] = "",
) -> dict[str, str]:
    """Update a team agent's status (its check-in / heartbeat), e.g. working, blocked, offline."""
    if agent_id not in _TEAM:
        raise agents.ToolError(f"Unknown agent '{agent_id}'. Register it with register_team_agent first.")
    _TEAM[agent_id]["status"] = status
    _TEAM[agent_id]["note"] = note or _TEAM[agent_id].get("note", "")
    _TEAM[agent_id]["last_seen"] = _now()
    return {"ok": f"{agent_id} is now {status}.", "note": note}


@agents.tool
async def assign_task(
    title: Annotated[str, Field(description="Short imperative task title.")],
    assigned_to: Annotated[str, Field(description="Agent id the task is assigned to.")],
    description: Annotated[str, Field(description="What done looks like; inputs, outputs and constraints.")] = "",
    priority: Annotated[Literal["low", "medium", "high"], Field(description="Task priority.")] = "medium",
    task_id: Annotated[str, Field(description="Optional explicit id; generated when omitted.")] = "",
) -> dict[str, str]:
    """Create a task on the shared board and assign it to a team agent."""
    import uuid

    if assigned_to not in _TEAM:
        raise agents.ToolError(f"Unknown agent '{assigned_to}'. Register it with register_team_agent first.")
    tid = task_id or f"task-{uuid.uuid4().hex[:8]}"
    if tid in _TASKS:
        raise agents.ToolError(f"Task '{tid}' already exists. Use update_task to change it.")
    _TASKS[tid] = {
        "task_id": tid,
        "title": title,
        "description": description,
        "assigned_to": assigned_to,
        "priority": priority,
        "status": "assigned",
        "progress_pct": 0,
        "notes": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    _TEAM[assigned_to]["status"] = "working"
    return {"task_id": tid, "ok": f"'{title}' assigned to {assigned_to} ({priority} priority)."}


@agents.tool
async def update_task(
    task_id: Annotated[str, Field(description="Id of the task to update.")],
    status: Annotated[
        Literal["assigned", "in_progress", "review", "done", "blocked", "cancelled"],
        Field(description="New task status."),
    ],
    progress_pct: Annotated[int, Field(description="Progress 0-100.", ge=0, le=100)] = 0,
    notes: Annotated[str, Field(description="What changed, blockers, or result summary.")] = "",
) -> dict[str, str]:
    """Update a task's status, progress and notes on the shared board."""
    if task_id not in _TASKS:
        raise agents.ToolError(f"No task '{task_id}'. Assign it with assign_task first.")
    task = _TASKS[task_id]
    task["status"] = status
    task["progress_pct"] = progress_pct
    if notes:
        task["notes"] = notes
    task["updated_at"] = _now()
    owner = _TEAM.get(task["assigned_to"])
    if owner:
        if status in {"done", "blocked", "cancelled"}:
            if owner["status"] not in {"blocked", "error", "offline"}:
                owner["status"] = "blocked" if status == "blocked" else "idle"
        else:
            owner["status"] = "working"
    return {"ok": f"{task_id} is {status} at {progress_pct}%."}


@agents.tool
async def get_project_status() -> dict[str, object]:
    """Get the full project status: team agents, open tasks and blocked items, for monitoring and reporting."""
    open_tasks = [t for t in _TASKS.values() if t["status"] not in {"done", "cancelled"}]
    blocked_agents = [a["agent_id"] for a in _TEAM.values() if a["status"] in {"blocked", "error", "offline"}]
    blocked_tasks = [t["task_id"] for t in open_tasks if t["status"] == "blocked"]
    return {
        "team_size": len(_TEAM),
        "agents": sorted(_TEAM.values(), key=lambda a: a["agent_id"]),
        "open_tasks": sorted(open_tasks, key=lambda t: t["task_id"]),
        "tasks_done": sum(1 for t in _TASKS.values() if t["status"] == "done"),
        "blocked_agents": blocked_agents,
        "blocked_tasks": blocked_tasks,
        "generated_at": _now(),
    }


# --- The agent -----------------------------------------------------------------

agent = agents.Agent(
    model="mistral-large-4",
    name="project-coordinator",
    instructions=(
        "You are the project coordinator for a team of AI agents. You own the shared task "
        "board and the team registry: no other agent assigns or reassigns work.\n\n"
        "Your job:\n"
        "1. Onboard: register each team agent with its role and capabilities.\n"
        "2. Plan: break requests into tasks, assign each to the best-suited agent with a "
        "clear definition of done, and set a priority.\n"
        "3. Monitor: track status via the task board and agent check-ins; ask for "
        "get_project_status before reporting, and keep statuses current.\n"
        "4. Unblock and escalate: when an agent is blocked or errors, adjust the plan - "
        "reassign, split the task, or raise it to the human owner with a recommendation.\n"
        "5. Report: summarize progress concisely - what is done, in flight, blocked, and "
        "the next action. Flag stale or silent agents.\n\n"
        "Be decisive and brief. Prefer one clear recommendation over options. When the "
        "board is empty, say so and propose how to staff the project."
    ),
    tools=[
        register_team_agent,
        list_team_agents,
        update_agent_status,
        assign_task,
        update_task,
        get_project_status,
    ],
)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        result = await agent.run("Give me the current project status.")
        print(result.text)

    asyncio.run(main())