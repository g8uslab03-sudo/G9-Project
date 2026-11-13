"""GCP Orchestrator: Flask API, LangGraph state graph, macro planner/dispatcher."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, jsonify, request

from shared.application.system import AutonomousSystem
from shared.contracts import (
    Checkpoint,
    CloudProvider,
    CreateJobCommand,
    EventType,
    Job,
    JobStatus,
    RestoreCheckpointCommand,
    SaveCheckpointCommand,
)
from shared.persistence import PersistenceManager


app = Flask(__name__)

SYSTEM: Optional[AutonomousSystem] = None
PERSISTENCE: Optional[PersistenceManager] = None


def init_system() -> AutonomousSystem:
    global SYSTEM, PERSISTENCE
    if SYSTEM is None:
        PERSISTENCE = PersistenceManager(base_path=os.getenv("DATA_PATH", "data"))
        SYSTEM = AutonomousSystem(persistence=PERSISTENCE)
    return SYSTEM


# ---------------------------------------------------------------------------
# LangGraph-style state graph (macro planner / dispatcher)
# ---------------------------------------------------------------------------

@dataclass
class GraphState:
    job_id: str
    step: str = "receive"
    status: str = "pending"
    understanding: Dict[str, Any] = field(default_factory=dict)
    sub_tasks: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    decision: Optional[dict] = None
    allocation: Optional[dict] = None
    audit: Optional[dict] = None
    shadow: List[str] = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    checkpoints: List[str] = field(default_factory=list)
    evaporated: bool = False


class MacroPlannerGraph:
    """LangGraph state graph driving the macro (GCP) planning/dispatch loop."""

    def __init__(self, system: AutonomousSystem):
        self.system = system

    def receive(self, state: GraphState) -> GraphState:
        state.step = "receive"
        state.status = "received"
        return state

    def understand(self, state: GraphState) -> GraphState:
        job = self._get_job(state.job_id)
        understanding = self.system.main_brain.understand_requirement(job.description)
        state.understanding = understanding
        state.step = "understand"
        return state

    def decompose(self, state: GraphState) -> GraphState:
        sub_tasks = self.system.main_brain.decompose_problem(state.understanding)
        state.sub_tasks = sub_tasks
        state.step = "decompose"
        return state

    def decide(self, state: GraphState) -> GraphState:
        job = self._get_job(state.job_id)
        decision = self.system.main_brain.decide(job, state.understanding, state.sub_tasks)
        state.decision = decision.__dict__
        state.skills = decision.skills
        state.step = "decide"
        self._save_checkpoint(state, "decide")
        return state

    def dispatch(self, state: GraphState) -> GraphState:
        job = self._get_job(state.job_id)
        decision = BrainDecision.from_dict(state.decision) if hasattr(BrainDecision, "from_dict") else None
        # Reconstruct minimal decision
        from shared.application.system import BrainDecision
        bd = BrainDecision(
            job_id=state.job_id,
            action=state.decision.get("action", "dispatch"),
            skills=state.skills,
            reasoning=state.decision.get("reasoning", ""),
            confidence=state.decision.get("confidence", 0.0),
        )
        allocation = self.system.assistant_brain.allocate_work(job, bd)
        state.allocation = allocation
        state.step = "dispatch"
        self._save_checkpoint(state, "dispatch")
        return state

    def audit_review(self, state: GraphState) -> GraphState:
        job = self._get_job(state.job_id)
        audit = self.system.audit_brain.approve(job, state.allocation or {})
        state.audit = audit.__dict__
        state.step = "audit"
        return state

    def shadow_review(self, state: GraphState) -> GraphState:
        from shared.application.system import BrainDecision
        bd = BrainDecision(
            job_id=state.job_id,
            action="shadow",
            skills=state.skills,
        )
        shadow = self.system.shadow_brain.challenge(bd, state.allocation or {})
        state.shadow = shadow
        state.step = "shadow"
        return state

    def evaporate(self, state: GraphState) -> GraphState:
        # One-hand-off: fully evaporate workspace before handing to Kaggle worker
        checkpoint = Checkpoint(
            job_id=state.job_id,
            step=state.step,
            state={
                "understanding": state.understanding,
                "sub_tasks": state.sub_tasks,
                "skills": state.skills,
                "decision": state.decision,
                "allocation": state.allocation,
            },
        )
        self.system.persistence.evaporator.evaporate(
            save_checkpoint=True,
            checkpoint_store=self.system.persistence.checkpoints,
            job_id=state.job_id,
            step=state.step,
        )
        state.evaporated = True
        state.step = "evaporated"
        self._save_checkpoint(state, "evaporate")
        return state

    def run(self, job: Job) -> GraphState:
        state = GraphState(job_id=str(job.id))
        state = self.receive(state)
        state = self.understand(state)
        state = self.decompose(state)
        state = self.decide(state)
        state = self.dispatch(state)
        state = self.audit_review(state)
        state = self.shadow_review(state)
        state = self.evaporate(state)
        state.status = "dispatched_to_kaggle"
        return state

    def _get_job(self, job_id: str) -> Job:
        # Reconstruct minimal job
        return Job(id=job_id, title="job", description="", status=JobStatus.PENDING)

    def _save_checkpoint(self, state: GraphState, step: str):
        checkpoint = Checkpoint(
            job_id=state.job_id,
            step=step,
            state={
                "step": state.step,
                "status": state.status,
                "understanding": state.understanding,
                "sub_tasks": state.sub_tasks,
                "skills": state.skills,
                "decision": state.decision,
                "allocation": state.allocation,
                "audit": state.audit,
                "shadow": state.shadow,
                "evaporated": state.evaporated,
            },
        )
        self.system.persistence.save_checkpoint(
            checkpoint,
            {
                "step": state.step,
                "status": state.status,
                "understanding": state.understanding,
                "sub_tasks": state.sub_tasks,
                "skills": state.skills,
                "decision": state.decision,
                "allocation": state.allocation,
                "audit": state.audit,
                "shadow": state.shadow,
                "evaporated": state.evaporated,
            },
        )


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

GRAPH: Optional[MacroPlannerGraph] = None


def get_graph() -> MacroPlannerGraph:
    global GRAPH
    if GRAPH is None:
        system = init_system()
        GRAPH = MacroPlannerGraph(system)
    return GRAPH


@app.route("/health", methods=["GET"])
def health():
    system = init_system()
    return jsonify(system.health().__dict__)


@app.route("/jobs", methods=["POST"])
def create_job():
    body = request.get_json(silent=True) or {}
    job = Job(
        title=body.get("title", "Untitled Job"),
        description=body.get("description", ""),
        provider=CloudProvider(body.get("provider", "gcp")),
        input_data=body.get("input_data", {}),
    )
    system = init_system()
    # Persist job as atomic memory
    system.persistence.store_atomic(
        content=json.dumps({
            "id": str(job.id),
            "title": job.title,
            "description": job.description,
            "provider": job.provider,
            "status": job.status,
        }, default=str),
        content_type="job",
        source="gcp_api",
    )
    # Run macro planner graph (evaporates at end, hands off to Kaggle)
    graph = get_graph()
    state = graph.run(job)
    return jsonify({
        "job_id": str(job.id),
        "status": state.status,
        "step": state.step,
        "trace_id": state.trace_id,
        "evaporated": state.evaporated,
        "skills": state.skills,
        "decision": state.decision,
    }), 202


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    system = init_system()
    # Try to restore last checkpoint
    ckpt = system.persistence.restore_checkpoint(job_id, "evaporate")
    return jsonify({
        "job_id": job_id,
        "checkpoint": ckpt,
    })


@app.route("/checkpoints/<job_id>", methods=["GET"])
def list_checkpoints(job_id):
    system = init_system()
    ckpts = system.persistence.checkpoints.list_checkpoints(job_id)
    return jsonify({"job_id": job_id, "checkpoints": ckpts})


@app.route("/research", methods=["POST"])
def research():
    body = request.get_json(silent=True) or {}
    topic = body.get("topic", "general research")
    system = init_system()

    def core_selector(role: str) -> str:
        return f"open_source_{role}_core_v1"

    loop = system.research_module.run_research_loop(topic, core_selector)
    return jsonify(loop), 202


@app.route("/dream", methods=["POST"])
def dream():
    body = request.get_json(silent=True) or {}
    sources = body.get("sources", ["arXiv", "WorldMonitor", "DuckDuckGo news"])
    system = init_system()
    hyps = system.dream_module.generate_daily_hypotheses(sources)
    results = []
    for h in hyps:
        job_ctx = {"source": "dream_daily"}
        audit = system.dream_module.submit_for_audit(h, job_ctx)
        results.append({
            "hypothesis": h.__dict__,
            "audit": audit.__dict__,
        })
    return jsonify({"hypotheses": results})


@app.route("/magpie/failure", methods=["POST"])
def magpie_failure():
    body = request.get_json(silent=True) or {}
    system = init_system()
    rec = system.magpie_module.record_failure(
        module=body.get("module", "unknown"),
        component=body.get("component", "unknown"),
        error_type=body.get("error_type", "error"),
        error_message=body.get("error_message", ""),
        stack_trace=body.get("stack_trace"),
        context=body.get("context", {}),
    )
    material = system.magpie_module.distill(rec)
    return jsonify({"failure_id": str(rec.id), "material": material})


@app.route("/magpie/finetune", methods=["POST"])
def magpie_finetune():
    body = request.get_json(silent=True) or {}
    system = init_system()
    schedule = system.magpie_module.schedule_finetune(
        ai_core=body.get("ai_core", "main_brain"),
        material_ids=body.get("material_ids", []),
    )
    return jsonify(schedule)


@app.route("/evaporate/<job_id>", methods=["POST"])
def evaporate(job_id):
    body = request.get_json(silent=True) or {}
    system = init_system()
    step = body.get("step", "manual_evaporate")
    result = system.persistence.evaporator.evaporate(
        save_checkpoint=True,
        checkpoint_store=system.persistence.checkpoints,
        job_id=job_id,
        step=step,
    )
    return jsonify(result)


@app.route("/skills", methods=["GET"])
def list_skills():
    from shared.skill_genome import SKILL_CATALOG
    return jsonify({
        "total": len(SKILL_CATALOG),
        "skills": [
            {
                "code": s.code,
                "name": s.name,
                "category": s.category,
                "status": s.status,
                "version": s.version,
            }
            for s in SKILL_CATALOG.values()
        ],
    })


def create_app():
    init_system()
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
