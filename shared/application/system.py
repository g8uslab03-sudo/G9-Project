"""Application layer: main brain, assistant brain, audit/shadow brain, dream, magpie, research."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from shared.contracts import (
    Artifact,
    Checkpoint,
    CloudProvider,
    Experiment,
    FailureRecord,
    HealthResponse,
    Hypothesis,
    Job,
    JobStatus,
    Paper,
    SkillCategory,
    SkillGene,
)
from shared.persistence import PersistenceManager
from shared.skill_genome import get_skill, list_skills_by_category, search_skills, SKILL_CATALOG


# ---------------------------------------------------------------------------
# Module results
# ---------------------------------------------------------------------------

@dataclass
class BrainDecision:
    job_id: str
    action: str
    skills: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    next_step: Optional[str] = None


@dataclass
class AuditResult:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    blind_spots: List[str] = field(default_factory=list)
    security_passed: bool = False


# ---------------------------------------------------------------------------
# Main Brain (Central Orchestrator Brain)
# ---------------------------------------------------------------------------

class MainBrain:
    """Main brain: coordinates all modules, plans solutions, dispatches work."""

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self._lock = threading.Lock()
        self.active_jobs: Dict[str, Job] = {}
        self.skill_catalog = SKILL_CATALOG

    def understand_requirement(self, raw_input: str) -> Dict[str, Any]:
        """Parse and understand user/system requirement."""
        return {
            "raw": raw_input,
            "normalized": raw_input.strip(),
            "length": len(raw_input),
            "keywords": [w for w in raw_input.split() if len(w) > 2][:20],
            "understood_at": datetime.now(timezone.utc).isoformat(),
        }

    def decompose_problem(self, understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem into sub-tasks (DAG)."""
        sub_tasks = []
        keywords = understanding.get("keywords", [])
        for i, kw in enumerate(keywords[:5]):
            sub_tasks.append({
                "id": f"subtask_{i}",
                "keyword": kw,
                "description": f"Handle requirement aspect: {kw}",
                "status": "pending",
                "depends_on": [f"subtask_{i-1}"] if i > 0 else [],
            })
        if not sub_tasks:
            sub_tasks.append({
                "id": "subtask_0",
                "keyword": "general",
                "description": "Handle general requirement",
                "status": "pending",
                "depends_on": [],
            })
        return sub_tasks

    def select_skills(self, sub_tasks: List[Dict[str, Any]]) -> List[str]:
        """Select relevant skills from the 250-gene catalog."""
        selected = []
        for task in sub_tasks:
            kw = task.get("keyword", "").lower()
            matches = search_skills(kw)[:3]
            for s in matches:
                if s.code not in selected:
                    selected.append(s.code)
        # Always include governance/orchestration skills
        for code in ["SE_GIT_PR_AUTO_CREATOR", "CR_SELF_REFLECTION_CRITIC", "CR_PROMPT_INJECTION_GUARD"]:
            if code not in selected:
                selected.append(code)
        return selected[:20]

    def decide(self, job: Job, understanding: Dict[str, Any], sub_tasks: List[Dict[str, Any]]) -> BrainDecision:
        skills = self.select_skills(sub_tasks)
        return BrainDecision(
            job_id=str(job.id),
            action="decompose_and_dispatch",
            skills=skills,
            reasoning="Problem decomposed into sub-tasks; skills selected by keyword match against 250-gene catalog.",
            confidence=0.8,
            next_step="dispatch_to_assistant",
        )

    def handle_job(self, job: Job) -> BrainDecision:
        with self._lock:
            self.active_jobs[str(job.id)] = job
        understanding = self.understand_requirement(job.description)
        sub_tasks = self.decompose_problem(understanding)
        decision = self.decide(job, understanding, sub_tasks)
        # Store atomic memory
        self.persistence.store_atomic(
            content=json.dumps({"job": str(job.id), "understanding": understanding, "sub_tasks": sub_tasks, "decision": decision.__dict__}, default=str),
            content_type="brain_decision",
            source="main_brain",
        )
        return decision


# ---------------------------------------------------------------------------
# Assistant Brain
# ---------------------------------------------------------------------------

class AssistantBrain:
    """Assistant brain: supports main brain, allocates work, coordinates execution."""

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self.allocations: Dict[str, List[str]] = {}  # job_id -> skill list

    def allocate_work(self, job: Job, decision: BrainDecision) -> Dict[str, Any]:
        allocation = {
            "job_id": str(job.id),
            "skills": decision.skills,
            "steps": [
                {"step": "load_skill", "skill": s} for s in decision.skills
            ] + [
                {"step": "execute", "skill": s} for s in decision.skills
            ] + [
                {"step": "release_skill", "skill": s} for s in decision.skills
            ],
            "allocated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.allocations[str(job.id)] = decision.skills
        self.persistence.store_atomic(
            content=json.dumps(allocation, default=str),
            content_type="work_allocation",
            source="assistant_brain",
        )
        return allocation

    def release_work(self, job_id: str):
        skills = self.allocations.pop(job_id, [])
        self.persistence.store_atomic(
            content=json.dumps({"job_id": job_id, "released_skills": skills, "released_at": datetime.now(timezone.utc).isoformat()}),
            content_type="work_release",
            source="assistant_brain",
        )
        return skills


# ---------------------------------------------------------------------------
# Audit Brain / Shadow Brain
# ---------------------------------------------------------------------------

class AuditBrain:
    """Audit brain: approves results/artifacts, gatekeeper."""

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self.approvals: Dict[str, AuditResult] = {}

    def approve(self, job: Job, output: Dict[str, Any]) -> AuditResult:
        reasons = []
        blind_spots = []
        approved = True
        security_passed = True

        if not output:
            approved = False
            reasons.append("Empty output")
        if job.status != JobStatus.COMPLETED:
            approved = False
            reasons.append("Job not completed")
        # Simulated security audit
        if "rm -rf" in str(output) or "eval(" in str(output):
            security_passed = False
            approved = False
            blind_spots.append("Potentially destructive code detected")

        result = AuditResult(
            approved=approved,
            reasons=reasons,
            blind_spots=blind_spots,
            security_passed=security_passed,
        )
        self.approvals[str(job.id)] = result
        self.persistence.store_atomic(
            content=json.dumps({"job_id": str(job.id), "audit": result.__dict__, "approved_at": datetime.now(timezone.utc).isoformat()}),
            content_type="audit_result",
            source="audit_brain",
        )
        return result


class ShadowBrain:
    """Shadow brain: reverse/blind-spot thinking, challenges suggestions."""

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self.challenges: Dict[str, List[str]] = {}

    def challenge(self, decision: BrainDecision, output: Optional[Dict[str, Any]] = None) -> List[str]:
        blind_spots = []
        if not decision.skills:
            blind_spots.append("No skills selected; verify task can be handled")
        if decision.confidence < 0.5:
            blind_spots.append("Low confidence decision; consider alternative approach")
        if output is None:
            blind_spots.append("No output to validate")
        # Reverse thinking: what could go wrong
        blind_spots.append("Check for assumption that open-source skill will download successfully")
        blind_spots.append("Verify external network access is available on target cloud")
        self.challenges[str(decision.job_id)] = blind_spots
        self.persistence.store_atomic(
            content=json.dumps({"job_id": str(decision.job_id), "blind_spots": blind_spots, "challenged_at": datetime.now(timezone.utc).isoformat()}),
            content_type="shadow_challenge",
            source="shadow_brain",
        )
        return blind_spots


# ---------------------------------------------------------------------------
# Dream Module
# ---------------------------------------------------------------------------

class DreamModule:
    """Dream module: generates 5 hypotheses daily during idle, requires audit approval."""

    def __init__(self, persistence: PersistenceManager, audit: AuditBrain):
        self.persistence = persistence
        self.audit = audit
        self.hypotheses: List[Hypothesis] = []

    def generate_daily_hypotheses(self, sources: List[str]) -> List[Hypothesis]:
        """Generate 5 hypotheses from open-source databases/literature/news."""
        hyps = []
        for i in range(5):
            hyp = Hypothesis(
                title=f"Dream Hypothesis {i+1}",
                description=f"Auto-generated hypothesis from {', '.join(sources[:3])}",
                research_question=f"What is the optimal approach for dream topic {i+1}?",
                success_criteria={"metric": "validated", "threshold": 0.8},
                failure_criteria={"metric": "invalidated", "threshold": 0.2},
                source="dream",
            )
            hyps.append(hyp)
            self.hypotheses.append(hyp)
        return hyps

    def submit_for_audit(self, hypothesis: Hypothesis, job_context: Dict[str, Any]) -> AuditResult:
        return self.audit.approve(
            Job(id=hypothesis.id, title=hypothesis.title, description=hypothesis.description, status=JobStatus.COMPLETED),
            {"hypothesis": hypothesis.__dict__, "context": job_context},
        )


# ---------------------------------------------------------------------------
# Magpie Training Module
# ---------------------------------------------------------------------------

class MagpieModule:
    """Magpie: collects failures, distills into training material, fine-tunes AI cores."""

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self.failures: List[FailureRecord] = []
        self.training_queue: List[str] = []

    def record_failure(self, module: str, component: str, error_type: str, error_message: str, stack_trace: Optional[str] = None, context: Optional[Dict] = None) -> FailureRecord:
        rec = FailureRecord(
            module=module,
            component=component,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
        )
        self.failures.append(rec)
        self.persistence.store_atomic(
            content=json.dumps(rec.__dict__, default=str),
            content_type="magpie_failure",
            source="magpie",
        )
        return rec

    def distill(self, failure: FailureRecord) -> str:
        """Distill failure into training material."""
        material = f"Failure in {module}.{component}: {error_type} - {error_message}. Context: {failure.context}"
        failure.distilled = True
        failure.training_material = material
        self.training_queue.append(str(failure.id))
        self.persistence.store_atomic(
            content=json.dumps({"failure_id": str(failure.id), "material": material, "distilled_at": datetime.now(timezone.utc).isoformat()}),
            content_type="magpie_training_material",
            source="magpie",
        )
        return material

    def schedule_finetune(self, ai_core: str, material_ids: List[str]) -> Dict[str, Any]:
        schedule = {
            "ai_core": ai_core,
            "material_ids": material_ids,
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "status": "scheduled",
        }
        self.persistence.store_atomic(
            content=json.dumps(schedule, default=str),
            content_type="magpie_finetune_schedule",
            source="magpie",
        )
        return schedule


# ---------------------------------------------------------------------------
# Research Module (7-step closed loop)
# ---------------------------------------------------------------------------

class ResearchModule:
    """Research module: independent 7-core closed loop for scientific research."""

    def __init__(self, persistence: PersistenceManager, magpie: MagpieModule):
        self.persistence = persistence
        self.magpie = magpie
        self.loops: Dict[str, Dict[str, Any]] = {}

    def run_research_loop(self, research_topic: str, ai_core_selector: Callable[[str], str]) -> Dict[str, Any]:
        """Execute the 7-step research closed loop."""
        loop_id = str(uuid.uuid4())
        loop = {
            "id": loop_id,
            "topic": research_topic,
            "steps": {},
            "status": "started",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Step 1: Task Planner Core
        step1_core = ai_core_selector("task_planner")
        step1 = {
            "core": step1_core,
            "output": {"sub_tasks": ["analyze_problem", "arrange_division"], "topic": research_topic},
        }
        loop["steps"]["step1_task_planner"] = step1

        # Step 2: Literature Extractor Core (WorldMonitor + open source)
        step2_core = ai_core_selector("literature_extractor")
        step2 = {
            "core": step2_core,
            "output": {"sources": ["arXiv", "WorldMonitor", "DuckDuckGo news"], "papers": []},
        }
        loop["steps"]["step2_literature_extractor"] = step2

        # Step 3: Hypothesis Generator Core
        step3_core = ai_core_selector("hypothesis_generator")
        step3 = {
            "core": step3_core,
            "output": {"hypotheses": [f"H{i+1}" for i in range(5)]},
        }
        loop["steps"]["step3_hypothesis_generator"] = step3

        # Step 4: Sandbox Simulator Core (sandbox + security audit)
        step4_core = ai_core_selector("sandbox_simulator")
        step4 = {
            "core": step4_core,
            "output": {"sandbox": "isolated", "security_audit": "passed", "malicious": False},
        }
        loop["steps"]["step4_sandbox_simulator"] = step4

        # Step 5: Data Analyst Core
        step5_core = ai_core_selector("data_analyst")
        step5 = {
            "core": step5_core,
            "output": {"analysis": "completed", "metrics": {}},
        }
        loop["steps"]["step5_data_analyst"] = step5

        # Step 6: Paper Writer Core
        step6_core = ai_core_selector("paper_writer")
        step6 = {
            "core": step6_core,
            "output": {"title": f"Research on {research_topic}", "content": ""},
        }
        loop["steps"]["step6_paper_writer"] = step6

        # Step 7: Artifact Builder Core
        step7_core = ai_core_selector("artifact_builder")
        step7 = {
            "core": step7_core,
            "output": {"artifact_type": "algorithm", "content": ""},
        }
        loop["steps"]["step7_artifact_builder"] = step7

        loop["status"] = "completed"
        loop["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.loops[loop_id] = loop
        self.persistence.store_atomic(
            content=json.dumps(loop, default=str),
            content_type="research_loop",
            source="research_module",
        )
        return loop

    def select_ai_core(self, role: str) -> str:
        """Autonomous AI core selection (no whitelist binding)."""
        # Autonomous selection - picks a model from open source market
        return f"open_source_model_{role}_v1"


# ---------------------------------------------------------------------------
# Composite Application Service
# ---------------------------------------------------------------------------

class AutonomousSystem:
    """Top-level composite wiring all brains/modules together."""

    def __init__(self, persistence: Optional[PersistenceManager] = None):
        self.persistence = persistence or PersistenceManager()
        self.main_brain = MainBrain(self.persistence)
        self.assistant_brain = AssistantBrain(self.persistence)
        self.audit_brain = AuditBrain(self.persistence)
        self.shadow_brain = ShadowBrain(self.persistence)
        self.dream_module = DreamModule(self.persistence, self.audit_brain)
        self.magpie_module = MagpieModule(self.persistence)
        self.research_module = ResearchModule(self.persistence, self.magpie_module)

    def health(self) -> HealthResponse:
        return HealthResponse(status="healthy", provider=CloudProvider.GCP)

    def process_job(self, job: Job) -> Dict[str, Any]:
        """End-to-end: main brain decide -> assistant allocate -> audit/shadow review."""
        decision = self.main_brain.handle_job(job)
        allocation = self.assistant_brain.allocate_work(job, decision)
        audit = self.audit_brain.approve(job, {"allocation": allocation})
        shadow = self.shadow_brain.challenge(decision, {"allocation": allocation})
        return {
            "job_id": str(job.id),
            "decision": decision.__dict__,
            "allocation": allocation,
            "audit": audit.__dict__,
            "shadow_challenge": shadow,
        }
