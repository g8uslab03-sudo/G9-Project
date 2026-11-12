"""
Domain Aggregates - Consistency boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from uuid import UUID, uuid4
from enum import Enum
from ..shared.kernel.base import AggregateRoot, DomainEvent
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import BusinessRuleError, ValidationError, ConcurrencyError
from .entities import (
    ProblemId, ProblemStatus, AgentId, AgentRole, SkillId, SkillCategory,
    GeneKey, SkillSpec, AgentCapability, ProblemContext, SubProblem,
    VerificationResult, VerificationStatus, DebateArgument, IntegratedAnswer,
    MemoryId, MemoryAtom, MemoryManifest, MemoryGraphEdge,
    ResearchId, ResearchPhase, ResearchPlan, Hypothesis, ExperimentResult,
    ResearchPaper, Artifact, DreamHypothesis, FailureRecord, TrainingSample
)


class ProblemAggregate(AggregateRoot[ProblemId]):
    """Aggregate for problem solving pipeline."""
    
    def __init__(self, problem_id: ProblemId, context: ProblemContext):
        super().__init__(problem_id)
        self._context = context
        self._status = ProblemStatus.RECEIVED
        self._understanding: Optional[str] = None
        self._sub_problems: Dict[UUID, SubProblem] = {}
        self._assignments: Dict[UUID, AgentId] = {}  # sub_problem_id -> agent_id
        self._reasoning_results: Dict[UUID, Any] = {}
        self._verifications: Dict[UUID, List[VerificationResult]] = {}
        self._debate_arguments: Dict[UUID, List[DebateArgument]] = {}
        self._integrated_answer: Optional[IntegratedAnswer] = None
        self._current_stage = 0
        self._stages = [
            "receive_problem",
            "understand_requirements",
            "decompose_problem",
            "dispatch_experts",
            "parallel_reasoning",
            "mutual_verification",
            "debate_evaluation",
            "integrate_answer",
            "continuous_learning",
            "optimize_future"
        ]
    
    @property
    def context(self) -> ProblemContext:
        return self._context
    
    @property
    def status(self) -> ProblemStatus:
        return self._status
    
    @property
    def sub_problems(self) -> List[SubProblem]:
        return list(self._sub_problems.values())
    
    @property
    def integrated_answer(self) -> Optional[IntegratedAnswer]:
        return self._integrated_answer
    
    def _get_snapshot_state(self) -> dict:
        return {
            "context": self._context.__dict__,
            "status": self._status.value,
            "understanding": self._understanding,
            "sub_problems": {str(k): v.__dict__ for k, v in self._sub_problems.items()},
            "assignments": {str(k): str(v) for k, v in self._assignments.items()},
            "reasoning_results": self._reasoning_results,
            "verifications": {str(k): [v.__dict__ for v in vs] for k, vs in self._verifications.items()},
            "debate_arguments": {str(k): [a.__dict__ for a in args] for k, args in self._debate_arguments.items()},
            "integrated_answer": self._integrated_answer.__dict__ if self._integrated_answer else None,
            "current_stage": self._current_stage
        }
    
    def _load_from_snapshot(self, state: dict, version: int) -> None:
        self._context = ProblemContext(**state["context"])
        self._status = ProblemStatus(state["status"])
        self._understanding = state["understanding"]
        self._sub_problems = {UUID(k): SubProblem(**v) for k, v in state["sub_problems"].items()}
        self._assignments = {UUID(k): AgentId(UUID(v)) for k, v in state["assignments"].items()}
        self._reasoning_results = state["reasoning_results"]
        self._verifications = {UUID(k): [VerificationResult(**v) for v in vs] for k, vs in state["verifications"].items()}
        self._debate_arguments = {UUID(k): [DebateArgument(**a) for a in args] for k, args in state["debate_arguments"].items()}
        self._integrated_answer = IntegratedAnswer(**state["integrated_answer"]) if state["integrated_answer"] else None
        self._current_stage = state["current_stage"]
        self._version = version
    
    def _apply_event(self, event: DomainEvent) -> None:
        event_handlers = {
            "ProblemReceived": self._on_problem_received,
            "ProblemUnderstood": self._on_problem_understood,
            "ProblemDecomposed": self._on_problem_decomposed,
            "ExpertsDispatched": self._on_experts_dispatched,
            "ReasoningCompleted": self._on_reasoning_completed,
            "VerificationCompleted": self._on_verification_completed,
            "DebateCompleted": self._on_debate_completed,
            "AnswerIntegrated": self._on_answer_integrated,
        }
        
        handler = event_handlers.get(event.event_type)
        if handler:
            handler(event)
    
    def _on_problem_received(self, event: DomainEvent) -> None:
        self._status = ProblemStatus.RECEIVED
        self._current_stage = 0
    
    def _on_problem_understood(self, event: DomainEvent) -> None:
        payload = event._payload()
        self._understanding = payload["understanding"]
        self._status = ProblemStatus.UNDERSTANDING
        self._current_stage = 1
    
    def _on_problem_decomposed(self, event: DomainEvent) -> None:
        payload = event._payload()
        for sp_data in payload["sub_problems"]:
            sub_problem = SubProblem(**sp_data)
            self._sub_problems[sub_problem.sub_id] = sub_problem
        self._status = ProblemStatus.DECOMPOSED
        self._current_stage = 2
    
    def _on_experts_dispatched(self, event: DomainEvent) -> None:
        payload = event._payload()
        for sub_id_str, agent_id_str in payload["assignments"].items():
            self._assignments[UUID(sub_id_str)] = AgentId(UUID(agent_id_str))
            if UUID(sub_id_str) in self._sub_problems:
                self._sub_problems[UUID(sub_id_str)] = self._sub_problems[UUID(sub_id_str)].with_changes(
                    assigned_agent=AgentId(UUID(agent_id_str)),
                    status=ProblemStatus.DISPATCHED
                )
        self._status = ProblemStatus.DISPATCHED
        self._current_stage = 3
    
    def _on_reasoning_completed(self, event: DomainEvent) -> None:
        payload = event._payload()
        sub_problem_id = UUID(payload["sub_problem_id"])
        self._reasoning_results[sub_problem_id] = payload["result"]
        if sub_problem_id in self._sub_problems:
            self._sub_problems[sub_problem_id] = self._sub_problems[sub_problem_id].with_changes(
                result=payload["result"],
                status=ProblemStatus.REASONING
            )
        self._status = ProblemStatus.REASONING
        self._current_stage = 4
    
    def _on_verification_completed(self, event: DomainEvent) -> None:
        payload = event._payload()
        verification = VerificationResult(**payload["verification"])
        sub_problem_id = verification.sub_problem_id
        if sub_problem_id not in self._verifications:
            self._verifications[sub_problem_id] = []
        self._verifications[sub_problem_id].append(verification)
        self._status = ProblemStatus.VERIFYING
        self._current_stage = 5
    
    def _on_debate_completed(self, event: DomainEvent) -> None:
        payload = event._payload()
        sub_problem_id = UUID(payload["sub_problem_id"])
        self._debate_arguments[sub_problem_id] = [DebateArgument(**a) for a in payload["arguments"]]
        self._status = ProblemStatus.DEBATING
        self._current_stage = 6
    
    def _on_answer_integrated(self, event: DomainEvent) -> None:
        payload = event._payload()
        self._integrated_answer = IntegratedAnswer(**payload["answer"])
        self._status = ProblemStatus.COMPLETED
        self._current_stage = 9
    
    # Commands
    def receive_problem(self) -> Result[None]:
        if self._status != ProblemStatus.RECEIVED:
            return failure(BusinessRuleError("Problem already received"))
        
        event = ProblemReceived(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            context=self._context
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def understand_requirements(self, understanding: str, constraints: List[str]) -> Result[None]:
        if self._status != ProblemStatus.RECEIVED:
            return failure(BusinessRuleError("Cannot understand: invalid status"))
        
        event = ProblemUnderstood(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            understanding=understanding,
            identified_constraints=constraints
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def decompose_problem(self, sub_problems: List[SubProblem]) -> Result[None]:
        if self._status != ProblemStatus.UNDERSTANDING:
            return failure(BusinessRuleError("Cannot decompose: invalid status"))
        
        # Validate all sub-problems have required skills
        for sp in sub_problems:
            if not sp.required_skills:
                return failure(ValidationError(f"Sub-problem {sp.sub_id} has no required skills"))
        
        event = ProblemDecomposed(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            sub_problems=sub_problems
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def dispatch_experts(self, assignments: Dict[UUID, AgentId]) -> Result[None]:
        if self._status != ProblemStatus.DECOMPOSED:
            return failure(BusinessRuleError("Cannot dispatch: invalid status"))
        
        # Validate all sub-problems are assigned
        for sub_id in self._sub_problems:
            if sub_id not in assignments:
                return failure(ValidationError(f"Sub-problem {sub_id} not assigned"))
        
        event = ExpertsDispatched(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            assignments=assignments
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def complete_reasoning(self, sub_problem_id: UUID, agent_id: AgentId, result: Any) -> Result[None]:
        if self._status not in [ProblemStatus.DISPATCHED, ProblemStatus.REASONING]:
            return failure(BusinessRuleError("Cannot complete reasoning: invalid status"))
        
        event = ReasoningCompleted(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            sub_problem_id=sub_problem_id,
            agent_id=agent_id,
            result=result
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def complete_verification(self, verification: VerificationResult) -> Result[None]:
        if self._status != ProblemStatus.REASONING:
            return failure(BusinessRuleError("Cannot verify: invalid status"))
        
        event = VerificationCompleted(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            verification=verification
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def complete_debate(self, sub_problem_id: UUID, arguments: List[DebateArgument], consensus: Optional[str]) -> Result[None]:
        if self._status != ProblemStatus.VERIFYING:
            return failure(BusinessRuleError("Cannot debate: invalid status"))
        
        event = DebateCompleted(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            sub_problem_id=sub_problem_id,
            arguments=arguments,
            consensus=consensus
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def integrate_answer(self, answer: IntegratedAnswer) -> Result[None]:
        if self._status != ProblemStatus.DEBATING:
            return failure(BusinessRuleError("Cannot integrate: invalid status"))
        
        event = AnswerIntegrated(
            aggregate_id=self.id.value,
            aggregate_type="ProblemAggregate",
            version=self.version + 1,
            problem_id=self.id,
            answer=answer
        )
        self.add_domain_event(event)
        self.mark_changes()
        return success(None)
    
    def advance_stage(self) -> Result[None]:
        if self._current_stage >= len(self._stages) - 1:
            return failure(BusinessRuleError("Already at final stage"))
        self._current_stage += 1
        self.mark_changes()
        return success(None)


class AgentAggregate(AggregateRoot[AgentId]):
    """Aggregate for agent lifecycle and capabilities."""
    
    def __init__(self, agent_id: AgentId, capability: AgentCapability):
        super().__init__(agent_id)
        self._capability = capability
        self._current_tasks: Set[UUID] = set()
        self._completed_tasks: List[UUID] = []
        self._performance_history: List[Dict[str, Any]] = []
        self._status = "idle"  # idle, busy, offline, error
    
    @property
    def capability(self) -> AgentCapability:
        return self._capability
    
    @property
    def current_load(self) -> int:
        return len(self._current_tasks)
    
    @property
    def max_load(self) -> int:
        return self._capability.max_concurrent_tasks
    
    @property
    def is_available(self) -> bool:
        return self._status == "idle" and self.current_load < self.max_load
    
    def _get_snapshot_state(self) -> dict:
        return {
            "capability": self._capability.__dict__,
            "current_tasks": [str(t) for t in self._current_tasks],
            "completed_tasks": [str(t) for t in self._completed_tasks],
            "performance_history": self._performance_history,
            "status": self._status
        }
    
    def _load_from_snapshot(self, state: dict, version: int) -> None:
        self._capability = AgentCapability(**state["capability"])
        self._current_tasks = {UUID(t) for t in state["current_tasks"]}
        self._completed_tasks = [UUID(t) for t in state["completed_tasks"]]
        self._performance_history = state["performance_history"]
        self._status = state["status"]
        self._version = version
    
    def _apply_event(self, event: DomainEvent) -> None:
        pass  # Events handled by commands
    
    def assign_task(self, task_id: UUID) -> Result[None]:
        if not self.is_available:
            return failure(BusinessRuleError("Agent not available"))
        
        self._current_tasks.add(task_id)
        self._status = "busy" if self.current_load >= self.max_load else "idle"
        self.mark_changes()
        return success(None)
    
    def complete_task(self, task_id: UUID, performance: Dict[str, float]) -> Result[None]:
        if task_id not in self._current_tasks:
            return failure(BusinessRuleError("Task not assigned to this agent"))
        
        self._current_tasks.remove(task_id)
        self._completed_tasks.append(task_id)
        self._performance_history.append({
            "task_id": str(task_id),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": performance
        })
        self._status = "idle" if not self._current_tasks else "busy"
        self.mark_changes()
        return success(None)
    
    def update_performance(self, metrics: Dict[str, float]) -> Result[None]:
        self._capability = self._capability.with_changes(performance_metrics=metrics)
        self.mark_changes()
        return success(None)
    
    def add_skill(self, skill_id: SkillId) -> Result[None]:
        if skill_id not in self._capability.skills:
            self._capability = self._capability.with_changes(
                skills=self._capability.skills + [skill_id]
            )
            self.mark_changes()
        return success(None)
    
    def go_offline(self) -> Result[None]:
        if self._current_tasks:
            return failure(BusinessRuleError("Cannot go offline with active tasks"))
        self._status = "offline"
        self.mark_changes()
        return success(None)
    
    def go_online(self) -> Result[None]:
        self._status = "idle"
        self.mark_changes()
        return success(None)


class SkillAggregate(AggregateRoot[SkillId]):
    """Aggregate for skill lifecycle."""
    
    def __init__(self, skill_id: SkillId, spec: SkillSpec):
        super().__init__(skill_id)
        self._spec = spec
        self._status = "available"  # available, deprecated, testing, promoted
        self._benchmarks: List[Dict[str, Any]] = []
        self._usage_count = 0
        self._success_count = 0
        self._failure_count = 0
    
    @property
    def spec(self) -> SkillSpec:
        return self._spec
    
    @property
    def status(self) -> str:
        return self._status
    
    @property
    def success_rate(self) -> float:
        total = self._success_count + self._failure_count
        return self._success_count / total if total > 0 else 0.0
    
    def _get_snapshot_state(self) -> dict:
        return {
            "spec": self._spec.__dict__,
            "status": self._status,
            "benchmarks": self._benchmarks,
            "usage_count": self._usage_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count
        }
    
    def _load_from_snapshot(self, state: dict, version: int) -> None:
        self._spec = SkillSpec(**state["spec"])
        self._status = state["status"]
        self._benchmarks = state["benchmarks"]
        self._usage_count = state["usage_count"]
        self._success_count = state["success_count"]
        self._failure_count = state["failure_count"]
        self._version = version
    
    def _apply_event(self, event: DomainEvent) -> None:
        pass
    
    def record_usage(self, success: bool, benchmark: Optional[Dict[str, Any]] = None) -> Result[None]:
        self._usage_count += 1
        if success:
            self._success_count += 1
        else:
            self._failure_count += 1
        
        if benchmark:
            self._benchmarks.append({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": benchmark
            })
        
        self.mark_changes()
        return success(None)
    
    def promote(self, benchmark_results: Dict[str, float]) -> Result[None]:
        if self._status != "testing":
            return failure(BusinessRuleError("Skill not in testing status"))
        
        self._status = "promoted"
        self._benchmarks.append({
            "timestamp": datetime.utcnow().isoformat(),
            "promotion_benchmark": benchmark_results
        })
        self.mark_changes()
        return success(None)
    
    def deprecate(self) -> Result[None]:
        self._status = "deprecated"
        self.mark_changes()
        return success(None)
    
    def update_spec(self, new_spec: SkillSpec) -> Result[None]:
        self._spec = new_spec
        self.mark_changes()
        return success(None)


class ResearchAggregate(AggregateRoot[ResearchId]):
    """Aggregate for research module 7-phase pipeline."""
    
    def __init__(self, research_id: ResearchId, problem_id: ProblemId, plan: ResearchPlan):
        super().__init__(research_id)
        self._problem_id = problem_id
        self._plan = plan
        self._current_phase_index = 0
        self._phase_outputs: Dict[ResearchPhase, Any] = {}
        self._hypotheses: List[Hypothesis] = []
        self._experiment_results: List[ExperimentResult] = []
        self._paper: Optional[ResearchPaper] = None
        self._artifacts: List[Artifact] = []
        self._status = "initialized"  # initialized, in_progress, completed, failed
    
    @property
    def current_phase(self) -> ResearchPhase:
        if self._current_phase_index < len(self._plan.phases):
            return self._plan.phases[self._current_phase_index]
        return ResearchPhase.ARTIFACT_PRODUCTION
    
    @property
    def is_complete(self) -> bool:
        return self._current_phase_index >= len(self._plan.phases)
    
    def _get_snapshot_state(self) -> dict:
        return {
            "problem_id": str(self._problem_id),
            "plan": self._plan.__dict__,
            "current_phase_index": self._current_phase_index,
            "phase_outputs": {k.value: v for k, v in self._phase_outputs.items()},
            "hypotheses": [h.__dict__ for h in self._hypotheses],
            "experiment_results": [er.__dict__ for er in self._experiment_results],
            "paper": self._paper.__dict__ if self._paper else None,
            "artifacts": [a.__dict__ for a in self._artifacts],
            "status": self._status
        }
    
    def _load_from_snapshot(self, state: dict, version: int) -> None:
        self._problem_id = ProblemId(UUID(state["problem_id"]))
        self._plan = ResearchPlan(**state["plan"])
        self._current_phase_index = state["current_phase_index"]
        self._phase_outputs = {ResearchPhase(k): v for k, v in state["phase_outputs"].items()}
        self._hypotheses = [Hypothesis(**h) for h in state["hypotheses"]]
        self._experiment_results = [ExperimentResult(**er) for er in state["experiment_results"]]
        self._paper = ResearchPaper(**state["paper"]) if state["paper"] else None
        self._artifacts = [Artifact(**a) for a in state["artifacts"]]
        self._status = state["status"]
        self._version = version
    
    def _apply_event(self, event: DomainEvent) -> None:
        pass
    
    def start(self) -> Result[None]:
        self._status = "in_progress"
        self.mark_changes()
        return success(None)
    
    def complete_phase(self, phase: ResearchPhase, output: Any) -> Result[None]:
        if phase != self.current_phase:
            return failure(BusinessRuleError(f"Expected phase {self.current_phase}, got {phase}"))
        
        self._phase_outputs[phase] = output
        self._current_phase_index += 1
        
        if self.is_complete:
            self._status = "completed"
        
        self.mark_changes()
        return success(None)
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> Result[None]:
        if self.current_phase != ResearchPhase.HYPOTHESIS_GENERATION:
            return failure(BusinessRuleError("Not in hypothesis generation phase"))
        
        self._hypotheses.append(hypothesis)
        self.mark_changes()
        return success(None)
    
    def add_experiment_result(self, result: ExperimentResult) -> Result[None]:
        if self.current_phase != ResearchPhase.EXPERIMENT_EXECUTION:
            return failure(BusinessRuleError("Not in experiment execution phase"))
        
        self._experiment_results.append(result)
        self.mark_changes()
        return success(None)
    
    def set_paper(self, paper: ResearchPaper) -> Result[None]:
        if self.current_phase != ResearchPhase.PAPER_GENERATION:
            return failure(BusinessRuleError("Not in paper generation phase"))
        
        self._paper = paper
        self.mark_changes()
        return success(None)
    
    def add_artifact(self, artifact: Artifact) -> Result[None]:
        if self.current_phase != ResearchPhase.ARTIFACT_PRODUCTION:
            return failure(BusinessRuleError("Not in artifact production phase"))
        
        self._artifacts.append(artifact)
        self.mark_changes()
        return success(None)