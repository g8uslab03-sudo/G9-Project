"""
Application Commands - Write operations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from ..shared.kernel.commands import Command
from ..shared.kernel.result import Result
from ..domain.entities import (
    ProblemId, ProblemContext, AgentId, AgentRole, SkillId,
    GeneKey, SkillSpec, AgentCapability, SubProblem,
    VerificationResult, DebateArgument, IntegratedAnswer,
    MemoryId, MemoryAtom, MemoryManifest, MemoryGraphEdge,
    ResearchId, ResearchPlan, Hypothesis, ExperimentResult,
    ResearchPaper, Artifact, DreamHypothesis, FailureRecord
)


# Problem Commands
@dataclass
class ReceiveProblemCommand(Command[ProblemId]):
    user_query: str
    domain: str = "general"
    constraints: Dict[str, Any] = field(default_factory=dict)
    available_resources: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnderstandProblemCommand(Command[None]):
    problem_id: ProblemId
    understanding: str
    identified_constraints: List[str]


@dataclass
class DecomposeProblemCommand(Command[List[SubProblem]]):
    problem_id: ProblemId
    sub_problems: List[SubProblem]


@dataclass
class DispatchExpertsCommand(Command[Dict[UUID, AgentId]]):
    problem_id: ProblemId
    assignments: Dict[UUID, AgentId]


@dataclass
class CompleteReasoningCommand(Command[None]):
    problem_id: ProblemId
    sub_problem_id: UUID
    agent_id: AgentId
    result: Any


@dataclass
class VerifyResultCommand(Command[VerificationResult]):
    problem_id: ProblemId
    verification: VerificationResult


@dataclass
class ConductDebateCommand(Command[Optional[str]]):
    problem_id: ProblemId
    sub_problem_id: UUID
    arguments: List[DebateArgument]


@dataclass
class IntegrateAnswerCommand(Command[IntegratedAnswer]):
    problem_id: ProblemId
    answer: IntegratedAnswer


# Agent Commands
@dataclass
class RegisterAgentCommand(Command[AgentId]):
    capability: AgentCapability


@dataclass
class AssignTaskCommand(Command[None]):
    agent_id: AgentId
    task_id: UUID


@dataclass
class CompleteTaskCommand(Command[None]):
    agent_id: AgentId
    task_id: UUID
    performance: Dict[str, float]


@dataclass
class UpdateAgentPerformanceCommand(Command[None]):
    agent_id: AgentId
    metrics: Dict[str, float]


@dataclass
class AddAgentSkillCommand(Command[None]):
    agent_id: AgentId
    skill_id: SkillId


# Skill Commands
@dataclass
class RegisterSkillCommand(Command[SkillId]):
    spec: SkillSpec


@dataclass
class AcquireSkillCommand(Command[SkillSpec]):
    gene_key: GeneKey


@dataclass
class PromoteSkillCommand(Command[None]):
    skill_id: SkillId
    benchmark_results: Dict[str, float]


@dataclass
class DeprecateSkillCommand(Command[None]):
    skill_id: SkillId


@dataclass
class RecordSkillUsageCommand(Command[None]):
    skill_id: SkillId
    success: bool
    benchmark: Optional[Dict[str, Any]] = None


# Memory Commands
@dataclass
class StoreMemoryAtomCommand(Command[MemoryId]):
    atom: MemoryAtom


@dataclass
class StoreMemoryManifestCommand(Command[UUID]):
    manifest: MemoryManifest


@dataclass
class AddMemoryRelationshipCommand(Command[None]):
    edge: MemoryGraphEdge


# Research Commands
@dataclass
class StartResearchCommand(Command[ResearchId]):
    problem_id: ProblemId
    plan: ResearchPlan


@dataclass
class CompleteResearchPhaseCommand(Command[None]):
    research_id: ResearchId
    phase: str
    output: Any


@dataclass
class AddHypothesisCommand(Command[None]):
    research_id: ResearchId
    hypothesis: Hypothesis


@dataclass
class AddExperimentResultCommand(Command[None]):
    research_id: ResearchId
    result: ExperimentResult


@dataclass
class SetResearchPaperCommand(Command[None]):
    research_id: ResearchId
    paper: ResearchPaper


@dataclass
class AddArtifactCommand(Command[None]):
    research_id: ResearchId
    artifact: Artifact


# Dream Commands
@dataclass
class GenerateDreamHypothesesCommand(Command[List[DreamHypothesis]]):
    count: int = 5


@dataclass
class AuditDreamHypothesisCommand(Command[None]):
    hypothesis_id: UUID
    approved: bool
    audit_notes: str


# Magpie Commands
@dataclass
class RecordFailureCommand(Command[None]):
    failure: FailureRecord


@dataclass
class DistillFailuresCommand(Command[List[Any]]):
    module: Optional[str] = None


@dataclass
class FineTuneModelCommand(Command[None]):
    agent_id: AgentId
    training_samples: List[UUID]


# Pipeline Commands
@dataclass
class ExecutePipelineCommand(Command[IntegratedAnswer]):
    problem_id: ProblemId


@dataclass
class SaveCheckpointCommand(Command[None]):
    pipeline_id: str
    stage: str
    state: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ResumePipelineCommand(Command[Optional[Dict[str, Any]]]):
    pipeline_id: str