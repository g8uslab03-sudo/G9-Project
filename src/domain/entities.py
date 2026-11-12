"""
Domain Entities - Core business objects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from uuid import UUID, uuid4
from enum import Enum
from ..shared.kernel.base import Entity, AggregateRoot, ValueObject, DomainEvent, Identifier, UUIDIdentifier
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import BusinessRuleError, ValidationError


class ProblemStatus(str, Enum):
    RECEIVED = "received"
    UNDERSTANDING = "understanding"
    DECOMPOSED = "decomposed"
    DISPATCHED = "dispatched"
    REASONING = "reasoning"
    VERIFYING = "verifying"
    DEBATING = "debating"
    INTEGRATING = "integrating"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(str, Enum):
    MAIN_BRAIN = "main_brain"
    SUB_BRAIN = "sub_brain"
    AUDIT_BRAIN = "audit_brain"
    SHADOW_BRAIN = "shadow_brain"
    DREAM_MODULE = "dream_module"
    MAGPIE_MODULE = "magpie_module"
    RESEARCH_MODULE = "research_module"
    EXPERT_AGENT = "expert_agent"


class SkillCategory(str, Enum):
    INFORMATION_RETRIEVAL = "information_retrieval"
    LOGICAL_REASONING = "logical_reasoning"
    CONTENT_CREATION = "content_creation"
    VISUAL_DESIGN = "visual_design"
    PROGRAMMING = "programming"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class ProblemId(Identifier):
    value: UUID
    
    def __init__(self, value: UUID = None):
        object.__setattr__(self, 'value', value or uuid4())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProblemId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class AgentId(Identifier):
    value: UUID
    
    def __init__(self, value: UUID = None):
        object.__setattr__(self, 'value', value or uuid4())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SkillId(Identifier):
    value: UUID
    
    def __init__(self, value: UUID = None):
        object.__setattr__(self, 'value', value or uuid4())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MemoryId(Identifier):
    value: UUID
    
    def __init__(self, value: UUID = None):
        object.__setattr__(self, 'value', value or uuid4())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ResearchId(Identifier):
    value: UUID
    
    def __init__(self, value: UUID = None):
        object.__setattr__(self, 'value', value or uuid4())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResearchId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class GeneKey(ValueObject):
    """Skill gene key - unique identifier for a skill."""
    category: SkillCategory
    index: int  # 1-50
    
    def _equality_components(self) -> tuple:
        return (self.category, self.index)
    
    def __str__(self) -> str:
        return f"SKILL_{self.category.value.upper()}_{self.index:03d}"
    
    @classmethod
    def from_string(cls, s: str) -> 'GeneKey':
        parts = s.split('_')
        if len(parts) != 3 or parts[0] != "SKILL":
            raise ValueError(f"Invalid gene key format: {s}")
        category = SkillCategory(parts[1].lower())
        index = int(parts[2])
        return cls(category, index)


@dataclass(frozen=True)
class SkillSpec(ValueObject):
    """Skill specification."""
    gene_key: GeneKey
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any]
    source: str  # "builtin", "downloaded", "researched"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.gene_key, self.version)


@dataclass(frozen=True)
class AgentCapability(ValueObject):
    """Agent capability profile."""
    agent_id: AgentId
    role: AgentRole
    skills: List[SkillId]
    max_concurrent_tasks: int
    supported_modalities: List[str]
    resource_limits: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.agent_id, self.role)


@dataclass(frozen=True)
class ProblemContext(ValueObject):
    """Context for problem solving."""
    problem_id: ProblemId
    user_query: str
    domain: str
    constraints: Dict[str, Any]
    available_resources: Dict[str, Any]
    priority: int = 1
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.problem_id, self.user_query)


@dataclass(frozen=True)
class SubProblem(ValueObject):
    """Decomposed sub-problem."""
    sub_id: UUID
    description: str
    required_skills: List[GeneKey]
    assigned_agent: Optional[AgentId] = None
    dependencies: List[UUID] = field(default_factory=list)
    status: ProblemStatus = ProblemStatus.RECEIVED
    result: Optional[Any] = None
    verification: VerificationStatus = VerificationStatus.PENDING
    
    def _equality_components(self) -> tuple:
        return (self.sub_id,)


@dataclass(frozen=True)
class VerificationResult(ValueObject):
    """Result of mutual verification."""
    verifier_id: AgentId
    target_id: AgentId
    sub_problem_id: UUID
    status: VerificationStatus
    confidence: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def _equality_components(self) -> tuple:
        return (self.verifier_id, self.target_id, self.sub_problem_id)


@dataclass(frozen=True)
class DebateArgument(ValueObject):
    """Argument in debate phase."""
    agent_id: AgentId
    position: str  # "support", "oppose", "neutral"
    argument: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def _equality_components(self) -> tuple:
        return (self.agent_id, self.position, self.argument)


@dataclass(frozen=True)
class IntegratedAnswer(ValueObject):
    """Final integrated answer."""
    problem_id: ProblemId
    answer: str
    confidence: float
    supporting_evidence: List[str]
    dissenting_opinions: List[str]
    skill_contributions: Dict[GeneKey, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.problem_id, self.answer)


@dataclass(frozen=True)
class MemoryAtom(ValueObject):
    """Atomic memory unit - smallest retrievable unit."""
    atom_id: MemoryId
    content: str
    content_type: str  # "text", "code", "data", "embedding"
    source: str
    source_id: str
    tags: List[str]
    embedding: Optional[List[float]] = None
    importance: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    
    def _equality_components(self) -> tuple:
        return (self.atom_id,)


@dataclass(frozen=True)
class MemoryManifest(ValueObject):
    """Manifest linking atoms to objects."""
    manifest_id: UUID
    object_type: str  # "document", "work_order", "experiment", "audit_evidence"
    object_id: str
    atom_ids: List[MemoryId]
    sequence: List[MemoryId]  # ordered atoms
    relationships: Dict[str, List[MemoryId]]  # relationship type -> atoms
    
    def _equality_components(self) -> tuple:
        return (self.manifest_id,)


@dataclass(frozen=True)
class MemoryGraphEdge(ValueObject):
    """Relationship between memory atoms."""
    source_id: MemoryId
    target_id: MemoryId
    relationship_type: str  # "references", "derives_from", "contradicts", "supports"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.source_id, self.target_id, self.relationship_type)


@dataclass(frozen=True)
class ResearchPhase(str, Enum):
    ANALYZE_AND_PLAN = "analyze_and_plan"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_EXECUTION = "experiment_execution"
    DATA_ANALYSIS = "data_analysis"
    PAPER_GENERATION = "paper_generation"
    ARTIFACT_PRODUCTION = "artifact_production"


@dataclass(frozen=True)
class ResearchPlan(ValueObject):
    """Research execution plan."""
    research_id: ResearchId
    problem_id: ProblemId
    phases: List[ResearchPhase]
    assigned_agents: Dict[ResearchPhase, List[AgentId]]
    tools_per_phase: Dict[ResearchPhase, List[str]]
    sandbox_config: Dict[str, Any]
    audit_checkpoints: List[ResearchPhase]
    
    def _equality_components(self) -> tuple:
        return (self.research_id,)


@dataclass(frozen=True)
class Hypothesis(ValueObject):
    """Research hypothesis."""
    hypothesis_id: UUID
    research_id: ResearchId
    statement: str
    anchor_papers: List[str]  # paper IDs
    confidence: float
    testable_predictions: List[str]
    status: str = "proposed"  # proposed, testing, validated, rejected
    
    def _equality_components(self) -> tuple:
        return (self.hypothesis_id,)


@dataclass(frozen=True)
class ExperimentResult(ValueObject):
    """Experiment execution result."""
    experiment_id: UUID
    hypothesis_id: UUID
    sandbox_id: str
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str]
    execution_time_seconds: float
    resource_usage: Dict[str, Any]
    
    def _equality_components(self) -> tuple:
        return (self.experiment_id,)


@dataclass(frozen=True)
class ResearchPaper(ValueObject):
    """Generated research paper."""
    paper_id: UUID
    research_id: ResearchId
    title: str
    abstract: str
    sections: Dict[str, str]  # section_name -> content
    references: List[str]
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def _equality_components(self) -> tuple:
        return (self.paper_id,)


@dataclass(frozen=True)
class Artifact(ValueObject):
    """Produced artifact (algorithm, tool, skill, etc.)."""
    artifact_id: UUID
    research_id: ResearchId
    artifact_type: str  # "algorithm", "compression", "skill", "tool", "model"
    name: str
    description: str
    code: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    benchmark_results: Optional[Dict[str, float]] = None
    validation_status: str = "pending"  # pending, passed, failed
    
    def _equality_components(self) -> tuple:
        return (self.artifact_id,)


@dataclass(frozen=True)
class DreamHypothesis(ValueObject):
    """Hypothesis generated by dream module."""
    hypothesis_id: UUID
    title: str
    description: str
    source_databases: List[str]
    anchor_evidence: List[str]
    novelty_score: float
    feasibility_score: float
    impact_score: float
    status: str = "proposed"  # proposed, audited, approved, rejected
    audit_notes: Optional[str] = None
    
    def _equality_components(self) -> tuple:
        return (self.hypothesis_id,)


@dataclass(frozen=True)
class FailureRecord(ValueObject):
    """Record of failure for Magpie training."""
    failure_id: UUID
    module: str
    agent_id: Optional[AgentId]
    skill_id: Optional[SkillId]
    error_type: str
    error_message: str
    context: Dict[str, Any]
    stack_trace: Optional[str]
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution: Optional[str] = None
    
    def _equality_components(self) -> tuple:
        return (self.failure_id,)


@dataclass(frozen=True)
class TrainingSample(ValueObject):
    """Distilled training sample from failures."""
    sample_id: UUID
    failure_ids: List[UUID]
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    failure_pattern: str
    correction: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _equality_components(self) -> tuple:
        return (self.sample_id,)


# Events
@dataclass
class ProblemReceived(DomainEvent):
    problem_id: ProblemId
    context: ProblemContext
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "context": self.context.__dict__
        }


@dataclass
class ProblemUnderstood(DomainEvent):
    problem_id: ProblemId
    understanding: str
    identified_constraints: List[str]
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "understanding": self.understanding,
            "identified_constraints": self.identified_constraints
        }


@dataclass
class ProblemDecomposed(DomainEvent):
    problem_id: ProblemId
    sub_problems: List[SubProblem]
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "sub_problems": [sp.__dict__ for sp in self.sub_problems]
        }


@dataclass
class ExpertsDispatched(DomainEvent):
    problem_id: ProblemId
    assignments: Dict[UUID, AgentId]  # sub_problem_id -> agent_id
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "assignments": {str(k): str(v) for k, v in self.assignments.items()}
        }


@dataclass
class ReasoningCompleted(DomainEvent):
    problem_id: ProblemId
    sub_problem_id: UUID
    agent_id: AgentId
    result: Any
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "sub_problem_id": str(self.sub_problem_id),
            "agent_id": str(self.agent_id),
            "result": self.result
        }


@dataclass
class VerificationCompleted(DomainEvent):
    problem_id: ProblemId
    verification: VerificationResult
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "verification": self.verification.__dict__
        }


@dataclass
class DebateCompleted(DomainEvent):
    problem_id: ProblemId
    sub_problem_id: UUID
    arguments: List[DebateArgument]
    consensus: Optional[str]
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "sub_problem_id": str(self.sub_problem_id),
            "arguments": [a.__dict__ for a in self.arguments],
            "consensus": self.consensus
        }


@dataclass
class AnswerIntegrated(DomainEvent):
    problem_id: ProblemId
    answer: IntegratedAnswer
    
    def _payload(self) -> dict:
        return {
            "problem_id": str(self.problem_id),
            "answer": self.answer.__dict__
        }


@dataclass
class SkillAcquired(DomainEvent):
    skill_id: SkillId
    gene_key: GeneKey
    source: str
    spec: SkillSpec
    
    def _payload(self) -> dict:
        return {
            "skill_id": str(self.skill_id),
            "gene_key": str(self.gene_key),
            "source": self.source,
            "spec": self.spec.__dict__
        }


@dataclass
class SkillPromoted(DomainEvent):
    skill_id: SkillId
    gene_key: GeneKey
    benchmark_results: Dict[str, float]
    
    def _payload(self) -> dict:
        return {
            "skill_id": str(self.skill_id),
            "gene_key": str(self.gene_key),
            "benchmark_results": self.benchmark_results
        }


@dataclass
class MemoryStored(DomainEvent):
    memory_id: MemoryId
    atom: MemoryAtom
    
    def _payload(self) -> dict:
        return {
            "memory_id": str(self.memory_id),
            "atom": self.atom.__dict__
        }


@dataclass
class ResearchStarted(DomainEvent):
    research_id: ResearchId
    problem_id: ProblemId
    plan: ResearchPlan
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "problem_id": str(self.problem_id),
            "plan": self.plan.__dict__
        }


@dataclass
class ResearchPhaseCompleted(DomainEvent):
    research_id: ResearchId
    phase: ResearchPhase
    output: Any
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "phase": self.phase.value,
            "output": self.output
        }


@dataclass
class HypothesisGenerated(DomainEvent):
    research_id: ResearchId
    hypothesis: Hypothesis
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "hypothesis": self.hypothesis.__dict__
        }


@dataclass
class ExperimentCompleted(DomainEvent):
    research_id: ResearchId
    result: ExperimentResult
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "result": self.result.__dict__
        }


@dataclass
class PaperGenerated(DomainEvent):
    research_id: ResearchId
    paper: ResearchPaper
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "paper": self.paper.__dict__
        }


@dataclass
class ArtifactProduced(DomainEvent):
    research_id: ResearchId
    artifact: Artifact
    
    def _payload(self) -> dict:
        return {
            "research_id": str(self.research_id),
            "artifact": self.artifact.__dict__
        }


@dataclass
class DreamHypothesisGenerated(DomainEvent):
    hypothesis: DreamHypothesis
    
    def _payload(self) -> dict:
        return {
            "hypothesis": self.hypothesis.__dict__
        }


@dataclass
class DreamHypothesisAudited(DomainEvent):
    hypothesis_id: UUID
    approved: bool
    audit_notes: str
    
    def _payload(self) -> dict:
        return {
            "hypothesis_id": str(self.hypothesis_id),
            "approved": self.approved,
            "audit_notes": self.audit_notes
        }


@dataclass
class FailureRecorded(DomainEvent):
    failure: FailureRecord
    
    def _payload(self) -> dict:
        return {
            "failure": self.failure.__dict__
        }


@dataclass
class TrainingSampleCreated(DomainEvent):
    sample: TrainingSample
    
    def _payload(self) -> dict:
        return {
            "sample": self.sample.__dict__
        }


@dataclass
class ModelFineTuned(DomainEvent):
    agent_id: AgentId
    training_samples: List[UUID]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    
    def _payload(self) -> dict:
        return {
            "agent_id": str(self.agent_id),
            "training_samples": [str(s) for s in self.training_samples],
            "performance_before": self.performance_before,
            "performance_after": self.performance_after
        }