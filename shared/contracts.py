"""Shared contracts: Pydantic models, CQRS commands/events, event sourcing foundation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, validator


T = TypeVar("T")


class TimestampMixin(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IdMixin(BaseModel):
    id: UUID = Field(default_factory=uuid4)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SkillCategory(str, Enum):
    DATA_SENSING = "data_sensing"
    COGNITIVE_REASONING = "cognitive_reasoning"
    CONTENT_GENERATION = "content_generation"
    SYSTEM_EXECUTION = "system_execution"
    GOVERNANCE_SECURITY = "governance_security"


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    RETIRED = "retired"


class JobStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    JOB_CREATED = "job.created"
    JOB_DISPATCHED = "job.dispatched"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    SKILL_LOADED = "skill.loaded"
    SKILL_RELEASED = "skill.released"
    HYPOTHESIS_GENERATED = "hypothesis.generated"
    HYPOTHESIS_APPROVED = "hypothesis.approved"
    HYPOTHESIS_REJECTED = "hypothesis.rejected"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    EXPERIMENT_FAILED = "experiment.failed"
    PAPER_WRITTEN = "paper.written"
    ARTIFACT_BUILT = "artifact.built"
    MAGPIE_FAILURE_RECORDED = "magpie.failure.recorded"
    MAGPIE_TRAINING_STARTED = "magpie.training.started"
    MAGPIE_TRAINING_COMPLETED = "magpie.training.completed"
    DREAM_HYPOTHESIS_CREATED = "dream.hypothesis.created"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_RESTORED = "checkpoint.restored"
    WORKSPACE_EVAPORATED = "workspace.evaporated"
    SKILL_AUDIT_PASSED = "skill.audit.passed"
    SKILL_AUDIT_FAILED = "skill.audit.failed"


class CloudProvider(str, Enum):
    GCP = "gcp"
    KAGGLE = "kaggle"


class PrecisionMode(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


# ---------------------------------------------------------------------------
# Base Event (Event Sourcing)
# ---------------------------------------------------------------------------

class BaseEvent(TimestampMixin, IdMixin, Generic[T]):
    event_type: EventType
    aggregate_id: UUID
    aggregate_type: str
    version: int = 1
    payload: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @validator("aggregate_type")
    def _non_empty_aggregate(cls, v):
        if not v or not v.strip():
            raise ValueError("aggregate_type must not be empty")
        return v


# ---------------------------------------------------------------------------
# CQRS Commands
# ---------------------------------------------------------------------------

class BaseCommand(TimestampMixin, IdMixin, Generic[T]):
    command_type: str
    payload: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))

    model_config = ConfigDict(extra="forbid")


class CreateJobCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "create_job"


class DispatchJobCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "dispatch_job"


class CompleteJobCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "complete_job"


class FailJobCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "fail_job"


class LoadSkillCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "load_skill"


class ReleaseSkillCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "release_skill"


class RecordMagpieFailureCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "record_magpie_failure"


class StartMagpieTrainingCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "start_magpie_training"


class CreateDreamHypothesisCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "create_dream_hypothesis"


class SaveCheckpointCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "save_checkpoint"


class RestoreCheckpointCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "restore_checkpoint"


class EvaporateWorkspaceCommand(BaseCommand[Dict[str, Any]]):
    command_type: str = "evaporate_workspace"


# ---------------------------------------------------------------------------
# CQRS Queries (Read Model)
# ---------------------------------------------------------------------------

class BaseQuery(TimestampMixin, IdMixin, Generic[T]):
    query_type: str
    payload: T
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class GetJobQuery(BaseQuery[Dict[str, Any]]):
    query_type: str = "get_job"


class ListJobsQuery(BaseQuery[Dict[str, Any]]):
    query_type: str = "list_jobs"


class GetSkillQuery(BaseQuery[Dict[str, Any]]):
    query_type: str = "get_skill"


class SearchSkillsQuery(BaseQuery[Dict[str, Any]]):
    query_type: str = "search_skills"


# ---------------------------------------------------------------------------
# Domain Aggregates
# ---------------------------------------------------------------------------

class SkillGene(TimestampMixin, IdMixin):
    """250 elite skill genes: 5 categories x 50 keys."""
    code: str = Field(..., description="CATEGORY_SKILLNAME e.g. DS_WEB_SCRAPER_DYNAMIC")
    name: str
    category: SkillCategory
    description: str
    open_source_url: Optional[str] = None
    download_command: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    status: SkillStatus = SkillStatus.CANDIDATE
    stability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    composability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    roi_score: float = Field(default=0.5, ge=0.0, le=1.0)
    uniqueness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    version: str = Field(default="1.0.0")
    sandbox_required: bool = True
    network_required: bool = False
    gpu_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Job(TimestampMixin, IdMixin):
    title: str
    description: str
    status: JobStatus = JobStatus.PENDING
    provider: CloudProvider = CloudProvider.GCP
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    skills_used: List[str] = Field(default_factory=list)
    checkpoints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Hypothesis(TimestampMixin, IdMixin):
    title: str
    description: str
    research_question: str
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    failure_criteria: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="pending")  # pending, approved, rejected, tested
    source: str = Field(default="dream")  # dream, research, user
    evidence_ids: List[UUID] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Experiment(TimestampMixin, IdMixin):
    hypothesis_id: UUID
    title: str
    plan: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="planned")  # planned, running, completed, failed
    results: Dict[str, Any] = Field(default_factory=dict)
    anchor_points: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Paper(TimestampMixin, IdMixin):
    experiment_id: UUID
    title: str
    content: str
    format: str = Field(default="markdown")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Artifact(TimestampMixin, IdMixin):
    paper_id: Optional[UUID] = None
    name: str
    type: str  # algorithm, compression, skill, tool, document
    content: str
    storage_path: Optional[str] = None
    sha256: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class FailureRecord(TimestampMixin, IdMixin):
    module: str
    component: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    distilled: bool = False
    training_material: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class Checkpoint(TimestampMixin, IdMixin):
    job_id: UUID
    step: str
    state: Dict[str, Any] = Field(default_factory=dict)
    storage_path: str
    sha256: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    id: UUID
    status: JobStatus
    trace_id: str
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SkillResponse(BaseModel):
    code: str
    name: str
    category: SkillCategory
    status: SkillStatus
    version: str


class HealthResponse(BaseModel):
    status: str
    provider: CloudProvider
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"


class EvaporationResponse(BaseModel):
    evaporated: bool
    workspace_cleared: bool
    checkpoint_saved: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

