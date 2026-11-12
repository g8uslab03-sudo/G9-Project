"""Typed contracts shared by the GCP controller and Kaggle workers.

Every structure is a plain dataclass with ``to_dict``/``from_dict`` so it can be
persisted as JSON without depending on pydantic or another runtime library.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from shared.jsonutil import jsonable


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def new_id(prefix: str) -> str:
    """Generate a short, human-readable identifier."""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clamp01(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _parse_dt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if not value:
        return utcnow()
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return utcnow()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class _StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def from_value(cls, value: Any):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            return None


class OnionLayer(_StrEnum):
    """Ordered analysis layers of the onion architecture."""

    PHENOMENON = "phenomenon"
    IMPACT = "impact"
    CONSTRAINT = "constraint"
    DIRECT_CAUSE = "direct_cause"
    ROOT_MECHANISM = "root_mechanism"
    VERIFICATION = "verification"


ONION_LAYERS: tuple[OnionLayer, ...] = tuple(OnionLayer)
ONION_LAYER_ORDER: dict[str, int] = {
    layer.value: index for index, layer in enumerate(ONION_LAYERS)
}


def next_onion_layer(layer: OnionLayer | str) -> OnionLayer | None:
    value = OnionLayer.from_value(layer)
    if value is None:
        return None
    index = ONION_LAYERS.index(value)
    if index + 1 >= len(ONION_LAYERS):
        return None
    return ONION_LAYERS[index + 1]


class ResearchLoop(_StrEnum):
    """The seven research loops."""

    PROBLEM_DEFINITION = "problem_definition"
    EVIDENCE_BASELINE = "evidence_baseline"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT_DESIGN = "experiment_design"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    REPLICATION_AUDIT_DELIVERY = "replication_audit_delivery"


LOOP_ORDER: tuple[ResearchLoop, ...] = tuple(ResearchLoop)
LOOP_NUMBERS: dict[str, int] = {
    loop.value: index + 1 for index, loop in enumerate(LOOP_ORDER)
}
TERMINAL_LOOPS: frozenset[str] = frozenset(
    {ResearchLoop.ANALYSIS.value, ResearchLoop.REPLICATION_AUDIT_DELIVERY.value}
)


class JobStatus(_StrEnum):
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


JOB_TERMINAL: frozenset[str] = frozenset(
    {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
)


class DispatchStatus(_StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class SkillCategory(_StrEnum):
    """Five skill classes used by the skill hierarchy."""

    RESEARCH = "research"
    DATA = "data"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


SKILL_CATEGORIES: tuple[SkillCategory, ...] = tuple(SkillCategory)


class SkillLifecycle(_StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    RETIRED = "retired"


class Severity(_StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StepStatus(_StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


SKILL_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
SKILL_CODE_MAX = 64


def make_skill_code(category: SkillCategory | str, name: str) -> str:
    category_value = SkillCategory.from_value(category)
    if category_value is None:
        raise ValueError(f"unknown skill category: {category!r}")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").upper()
    if not normalized:
        raise ValueError("skill name must not be empty")
    code = f"{category_value.value.upper()}_{normalized}"
    if len(code) > SKILL_CODE_MAX:
        raise ValueError(f"skill code exceeds {SKILL_CODE_MAX} characters")
    if not SKILL_CODE_RE.match(code):
        raise ValueError(f"invalid skill code: {code!r}")
    return code


def validate_skill_code(code: str) -> str:
    if not isinstance(code, str) or not SKILL_CODE_RE.match(code):
        raise ValueError(f"invalid skill code: {code!r}")
    if len(code) > SKILL_CODE_MAX:
        raise ValueError(f"skill code exceeds {SKILL_CODE_MAX} characters")
    return code


def validate_problem(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    if len(problem) > 20_000:
        raise ValueError("problem exceeds 20,000 characters")
    return problem.strip()


def _dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value)


@dataclass
class Evidence:
    evidence_id: str
    claim: str
    source: str
    source_type: str = "unknown"
    quote: str = ""
    url: str = ""
    retrieved_at: str = field(default_factory=utcnow)
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = new_id("ev")
        self.confidence = clamp01(self.confidence)
        self.retrieved_at = _parse_dt(self.retrieved_at)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        return cls(
            evidence_id=str(data.get("evidence_id") or new_id("ev")),
            claim=str(data.get("claim", "")),
            source=str(data.get("source", "unknown")),
            source_type=str(data.get("source_type", "unknown")),
            quote=str(data.get("quote", "")),
            url=str(data.get("url", "")),
            retrieved_at=_parse_dt(data.get("retrieved_at")),
            confidence=float(data.get("confidence", 0.5)),
            tags=list(data.get("tags") or []),
            fingerprint=str(data.get("fingerprint", "")),
        )


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    mechanism: str = ""
    rationale: str = ""
    testable_prediction: str = ""
    confidence: float = 0.5
    status: str = "proposed"
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            self.hypothesis_id = new_id("hyp")
        self.confidence = clamp01(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=str(data.get("hypothesis_id") or new_id("hyp")),
            statement=str(data.get("statement", "")),
            mechanism=str(data.get("mechanism", "")),
            rationale=str(data.get("rationale", "")),
            testable_prediction=str(data.get("testable_prediction", "")),
            confidence=float(data.get("confidence", 0.5)),
            status=str(data.get("status", "proposed")),
            evidence_ids=list(data.get("evidence_ids") or []),
        )


@dataclass
class Experiment:
    experiment_id: str
    hypothesis_id: str
    design: str = ""
    independent_variable: str = ""
    dependent_variables: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    success_criteria: str = ""
    sample_size: int = 0
    duration_days: int = 0
    status: str = "planned"
    result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = new_id("exp")
        try:
            self.sample_size = max(0, int(self.sample_size))
            self.duration_days = max(0, int(self.duration_days))
        except (TypeError, ValueError):
            self.sample_size = 0
            self.duration_days = 0

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Experiment":
        return cls(
            experiment_id=str(data.get("experiment_id") or new_id("exp")),
            hypothesis_id=str(data.get("hypothesis_id", "")),
            design=str(data.get("design", "")),
            independent_variable=str(data.get("independent_variable", "")),
            dependent_variables=list(data.get("dependent_variables") or []),
            metrics=list(data.get("metrics") or []),
            success_criteria=str(data.get("success_criteria", "")),
            sample_size=int(data.get("sample_size", 0) or 0),
            duration_days=int(data.get("duration_days", 0) or 0),
            status=str(data.get("status", "planned")),
            result=data.get("result"),
        )


@dataclass
class ExperimentResult:
    experiment_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_refs: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    interpretation: str = ""
    passed: bool | None = None
    executed_at: str = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        self.executed_at = _parse_dt(self.executed_at)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            metrics=dict(data.get("metrics") or {}),
            raw_refs=list(data.get("raw_refs") or []),
            artifacts=list(data.get("artifacts") or []),
            interpretation=str(data.get("interpretation", "")),
            passed=data.get("passed"),
            executed_at=_parse_dt(data.get("executed_at")),
        )


@dataclass
class OnionAnalysis:
    job_id: str
    layers: dict[str, str] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    verification: str = ""
    updated_at: str = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        self.updated_at = _parse_dt(self.updated_at)
        for key in list(self.layers):
            layer = OnionLayer.from_value(key)
            if layer is not None:
                self.layers[layer.value] = self.layers.pop(key)

    def set_layer(self, layer: OnionLayer | str, text: str) -> None:
        value = OnionLayer.from_value(layer)
        if value is None:
            raise ValueError(f"unknown onion layer: {layer!r}")
        self.layers[value.value] = str(text)
        self.updated_at = utcnow()

    def missing_layers(self) -> list[str]:
        return [layer.value for layer in ONION_LAYERS if layer.value not in self.layers]

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OnionAnalysis":
        return cls(
            job_id=str(data.get("job_id", "")),
            layers=dict(data.get("layers") or {}),
            evidence_ids=list(data.get("evidence_ids") or []),
            assumptions=list(data.get("assumptions") or []),
            verification=str(data.get("verification", "")),
            updated_at=_parse_dt(data.get("updated_at")),
        )


@dataclass
class ResearchStep:
    step_id: str
    loop: str
    skill_code: str
    status: str = StepStatus.PENDING.value
    started_at: str = field(default_factory=utcnow)
    finished_at: str = ""
    output_ref: str = ""
    error: str = ""
    attempt: int = 1

    def __post_init__(self) -> None:
        if not self.step_id:
            self.step_id = new_id("step")
        self.started_at = _parse_dt(self.started_at)
        if self.finished_at:
            self.finished_at = _parse_dt(self.finished_at)

    def finish(self, status: str, *, output_ref: str = "", error: str = "") -> None:
        self.status = status
        self.finished_at = utcnow()
        self.output_ref = output_ref
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchStep":
        return cls(
            step_id=str(data.get("step_id") or new_id("step")),
            loop=str(data.get("loop", "")),
            skill_code=str(data.get("skill_code", "")),
            status=str(data.get("status", StepStatus.PENDING.value)),
            started_at=_parse_dt(data.get("started_at")),
            finished_at=_parse_dt(data.get("finished_at")) if data.get("finished_at") else "",
            output_ref=str(data.get("output_ref", "")),
            error=str(data.get("error", "")),
            attempt=int(data.get("attempt", 1) or 1),
        )


@dataclass
class Job:
    job_id: str
    problem: str
    handler: str = "market_research"
    params: dict[str, Any] = field(default_factory=dict)
    status: str = JobStatus.QUEUED.value
    priority: int = 0
    owner: str = ""
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    steps: list[ResearchStep] = field(default_factory=list)
    result_ref: str = ""
    checkpoint_ref: str = ""
    retry_count: int = 0
    max_retries: int = 2
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = new_id("job")
        self.created_at = _parse_dt(self.created_at)
        self.updated_at = _parse_dt(self.updated_at)
        try:
            self.priority = int(self.priority)
            self.retry_count = max(0, int(self.retry_count))
            self.max_retries = max(0, int(self.max_retries))
        except (TypeError, ValueError):
            self.priority = 0
            self.retry_count = 0
            self.max_retries = 2

    def touch(self) -> None:
        self.updated_at = utcnow()

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Job":
        return cls(
            job_id=str(data.get("job_id") or new_id("job")),
            problem=str(data.get("problem", "")),
            handler=str(data.get("handler", "market_research")),
            params=dict(data.get("params") or {}),
            status=str(data.get("status", JobStatus.QUEUED.value)),
            priority=int(data.get("priority", 0) or 0),
            owner=str(data.get("owner", "")),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            steps=[ResearchStep.from_dict(item) for item in data.get("steps") or []],
            result_ref=str(data.get("result_ref", "")),
            checkpoint_ref=str(data.get("checkpoint_ref", "")),
            retry_count=int(data.get("retry_count", 0) or 0),
            max_retries=int(data.get("max_retries", 2) or 2),
            error=str(data.get("error", "")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Dispatch:
    job_id: str
    dispatch_id: str = ""
    status: str = DispatchStatus.PENDING.value
    attempt: int = 1
    worker_id: str = ""
    created_at: str = field(default_factory=utcnow)
    claimed_at: str = ""
    lease_expires_at: str = ""
    heartbeat_at: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if not self.dispatch_id:
            self.dispatch_id = new_id("dsp")
        self.created_at = _parse_dt(self.created_at)
        for name in ("claimed_at", "lease_expires_at", "heartbeat_at"):
            value = getattr(self, name)
            if value:
                setattr(self, name, _parse_dt(value))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dispatch":
        return cls(
            job_id=str(data.get("job_id", "")),
            dispatch_id=str(data.get("dispatch_id") or new_id("dsp")),
            status=str(data.get("status", DispatchStatus.PENDING.value)),
            attempt=int(data.get("attempt", 1) or 1),
            worker_id=str(data.get("worker_id", "")),
            created_at=_parse_dt(data.get("created_at")),
            claimed_at=_parse_dt(data.get("claimed_at")) if data.get("claimed_at") else "",
            lease_expires_at=_parse_dt(data.get("lease_expires_at")) if data.get("lease_expires_at") else "",
            heartbeat_at=_parse_dt(data.get("heartbeat_at")) if data.get("heartbeat_at") else "",
            error=str(data.get("error", "")),
        )


@dataclass
class Checkpoint:
    checkpoint_id: str
    job_id: str
    step_index: int = 0
    state_ref: str = ""
    state_hash: str = ""
    parent_checkpoint_id: str = ""
    created_at: str = field(default_factory=utcnow)
    merkle_root: str = ""
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            self.checkpoint_id = new_id("ckpt")
        self.created_at = _parse_dt(self.created_at)
        try:
            self.step_index = int(self.step_index)
        except (TypeError, ValueError):
            self.step_index = 0

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=str(data.get("checkpoint_id") or new_id("ckpt")),
            job_id=str(data.get("job_id", "")),
            step_index=int(data.get("step_index", 0) or 0),
            state_ref=str(data.get("state_ref", "")),
            state_hash=str(data.get("state_hash", "")),
            parent_checkpoint_id=str(data.get("parent_checkpoint_id", "")),
            created_at=_parse_dt(data.get("created_at")),
            merkle_root=str(data.get("merkle_root", "")),
            verified=bool(data.get("verified", False)),
        )


@dataclass
class ShadowFinding:
    finding_id: str
    category: str
    severity: str = Severity.MEDIUM.value
    question: str = ""
    evidence: str = ""
    recommendation: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = new_id("sf")
        self.severity = str(Severity.from_value(self.severity) or Severity.MEDIUM)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowFinding":
        return cls(
            finding_id=str(data.get("finding_id") or new_id("sf")),
            category=str(data.get("category", "general")),
            severity=str(data.get("severity", Severity.MEDIUM.value)),
            question=str(data.get("question", "")),
            evidence=str(data.get("evidence", "")),
            recommendation=str(data.get("recommendation", "")),
        )


@dataclass
class ShadowAudit:
    audit_id: str
    job_id: str
    findings: list[ShadowFinding] = field(default_factory=list)
    anchor_points: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "pending"
    created_at: str = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.audit_id:
            self.audit_id = new_id("audit")
        self.created_at = _parse_dt(self.created_at)

    @property
    def has_blocking_findings(self) -> bool:
        return any(
            finding.severity in {Severity.HIGH.value, Severity.CRITICAL.value}
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowAudit":
        return cls(
            audit_id=str(data.get("audit_id") or new_id("audit")),
            job_id=str(data.get("job_id", "")),
            findings=[ShadowFinding.from_dict(item) for item in data.get("findings") or []],
            anchor_points=list(data.get("anchor_points") or []),
            verdict=str(data.get("verdict", "pending")),
            created_at=_parse_dt(data.get("created_at")),
        )


@dataclass
class MarketSource:
    name: str
    source_type: str = "web"
    url: str = ""
    cadence: str = "manual"
    reliability: float = 0.5
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.reliability = clamp01(self.reliability)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketSource":
        return cls(
            name=str(data.get("name", "")),
            source_type=str(data.get("source_type", "web")),
            url=str(data.get("url", "")),
            cadence=str(data.get("cadence", "manual")),
            reliability=float(data.get("reliability", 0.5)),
            tags=list(data.get("tags") or []),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class SkillScore:
    stability: float = 0.0
    composability: float = 0.0
    roi: float = 0.0
    uniqueness: float = 0.0

    def __post_init__(self) -> None:
        self.stability = clamp01(self.stability)
        self.composability = clamp01(self.composability)
        self.roi = clamp01(self.roi)
        self.uniqueness = clamp01(self.uniqueness)

    def weighted(self, weights: Mapping[str, float] | None = None) -> float:
        weights = weights or {
            "stability": 0.30,
            "composability": 0.20,
            "roi": 0.35,
            "uniqueness": 0.15,
        }
        total = sum(float(weights.get(name, 0.0)) for name in
                    ("stability", "composability", "roi", "uniqueness"))
        if total <= 0:
            return 0.0
        return sum(
            getattr(self, name) * float(weights.get(name, 0.0))
            for name in ("stability", "composability", "roi", "uniqueness")
        ) / total

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillScore":
        return cls(
            stability=float(data.get("stability", 0.0)),
            composability=float(data.get("composability", 0.0)),
            roi=float(data.get("roi", 0.0)),
            uniqueness=float(data.get("uniqueness", 0.0)),
        )


def job_progress(job: Job) -> dict[str, Any]:
    total = len(job.steps)
    done = sum(1 for step in job.steps if step.status == StepStatus.SUCCEEDED.value)
    failed = sum(1 for step in job.steps if step.status == StepStatus.FAILED.value)
    return {
        "steps": total,
        "succeeded": done,
        "failed": failed,
        "percent": int(100 * done / total) if total else 0,
    }


def validate_job_transition(current: str, new: str) -> bool:
    current_value = JobStatus.from_value(current)
    new_value = JobStatus.from_value(new)
    if current_value is None or new_value is None:
        return False
    if current_value == new_value:
        return True
    allowed = {
        JobStatus.QUEUED: {JobStatus.DISPATCHED, JobStatus.CANCELLED, JobStatus.FAILED},
        JobStatus.DISPATCHED: {JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.FAILED},
        JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RECOVERING, JobStatus.CANCELLED},
        JobStatus.RECOVERING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
        JobStatus.COMPLETED: set(),
        JobStatus.FAILED: {JobStatus.QUEUED},
        JobStatus.CANCELLED: set(),
    }
    return new_value in allowed.get(current_value, set())


def _filter_dataclass_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


__all__ = [
    "Dispatch",
    "DispatchStatus",
    "Evidence",
    "Experiment",
    "ExperimentResult",
    "Hypothesis",
    "JOB_TERMINAL",
    "Job",
    "JobStatus",
    "LOOP_NUMBERS",
    "LOOP_ORDER",
    "MarketSource",
    "ONION_LAYERS",
    "ONION_LAYER_ORDER",
    "OnionAnalysis",
    "OnionLayer",
    "ResearchLoop",
    "ResearchStep",
    "SKILL_CATEGORIES",
    "SKILL_CODE_RE",
    "ShadowAudit",
    "ShadowFinding",
    "SkillCategory",
    "SkillLifecycle",
    "SkillScore",
    "StepStatus",
    "TERMINAL_LOOPS",
    "clamp01",
    "job_progress",
    "make_skill_code",
    "new_id",
    "next_onion_layer",
    "utcnow",
    "validate_job_transition",
    "validate_problem",
    "validate_skill_code",
]