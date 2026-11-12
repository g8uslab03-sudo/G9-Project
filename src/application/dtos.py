"""
Application DTOs - Data Transfer Objects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime
from ..domain.entities import ProblemStatus, AgentRole, SkillCategory, ResearchPhase


@dataclass
class ProblemDTO:
    problem_id: str
    user_query: str
    domain: str
    status: str
    constraints: Dict[str, Any]
    available_resources: Dict[str, Any]
    priority: int
    created_at: datetime
    updated_at: datetime
    answer: Optional[str] = None
    confidence: Optional[float] = None
    sub_problems: List['SubProblemDTO'] = field(default_factory=list)


@dataclass
class SubProblemDTO:
    sub_id: str
    description: str
    required_skills: List[str]
    assigned_agent: Optional[str] = None
    status: str = "received"
    result: Optional[Any] = None
    verification_status: str = "pending"
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentDTO:
    agent_id: str
    role: str
    skills: List[str]
    max_concurrent_tasks: int
    current_load: int
    supported_modalities: List[str]
    resource_limits: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: str


@dataclass
class SkillDTO:
    skill_id: str
    gene_key: str
    name: str
    description: str
    category: str
    source: str
    version: str
    status: str
    usage_count: int
    success_rate: float
    tags: List[str]
    resource_requirements: Dict[str, Any]


@dataclass
class ResearchDTO:
    research_id: str
    problem_id: str
    current_phase: str
    is_complete: bool
    phases: List[str]
    hypotheses_count: int
    experiments_count: int
    has_paper: bool
    artifacts_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class HypothesisDTO:
    hypothesis_id: str
    research_id: str
    statement: str
    anchor_papers: List[str]
    confidence: float
    testable_predictions: List[str]
    status: str


@dataclass
class ExperimentResultDTO:
    experiment_id: str
    hypothesis_id: str
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time_seconds: float
    resource_usage: Dict[str, Any]


@dataclass
class ResearchPaperDTO:
    paper_id: str
    research_id: str
    title: str
    abstract: str
    sections: Dict[str, str]
    references: List[str]
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    generated_at: datetime


@dataclass
class ArtifactDTO:
    artifact_id: str
    research_id: str
    artifact_type: str
    name: str
    description: str
    code: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    benchmark_results: Optional[Dict[str, float]] = None
    validation_status: str = "pending"


@dataclass
class DreamHypothesisDTO:
    hypothesis_id: str
    title: str
    description: str
    source_databases: List[str]
    anchor_evidence: List[str]
    novelty_score: float
    feasibility_score: float
    impact_score: float
    status: str
    audit_notes: Optional[str] = None


@dataclass
class PipelineStatusDTO:
    pipeline_id: str
    problem_id: str
    current_stage: str
    stages_completed: List[str]
    stage_results: Dict[str, Any]
    started_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class CheckpointDTO:
    pipeline_id: str
    stage: str
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class SystemHealthDTO:
    status: str
    components: Dict[str, str]
    counts: Dict[str, int]
    timestamp: datetime


@dataclass
class SystemMetricsDTO:
    problems: Dict[str, Any]
    agents: List[Dict[str, Any]]
    timestamp: str


@dataclass
class ResourceUsageDTO:
    cpu_usage_percent: float
    memory_usage_gb: float
    gpu_usage_percent: float
    gpu_memory_gb: float
    disk_usage_gb: float
    network_io_mbps: float


@dataclass
class SkillGeneDTO:
    gene_key: str
    category: str
    index: int
    status: str
    skill_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Mapping functions
def problem_to_dto(problem: Any) -> ProblemDTO:
    """Convert ProblemAggregate to DTO."""
    return ProblemDTO(
        problem_id=str(problem.id),
        user_query=problem.context.user_query,
        domain=problem.context.domain,
        status=problem.status.value,
        constraints=problem.context.constraints,
        available_resources=problem.context.available_resources,
        priority=problem.context.priority,
        created_at=problem.created_at,
        updated_at=problem.updated_at,
        answer=problem.integrated_answer.answer if problem.integrated_answer else None,
        confidence=problem.integrated_answer.confidence if problem.integrated_answer else None,
        sub_problems=[sub_problem_to_dto(sp) for sp in problem.sub_problems]
    )


def sub_problem_to_dto(sp: Any) -> SubProblemDTO:
    """Convert SubProblem to DTO."""
    return SubProblemDTO(
        sub_id=str(sp.sub_id),
        description=sp.description,
        required_skills=[str(s) for s in sp.required_skills],
        assigned_agent=str(sp.assigned_agent) if sp.assigned_agent else None,
        status=sp.status.value,
        result=sp.result,
        verification_status=sp.verification.value,
        dependencies=[str(d) for d in sp.dependencies]
    )


def agent_to_dto(agent: Any) -> AgentDTO:
    """Convert AgentAggregate to DTO."""
    return AgentDTO(
        agent_id=str(agent.id),
        role=agent.capability.role.value,
        skills=[str(s) for s in agent.capability.skills],
        max_concurrent_tasks=agent.capability.max_concurrent_tasks,
        current_load=agent.current_load,
        supported_modalities=agent.capability.supported_modalities,
        resource_limits=agent.capability.resource_limits,
        performance_metrics=agent.capability.performance_metrics,
        status="available" if agent.is_available else "busy"
    )


def skill_to_dto(skill: Any) -> SkillDTO:
    """Convert SkillAggregate to DTO."""
    return SkillDTO(
        skill_id=str(skill.id),
        gene_key=str(skill.spec.gene_key),
        name=skill.spec.name,
        description=skill.spec.description,
        category=skill.spec.gene_key.category.value,
        source=skill.spec.source,
        version=skill.spec.version,
        status=skill.status,
        usage_count=skill._usage_count,
        success_rate=skill.success_rate,
        tags=skill.spec.tags,
        resource_requirements=skill.spec.resource_requirements
    )


def research_to_dto(research: Any) -> ResearchDTO:
    """Convert ResearchAggregate to DTO."""
    return ResearchDTO(
        research_id=str(research.id),
        problem_id=str(research._problem_id),
        current_phase=research.current_phase.value,
        is_complete=research.is_complete,
        phases=[p.value for p in research._plan.phases],
        hypotheses_count=len(research._hypotheses),
        experiments_count=len(research._experiment_results),
        has_paper=research._paper is not None,
        artifacts_count=len(research._artifacts),
        created_at=research.created_at,
        updated_at=research.updated_at
    )


def hypothesis_to_dto(h: Any) -> HypothesisDTO:
    """Convert Hypothesis to DTO."""
    return HypothesisDTO(
        hypothesis_id=str(h.hypothesis_id),
        research_id=str(h.research_id),
        statement=h.statement,
        anchor_papers=h.anchor_papers,
        confidence=h.confidence,
        testable_predictions=h.testable_predictions,
        status=h.status
    )


def experiment_to_dto(exp: Any) -> ExperimentResultDTO:
    """Convert ExperimentResult to DTO."""
    return ExperimentResultDTO(
        experiment_id=str(exp.experiment_id),
        hypothesis_id=str(exp.hypothesis_id),
        success=exp.success,
        data=exp.data,
        metrics=exp.metrics,
        execution_time_seconds=exp.execution_time_seconds,
        resource_usage=exp.resource_usage
    )


def paper_to_dto(paper: Any) -> ResearchPaperDTO:
    """Convert ResearchPaper to DTO."""
    return ResearchPaperDTO(
        paper_id=str(paper.paper_id),
        research_id=str(paper.research_id),
        title=paper.title,
        abstract=paper.abstract,
        sections=paper.sections,
        references=paper.references,
        figures=paper.figures,
        tables=paper.tables,
        generated_at=paper.generated_at
    )


def artifact_to_dto(artifact: Any) -> ArtifactDTO:
    """Convert Artifact to DTO."""
    return ArtifactDTO(
        artifact_id=str(artifact.artifact_id),
        research_id=str(artifact.research_id),
        artifact_type=artifact.artifact_type,
        name=artifact.name,
        description=artifact.description,
        code=artifact.code,
        config=artifact.config,
        benchmark_results=artifact.benchmark_results,
        validation_status=artifact.validation_status
    )


def dream_hypothesis_to_dto(dh: Any) -> DreamHypothesisDTO:
    """Convert DreamHypothesis to DTO."""
    return DreamHypothesisDTO(
        hypothesis_id=str(dh.hypothesis_id),
        title=dh.title,
        description=dh.description,
        source_databases=dh.source_databases,
        anchor_evidence=dh.anchor_evidence,
        novelty_score=dh.novelty_score,
        feasibility_score=dh.feasibility_score,
        impact_score=dh.impact_score,
        status=dh.status,
        audit_notes=dh.audit_notes
    )