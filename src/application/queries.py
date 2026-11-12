"""
Application Queries - Read operations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID
from ..shared.kernel.commands import Query
from ..domain.entities import (
    ProblemId, AgentId, SkillId, MemoryId, ResearchId,
    ProblemStatus, AgentRole, SkillCategory, GeneKey
)
from ..domain.aggregates import ProblemAggregate, AgentAggregate, SkillAggregate, ResearchAggregate


# Problem Queries
@dataclass
class GetProblemQuery(Query[ProblemAggregate]):
    problem_id: ProblemId


@dataclass
class GetProblemsByStatusQuery(Query[List[ProblemAggregate]]):
    status: ProblemStatus
    limit: int = 100


@dataclass
class GetRecentProblemsQuery(Query[List[ProblemAggregate]]):
    limit: int = 100


@dataclass
class GetProblemStatusQuery(Query[ProblemStatus]):
    problem_id: ProblemId


@dataclass
class GetProblemAnswerQuery(Query[Optional[Any]]):
    problem_id: ProblemId


# Agent Queries
@dataclass
class GetAgentQuery(Query[AgentAggregate]):
    agent_id: AgentId


@dataclass
class GetAvailableAgentsQuery(Query[List[AgentAggregate]]):
    role: Optional[AgentRole] = None


@dataclass
class GetAgentsByRoleQuery(Query[List[AgentAggregate]]):
    role: AgentRole


@dataclass
class GetAllAgentsQuery(Query[List[AgentAggregate]]):
    pass


@dataclass
class GetAgentCapabilitiesQuery(Query[Dict[str, Any]]):
    agent_id: AgentId


# Skill Queries
@dataclass
class GetSkillQuery(Query[SkillAggregate]):
    skill_id: SkillId


@dataclass
class GetSkillByGeneKeyQuery(Query[Optional[SkillAggregate]]):
    gene_key: GeneKey


@dataclass
class GetSkillsByCategoryQuery(Query[List[SkillAggregate]]):
    category: SkillCategory


@dataclass
class GetAvailableSkillsQuery(Query[List[SkillAggregate]]):
    pass


@dataclass
class GetAllSkillsQuery(Query[List[SkillAggregate]]):
    pass


@dataclass
class SearchSkillsQuery(Query[List[SkillAggregate]]):
    query: str
    limit: int = 10


@dataclass
class GetSkillGeneKeysQuery(Query[List[GeneKey]]):
    category: Optional[SkillCategory] = None


# Research Queries
@dataclass
class GetResearchQuery(Query[ResearchAggregate]):
    research_id: ResearchId


@dataclass
class GetResearchByProblemQuery(Query[List[ResearchAggregate]]):
    problem_id: ProblemId


@dataclass
class GetInProgressResearchQuery(Query[List[ResearchAggregate]]):
    pass


@dataclass
class GetCompletedResearchQuery(Query[List[ResearchAggregate]]):
    limit: int = 100


# Memory Queries
@dataclass
class SearchMemoryQuery(Query[List[Any]]):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


@dataclass
class GetMemoryByManifestQuery(Query[List[Any]]):
    object_type: str
    object_id: str


@dataclass
class GetMemoryAtomQuery(Query[Optional[Any]]):
    atom_id: MemoryId


# Pipeline Queries
@dataclass
class GetPipelineStatusQuery(Query[Dict[str, Any]]):
    pipeline_id: str


@dataclass
class GetPipelineCheckpointsQuery(Query[List[Dict[str, Any]]]):
    pipeline_id: str


@dataclass
class GetLatestCheckpointQuery(Query[Optional[Dict[str, Any]]]):
    pipeline_id: str


# System Queries
@dataclass
class GetSystemHealthQuery(Query[Dict[str, Any]]):
    pass


@dataclass
class GetSystemMetricsQuery(Query[Dict[str, Any]]):
    pass


@dataclass
class GetResourceUsageQuery(Query[Dict[str, Any]]):
    pass