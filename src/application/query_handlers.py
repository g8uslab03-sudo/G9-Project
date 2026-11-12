"""
Query Handlers - Handle read operations.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from uuid import UUID
from ..shared.kernel.commands import QueryHandler
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import NotFoundError
from ..domain.entities import (
    ProblemId, AgentId, SkillId, MemoryId, ResearchId,
    ProblemStatus, AgentRole, SkillCategory, GeneKey
)
from ..domain.aggregates import ProblemAggregate, AgentAggregate, SkillAggregate, ResearchAggregate
from .queries import *


class ProblemQueryHandlers:
    """Handlers for problem-related queries."""
    
    def __init__(self, problem_repo: 'ProblemRepository'):
        self._problem_repo = problem_repo
    
    async def handle_get_problem(self, query: GetProblemQuery) -> Result[ProblemAggregate]:
        return await self._problem_repo.get(query.problem_id)
    
    async def handle_get_problems_by_status(self, query: GetProblemsByStatusQuery) -> Result[List[ProblemAggregate]]:
        return await self._problem_repo.get_by_status(query.status, query.limit)
    
    async def handle_get_recent_problems(self, query: GetRecentProblemsQuery) -> Result[List[ProblemAggregate]]:
        return await self._problem_repo.get_recent(query.limit)
    
    async def handle_get_problem_status(self, query: GetProblemStatusQuery) -> Result[ProblemStatus]:
        aggregate_result = await self._problem_repo.get(query.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {query.problem_id} not found"))
        return success(aggregate_result.unwrap().status)
    
    async def handle_get_problem_answer(self, query: GetProblemAnswerQuery) -> Result[Optional[Any]]:
        aggregate_result = await self._problem_repo.get(query.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {query.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        if aggregate.integrated_answer:
            return success(aggregate.integrated_answer.answer)
        return success(None)


class AgentQueryHandlers:
    """Handlers for agent-related queries."""
    
    def __init__(self, agent_repo: 'AgentRepository'):
        self._agent_repo = agent_repo
    
    async def handle_get_agent(self, query: GetAgentQuery) -> Result[AgentAggregate]:
        return await self._agent_repo.get(query.agent_id)
    
    async def handle_get_available_agents(self, query: GetAvailableAgentsQuery) -> Result[List[AgentAggregate]]:
        return await self._agent_repo.get_available(query.role)
    
    async def handle_get_agents_by_role(self, query: GetAgentsByRoleQuery) -> Result[List[AgentAggregate]]:
        return await self._agent_repo.get_by_role(query.role)
    
    async def handle_get_all_agents(self, query: GetAllAgentsQuery) -> Result[List[AgentAggregate]]:
        return await self._agent_repo.get_all()
    
    async def handle_get_agent_capabilities(self, query: GetAgentCapabilitiesQuery) -> Result[Dict[str, Any]]:
        aggregate_result = await self._agent_repo.get(query.agent_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Agent {query.agent_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        return success({
            "agent_id": str(aggregate.id),
            "role": aggregate.capability.role.value,
            "skills": [str(s) for s in aggregate.capability.skills],
            "max_concurrent_tasks": aggregate.capability.max_concurrent_tasks,
            "current_load": aggregate.current_load,
            "supported_modalities": aggregate.capability.supported_modalities,
            "resource_limits": aggregate.capability.resource_limits,
            "performance_metrics": aggregate.capability.performance_metrics,
            "status": "available" if aggregate.is_available else "busy"
        })


class SkillQueryHandlers:
    """Handlers for skill-related queries."""
    
    def __init__(self, skill_repo: 'SkillRepository', gene_repo: 'SkillGeneRepository'):
        self._skill_repo = skill_repo
        self._gene_repo = gene_repo
    
    async def handle_get_skill(self, query: GetSkillQuery) -> Result[SkillAggregate]:
        return await self._skill_repo.get(query.skill_id)
    
    async def handle_get_skill_by_gene_key(self, query: GetSkillByGeneKeyQuery) -> Result[Optional[SkillAggregate]]:
        return await self._skill_repo.get_by_gene_key(query.gene_key)
    
    async def handle_get_skills_by_category(self, query: GetSkillsByCategoryQuery) -> Result[List[SkillAggregate]]:
        return await self._skill_repo.get_by_category(query.category)
    
    async def handle_get_available_skills(self, query: GetAvailableSkillsQuery) -> Result[List[SkillAggregate]]:
        return await self._skill_repo.get_available()
    
    async def handle_get_all_skills(self, query: GetAllSkillsQuery) -> Result[List[SkillAggregate]]:
        return await self._skill_repo.get_all()
    
    async def handle_search_skills(self, query: SearchSkillsQuery) -> Result[List[SkillAggregate]]:
        return await self._skill_repo.search(query.query, query.limit)
    
    async def handle_get_skill_gene_keys(self, query: GetSkillGeneKeysQuery) -> Result[List[GeneKey]]:
        if query.category:
            result = await self._gene_repo.get_genes_by_category(query.category)
        else:
            result = await self._gene_repo.get_all_genes()
        
        if result.is_failure():
            return failure(result.unwrap_err())
        
        gene_keys = [GeneKey.from_string(g["gene_key"]) for g in result.unwrap()]
        return success(gene_keys)


class ResearchQueryHandlers:
    """Handlers for research-related queries."""
    
    def __init__(self, research_repo: 'ResearchRepository'):
        self._research_repo = research_repo
    
    async def handle_get_research(self, query: GetResearchQuery) -> Result[ResearchAggregate]:
        return await self._research_repo.get(query.research_id)
    
    async def handle_get_research_by_problem(self, query: GetResearchByProblemQuery) -> Result[List[ResearchAggregate]]:
        return await self._research_repo.get_by_problem(query.problem_id)
    
    async def handle_get_in_progress_research(self, query: GetInProgressResearchQuery) -> Result[List[ResearchAggregate]]:
        return await self._research_repo.get_in_progress()
    
    async def handle_get_completed_research(self, query: GetCompletedResearchQuery) -> Result[List[ResearchAggregate]]:
        return await self._research_repo.get_completed(query.limit)


class MemoryQueryHandlers:
    """Handlers for memory-related queries."""
    
    def __init__(self, memory_repo: 'MemoryRepository'):
        self._memory_repo = memory_repo
    
    async def handle_search_memory(self, query: SearchMemoryQuery) -> Result[List[Any]]:
        return await self._memory_repo.search(query.query, query.limit, query.filters)
    
    async def handle_get_memory_by_manifest(self, query: GetMemoryByManifestQuery) -> Result[List[Any]]:
        return await self._memory_repo.get_by_manifest(query.object_type, query.object_id)
    
    async def handle_get_memory_atom(self, query: GetMemoryAtomQuery) -> Result[Optional[Any]]:
        return await self._memory_repo.get_atom(query.atom_id)


class PipelineQueryHandlers:
    """Handlers for pipeline-related queries."""
    
    def __init__(self, checkpoint_repo: 'CheckpointRepository'):
        self._checkpoint_repo = checkpoint_repo
    
    async def handle_get_pipeline_status(self, query: GetPipelineStatusQuery) -> Result[Dict[str, Any]]:
        checkpoint_result = await self._checkpoint_repo.get_latest_checkpoint(query.pipeline_id)
        if checkpoint_result.is_failure():
            return failure(checkpoint_result.unwrap_err())
        
        checkpoint = checkpoint_result.unwrap()
        if checkpoint:
            return success({
                "pipeline_id": query.pipeline_id,
                "current_stage": checkpoint.get("stage"),
                "last_updated": checkpoint.get("metadata", {}).get("timestamp"),
                "state": checkpoint.get("state")
            })
        return success({
            "pipeline_id": query.pipeline_id,
            "current_stage": "not_started",
            "last_updated": None,
            "state": None
        })
    
    async def handle_get_pipeline_checkpoints(self, query: GetPipelineCheckpointsQuery) -> Result[List[Dict[str, Any]]]:
        return await self._checkpoint_repo.list_checkpoints(query.pipeline_id)
    
    async def handle_get_latest_checkpoint(self, query: GetLatestCheckpointQuery) -> Result[Optional[Dict[str, Any]]]:
        return await self._checkpoint_repo.get_latest_checkpoint(query.pipeline_id)


class SystemQueryHandlers:
    """Handlers for system-related queries."""
    
    def __init__(
        self,
        problem_repo: 'ProblemRepository',
        agent_repo: 'AgentRepository',
        skill_repo: 'SkillRepository',
        research_repo: 'ResearchRepository'
    ):
        self._problem_repo = problem_repo
        self._agent_repo = agent_repo
        self._skill_repo = skill_repo
        self._research_repo = research_repo
    
    async def handle_get_system_health(self, query: GetSystemHealthQuery) -> Result[Dict[str, Any]]:
        # Check various system components
        problems_result = await self._problem_repo.get_recent(10)
        agents_result = await self._agent_repo.get_all()
        skills_result = await self._skill_repo.get_all()
        research_result = await self._research_repo.get_in_progress()
        
        return success({
            "status": "healthy",
            "components": {
                "problems": "ok" if problems_result.is_success() else "degraded",
                "agents": "ok" if agents_result.is_success() else "degraded",
                "skills": "ok" if skills_result.is_success() else "degraded",
                "research": "ok" if research_result.is_success() else "degraded"
            },
            "counts": {
                "recent_problems": len(problems_result.unwrap()) if problems_result.is_success() else 0,
                "total_agents": len(agents_result.unwrap()) if agents_result.is_success() else 0,
                "total_skills": len(skills_result.unwrap()) if skills_result.is_success() else 0,
                "in_progress_research": len(research_result.unwrap()) if research_result.is_success() else 0
            }
        })
    
    async def handle_get_system_metrics(self, query: GetSystemMetricsQuery) -> Result[Dict[str, Any]]:
        # Get metrics from all components
        problems_result = await self._problem_repo.get_recent(100)
        agents_result = await self._agent_repo.get_all()
        
        completed_problems = 0
        failed_problems = 0
        if problems_result.is_success():
            for p in problems_result.unwrap():
                if p.status == ProblemStatus.COMPLETED:
                    completed_problems += 1
                elif p.status == ProblemStatus.FAILED:
                    failed_problems += 1
        
        agent_loads = []
        if agents_result.is_success():
            for a in agents_result.unwrap():
                agent_loads.append({
                    "agent_id": str(a.id),
                    "role": a.capability.role.value,
                    "current_load": a.current_load,
                    "max_load": a.max_load
                })
        
        return success({
            "problems": {
                "completed": completed_problems,
                "failed": failed_problems,
                "success_rate": completed_problems / (completed_problems + failed_problems) if (completed_problems + failed_problems) > 0 else 0
            },
            "agents": agent_loads,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual clock
        })
    
    async def handle_get_resource_usage(self, query: GetResourceUsageQuery) -> Result[Dict[str, Any]]:
        # This would integrate with actual resource monitoring
        return success({
            "cpu_usage_percent": 0.0,
            "memory_usage_gb": 0.0,
            "gpu_usage_percent": 0.0,
            "gpu_memory_gb": 0.0,
            "disk_usage_gb": 0.0,
            "network_io_mbps": 0.0
        })


def register_query_handlers(bus: 'QueryBus', handlers: Dict[str, Any]) -> None:
    """Register all query handlers with the bus."""
    # Problem handlers
    problem_handlers = handlers.get('problem')
    if problem_handlers:
        bus.register(GetProblemHandler(problem_handlers))
        bus.register(GetProblemsByStatusHandler(problem_handlers))
        bus.register(GetRecentProblemsHandler(problem_handlers))
        bus.register(GetProblemStatusHandler(problem_handlers))
        bus.register(GetProblemAnswerHandler(problem_handlers))
    
    # Agent handlers
    agent_handlers = handlers.get('agent')
    if agent_handlers:
        bus.register(GetAgentHandler(agent_handlers))
        bus.register(GetAvailableAgentsHandler(agent_handlers))
        bus.register(GetAgentsByRoleHandler(agent_handlers))
        bus.register(GetAllAgentsHandler(agent_handlers))
        bus.register(GetAgentCapabilitiesHandler(agent_handlers))
    
    # Skill handlers
    skill_handlers = handlers.get('skill')
    if skill_handlers:
        bus.register(GetSkillHandler(skill_handlers))
        bus.register(GetSkillByGeneKeyHandler(skill_handlers))
        bus.register(GetSkillsByCategoryHandler(skill_handlers))
        bus.register(GetAvailableSkillsHandler(skill_handlers))
        bus.register(GetAllSkillsHandler(skill_handlers))
        bus.register(SearchSkillsHandler(skill_handlers))
        bus.register(GetSkillGeneKeysHandler(skill_handlers))
    
    # Research handlers
    research_handlers = handlers.get('research')
    if research_handlers:
        bus.register(GetResearchHandler(research_handlers))
        bus.register(GetResearchByProblemHandler(research_handlers))
        bus.register(GetInProgressResearchHandler(research_handlers))
        bus.register(GetCompletedResearchHandler(research_handlers))
    
    # Memory handlers
    memory_handlers = handlers.get('memory')
    if memory_handlers:
        bus.register(SearchMemoryHandler(memory_handlers))
        bus.register(GetMemoryByManifestHandler(memory_handlers))
        bus.register(GetMemoryAtomHandler(memory_handlers))
    
    # Pipeline handlers
    pipeline_handlers = handlers.get('pipeline')
    if pipeline_handlers:
        bus.register(GetPipelineStatusHandler(pipeline_handlers))
        bus.register(GetPipelineCheckpointsHandler(pipeline_handlers))
        bus.register(GetLatestCheckpointHandler(pipeline_handlers))
    
    # System handlers
    system_handlers = handlers.get('system')
    if system_handlers:
        bus.register(GetSystemHealthHandler(system_handlers))
        bus.register(GetSystemMetricsHandler(system_handlers))
        bus.register(GetResourceUsageHandler(system_handlers))


# Handler adapter classes
class GetProblemHandler(QueryHandler[GetProblemQuery, ProblemAggregate]):
    def __init__(self, handler: ProblemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetProblemQuery
    
    async def handle(self, query: GetProblemQuery) -> ProblemAggregate:
        result = await self._handler.handle_get_problem(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetProblemsByStatusHandler(QueryHandler[GetProblemsByStatusQuery, List[ProblemAggregate]]):
    def __init__(self, handler: ProblemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetProblemsByStatusQuery
    
    async def handle(self, query: GetProblemsByStatusQuery) -> List[ProblemAggregate]:
        result = await self._handler.handle_get_problems_by_status(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetRecentProblemsHandler(QueryHandler[GetRecentProblemsQuery, List[ProblemAggregate]]):
    def __init__(self, handler: ProblemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetRecentProblemsQuery
    
    async def handle(self, query: GetRecentProblemsQuery) -> List[ProblemAggregate]:
        result = await self._handler.handle_get_recent_problems(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetProblemStatusHandler(QueryHandler[GetProblemStatusQuery, ProblemStatus]):
    def __init__(self, handler: ProblemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetProblemStatusQuery
    
    async def handle(self, query: GetProblemStatusQuery) -> ProblemStatus:
        result = await self._handler.handle_get_problem_status(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetProblemAnswerHandler(QueryHandler[GetProblemAnswerQuery, Optional[Any]]):
    def __init__(self, handler: ProblemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetProblemAnswerQuery
    
    async def handle(self, query: GetProblemAnswerQuery) -> Optional[Any]:
        result = await self._handler.handle_get_problem_answer(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAgentHandler(QueryHandler[GetAgentQuery, AgentAggregate]):
    def __init__(self, handler: AgentQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAgentQuery
    
    async def handle(self, query: GetAgentQuery) -> AgentAggregate:
        result = await self._handler.handle_get_agent(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAvailableAgentsHandler(QueryHandler[GetAvailableAgentsQuery, List[AgentAggregate]]):
    def __init__(self, handler: AgentQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAvailableAgentsQuery
    
    async def handle(self, query: GetAvailableAgentsQuery) -> List[AgentAggregate]:
        result = await self._handler.handle_get_available_agents(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAgentsByRoleHandler(QueryHandler[GetAgentsByRoleQuery, List[AgentAggregate]]):
    def __init__(self, handler: AgentQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAgentsByRoleQuery
    
    async def handle(self, query: GetAgentsByRoleQuery) -> List[AgentAggregate]:
        result = await self._handler.handle_get_agents_by_role(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAllAgentsHandler(QueryHandler[GetAllAgentsQuery, List[AgentAggregate]]):
    def __init__(self, handler: AgentQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAllAgentsQuery
    
    async def handle(self, query: GetAllAgentsQuery) -> List[AgentAggregate]:
        result = await self._handler.handle_get_all_agents(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAgentCapabilitiesHandler(QueryHandler[GetAgentCapabilitiesQuery, Dict[str, Any]]):
    def __init__(self, handler: AgentQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAgentCapabilitiesQuery
    
    async def handle(self, query: GetAgentCapabilitiesQuery) -> Dict[str, Any]:
        result = await self._handler.handle_get_agent_capabilities(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSkillHandler(QueryHandler[GetSkillQuery, SkillAggregate]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSkillQuery
    
    async def handle(self, query: GetSkillQuery) -> SkillAggregate:
        result = await self._handler.handle_get_skill(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSkillByGeneKeyHandler(QueryHandler[GetSkillByGeneKeyQuery, Optional[SkillAggregate]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSkillByGeneKeyQuery
    
    async def handle(self, query: GetSkillByGeneKeyQuery) -> Optional[SkillAggregate]:
        result = await self._handler.handle_get_skill_by_gene_key(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSkillsByCategoryHandler(QueryHandler[GetSkillsByCategoryQuery, List[SkillAggregate]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSkillsByCategoryQuery
    
    async def handle(self, query: GetSkillsByCategoryQuery) -> List[SkillAggregate]:
        result = await self._handler.handle_get_skills_by_category(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAvailableSkillsHandler(QueryHandler[GetAvailableSkillsQuery, List[SkillAggregate]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAvailableSkillsQuery
    
    async def handle(self, query: GetAvailableSkillsQuery) -> List[SkillAggregate]:
        result = await self._handler.handle_get_available_skills(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetAllSkillsHandler(QueryHandler[GetAllSkillsQuery, List[SkillAggregate]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetAllSkillsQuery
    
    async def handle(self, query: GetAllSkillsQuery) -> List[SkillAggregate]:
        result = await self._handler.handle_get_all_skills(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class SearchSkillsHandler(QueryHandler[SearchSkillsQuery, List[SkillAggregate]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return SearchSkillsQuery
    
    async def handle(self, query: SearchSkillsQuery) -> List[SkillAggregate]:
        result = await self._handler.handle_search_skills(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSkillGeneKeysHandler(QueryHandler[GetSkillGeneKeysQuery, List[GeneKey]]):
    def __init__(self, handler: SkillQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSkillGeneKeysQuery
    
    async def handle(self, query: GetSkillGeneKeysQuery) -> List[GeneKey]:
        result = await self._handler.handle_get_skill_gene_keys(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetResearchHandler(QueryHandler[GetResearchQuery, ResearchAggregate]):
    def __init__(self, handler: ResearchQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetResearchQuery
    
    async def handle(self, query: GetResearchQuery) -> ResearchAggregate:
        result = await self._handler.handle_get_research(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetResearchByProblemHandler(QueryHandler[GetResearchByProblemQuery, List[ResearchAggregate]]):
    def __init__(self, handler: ResearchQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetResearchByProblemQuery
    
    async def handle(self, query: GetResearchByProblemQuery) -> List[ResearchAggregate]:
        result = await self._handler.handle_get_research_by_problem(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetInProgressResearchHandler(QueryHandler[GetInProgressResearchQuery, List[ResearchAggregate]]):
    def __init__(self, handler: ResearchQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetInProgressResearchQuery
    
    async def handle(self, query: GetInProgressResearchQuery) -> List[ResearchAggregate]:
        result = await self._handler.handle_get_in_progress_research(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetCompletedResearchHandler(QueryHandler[GetCompletedResearchQuery, List[ResearchAggregate]]):
    def __init__(self, handler: ResearchQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetCompletedResearchQuery
    
    async def handle(self, query: GetCompletedResearchQuery) -> List[ResearchAggregate]:
        result = await self._handler.handle_get_completed_research(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class SearchMemoryHandler(QueryHandler[SearchMemoryQuery, List[Any]]):
    def __init__(self, handler: MemoryQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return SearchMemoryQuery
    
    async def handle(self, query: SearchMemoryQuery) -> List[Any]:
        result = await self._handler.handle_search_memory(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetMemoryByManifestHandler(QueryHandler[GetMemoryByManifestQuery, List[Any]]):
    def __init__(self, handler: MemoryQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetMemoryByManifestQuery
    
    async def handle(self, query: GetMemoryByManifestQuery) -> List[Any]:
        result = await self._handler.handle_get_memory_by_manifest(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetMemoryAtomHandler(QueryHandler[GetMemoryAtomQuery, Optional[Any]]):
    def __init__(self, handler: MemoryQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetMemoryAtomQuery
    
    async def handle(self, query: GetMemoryAtomQuery) -> Optional[Any]:
        result = await self._handler.handle_get_memory_atom(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetPipelineStatusHandler(QueryHandler[GetPipelineStatusQuery, Dict[str, Any]]):
    def __init__(self, handler: PipelineQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetPipelineStatusQuery
    
    async def handle(self, query: GetPipelineStatusQuery) -> Dict[str, Any]:
        result = await self._handler.handle_get_pipeline_status(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetPipelineCheckpointsHandler(QueryHandler[GetPipelineCheckpointsQuery, List[Dict[str, Any]]]):
    def __init__(self, handler: PipelineQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetPipelineCheckpointsQuery
    
    async def handle(self, query: GetPipelineCheckpointsQuery) -> List[Dict[str, Any]]:
        result = await self._handler.handle_get_pipeline_checkpoints(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetLatestCheckpointHandler(QueryHandler[GetLatestCheckpointQuery, Optional[Dict[str, Any]]]):
    def __init__(self, handler: PipelineQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetLatestCheckpointQuery
    
    async def handle(self, query: GetLatestCheckpointQuery) -> Optional[Dict[str, Any]]:
        result = await self._handler.handle_get_latest_checkpoint(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSystemHealthHandler(QueryHandler[GetSystemHealthQuery, Dict[str, Any]]):
    def __init__(self, handler: SystemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSystemHealthQuery
    
    async def handle(self, query: GetSystemHealthQuery) -> Dict[str, Any]:
        result = await self._handler.handle_get_system_health(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetSystemMetricsHandler(QueryHandler[GetSystemMetricsQuery, Dict[str, Any]]):
    def __init__(self, handler: SystemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetSystemMetricsQuery
    
    async def handle(self, query: GetSystemMetricsQuery) -> Dict[str, Any]:
        result = await self._handler.handle_get_system_metrics(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class GetResourceUsageHandler(QueryHandler[GetResourceUsageQuery, Dict[str, Any]]):
    def __init__(self, handler: SystemQueryHandlers):
        self._handler = handler
    
    @property
    def query_type(self) -> type:
        return GetResourceUsageQuery
    
    async def handle(self, query: GetResourceUsageQuery) -> Dict[str, Any]:
        result = await self._handler.handle_get_resource_usage(query)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()