"""
Domain Services - Business logic that doesn't belong to a single entity.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from uuid import UUID, uuid4
from enum import Enum
from ..shared.kernel.base import DomainEvent
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import BusinessRuleError, ValidationError
from .entities import (
    ProblemId, ProblemStatus, AgentId, AgentRole, SkillId, SkillCategory,
    GeneKey, SkillSpec, AgentCapability, ProblemContext, SubProblem,
    VerificationResult, VerificationStatus, DebateArgument, IntegratedAnswer,
    MemoryId, MemoryAtom, MemoryManifest, MemoryGraphEdge,
    ResearchId, ResearchPhase, ResearchPlan, Hypothesis, ExperimentResult,
    ResearchPaper, Artifact, DreamHypothesis, FailureRecord, TrainingSample
)
from .aggregates import ProblemAggregate, AgentAggregate, SkillAggregate, ResearchAggregate


class ProblemDecompositionService:
    """Service for decomposing problems into sub-problems."""
    
    def __init__(self, skill_registry: 'SkillRegistry'):
        self._skill_registry = skill_registry
    
    def decompose(
        self,
        problem_context: ProblemContext,
        available_skills: List[GeneKey]
    ) -> Result[List[SubProblem]]:
        """Decompose problem based on understanding and available skills."""
        # This would use LLM to analyze and decompose
        # For now, return a structured decomposition
        sub_problems = []
        
        # Analyze problem domain and required skills
        required_skills = self._analyze_required_skills(problem_context, available_skills)
        
        # Create sub-problems based on skill groups
        skill_groups = self._group_skills(required_skills)
        
        for i, (group_name, skills) in enumerate(skill_groups.items()):
            sub_id = uuid4()
            sub_problems.append(SubProblem(
                sub_id=sub_id,
                description=f"{group_name}: {problem_context.user_query[:100]}",
                required_skills=skills,
                dependencies=[]
            ))
        
        # Add dependencies based on skill flow
        for i in range(1, len(sub_problems)):
            sub_problems[i] = sub_problems[i].with_changes(
                dependencies=[sub_problems[i-1].sub_id]
            )
        
        return success(sub_problems)
    
    def _analyze_required_skills(
        self,
        context: ProblemContext,
        available_skills: List[GeneKey]
    ) -> List[GeneKey]:
        """Analyze which skills are needed for the problem."""
        # This would use semantic matching between problem and skills
        # For now, return all available skills as a simplification
        return available_skills[:10]  # Limit to top 10
    
    def _group_skills(self, skills: List[GeneKey]) -> Dict[str, List[GeneKey]]:
        """Group skills by category."""
        groups: Dict[str, List[GeneKey]] = {}
        for skill in skills:
            category = skill.category.value
            if category not in groups:
                groups[category] = []
            groups[category].append(skill)
        return groups


class AgentAssignmentService:
    """Service for assigning agents to sub-problems."""
    
    def __init__(self, agent_registry: 'AgentRegistry'):
        self._agent_registry = agent_registry
    
    def assign(
        self,
        sub_problems: List[SubProblem],
        available_agents: List[AgentCapability]
    ) -> Result[Dict[UUID, AgentId]]:
        """Assign best agents to sub-problems based on skills and load."""
        assignments = {}
        agent_loads: Dict[AgentId, int] = {agent.agent_id: 0 for agent in available_agents}
        
        for sub_problem in sub_problems:
            best_agent = self._find_best_agent(sub_problem, available_agents, agent_loads)
            if best_agent:
                assignments[sub_problem.sub_id] = best_agent.agent_id
                agent_loads[best_agent.agent_id] += 1
            else:
                return failure(BusinessRuleError(f"No suitable agent for sub-problem {sub_problem.sub_id}"))
        
        return success(assignments)
    
    def _find_best_agent(
        self,
        sub_problem: SubProblem,
        available_agents: List[AgentCapability],
        agent_loads: Dict[AgentId, int]
    ) -> Optional[AgentCapability]:
        """Find best agent for sub-problem."""
        scored_agents = []
        
        for agent in available_agents:
            if agent_loads[agent.agent_id] >= agent.max_concurrent_tasks:
                continue
            
            # Score based on skill match
            skill_match = len(set(sub_problem.required_skills) & set(agent.skills))
            if skill_match == 0:
                continue
            
            # Score based on performance
            perf_score = agent.performance_metrics.get("success_rate", 0.5)
            
            # Score based on current load (prefer less loaded)
            load_factor = 1.0 - (agent_loads[agent.agent_id] / agent.max_concurrent_tasks)
            
            total_score = skill_match * 0.5 + perf_score * 0.3 + load_factor * 0.2
            scored_agents.append((total_score, agent))
        
        if not scored_agents:
            return None
        
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]


class VerificationService:
    """Service for mutual verification between agents."""
    
    def verify(
        self,
        verifier_id: AgentId,
        target_id: AgentId,
        sub_problem_id: UUID,
        target_result: Any,
        criteria: Dict[str, Any]
    ) -> Result[VerificationResult]:
        """Verify another agent's result."""
        # This would implement actual verification logic
        # For now, return a mock verification
        
        issues = []
        suggestions = []
        confidence = 0.8
        status = VerificationStatus.PASSED
        
        # Check criteria
        for criterion, expected in criteria.items():
            actual = target_result.get(criterion) if isinstance(target_result, dict) else None
            if actual != expected:
                issues.append(f"Criterion '{criterion}' mismatch: expected {expected}, got {actual}")
                confidence -= 0.1
        
        if issues:
            status = VerificationStatus.NEEDS_REVIEW
            if len(issues) > 3:
                status = VerificationStatus.FAILED
        
        return success(VerificationResult(
            verifier_id=verifier_id,
            target_id=target_id,
            sub_problem_id=sub_problem_id,
            status=status,
            confidence=max(0.0, confidence),
            issues=issues,
            suggestions=suggestions
        ))


class DebateService:
    """Service for managing debate between agents."""
    
    def conduct_debate(
        self,
        sub_problem_id: UUID,
        arguments: List[DebateArgument]
    ) -> Result[Optional[str]]:
        """Conduct debate and find consensus."""
        if not arguments:
            return success(None)
        
        # Group by position
        support_args = [a for a in arguments if a.position == "support"]
        oppose_args = [a for a in arguments if a.position == "oppose"]
        neutral_args = [a for a in arguments if a.position == "neutral"]
        
        # Calculate weighted scores
        support_score = sum(a.confidence for a in support_args)
        oppose_score = sum(a.confidence for a in oppose_args)
        
        # Determine consensus
        if support_score > oppose_score * 1.5:
            consensus = "support"
        elif oppose_score > support_score * 1.5:
            consensus = "oppose"
        else:
            consensus = "mixed"
        
        return success(consensus)


class IntegrationService:
    """Service for integrating multiple agent outputs into final answer."""
    
    def integrate(
        self,
        problem_id: ProblemId,
        sub_problem_results: Dict[UUID, Any],
        verifications: Dict[UUID, List[VerificationResult]],
        debates: Dict[UUID, List[DebateArgument]],
        skill_contributions: Dict[GeneKey, float]
    ) -> Result[IntegratedAnswer]:
        """Integrate all results into final answer."""
        # Collect evidence
        supporting_evidence = []
        dissenting_opinions = []
        
        for sub_id, result in sub_problem_results.items():
            if isinstance(result, dict):
                supporting_evidence.append(f"Sub-problem {sub_id}: {result.get('summary', 'Completed')}")
            else:
                supporting_evidence.append(f"Sub-problem {sub_id}: {str(result)[:200]}")
        
        for sub_id, vers in verifications.items():
            for v in vers:
                if v.status == VerificationStatus.FAILED:
                    dissenting_opinions.append(f"Verification failed for {sub_id}: {v.issues}")
                elif v.status == VerificationStatus.NEEDS_REVIEW:
                    dissenting_opinions.append(f"Verification needs review for {sub_id}: {v.issues}")
        
        for sub_id, args in debates.items():
            for arg in args:
                if arg.position == "oppose":
                    dissenting_opinions.append(f"Opposing argument for {sub_id}: {arg.argument}")
        
        # Calculate overall confidence
        confidence = 0.7  # Base confidence
        if dissenting_opinions:
            confidence -= min(0.3, len(dissenting_opinions) * 0.05)
        
        # Generate answer (would use LLM in practice)
        answer_text = self._synthesize_answer(sub_problem_results, verifications, debates)
        
        return success(IntegratedAnswer(
            problem_id=problem_id,
            answer=answer_text,
            confidence=max(0.0, min(1.0, confidence)),
            supporting_evidence=supporting_evidence,
            dissenting_opinions=dissenting_opinions,
            skill_contributions=skill_contributions
        ))
    
    def _synthesize_answer(
        self,
        results: Dict[UUID, Any],
        verifications: Dict[UUID, List[VerificationResult]],
        debates: Dict[UUID, List[DebateArgument]]
    ) -> str:
        """Synthesize final answer from components."""
        parts = ["## Integrated Solution\n"]
        
        for sub_id, result in results.items():
            parts.append(f"### Sub-problem {sub_id}\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    parts.append(f"- **{key}**: {value}\n")
            else:
                parts.append(f"{result}\n")
            parts.append("\n")
        
        if dissenting_opinions := [op for sub_id, vers in verifications.items() 
                                     for v in vers if v.status != VerificationStatus.PASSED
                                     for op in v.issues]:
            parts.append("### Caveats and Limitations\n")
            for op in dissenting_opinions:
                parts.append(f"- {op}\n")
        
        return "".join(parts)


class SkillAcquisitionService:
    """Service for acquiring skills from open source or research."""
    
    def __init__(
        self,
        skill_registry: 'SkillRegistry',
        open_source_client: 'OpenSourceClient',
        research_module: 'ResearchModule'
    ):
        self._skill_registry = skill_registry
        self._open_source_client = open_source_client
        self._research_module = research_module
    
    async def acquire_skill(self, gene_key: GeneKey) -> Result[SkillSpec]:
        """Acquire skill - try open source first, then research."""
        # Step 1: Check if already in registry
        existing = self._skill_registry.get_by_gene_key(gene_key)
        if existing:
            return success(existing.spec)
        
        # Step 2: Try to download from open source
        download_result = await self._open_source_client.search_and_download(gene_key)
        if download_result.is_success():
            skill_spec = download_result.unwrap()
            return success(skill_spec)
        
        # Step 3: Request research module to create
        research_result = await self._research_module.create_skill(gene_key)
        if research_result.is_success():
            skill_spec = research_result.unwrap()
            return success(skill_spec)
        
        return failure(BusinessRuleError(f"Failed to acquire skill {gene_key}"))


class MemoryManagementService:
    """Service for managing memory system."""
    
    def __init__(self, memory_store: 'MemoryStore'):
        self._memory_store = memory_store
    
    def store_atom(self, atom: MemoryAtom) -> Result[MemoryId]:
        """Store atomic memory."""
        return self._memory_store.store_atom(atom)
    
    def store_manifest(self, manifest: MemoryManifest) -> Result[UUID]:
        """Store memory manifest."""
        return self._memory_store.store_manifest(manifest)
    
    def add_relationship(self, edge: MemoryGraphEdge) -> Result[None]:
        """Add relationship between atoms."""
        return self._memory_store.add_edge(edge)
    
    def retrieve_by_query(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Result[List[MemoryAtom]]:
        """Retrieve atoms by semantic query."""
        return self._memory_store.search(query, limit, filters)
    
    def get_object_memory(self, object_type: str, object_id: str) -> Result[List[MemoryAtom]]:
        """Get all atoms for an object via manifest."""
        return self._memory_store.get_by_manifest(object_type, object_id)


class ResearchPlanningService:
    """Service for planning research execution."""
    
    def create_plan(
        self,
        problem_id: ProblemId,
        problem_context: ProblemContext,
        available_agents: List[AgentCapability],
        available_tools: List[str]
    ) -> Result[ResearchPlan]:
        """Create research plan with 7 phases."""
        research_id = ResearchId()
        
        # Assign agents to phases based on capabilities
        agents_per_phase = {}
        for phase in ResearchPhase:
            suitable_agents = [
                a for a in available_agents
                if self._is_agent_suitable_for_phase(a, phase)
            ]
            agents_per_phase[phase] = suitable_agents[:3]  # Max 3 per phase
        
        # Tools per phase
        tools_per_phase = {
            ResearchPhase.ANALYZE_AND_PLAN: ["llm_reasoning", "task_decomposition"],
            ResearchPhase.LITERATURE_REVIEW: ["arxiv_search", "semantic_scholar", "pubmed", "crossref"],
            ResearchPhase.HYPOTHESIS_GENERATION: ["llm_reasoning", "knowledge_graph"],
            ResearchPhase.EXPERIMENT_EXECUTION: ["python_sandbox", "jupyter", "docker"],
            ResearchPhase.DATA_ANALYSIS: ["pandas", "numpy", "scipy", "visualization"],
            ResearchPhase.PAPER_GENERATION: ["llm_writing", "latex", "reference_manager"],
            ResearchPhase.ARTIFACT_PRODUCTION: ["code_generator", "docker", "benchmark_runner"]
        }
        
        # Audit checkpoints - every phase needs audit
        audit_checkpoints = list(ResearchPhase)
        
        # Sandbox config
        sandbox_config = {
            "isolation": "full",
            "network_access": False,
            "resource_limits": {"cpu": 2, "memory_gb": 8, "disk_gb": 10},
            "timeout_seconds": 3600
        }
        
        plan = ResearchPlan(
            research_id=research_id,
            problem_id=problem_id,
            phases=list(ResearchPhase),
            assigned_agents=agents_per_phase,
            tools_per_phase=tools_per_phase,
            sandbox_config=sandbox_config,
            audit_checkpoints=audit_checkpoints
        )
        
        return success(plan)
    
    def _is_agent_suitable_for_phase(self, agent: AgentCapability, phase: ResearchPhase) -> bool:
        """Check if agent has skills for phase."""
        phase_skills = {
            ResearchPhase.ANALYZE_AND_PLAN: [SkillCategory.LOGICAL_REASONING, SkillCategory.PROGRAMMING],
            ResearchPhase.LITERATURE_REVIEW: [SkillCategory.INFORMATION_RETRIEVAL],
            ResearchPhase.HYPOTHESIS_GENERATION: [SkillCategory.LOGICAL_REASONING, SkillCategory.INFORMATION_RETRIEVAL],
            ResearchPhase.EXPERIMENT_EXECUTION: [SkillCategory.PROGRAMMING],
            ResearchPhase.DATA_ANALYSIS: [SkillCategory.LOGICAL_REASONING, SkillCategory.PROGRAMMING],
            ResearchPhase.PAPER_GENERATION: [SkillCategory.CONTENT_CREATION],
            ResearchPhase.ARTIFACT_PRODUCTION: [SkillCategory.PROGRAMMING]
        }
        
        required_categories = phase_skills.get(phase, [])
        agent_categories = set()
        for skill_id in agent.skills:
            # Would need to look up skill category
            pass
        
        return True  # Simplified


class DreamGenerationService:
    """Service for generating hypotheses during idle time."""
    
    def __init__(
        self,
        open_source_client: 'OpenSourceClient',
        memory_service: MemoryManagementService
    ):
        self._open_source_client = open_source_client
        self._memory_service = memory_service
    
    async def generate_hypotheses(self, count: int = 5) -> Result[List[DreamHypothesis]]:
        """Generate novel hypotheses from open source data."""
        hypotheses = []
        
        # Get recent papers, trends, news
        sources = [
            "arxiv", "pubmed", "semantic_scholar", "openalex",
            "huggingface_papers", "papers_with_code", "github_trending",
            "hacker_news", "lobsteers"
        ]
        
        for i in range(count):
            # Select random source
            import random
            source = random.choice(sources)
            
            # Fetch recent items
            items = await self._open_source_client.fetch_recent(source, limit=10)
            
            if items:
                # Generate hypothesis from items
                hypothesis = self._synthesize_hypothesis(items, source)
                hypotheses.append(hypothesis)
        
        return success(hypotheses)
    
    def _synthesize_hypothesis(self, items: List[Dict], source: str) -> DreamHypothesis:
        """Synthesize hypothesis from source items."""
        # This would use LLM to generate novel hypotheses
        # For now, create a template hypothesis
        return DreamHypothesis(
            hypothesis_id=uuid4(),
            title=f"Novel approach from {source} trends",
            description=f"Based on recent developments in {source}, we hypothesize...",
            source_databases=[source],
            anchor_evidence=[item.get("id", "") for item in items[:3]],
            novelty_score=0.7,
            feasibility_score=0.6,
            impact_score=0.8
        )


class MagpieDistillationService:
    """Service for distilling failures into training data."""
    
    def __init__(self, failure_store: 'FailureStore', training_store: 'TrainingStore'):
        self._failure_store = failure_store
        self._training_store = training_store
    
    def distill_failures(self, module: Optional[str] = None) -> Result[List[TrainingSample]]:
        """Distill failure records into training samples."""
        failures = self._failure_store.get_unresolved(module)
        if not failures:
            return success([])
        
        # Group by failure pattern
        patterns = self._group_by_pattern(failures)
        samples = []
        
        for pattern, pattern_failures in patterns.items():
            sample = self._create_training_sample(pattern, pattern_failures)
            samples.append(sample)
        
        return success(samples)
    
    def _group_by_pattern(self, failures: List[FailureRecord]) -> Dict[str, List[FailureRecord]]:
        """Group failures by pattern."""
        patterns: Dict[str, List[FailureRecord]] = {}
        
        for failure in failures:
            pattern = f"{failure.module}:{failure.error_type}"
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(failure)
        
        return patterns
    
    def _create_training_sample(self, pattern: str, failures: List[FailureRecord]) -> TrainingSample:
        """Create training sample from failure pattern."""
        # Extract common context and create correction
        sample_id = uuid4()
        
        # Use first failure as base
        base = failures[0]
        
        return TrainingSample(
            sample_id=sample_id,
            failure_ids=[f.failure_id for f in failures],
            input_data=base.context,
            expected_output={"success": True, "error": None},
            failure_pattern=pattern,
            correction=self._generate_correction(pattern, failures),
            metadata={
                "failure_count": len(failures),
                "modules": list(set(f.module for f in failures)),
                "created_at": datetime.utcnow().isoformat()
            }
        )
    
    def _generate_correction(self, pattern: str, failures: List[FailureRecord]) -> str:
        """Generate correction for failure pattern."""
        # This would use LLM to generate corrections
        return f"To avoid {pattern}: implement proper error handling, validation, and retry logic"


# Registry interfaces (to be implemented in infrastructure)
class SkillRegistry(ABC):
    @abstractmethod
    def get_by_gene_key(self, gene_key: GeneKey) -> Optional[SkillAggregate]:
        pass
    
    @abstractmethod
    def get_all(self) -> List[SkillAggregate]:
        pass


class AgentRegistry(ABC):
    @abstractmethod
    def get_available(self) -> List[AgentCapability]:
        pass
    
    @abstractmethod
    def get_by_role(self, role: AgentRole) -> List[AgentCapability]:
        pass


class OpenSourceClient(ABC):
    @abstractmethod
    async def search_and_download(self, gene_key: GeneKey) -> Result[SkillSpec]:
        pass
    
    @abstractmethod
    async def fetch_recent(self, source: str, limit: int) -> List[Dict]:
        pass


class ResearchModule(ABC):
    @abstractmethod
    async def create_skill(self, gene_key: GeneKey) -> Result[SkillSpec]:
        pass


class MemoryStore(ABC):
    @abstractmethod
    def store_atom(self, atom: MemoryAtom) -> Result[MemoryId]:
        pass
    
    @abstractmethod
    def store_manifest(self, manifest: MemoryManifest) -> Result[UUID]:
        pass
    
    @abstractmethod
    def add_edge(self, edge: MemoryGraphEdge) -> Result[None]:
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int, filters: Optional[Dict]) -> Result[List[MemoryAtom]]:
        pass
    
    @abstractmethod
    def get_by_manifest(self, object_type: str, object_id: str) -> Result[List[MemoryAtom]]:
        pass


class FailureStore(ABC):
    @abstractmethod
    def get_unresolved(self, module: Optional[str]) -> List[FailureRecord]:
        pass


class TrainingStore(ABC):
    @abstractmethod
    def store(self, sample: TrainingSample) -> Result[None]:
        pass