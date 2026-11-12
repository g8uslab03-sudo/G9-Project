"""
Application Services - Orchestration and coordination.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import asyncio

from ..shared.kernel.commands import CommandBus, QueryBus
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import BusinessRuleError, ResourceExhaustedError
from ..shared.observability import get_logger, get_metrics, traced, metered
from ..domain.entities import (
    ProblemId, ProblemContext, AgentId, AgentRole, SkillId,
    GeneKey, SkillSpec, AgentCapability, SubProblem,
    VerificationResult, VerificationStatus, DebateArgument, IntegratedAnswer,
    ResearchId, ResearchPlan, ResearchPhase
)
from ..domain.aggregates import ProblemAggregate
from ..domain.services import (
    ProblemDecompositionService, AgentAssignmentService, VerificationService,
    DebateService, IntegrationService, SkillAcquisitionService,
    ResearchPlanningService, DreamGenerationService, MagpieDistillationService
)
from .commands import *
from .queries import *


logger = get_logger("application.orchestrator")
metrics = get_metrics()


class PipelineStage(str, Enum):
    RECEIVE_PROBLEM = "receive_problem"
    UNDERSTAND_REQUIREMENTS = "understand_requirements"
    DECOMPOSE_PROBLEM = "decompose_problem"
    DISPATCH_EXPERTS = "dispatch_experts"
    PARALLEL_REASONING = "parallel_reasoning"
    MUTUAL_VERIFICATION = "mutual_verification"
    DEBATE_EVALUATION = "debate_evaluation"
    INTEGRATE_ANSWER = "integrate_answer"
    CONTINUOUS_LEARNING = "continuous_learning"
    OPTIMIZE_FUTURE = "optimize_future"


@dataclass
class PipelineState:
    """State of pipeline execution."""
    pipeline_id: str
    problem_id: ProblemId
    current_stage: PipelineStage
    stage_results: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineOrchestrator:
    """Main orchestrator for the multi-agent problem solving pipeline."""
    
    def __init__(
        self,
        command_bus: CommandBus,
        query_bus: QueryBus,
        decomposition_service: ProblemDecompositionService,
        assignment_service: AgentAssignmentService,
        verification_service: VerificationService,
        debate_service: DebateService,
        integration_service: IntegrationService,
        skill_acquisition_service: SkillAcquisitionService,
        research_planning_service: ResearchPlanningService,
        dream_generation_service: DreamGenerationService,
        magpie_distillation_service: MagpieDistillationService,
        checkpoint_repo: 'CheckpointRepository',
        agent_repo: 'AgentRepository',
        skill_repo: 'SkillRepository',
        memory_service: 'MemoryManagementService'
    ):
        self._command_bus = command_bus
        self._query_bus = query_bus
        self._decomposition_service = decomposition_service
        self._assignment_service = assignment_service
        self._verification_service = verification_service
        self._debate_service = debate_service
        self._integration_service = integration_service
        self._skill_acquisition_service = skill_acquisition_service
        self._research_planning_service = research_planning_service
        self._dream_generation_service = dream_generation_service
        self._magpie_distillation_service = magpie_distillation_service
        self._checkpoint_repo = checkpoint_repo
        self._agent_repo = agent_repo
        self._skill_repo = skill_repo
        self._memory_service = memory_service
        
        self._pipelines: Dict[str, PipelineState] = {}
        self._stage_handlers: Dict[PipelineStage, Callable] = {
            PipelineStage.RECEIVE_PROBLEM: self._stage_receive_problem,
            PipelineStage.UNDERSTAND_REQUIREMENTS: self._stage_understand_requirements,
            PipelineStage.DECOMPOSE_PROBLEM: self._stage_decompose_problem,
            PipelineStage.DISPATCH_EXPERTS: self._stage_dispatch_experts,
            PipelineStage.PARALLEL_REASONING: self._stage_parallel_reasoning,
            PipelineStage.MUTUAL_VERIFICATION: self._stage_mutual_verification,
            PipelineStage.DEBATE_EVALUATION: self._stage_debate_evaluation,
            PipelineStage.INTEGRATE_ANSWER: self._stage_integrate_answer,
            PipelineStage.CONTINUOUS_LEARNING: self._stage_continuous_learning,
            PipelineStage.OPTIMIZE_FUTURE: self._stage_optimize_future,
        }
    
    @traced("orchestrator.execute_pipeline")
    @metered("pipeline_execution", "histogram")
    async def execute_pipeline(
        self,
        user_query: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_resources: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        resume_from: Optional[str] = None
    ) -> Result[IntegratedAnswer]:
        """Execute the full problem-solving pipeline."""
        pipeline_id = str(uuid4())
        
        # Check for resume
        if resume_from:
            checkpoint_result = await self._checkpoint_repo.get_latest_checkpoint(resume_from)
            if checkpoint_result.is_success() and checkpoint_result.unwrap():
                return await self._resume_pipeline(resume_from, checkpoint_result.unwrap())
        
        # Initialize pipeline state
        problem_id = ProblemId()
        context = ProblemContext(
            problem_id=problem_id,
            user_query=user_query,
            domain=domain,
            constraints=constraints or {},
            available_resources=available_resources or {},
            priority=priority
        )
        
        state = PipelineState(
            pipeline_id=pipeline_id,
            problem_id=problem_id,
            current_stage=PipelineStage.RECEIVE_PROBLEM
        )
        self._pipelines[pipeline_id] = state
        
        try:
            # Execute stages sequentially
            stages = list(PipelineStage)
            for i, stage in enumerate(stages):
                state.current_stage = stage
                state.updated_at = datetime.utcnow()
                
                logger.info(f"Pipeline {pipeline_id}: Executing stage {stage.value}")
                metrics.counter("pipeline_stage_started", 1, pipeline_id=pipeline_id, stage=stage.value)
                
                # Execute stage handler
                handler = self._stage_handlers[stage]
                stage_result = await handler(state, context)
                
                if stage_result.is_failure():
                    logger.error(f"Pipeline {pipeline_id}: Stage {stage.value} failed: {stage_result.unwrap_err()}")
                    metrics.counter("pipeline_stage_failed", 1, pipeline_id=pipeline_id, stage=stage.value)
                    return failure(stage_result.unwrap_err())
                
                state.stage_results[stage.value] = stage_result.unwrap()
                
                # Save checkpoint after each stage
                await self._save_checkpoint(pipeline_id, stage, state.stage_results)
                
                metrics.counter("pipeline_stage_completed", 1, pipeline_id=pipeline_id, stage=stage.value)
            
            # Get final answer
            final_answer = state.stage_results.get(PipelineStage.INTEGRATE_ANSWER.value)
            if final_answer:
                logger.info(f"Pipeline {pipeline_id}: Completed successfully")
                metrics.counter("pipeline_completed", 1, pipeline_id=pipeline_id)
                return success(final_answer)
            
            return failure(BusinessRuleError("No answer generated"))
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id}: Unexpected error: {e}")
            metrics.counter("pipeline_error", 1, pipeline_id=pipeline_id)
            return failure(e)
        finally:
            # Cleanup pipeline state
            self._pipelines.pop(pipeline_id, None)
    
    async def _resume_pipeline(
        self,
        pipeline_id: str,
        checkpoint: Dict[str, Any]
    ) -> Result[IntegratedAnswer]:
        """Resume pipeline from checkpoint."""
        logger.info(f"Resuming pipeline {pipeline_id} from stage {checkpoint.get('stage')}")
        
        # Restore state
        stage = PipelineStage(checkpoint["stage"])
        stage_results = checkpoint.get("state", {})
        
        # Reconstruct context
        problem_id = ProblemId(UUID(checkpoint["metadata"]["problem_id"]))
        context = ProblemContext(
            problem_id=problem_id,
            user_query=checkpoint["metadata"]["user_query"],
            domain=checkpoint["metadata"]["domain"],
            constraints=checkpoint["metadata"]["constraints"],
            available_resources=checkpoint["metadata"]["available_resources"],
            priority=checkpoint["metadata"]["priority"]
        )
        
        state = PipelineState(
            pipeline_id=pipeline_id,
            problem_id=problem_id,
            current_stage=stage,
            stage_results=stage_results
        )
        self._pipelines[pipeline_id] = state
        
        # Continue from next stage
        stages = list(PipelineStage)
        start_index = stages.index(stage) + 1
        
        for stage in stages[start_index:]:
            state.current_stage = stage
            state.updated_at = datetime.utcnow()
            
            handler = self._stage_handlers[stage]
            stage_result = await handler(state, context)
            
            if stage_result.is_failure():
                return failure(stage_result.unwrap_err())
            
            state.stage_results[stage.value] = stage_result.unwrap()
            await self._save_checkpoint(pipeline_id, stage, state.stage_results)
        
        final_answer = state.stage_results.get(PipelineStage.INTEGRATE_ANSWER.value)
        if final_answer:
            return success(final_answer)
        
        return failure(BusinessRuleError("No answer generated after resume"))
    
    async def _save_checkpoint(
        self,
        pipeline_id: str,
        stage: PipelineStage,
        stage_results: Dict[str, Any]
    ) -> None:
        """Save pipeline checkpoint."""
        state = self._pipelines.get(pipeline_id)
        if not state:
            return
        
        checkpoint = {
            "pipeline_id": pipeline_id,
            "stage": stage.value,
            "state": stage_results,
            "metadata": {
                "problem_id": str(state.problem_id),
                "user_query": state.stage_results.get(PipelineStage.RECEIVE_PROBLEM.value, {}).get("user_query", ""),
                "domain": state.stage_results.get(PipelineStage.RECEIVE_PROBLEM.value, {}).get("domain", ""),
                "constraints": state.stage_results.get(PipelineStage.RECEIVE_PROBLEM.value, {}).get("constraints", {}),
                "available_resources": state.stage_results.get(PipelineStage.RECEIVE_PROBLEM.value, {}).get("available_resources", {}),
                "priority": state.stage_results.get(PipelineStage.RECEIVE_PROBLEM.value, {}).get("priority", 1),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._checkpoint_repo.save_checkpoint(pipeline_id, stage.value, stage_results, checkpoint["metadata"])
    
    # Stage Handlers
    async def _stage_receive_problem(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[str, Any]]:
        """Stage 1: Receive and validate problem."""
        command = ReceiveProblemCommand(
            user_query=context.user_query,
            domain=context.domain,
            constraints=context.constraints,
            available_resources=context.available_resources,
            priority=context.priority
        )
        
        result = await self._command_bus.dispatch(command)
        if isinstance(result, Result):
            if result.is_failure():
                return failure(result.unwrap_err())
            problem_id = result.unwrap()
        else:
            problem_id = result
        
        return success({
            "problem_id": str(problem_id),
            "user_query": context.user_query,
            "domain": context.domain,
            "constraints": context.constraints,
            "available_resources": context.available_resources,
            "priority": context.priority
        })
    
    async def _stage_understand_requirements(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[str, Any]]:
        """Stage 2: Understand problem requirements using main brain."""
        # Use main brain agent to analyze problem
        problem_id = state.problem_id
        
        # Get main brain agent
        agents_result = await self._agent_repo.get_available(AgentRole.MAIN_BRAIN)
        if agents_result.is_failure() or not agents_result.unwrap():
            return failure(BusinessRuleError("No main brain agent available"))
        
        main_brain = agents_result.unwrap()[0]
        
        # Use main brain to understand (would call LLM in practice)
        understanding = f"Analyzed problem: {context.user_query}. Domain: {context.domain}. Constraints: {context.constraints}"
        identified_constraints = list(context.constraints.keys())
        
        command = UnderstandProblemCommand(
            problem_id=problem_id,
            understanding=understanding,
            identified_constraints=identified_constraints
        )
        
        result = await self._command_bus.dispatch(command)
        if isinstance(result, Result) and result.is_failure():
            return failure(result.unwrap_err())
        
        return success({
            "understanding": understanding,
            "identified_constraints": identified_constraints
        })
    
    async def _stage_decompose_problem(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[List[SubProblem]]:
        """Stage 3: Decompose problem into sub-problems."""
        problem_id = state.problem_id
        
        # Get available skills
        skills_result = await self._skill_repo.get_available()
        if skills_result.is_failure():
            return failure(skills_result.unwrap_err())
        
        available_skills = []
        for skill in skills_result.unwrap():
            # Would need to get gene key from skill
            pass
        
        # Use decomposition service
        result = self._decomposition_service.decompose(context, available_skills)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        sub_problems = result.unwrap()
        
        # Send command to persist
        command = DecomposeProblemCommand(
            problem_id=problem_id,
            sub_problems=sub_problems
        )
        
        cmd_result = await self._command_bus.dispatch(command)
        if isinstance(cmd_result, Result) and cmd_result.is_failure():
            return failure(cmd_result.unwrap_err())
        
        return success(sub_problems)
    
    async def _stage_dispatch_experts(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[UUID, AgentId]]:
        """Stage 4: Dispatch expert agents to sub-problems."""
        problem_id = state.problem_id
        sub_problems = state.stage_results.get(PipelineStage.DECOMPOSE_PROBLEM.value, [])
        
        if not sub_problems:
            return failure(BusinessRuleError("No sub-problems to dispatch"))
        
        # Get available agents
        agents_result = await self._agent_repo.get_available()
        if agents_result.is_failure():
            return failure(agents_result.unwrap_err())
        
        available_agents = [a.capability for a in agents_result.unwrap()]
        
        # Use assignment service
        result = self._assignment_service.assign(sub_problems, available_agents)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        assignments = result.unwrap()
        
        # Send command to persist
        command = DispatchExpertsCommand(
            problem_id=problem_id,
            assignments=assignments
        )
        
        cmd_result = await self._command_bus.dispatch(command)
        if isinstance(cmd_result, Result) and cmd_result.is_failure():
            return failure(cmd_result.unwrap_err())
        
        return success(assignments)
    
    async def _stage_parallel_reasoning(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[UUID, Any]]:
        """Stage 5: Parallel reasoning by expert agents."""
        problem_id = state.problem_id
        assignments = state.stage_results.get(PipelineStage.DISPATCH_EXPERTS.value, {})
        
        if not assignments:
            return failure(BusinessRuleError("No assignments for reasoning"))
        
        # Execute reasoning in parallel for each sub-problem
        reasoning_tasks = []
        for sub_problem_id, agent_id in assignments.items():
            task = self._execute_reasoning(problem_id, sub_problem_id, agent_id, context)
            reasoning_tasks.append(task)
        
        results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)
        
        reasoning_results = {}
        for i, (sub_problem_id, agent_id) in enumerate(assignments.items()):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Reasoning failed for {sub_problem_id}: {result}")
                reasoning_results[sub_problem_id] = {"error": str(result), "success": False}
            else:
                reasoning_results[sub_problem_id] = result
        
        return success(reasoning_results)
    
    async def _execute_reasoning(
        self,
        problem_id: ProblemId,
        sub_problem_id: UUID,
        agent_id: AgentId,
        context: ProblemContext
    ) -> Dict[str, Any]:
        """Execute reasoning for a single sub-problem."""
        # Get agent
        agent_result = await self._agent_repo.get(agent_id)
        if agent_result.is_failure():
            raise BusinessRuleError(f"Agent {agent_id} not found")
        
        agent = agent_result.unwrap()
        
        # Assign task
        agent.assign_task(sub_problem_id)
        await self._agent_repo.save(agent)
        
        # Execute reasoning (would use LLM + skills in practice)
        # For now, simulate reasoning
        sub_problem = None
        for sp in state.stage_results.get(PipelineStage.DECOMPOSE_PROBLEM.value, []):
            if sp.sub_id == sub_problem_id:
                sub_problem = sp
                break
        
        if not sub_problem:
            raise BusinessRuleError(f"Sub-problem {sub_problem_id} not found")
        
        # Simulate skill execution
        skill_results = {}
        for gene_key in sub_problem.required_skills:
            skill_result = await self._execute_skill(gene_key, context, sub_problem)
            skill_results[str(gene_key)] = skill_result
        
        # Complete task
        agent.complete_task(sub_problem_id, {"skills_executed": len(skill_results)})
        await self._agent_repo.save(agent)
        
        # Send command to persist
        command = CompleteReasoningCommand(
            problem_id=problem_id,
            sub_problem_id=sub_problem_id,
            agent_id=agent_id,
            result={"skill_results": skill_results, "summary": f"Completed {sub_problem.description}"}
        )
        
        await self._command_bus.dispatch(command)
        
        return {"skill_results": skill_results, "summary": f"Completed {sub_problem.description}", "success": True}
    
    async def _execute_skill(
        self,
        gene_key: GeneKey,
        context: ProblemContext,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """Execute a single skill."""
        # Get skill
        skill_result = await self._skill_repo.get_by_gene_key(gene_key)
        if skill_result.is_failure() or not skill_result.unwrap():
            # Try to acquire skill
            acquire_result = await self._skill_acquisition_service.acquire_skill(gene_key)
            if acquire_result.is_failure():
                return {"success": False, "error": f"Skill {gene_key} not available"}
            
            skill_spec = acquire_result.unwrap()
            # Register new skill
            register_cmd = RegisterSkillCommand(spec=skill_spec)
            await self._command_bus.dispatch(register_cmd)
        else:
            skill = skill_result.unwrap()
            skill_spec = skill.spec
        
        # Record usage
        usage_cmd = RecordSkillUsageCommand(
            skill_id=skill.id if skill_result.is_success() else SkillId(),
            success=True,
            benchmark={"executed_at": datetime.utcnow().isoformat()}
        )
        await self._command_bus.dispatch(usage_cmd)
        
        return {"success": True, "skill": str(gene_key), "spec": skill_spec.name}
    
    async def _stage_mutual_verification(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[UUID, List[VerificationResult]]]:
        """Stage 6: Mutual verification between agents."""
        problem_id = state.problem_id
        reasoning_results = state.stage_results.get(PipelineStage.PARALLEL_REASONING.value, {})
        assignments = state.stage_results.get(PipelineStage.DISPATCH_EXPERTS.value, {})
        
        verifications = {}
        
        for sub_problem_id, result in reasoning_results.items():
            if not result.get("success"):
                continue
            
            target_agent_id = assignments.get(sub_problem_id)
            if not target_agent_id:
                continue
            
            # Get verifier agents (different from target)
            agents_result = await self._agent_repo.get_available()
            if agents_result.is_failure():
                continue
            
            verifiers = [a for a in agents_result.unwrap() if a.id != target_agent_id][:2]
            
            sub_verifications = []
            for verifier in verifiers:
                verification_result = self._verification_service.verify(
                    verifier_id=verifier.id,
                    target_id=target_agent_id,
                    sub_problem_id=sub_problem_id,
                    target_result=result,
                    criteria={"success": True}
                )
                
                if verification_result.is_success():
                    verification = verification_result.unwrap()
                    sub_verifications.append(verification)
                    
                    # Persist verification
                    command = VerifyResultCommand(
                        problem_id=problem_id,
                        verification=verification
                    )
                    await self._command_bus.dispatch(command)
            
            verifications[sub_problem_id] = sub_verifications
        
        return success(verifications)
    
    async def _stage_debate_evaluation(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[UUID, Optional[str]]]:
        """Stage 7: Debate evaluation."""
        problem_id = state.problem_id
        reasoning_results = state.stage_results.get(PipelineStage.PARALLEL_REASONING.value, {})
        
        debates = {}
        
        for sub_problem_id, result in reasoning_results.items():
            if not result.get("success"):
                continue
            
            # Generate arguments from different perspectives
            arguments = [
                DebateArgument(
                    agent_id=AgentId(),  # Would be actual agent
                    position="support",
                    argument=f"Supporting result for {sub_problem_id}: {result.get('summary')}",
                    confidence=0.8
                ),
                DebateArgument(
                    agent_id=AgentId(),
                    position="oppose",
                    argument=f"Potential issues with {sub_problem_id} approach",
                    confidence=0.6
                ),
                DebateArgument(
                    agent_id=AgentId(),
                    position="neutral",
                    argument=f"Need more validation for {sub_problem_id}",
                    confidence=0.5
                )
            ]
            
            # Conduct debate
            consensus_result = self._debate_service.conduct_debate(sub_problem_id, arguments)
            if consensus_result.is_success():
                consensus = consensus_result.unwrap()
                debates[sub_problem_id] = consensus
                
                # Persist debate
                command = ConductDebateCommand(
                    problem_id=problem_id,
                    sub_problem_id=sub_problem_id,
                    arguments=arguments
                )
                await self._command_bus.dispatch(command)
        
        return success(debates)
    
    async def _stage_integrate_answer(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[IntegratedAnswer]:
        """Stage 8: Integrate final answer."""
        problem_id = state.problem_id
        reasoning_results = state.stage_results.get(PipelineStage.PARALLEL_REASONING.value, {})
        verifications = state.stage_results.get(PipelineStage.MUTUAL_VERIFICATION.value, {})
        debates = state.stage_results.get(PipelineStage.DEBATE_EVALUATION.value, {})
        
        # Calculate skill contributions
        skill_contributions = {}
        for sub_id, result in reasoning_results.items():
            for skill_key, skill_result in result.get("skill_results", {}).items():
                if skill_result.get("success"):
                    gene_key = GeneKey.from_string(skill_key)
                    skill_contributions[gene_key] = skill_contributions.get(gene_key, 0) + 1
        
        # Normalize contributions
        total = sum(skill_contributions.values())
        if total > 0:
            skill_contributions = {k: v/total for k, v in skill_contributions.items()}
        
        # Integrate
        integration_result = self._integration_service.integrate(
            problem_id=problem_id,
            sub_problem_results=reasoning_results,
            verifications=verifications,
            debates=debates,
            skill_contributions=skill_contributions
        )
        
        if integration_result.is_failure():
            return failure(integration_result.unwrap_err())
        
        answer = integration_result.unwrap()
        
        # Persist answer
        command = IntegrateAnswerCommand(
            problem_id=problem_id,
            answer=answer
        )
        await self._command_bus.dispatch(command)
        
        return success(answer)
    
    async def _stage_continuous_learning(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[str, Any]]:
        """Stage 9: Continuous learning from this problem."""
        # Record failures for Magpie
        reasoning_results = state.stage_results.get(PipelineStage.PARALLEL_REASONING.value, {})
        
        failures_recorded = 0
        for sub_id, result in reasoning_results.items():
            if not result.get("success"):
                failure_record = FailureRecord(
                    failure_id=uuid4(),
                    module="reasoning",
                    agent_id=None,
                    skill_id=None,
                    error_type="reasoning_failed",
                    error_message=result.get("error", "Unknown error"),
                    context={"sub_problem_id": str(sub_id), "problem_id": str(state.problem_id)}
                )
                
                command = RecordFailureCommand(failure=failure_record)
                await self._command_bus.dispatch(command)
                failures_recorded += 1
        
        # Trigger Magpie distillation if enough failures
        if failures_recorded > 0:
            distill_cmd = DistillFailuresCommand(module="reasoning")
            await self._command_bus.dispatch(distill_cmd)
        
        return success({"failures_recorded": failures_recorded, "learning_triggered": failures_recorded > 0})
    
    async def _stage_optimize_future(
        self,
        state: PipelineState,
        context: ProblemContext
    ) -> Result[Dict[str, Any]]:
        """Stage 10: Optimize future problem solving."""
        # Analyze performance and update agent skills
        reasoning_results = state.stage_results.get(PipelineStage.PARALLEL_REASONING.value, {})
        assignments = state.stage_results.get(PipelineStage.DISPATCH_EXPERTS.value, {})
        
        optimizations = []
        
        for sub_id, agent_id in assignments.items():
            result = reasoning_results.get(sub_id, {})
            if result.get("success"):
                # Update agent performance
                agent_result = await self._agent_repo.get(agent_id)
                if agent_result.is_success():
                    agent = agent_result.unwrap()
                    agent.update_performance({"success_rate": 0.9})  # Would calculate actual
                    await self._agent_repo.save(agent)
                    optimizations.append(f"Updated performance for agent {agent_id}")
        
        return success({"optimizations": optimizations})


class SkillManagementService:
    """Service for managing skill lifecycle."""
    
    def __init__(
        self,
        command_bus: CommandBus,
        query_bus: QueryBus,
        skill_repo: 'SkillRepository',
        gene_repo: 'SkillGeneRepository',
        acquisition_service: SkillAcquisitionService
    ):
        self._command_bus = command_bus
        self._query_bus = query_bus
        self._skill_repo = skill_repo
        self._gene_repo = gene_repo
        self._acquisition_service = acquisition_service
    
    async def initialize_skill_genes(self, genes: List[GeneKey]) -> Result[None]:
        """Initialize the 250 elite skill genes."""
        return await self._gene_repo.initialize_genes(genes)
    
    async def ensure_skill_available(self, gene_key: GeneKey) -> Result[SkillSpec]:
        """Ensure skill is available, acquire if needed."""
        # Check registry
        skill_result = await self._skill_repo.get_by_gene_key(gene_key)
        if skill_result.is_success() and skill_result.unwrap():
            return success(skill_result.unwrap().spec)
        
        # Acquire skill
        return await self._acquisition_service.acquire_skill(gene_key)
    
    async def promote_skill(self, skill_id: SkillId, benchmarks: Dict[str, float]) -> Result[None]:
        """Promote skill after successful benchmarking."""
        command = PromoteSkillCommand(skill_id=skill_id, benchmark_results=benchmarks)
        result = await self._command_bus.dispatch(command)
        if isinstance(result, Result) and result.is_failure():
            return failure(result.unwrap_err())
        return success(None)
    
    async def get_skill_stats(self) -> Result[Dict[str, Any]]:
        """Get skill system statistics."""
        all_skills_result = await self._skill_repo.get_all()
        if all_skills_result.is_failure():
            return failure(all_skills_result.unwrap_err())
        
        skills = all_skills_result.unwrap()
        
        by_category = {}
        by_status = {}
        total_usage = 0
        total_success = 0
        
        for skill in skills:
            cat = skill.spec.gene_key.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            status = skill.status
            by_status[status] = by_status.get(status, 0) + 1
            
            total_usage += skill._usage_count
            total_success += skill._success_count
        
        return success({
            "total_skills": len(skills),
            "by_category": by_category,
            "by_status": by_status,
            "total_usage": total_usage,
            "success_rate": total_success / total_usage if total_usage > 0 else 0
        })


class AgentManagementService:
    """Service for managing agent lifecycle."""
    
    def __init__(
        self,
        command_bus: CommandBus,
        query_bus: QueryBus,
        agent_repo: 'AgentRepository'
    ):
        self._command_bus = command_bus
        self._query_bus = query_bus
        self._agent_repo = agent_repo
    
    async def register_agent(self, capability: AgentCapability) -> Result[AgentId]:
        """Register a new agent."""
        command = RegisterAgentCommand(capability=capability)
        result = await self._command_bus.dispatch(command)
        if isinstance(result, Result):
            if result.is_failure():
                return failure(result.unwrap_err())
            return success(result.unwrap())
        return success(result)
    
    async def get_agent_load(self) -> Result[Dict[str, Any]]:
        """Get current agent load statistics."""
        agents_result = await self._agent_repo.get_all()
        if agents_result.is_failure():
            return failure(agents_result.unwrap_err())
        
        agents = agents_result.unwrap()
        
        loads = []
        for agent in agents:
            loads.append({
                "agent_id": str(agent.id),
                "role": agent.capability.role.value,
                "current_load": agent.current_load,
                "max_load": agent.max_load,
                "utilization": agent.current_load / agent.max_load if agent.max_load > 0 else 0,
                "is_available": agent.is_available
            })
        
        return success({
            "total_agents": len(agents),
            "available_agents": sum(1 for a in agents if a.is_available),
            "loads": loads
        })


class ResearchManagementService:
    """Service for managing research projects."""
    
    def __init__(
        self,
        command_bus: CommandBus,
        query_bus: QueryBus,
        research_repo: 'ResearchRepository',
        planning_service: ResearchPlanningService,
        agent_repo: 'AgentRepository'
    ):
        self._command_bus = command_bus
        self._query_bus = query_bus
        self._research_repo = research_repo
        self._planning_service = planning_service
        self._agent_repo = agent_repo
    
    async def start_research(
        self,
        problem_id: ProblemId,
        problem_context: ProblemContext
    ) -> Result[ResearchId]:
        """Start a new research project."""
        # Get available agents and tools
        agents_result = await self._agent_repo.get_all()
        if agents_result.is_failure():
            return failure(agents_result.unwrap_err())
        
        available_agents = [a.capability for a in agents_result.unwrap()]
        available_tools = [
            "arxiv_search", "semantic_scholar", "pubmed", "crossref",
            "python_sandbox", "jupyter", "docker",
            "pandas", "numpy", "scipy", "visualization",
            "llm_writing", "latex", "reference_manager",
            "code_generator", "benchmark_runner"
        ]
        
        # Create plan
        plan_result = self._planning_service.create_plan(
            problem_id, problem_context, available_agents, available_tools
        )
        if plan_result.is_failure():
            return failure(plan_result.unwrap_err())
        
        plan = plan_result.unwrap()
        
        # Start research
        command = StartResearchCommand(problem_id=problem_id, plan=plan)
        result = await self._command_bus.dispatch(command)
        if isinstance(result, Result):
            if result.is_failure():
                return failure(result.unwrap_err())
            return success(result.unwrap())
        return success(result)
    
    async def get_research_progress(self, research_id: ResearchId) -> Result[Dict[str, Any]]:
        """Get research progress."""
        research_result = await self._research_repo.get(research_id)
        if research_result.is_failure():
            return failure(research_result.unwrap_err())
        
        research = research_result.unwrap()
        
        return success({
            "research_id": str(research_id),
            "current_phase": research.current_phase.value,
            "is_complete": research.is_complete,
            "phases_completed": list(research._phase_outputs.keys()),
            "hypotheses_count": len(research._hypotheses),
            "experiments_count": len(research._experiment_results),
            "has_paper": research._paper is not None,
            "artifacts_count": len(research._artifacts)
        })