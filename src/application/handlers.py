"""
Command Handlers - Handle write operations.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from uuid import UUID
from ..shared.kernel.commands import CommandHandler, CommandBus
from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import BusinessRuleError, NotFoundError
from ..domain.aggregates import ProblemAggregate, AgentAggregate, SkillAggregate, ResearchAggregate
from ..domain.entities import (
    ProblemId, ProblemContext, AgentId, AgentCapability, SkillId,
    GeneKey, SkillSpec, SubProblem, VerificationResult,
    DebateArgument, IntegratedAnswer, MemoryAtom, MemoryManifest,
    MemoryGraphEdge, ResearchId, ResearchPlan, Hypothesis,
    ExperimentResult, ResearchPaper, Artifact, DreamHypothesis, FailureRecord
)
from .commands import *


class ProblemCommandHandlers:
    """Handlers for problem-related commands."""
    
    def __init__(
        self,
        problem_repo: 'ProblemRepository',
        decomposition_service: 'ProblemDecompositionService',
        assignment_service: 'AgentAssignmentService',
        verification_service: 'VerificationService',
        debate_service: 'DebateService',
        integration_service: 'IntegrationService',
        agent_repo: 'AgentRepository'
    ):
        self._problem_repo = problem_repo
        self._decomposition_service = decomposition_service
        self._assignment_service = assignment_service
        self._verification_service = verification_service
        self._debate_service = debate_service
        self._integration_service = integration_service
        self._agent_repo = agent_repo
    
    async def handle_receive_problem(self, command: ReceiveProblemCommand) -> Result[ProblemId]:
        problem_id = ProblemId()
        context = ProblemContext(
            problem_id=problem_id,
            user_query=command.user_query,
            domain=command.domain,
            constraints=command.constraints,
            available_resources=command.available_resources,
            priority=command.priority,
            deadline=command.deadline,
            metadata=command.metadata
        )
        
        aggregate = ProblemAggregate(problem_id, context)
        result = aggregate.receive_problem()
        if result.is_failure():
            return failure(result.unwrap_err())
        
        save_result = await self._problem_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        
        return success(problem_id)
    
    async def handle_understand_problem(self, command: UnderstandProblemCommand) -> Result[None]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.understand_requirements(
            command.understanding,
            command.identified_constraints
        )
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._problem_repo.save(aggregate)
    
    async def handle_decompose_problem(self, command: DecomposeProblemCommand) -> Result[List[SubProblem]]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.decompose_problem(command.sub_problems)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        save_result = await self._problem_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        
        return success(command.sub_problems)
    
    async def handle_dispatch_experts(self, command: DispatchExpertsCommand) -> Result[Dict[UUID, AgentId]]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.dispatch_experts(command.assignments)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Assign tasks to agents
        for sub_id, agent_id in command.assignments.items():
            agent_result = await self._agent_repo.get(agent_id)
            if agent_result.is_success():
                agent = agent_result.unwrap()
                agent.assign_task(sub_id)
                await self._agent_repo.save(agent)
        
        return await self._problem_repo.save(aggregate).map(lambda _: command.assignments)
    
    async def handle_complete_reasoning(self, command: CompleteReasoningCommand) -> Result[None]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.complete_reasoning(
            command.sub_problem_id,
            command.agent_id,
            command.result
        )
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Complete task for agent
        agent_result = await self._agent_repo.get(command.agent_id)
        if agent_result.is_success():
            agent = agent_result.unwrap()
            agent.complete_task(command.sub_problem_id, {"completed": True})
            await self._agent_repo.save(agent)
        
        return await self._problem_repo.save(aggregate)
    
    async def handle_verify_result(self, command: VerifyResultCommand) -> Result[VerificationResult]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.complete_verification(command.verification)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        save_result = await self._problem_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        
        return success(command.verification)
    
    async def handle_conduct_debate(self, command: ConductDebateCommand) -> Result[Optional[str]]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        consensus = self._debate_service.conduct_debate(
            command.sub_problem_id,
            command.arguments
        )
        if consensus.is_failure():
            return failure(consensus.unwrap_err())
        
        result = aggregate.complete_debate(
            command.sub_problem_id,
            command.arguments,
            consensus.unwrap()
        )
        if result.is_failure():
            return failure(result.unwrap_err())
        
        await self._problem_repo.save(aggregate)
        return consensus
    
    async def handle_integrate_answer(self, command: IntegrateAnswerCommand) -> Result[IntegratedAnswer]:
        aggregate_result = await self._problem_repo.get(command.problem_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Problem {command.problem_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.integrate_answer(command.answer)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        await self._problem_repo.save(aggregate)
        return success(command.answer)


class AgentCommandHandlers:
    """Handlers for agent-related commands."""
    
    def __init__(self, agent_repo: 'AgentRepository'):
        self._agent_repo = agent_repo
    
    async def handle_register_agent(self, command: RegisterAgentCommand) -> Result[AgentId]:
        aggregate = AgentAggregate(command.capability.agent_id, command.capability)
        save_result = await self._agent_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        return success(command.capability.agent_id)
    
    async def handle_assign_task(self, command: AssignTaskCommand) -> Result[None]:
        aggregate_result = await self._agent_repo.get(command.agent_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Agent {command.agent_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.assign_task(command.task_id)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._agent_repo.save(aggregate)
    
    async def handle_complete_task(self, command: CompleteTaskCommand) -> Result[None]:
        aggregate_result = await self._agent_repo.get(command.agent_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Agent {command.agent_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.complete_task(command.task_id, command.performance)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._agent_repo.save(aggregate)
    
    async def handle_update_performance(self, command: UpdateAgentPerformanceCommand) -> Result[None]:
        aggregate_result = await self._agent_repo.get(command.agent_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Agent {command.agent_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.update_performance(command.metrics)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._agent_repo.save(aggregate)
    
    async def handle_add_skill(self, command: AddAgentSkillCommand) -> Result[None]:
        aggregate_result = await self._agent_repo.get(command.agent_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Agent {command.agent_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.add_skill(command.skill_id)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._agent_repo.save(aggregate)


class SkillCommandHandlers:
    """Handlers for skill-related commands."""
    
    def __init__(
        self,
        skill_repo: 'SkillRepository',
        acquisition_service: 'SkillAcquisitionService'
    ):
        self._skill_repo = skill_repo
        self._acquisition_service = acquisition_service
    
    async def handle_register_skill(self, command: RegisterSkillCommand) -> Result[SkillId]:
        skill_id = SkillId()
        from ..domain.aggregates import SkillAggregate
        aggregate = SkillAggregate(skill_id, command.spec)
        save_result = await self._skill_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        return success(skill_id)
    
    async def handle_acquire_skill(self, command: AcquireSkillCommand) -> Result[SkillSpec]:
        return await self._acquisition_service.acquire_skill(command.gene_key)
    
    async def handle_promote_skill(self, command: PromoteSkillCommand) -> Result[None]:
        aggregate_result = await self._skill_repo.get(command.skill_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Skill {command.skill_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.promote(command.benchmark_results)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._skill_repo.save(aggregate)
    
    async def handle_deprecate_skill(self, command: DeprecateSkillCommand) -> Result[None]:
        aggregate_result = await self._skill_repo.get(command.skill_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Skill {command.skill_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.deprecate()
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._skill_repo.save(aggregate)
    
    async def handle_record_usage(self, command: RecordSkillUsageCommand) -> Result[None]:
        aggregate_result = await self._skill_repo.get(command.skill_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Skill {command.skill_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.record_usage(command.success, command.benchmark)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._skill_repo.save(aggregate)


class ResearchCommandHandlers:
    """Handlers for research-related commands."""
    
    def __init__(
        self,
        research_repo: 'ResearchRepository',
        planning_service: 'ResearchPlanningService'
    ):
        self._research_repo = research_repo
        self._planning_service = planning_service
    
    async def handle_start_research(self, command: StartResearchCommand) -> Result[ResearchId]:
        aggregate = ResearchAggregate(command.plan.research_id, command.problem_id, command.plan)
        result = aggregate.start()
        if result.is_failure():
            return failure(result.unwrap_err())
        
        save_result = await self._research_repo.save(aggregate)
        if save_result.is_failure():
            return failure(save_result.unwrap_err())
        
        return success(command.plan.research_id)
    
    async def handle_complete_phase(self, command: CompleteResearchPhaseCommand) -> Result[None]:
        aggregate_result = await self._research_repo.get(command.research_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Research {command.research_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        from ..domain.entities import ResearchPhase
        phase = ResearchPhase(command.phase)
        result = aggregate.complete_phase(phase, command.output)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._research_repo.save(aggregate)
    
    async def handle_add_hypothesis(self, command: AddHypothesisCommand) -> Result[None]:
        aggregate_result = await self._research_repo.get(command.research_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Research {command.research_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.add_hypothesis(command.hypothesis)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._research_repo.save(aggregate)
    
    async def handle_add_experiment_result(self, command: AddExperimentResultCommand) -> Result[None]:
        aggregate_result = await self._research_repo.get(command.research_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Research {command.research_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.add_experiment_result(command.result)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._research_repo.save(aggregate)
    
    async def handle_set_paper(self, command: SetResearchPaperCommand) -> Result[None]:
        aggregate_result = await self._research_repo.get(command.research_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Research {command.research_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.set_paper(command.paper)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._research_repo.save(aggregate)
    
    async def handle_add_artifact(self, command: AddArtifactCommand) -> Result[None]:
        aggregate_result = await self._research_repo.get(command.research_id)
        if aggregate_result.is_failure():
            return failure(NotFoundError(f"Research {command.research_id} not found"))
        
        aggregate = aggregate_result.unwrap()
        result = aggregate.add_artifact(command.artifact)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return await self._research_repo.save(aggregate)


def register_command_handlers(bus: CommandBus, handlers: Dict[str, Any]) -> None:
    """Register all command handlers with the bus."""
    # Problem handlers
    problem_handlers = handlers.get('problem')
    if problem_handlers:
        bus.register(ReceiveProblemHandler(problem_handlers))
        bus.register(UnderstandProblemHandler(problem_handlers))
        bus.register(DecomposeProblemHandler(problem_handlers))
        bus.register(DispatchExpertsHandler(problem_handlers))
        bus.register(CompleteReasoningHandler(problem_handlers))
        bus.register(VerifyResultHandler(problem_handlers))
        bus.register(ConductDebateHandler(problem_handlers))
        bus.register(IntegrateAnswerHandler(problem_handlers))
    
    # Agent handlers
    agent_handlers = handlers.get('agent')
    if agent_handlers:
        bus.register(RegisterAgentHandler(agent_handlers))
        bus.register(AssignTaskHandler(agent_handlers))
        bus.register(CompleteTaskHandler(agent_handlers))
        bus.register(UpdateAgentPerformanceHandler(agent_handlers))
        bus.register(AddAgentSkillHandler(agent_handlers))
    
    # Skill handlers
    skill_handlers = handlers.get('skill')
    if skill_handlers:
        bus.register(RegisterSkillHandler(skill_handlers))
        bus.register(AcquireSkillHandler(skill_handlers))
        bus.register(PromoteSkillHandler(skill_handlers))
        bus.register(DeprecateSkillHandler(skill_handlers))
        bus.register(RecordSkillUsageHandler(skill_handlers))
    
    # Research handlers
    research_handlers = handlers.get('research')
    if research_handlers:
        bus.register(StartResearchHandler(research_handlers))
        bus.register(CompleteResearchPhaseHandler(research_handlers))
        bus.register(AddHypothesisHandler(research_handlers))
        bus.register(AddExperimentResultHandler(research_handlers))
        bus.register(SetResearchPaperHandler(research_handlers))
        bus.register(AddArtifactHandler(research_handlers))


# Handler adapter classes
class ReceiveProblemHandler(CommandHandler[ReceiveProblemCommand, ProblemId]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return ReceiveProblemCommand
    
    async def handle(self, command: ReceiveProblemCommand) -> ProblemId:
        result = await self._handler.handle_receive_problem(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class UnderstandProblemHandler(CommandHandler[UnderstandProblemCommand, None]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return UnderstandProblemCommand
    
    async def handle(self, command: UnderstandProblemCommand) -> None:
        result = await self._handler.handle_understand_problem(command)
        if result.is_failure():
            raise result.unwrap_err()


class DecomposeProblemHandler(CommandHandler[DecomposeProblemCommand, List[SubProblem]]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return DecomposeProblemCommand
    
    async def handle(self, command: DecomposeProblemCommand) -> List[SubProblem]:
        result = await self._handler.handle_decompose_problem(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class DispatchExpertsHandler(CommandHandler[DispatchExpertsCommand, Dict[UUID, AgentId]]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return DispatchExpertsCommand
    
    async def handle(self, command: DispatchExpertsCommand) -> Dict[UUID, AgentId]:
        result = await self._handler.handle_dispatch_experts(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class CompleteReasoningHandler(CommandHandler[CompleteReasoningCommand, None]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return CompleteReasoningCommand
    
    async def handle(self, command: CompleteReasoningCommand) -> None:
        result = await self._handler.handle_complete_reasoning(command)
        if result.is_failure():
            raise result.unwrap_err()


class VerifyResultHandler(CommandHandler[VerifyResultCommand, VerificationResult]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return VerifyResultCommand
    
    async def handle(self, command: VerifyResultCommand) -> VerificationResult:
        result = await self._handler.handle_verify_result(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class ConductDebateHandler(CommandHandler[ConductDebateCommand, Optional[str]]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return ConductDebateCommand
    
    async def handle(self, command: ConductDebateCommand) -> Optional[str]:
        result = await self._handler.handle_conduct_debate(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class IntegrateAnswerHandler(CommandHandler[IntegrateAnswerCommand, IntegratedAnswer]):
    def __init__(self, handler: ProblemCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return IntegrateAnswerCommand
    
    async def handle(self, command: IntegrateAnswerCommand) -> IntegratedAnswer:
        result = await self._handler.handle_integrate_answer(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class RegisterAgentHandler(CommandHandler[RegisterAgentCommand, AgentId]):
    def __init__(self, handler: AgentCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return RegisterAgentCommand
    
    async def handle(self, command: RegisterAgentCommand) -> AgentId:
        result = await self._handler.handle_register_agent(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class AssignTaskHandler(CommandHandler[AssignTaskCommand, None]):
    def __init__(self, handler: AgentCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AssignTaskCommand
    
    async def handle(self, command: AssignTaskCommand) -> None:
        result = await self._handler.handle_assign_task(command)
        if result.is_failure():
            raise result.unwrap_err()


class CompleteTaskHandler(CommandHandler[CompleteTaskCommand, None]):
    def __init__(self, handler: AgentCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return CompleteTaskCommand
    
    async def handle(self, command: CompleteTaskCommand) -> None:
        result = await self._handler.handle_complete_task(command)
        if result.is_failure():
            raise result.unwrap_err()


class UpdateAgentPerformanceHandler(CommandHandler[UpdateAgentPerformanceCommand, None]):
    def __init__(self, handler: AgentCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return UpdateAgentPerformanceCommand
    
    async def handle(self, command: UpdateAgentPerformanceCommand) -> None:
        result = await self._handler.handle_update_performance(command)
        if result.is_failure():
            raise result.unwrap_err()


class AddAgentSkillHandler(CommandHandler[AddAgentSkillCommand, None]):
    def __init__(self, handler: AgentCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AddAgentSkillCommand
    
    async def handle(self, command: AddAgentSkillCommand) -> None:
        result = await self._handler.handle_add_skill(command)
        if result.is_failure():
            raise result.unwrap_err()


class RegisterSkillHandler(CommandHandler[RegisterSkillCommand, SkillId]):
    def __init__(self, handler: SkillCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return RegisterSkillCommand
    
    async def handle(self, command: RegisterSkillCommand) -> SkillId:
        result = await self._handler.handle_register_skill(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class AcquireSkillHandler(CommandHandler[AcquireSkillCommand, SkillSpec]):
    def __init__(self, handler: SkillCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AcquireSkillCommand
    
    async def handle(self, command: AcquireSkillCommand) -> SkillSpec:
        result = await self._handler.handle_acquire_skill(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class PromoteSkillHandler(CommandHandler[PromoteSkillCommand, None]):
    def __init__(self, handler: SkillCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return PromoteSkillCommand
    
    async def handle(self, command: PromoteSkillCommand) -> None:
        result = await self._handler.handle_promote_skill(command)
        if result.is_failure():
            raise result.unwrap_err()


class DeprecateSkillHandler(CommandHandler[DeprecateSkillCommand, None]):
    def __init__(self, handler: SkillCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return DeprecateSkillCommand
    
    async def handle(self, command: DeprecateSkillCommand) -> None:
        result = await self._handler.handle_deprecate_skill(command)
        if result.is_failure():
            raise result.unwrap_err()


class RecordSkillUsageHandler(CommandHandler[RecordSkillUsageCommand, None]):
    def __init__(self, handler: SkillCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return RecordSkillUsageCommand
    
    async def handle(self, command: RecordSkillUsageCommand) -> None:
        result = await self._handler.handle_record_usage(command)
        if result.is_failure():
            raise result.unwrap_err()


class StartResearchHandler(CommandHandler[StartResearchCommand, ResearchId]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return StartResearchCommand
    
    async def handle(self, command: StartResearchCommand) -> ResearchId:
        result = await self._handler.handle_start_research(command)
        if result.is_failure():
            raise result.unwrap_err()
        return result.unwrap()


class CompleteResearchPhaseHandler(CommandHandler[CompleteResearchPhaseCommand, None]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return CompleteResearchPhaseCommand
    
    async def handle(self, command: CompleteResearchPhaseCommand) -> None:
        result = await self._handler.handle_complete_phase(command)
        if result.is_failure():
            raise result.unwrap_err()


class AddHypothesisHandler(CommandHandler[AddHypothesisCommand, None]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AddHypothesisCommand
    
    async def handle(self, command: AddHypothesisCommand) -> None:
        result = await self._handler.handle_add_hypothesis(command)
        if result.is_failure():
            raise result.unwrap_err()


class AddExperimentResultHandler(CommandHandler[AddExperimentResultCommand, None]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AddExperimentResultCommand
    
    async def handle(self, command: AddExperimentResultCommand) -> None:
        result = await self._handler.handle_add_experiment_result(command)
        if result.is_failure():
            raise result.unwrap_err()


class SetResearchPaperHandler(CommandHandler[SetResearchPaperCommand, None]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return SetResearchPaperCommand
    
    async def handle(self, command: SetResearchPaperCommand) -> None:
        result = await self._handler.handle_set_paper(command)
        if result.is_failure():
            raise result.unwrap_err()


class AddArtifactHandler(CommandHandler[AddArtifactCommand, None]):
    def __init__(self, handler: ResearchCommandHandlers):
        self._handler = handler
    
    @property
    def command_type(self) -> type:
        return AddArtifactCommand
    
    async def handle(self, command: AddArtifactCommand) -> None:
        result = await self._handler.handle_add_artifact(command)
        if result.is_failure():
            raise result.unwrap_err()