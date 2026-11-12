"""
Main Application Entry Point.
"""
from __future__ import annotations
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..shared.kernel.clock import SystemClock, set_clock
from ..shared.kernel.id_generator import UUIDGenerator, set_id_generator
from ..shared.observability import (
    get_logger, get_metrics, get_tracer, get_health_registry,
    StructuredLogger, InMemoryMetricsCollector, NoOpTracer, SystemClock
)
from ..shared.kernel.commands import CommandBus, QueryBus
from ..application.services import (
    PipelineOrchestrator, SkillManagementService, AgentManagementService,
    ResearchManagementService
)
from ..application.ports import (
    LLMProviderPort, VectorSearchPort, OpenSourceSearchPort,
    SandboxPort, ModelTrainingPort, CompressionPort, MonitoringPort,
    StoragePort, MessageQueuePort, CachePort, EmbeddingPort
)
from ..infrastructure.orchestrator.brains import (
    BrainModuleFactory, MainBrain, SubBrain, AuditBrain, ShadowBrain,
    DreamModule, MagpieModule, ResearchModule
)
from ..infrastructure.memory.memory_system import (
    MemoryStoreImpl, MemoryConfig, TieredStorage
)
from ..infrastructure.rag.rag_system import (
    RAGPipeline, OpenSourceSearchImpl, EmbeddingPortImpl
)
from ..infrastructure.skills.skill_system import (
    SkillRegistry, SkillExecutor, SkillAcquisitionManager, SkillValidator
)
from ..infrastructure.research.research_pipeline import (
    ResearchPipeline, ResearchSandboxManager, LiteratureRetriever,
    HypothesisGenerator, ExperimentExecutor, DataAnalyzer,
    PaperGenerator, ArtifactProducer
)
from ..infrastructure.compression.compression import (
    ModelCompressor, GradientCompressor, VectorIndexCompressor,
    MixedPrecisionTrainer, MemoryOptimizer, CompressionConfig,
    PrecisionMode, GradientCompressionConfig, TurboQuantBackend
)
from ..infrastructure.deployment.deployment_config import (
    DeploymentTarget, get_deployment_config, generate_docker_compose
)


logger = get_logger("main")
metrics = get_metrics()


class AppConfig:
    """Application configuration."""
    
    def __init__(self):
        self.deployment_target = DeploymentTarget.LOCAL_DEV
        self.config = get_deployment_config(self.deployment_target)
        
        # Service configurations
        self.memory_config = MemoryConfig()
        self.compression_config = CompressionConfig(
            precision=PrecisionMode.FP16,
            quantization=None
        )
        self.gradient_compression_config = GradientCompressionConfig(
            method="1bit_adam"
        )


class Application:
    """Main application class."""
    
    def __init__(self, config: AppConfig):
        self._config = config
        self._command_bus = CommandBus()
        self._query_bus = QueryBus()
        self._running = False
        self._tasks: list = []
        
        # Core infrastructure
        self._memory_store: Optional[MemoryStoreImpl] = None
        self._vector_search: Optional[VectorSearchPort] = None
        self._open_source_search: Optional[OpenSourceSearchPort] = None
        self._llm_provider: Optional[LLMProviderPort] = None
        self._sandbox: Optional[SandboxPort] = None
        self._model_training: Optional[ModelTrainingPort] = None
        self._embedding_port: Optional[EmbeddingPort] = None
        self._compression_port: Optional[CompressionPort] = None
        self._storage_port: Optional[StoragePort] = None
        self._message_queue: Optional[MessageQueuePort] = None
        self._cache: Optional[CachePort] = None
        
        # Brain modules
        self._brain_factory: Optional[BrainModuleFactory] = None
        self._main_brain: Optional[MainBrain] = None
        self._sub_brain: Optional[SubBrain] = None
        self._audit_brain: Optional[AuditBrain] = None
        self._shadow_brain: Optional[ShadowBrain] = None
        self._dream_module: Optional[DreamModule] = None
        self._magpie_module: Optional[MagpieModule] = None
        self._research_module: Optional[ResearchModule] = None
        
        # Services
        self._pipeline_orchestrator: Optional[PipelineOrchestrator] = None
        self._skill_management: Optional[SkillManagementService] = None
        self._agent_management: Optional[AgentManagementService] = None
        self._research_management: Optional[ResearchManagementService] = None
        
        # Skill system
        self._skill_registry: Optional[SkillRegistry] = None
        self._skill_executor: Optional[SkillExecutor] = None
        self._skill_acquisition: Optional[SkillAcquisitionManager] = None
        self._skill_validator: Optional[SkillValidator] = None
        
        # Research pipeline
        self._research_pipeline: Optional[ResearchPipeline] = None
        
        # Compression
        self._model_compressor: Optional[ModelCompressor] = None
        self._gradient_compressor: Optional[GradientCompressor] = None
        self._vector_compressor: Optional[VectorIndexCompressor] = None
        self._mixed_precision_trainer: Optional[MixedPrecisionTrainer] = None
        self._memory_optimizer: Optional[MemoryOptimizer] = None
        self._turboquant: Optional[TurboQuantBackend] = None
    
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Multi-Agent Autonomous Problem Solving System")
        
        # Initialize core infrastructure
        await self._initialize_infrastructure()
        
        # Initialize brain modules
        await self._initialize_brains()
        
        # Initialize services
        await self._initialize_services()
        
        # Initialize skill system
        await self._initialize_skills()
        
        # Initialize research pipeline
        await self._initialize_research()
        
        # Initialize compression
        await self._initialize_compression()
        
        # Register health checks
        self._register_health_checks()
        
        # Initialize skill genes
        await self._initialize_skill_genes()
        
        logger.info("System initialization complete")
    
    async def _initialize_infrastructure(self) -> None:
        """Initialize core infrastructure components."""
        # Memory store
        self._memory_store = MemoryStoreImpl(
            config=self._config.memory_config,
            vector_search=self._vector_search,  # Will be set after vector search init
            compression=None,  # Will be set after compression init
            storage=None  # Will be set after storage init
        )
        await self._memory_store.start()
        
        # Embedding port
        self._embedding_port = EmbeddingPortImpl()
        await self._embedding_port.initialize()
        
        # Vector search (mock for now)
        from ..application.ports import VectorSearchPort
        class MockVectorSearch(VectorSearchPort):
            async def add(self, ids, vectors, metadata):
                return Result.success(None)
            async def search(self, vector, top_k, filters=None):
                return Result.success([])
            async def delete(self, ids):
                return Result.success(None)
            async def rebuild(self):
                return Result.success(None)
            async def get_stats(self):
                return Result.success({})
        
        self._vector_search = MockVectorSearch()
        
        # Update memory store with vector search
        # (In real implementation, would use proper dependency injection)
        
        # Open source search
        import aiohttp
        session = aiohttp.ClientSession()
        self._open_source_search = OpenSourceSearchImpl(
            session=session,
            embedding_port=self._embedding_port,
            vector_search=self._vector_search,
            config={}
        )
        
        # LLM Provider (mock for now)
        class MockLLMProvider(LLMProviderPort):
            async def complete(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2048, model=None):
                return Result.success(f"Mock LLM response for: {prompt[:100]}")
            async def complete_with_tools(self, prompt, tools, system_prompt=None):
                return Result.success({})
            async def embed(self, texts, model=None):
                return Result.success([[0.1] * 384 for _ in texts])
        
        self._llm_provider = MockLLMProvider()
        
        # Sandbox (mock for now)
        class MockSandbox(SandboxPort):
            async def execute(self, code, language="python", timeout=300, resources=None):
                return Result.success({"stdout": "Mock execution result", "stderr": "", "returncode": 0})
            async def create_session(self, config):
                return Result.success("mock-session")
            async def close_session(self, session_id):
                return Result.success(None)
        
        self._sandbox = MockSandbox()
        
        # Model training (mock)
        class MockModelTraining(ModelTrainingPort):
            async def fine_tune(self, model_id, training_data, config):
                return Result.success(f"model_{model_id}_finetuned")
            async def evaluate(self, model_id, eval_data):
                return Result.success({"accuracy": 0.9})
        
        self._model_training = MockModelTraining()
        
        # Compression port
        class MockCompressionPort(CompressionPort):
            async def compress_vectors(self, vectors, algorithm="RaBitQ"):
                return Result.success([b"compressed" for _ in vectors])
            async def decompress_vectors(self, compressed, algorithm="RaBitQ", original_shape=None):
                return Result.success([[0.1] * 384 for _ in compressed])
            async def quantize_model(self, model_path, quantization="FP16"):
                return Result.success(f"{model_path}_quantized")
        
        self._compression_port = MockCompressionPort()
        
        # Storage (mock)
        class MockStorage(StoragePort):
            async def save(self, key, data, metadata):
                return Result.success(key)
            async def load(self, key):
                return Result.success(b"data")
            async def delete(self, key):
                return Result.success(None)
            async def list(self, prefix):
                return Result.success([])
        
        self._storage_port = MockStorage()
        
        # Message queue (mock)
        class MockMessageQueue(MessageQueuePort):
            async def publish(self, topic, message):
                return Result.success(None)
            async def subscribe(self, topic, handler):
                return Result.success("sub_1")
            async def unsubscribe(self, subscription_id):
                return Result.success(None)
        
        self._message_queue = MockMessageQueue()
        
        # Cache (mock)
        class MockCache(CachePort):
            def __init__(self):
                self._data = {}
            async def get(self, key):
                return Result.success(self._data.get(key))
            async def set(self, key, value, ttl=3600):
                self._data[key] = value
                return Result.success(None)
            async def delete(self, key):
                self._data.pop(key, None)
                return Result.success(None)
            async def clear(self):
                self._data.clear()
                return Result.success(None)
        
        self._cache = MockCache()
    
    async def _initialize_brains(self) -> None:
        """Initialize brain modules."""
        # Create audit brain first (needed by others)
        audit_agent_id = AgentId()
        self._audit_brain = AuditBrain(
            agent_id=audit_agent_id,
            llm_provider=self._llm_provider,
            quality_criteria={
                "accuracy": 0.8,
                "completeness": 0.7,
                "clarity": 0.7,
                "relevance": 0.8
            }
        )
        
        # Create brain factory
        self._brain_factory = BrainModuleFactory(
            llm_provider=self._llm_provider,
            open_source_search=self._open_source_search,
            sandbox=self._sandbox,
            model_training=self._model_training,
            agent_repo=None,  # Would inject
            failure_store=None,
            training_store=None,
            memory_service=None  # Would inject
        )
        
        # Create main brain
        main_agent_id = AgentId()
        self._main_brain = self._brain_factory.create_main_brain(main_agent_id)
        
        # Create sub brain
        sub_agent_id = AgentId()
        self._sub_brain = self._brain_factory.create_sub_brain(sub_agent_id)
        
        # Create shadow brain
        shadow_agent_id = AgentId()
        self._shadow_brain = self._brain_factory.create_shadow_brain(shadow_agent_id)
        
        # Create dream module
        dream_agent_id = AgentId()
        self._dream_module = self._brain_factory.create_dream_module(dream_agent_id, self._audit_brain)
        
        # Create magpie module
        magpie_agent_id = AgentId()
        self._magpie_module = self._brain_factory.create_magpie_module(magpie_agent_id)
        
        # Create research module
        research_agent_id = AgentId()
        self._research_module = self._brain_factory.create_research_module(research_agent_id, self._audit_brain)
        
        # Register agents
        # Would register with agent repository
    
    async def _initialize_services(self) -> None:
        """Initialize application services."""
        # Pipeline orchestrator
        from ..domain.services import (
            ProblemDecompositionService, AgentAssignmentService,
            VerificationService, DebateService, IntegrationService,
            SkillAcquisitionService, MemoryManagementService,
            ResearchPlanningService, DreamGenerationService,
            MagpieDistillationService
        )
        
        # These would be properly injected
        self._pipeline_orchestrator = PipelineOrchestrator(
            command_bus=self._command_bus,
            query_bus=self._query_bus,
            decomposition_service=ProblemDecompositionService(None),
            assignment_service=AgentAssignmentService(None),
            verification_service=VerificationService(),
            debate_service=DebateService(),
            integration_service=IntegrationService(),
            skill_acquisition_service=SkillAcquisitionService(None, None, None),
            research_planning_service=ResearchPlanningService(),
            dream_generation_service=DreamGenerationService(None, None),
            magpie_distillation_service=MagpieDistillationService(None, None),
            checkpoint_repo=None,  # Would inject
            agent_repo=None,
            skill_repo=None,
            memory_service=None
        )
        
        self._skill_management = SkillManagementService(
            command_bus=self._command_bus,
            query_bus=self._query_bus,
            skill_repo=None,
            gene_repo=None,
            acquisition_service=SkillAcquisitionService(None, None, None)
        )
        
        self._agent_management = AgentManagementService(
            command_bus=self._command_bus,
            query_bus=self._query_bus,
            agent_repo=None
        )
        
        self._research_management = ResearchManagementService(
            command_bus=self._command_bus,
            query_bus=self._query_bus,
            research_repo=None,
            planning_service=ResearchPlanningService(),
            agent_repo=None
        )
    
    async def _initialize_skills(self) -> None:
        """Initialize skill system."""
        self._skill_registry = SkillRegistry()
        
        self._skill_executor = SkillExecutor(
            registry=self._skill_registry,
            llm_provider=self._llm_provider,
            sandbox=self._sandbox,
            open_source=self._open_source_search
        )
        
        self._skill_acquisition = SkillAcquisitionManager(
            registry=self._skill_registry,
            executor=self._skill_executor,
            open_source=self._open_source_search,
            llm_provider=self._llm_provider,
            sandbox=self._sandbox,
            audit_brain=self._audit_brain
        )
        
        self._skill_validator = SkillValidator(
            sandbox=self._sandbox,
            audit_brain=self._audit_brain
        )
    
    async def _initialize_research(self) -> None:
        """Initialize research pipeline."""
        # Create components
        sandbox_manager = ResearchSandboxManager(self._sandbox, {})
        literature_retriever = LiteratureRetriever(
            open_source=self._open_source_search,
            embedding_port=self._embedding_port,
            vector_search=self._vector_search,
            memory_port=None  # Would inject memory management port
        )
        
        rag_pipeline = RAGPipeline(
            open_source_search=self._open_source_search,
            embedding_port=self._embedding_port,
            vector_search=self._vector_search,
            llm_provider=self._llm_provider,
            memory_store=self._memory_store,
            config={}
        )
        
        hypothesis_generator = HypothesisGenerator(
            llm_provider=self._llm_provider,
            rag_pipeline=rag_pipeline
        )
        
        experiment_executor = ExperimentExecutor(
            sandbox_manager=sandbox_manager,
            llm_provider=self._llm_provider
        )
        
        data_analyzer = DataAnalyzer(self._llm_provider)
        paper_generator = PaperGenerator(self._llm_provider)
        artifact_producer = ArtifactProducer(self._llm_provider, sandbox_manager)
        
        self._research_pipeline = ResearchPipeline(
            literature_retriever=literature_retriever,
            hypothesis_generator=hypothesis_generator,
            experiment_executor=experiment_executor,
            data_analyzer=data_analyzer,
            paper_generator=paper_generator,
            artifact_producer=artifact_producer,
            audit_brain=self._audit_brain,
            memory_port=None,
            rag_pipeline=rag_pipeline
        )
    
    async def _initialize_compression(self) -> None:
        """Initialize compression components."""
        self._model_compressor = ModelCompressor(self._config.compression_config)
        self._gradient_compressor = GradientCompressor(self._config.gradient_compression_config)
        self._vector_compressor = VectorIndexCompressor(self._config.compression_config)
        self._mixed_precision_trainer = MixedPrecisionTrainer(self._config.compression_config)
        self._memory_optimizer = MemoryOptimizer(self._config.compression_config)
        self._turboquant = TurboQuantBackend()
    
    async def _initialize_skill_genes(self) -> None:
        """Initialize 250 elite skill genes."""
        gene_keys = self._skill_registry.get_gene_keys()
        await self._skill_management.initialize_skill_genes(gene_keys)
        logger.info(f"Initialized {len(gene_keys)} elite skill genes")
    
    def _register_health_checks(self) -> None:
        """Register health checks."""
        health_registry = get_health_registry()
        
        # System health
        class SystemHealthCheck:
            async def check(self):
                from ..shared.observability import HealthCheckResult
                return HealthCheckResult(
                    component="system",
                    status="healthy",
                    message="System operational",
                    timestamp=SystemClock().now_utc()
                )
        
        health_registry.register("system", SystemHealthCheck())
        
        # Memory health
        class MemoryHealthCheck:
            async def check(self):
                from ..shared.observability import HealthCheckResult
                if self._memory_store:
                    stats = self._memory_store.get_stats()
                    return HealthCheckResult(
                        component="memory",
                        status="healthy",
                        message=f"Memory: {stats}",
                        timestamp=SystemClock().now_utc()
                    )
                return HealthCheckResult(
                    component="memory",
                    status="unknown",
                    message="Memory store not initialized",
                    timestamp=SystemClock().now_utc()
                )
        
        health_registry.register("memory", MemoryHealthCheck())
    
    async def start(self) -> None:
        """Start the application."""
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._dream_cycle()),
            asyncio.create_task(self._magpie_cycle()),
            asyncio.create_task(self._health_check_cycle()),
            asyncio.create_task(self._cleanup_cycle())
        ]
        
        logger.info("Application started")
        
        # Wait for tasks
        await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def stop(self) -> None:
        """Stop the application."""
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop memory store
        if self._memory_store:
            await self._memory_store.stop()
        
        logger.info("Application stopped")
    
    async def _dream_cycle(self) -> None:
        """Daily dream cycle."""
        while self._running:
            try:
                # Run at 2 AM daily
                now = SystemClock().now_utc()
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run = next_run.replace(day=next_run.day + 1)
                
                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                if self._running:
                    logger.info("Starting daily dream cycle")
                    result = await self._dream_module.process({"action": "daily_dream_cycle"})
                    if result.is_success():
                        logger.info(f"Dream cycle completed: {result.unwrap()}")
                    else:
                        logger.error(f"Dream cycle failed: {result.unwrap_err()}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dream cycle error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _magpie_cycle(self) -> None:
        """Daily Magpie training cycle."""
        while self._running:
            try:
                # Run at 3 AM daily
                now = SystemClock().now_utc()
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run = next_run.replace(day=next_run.day + 1)
                
                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                if self._running:
                    logger.info("Starting daily Magpie training cycle")
                    result = await self._magpie_module.process({"action": "daily_training_cycle"})
                    if result.is_success():
                        logger.info(f"Magpie cycle completed: {result.unwrap()}")
                    else:
                        logger.error(f"Magpie cycle failed: {result.unwrap_err()}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Magpie cycle error: {e}")
                await asyncio.sleep(3600)
    
    async def _health_check_cycle(self) -> None:
        """Periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                health_registry = get_health_registry()
                results = await health_registry.run_all()
                
                for component, result in results.items():
                    if result.status != "healthy":
                        logger.warning(f"Health check failed for {component}: {result.message}")
                        metrics.counter("health_check_failed", 1, component=component)
                    else:
                        metrics.counter("health_check_passed", 1, component=component)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check cycle error: {e}")
    
    async def _cleanup_cycle(self) -> None:
        """Periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Cleanup idle sandbox sessions
                # Would call sandbox manager cleanup
                
                # Cleanup old checkpoints
                # Would call checkpoint repo cleanup
                
                metrics.gauge("cleanup_cycle", 1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup cycle error: {e}")
    
    # Public API methods
    async def solve_problem(
        self,
        user_query: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_resources: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> Result[Any]:
        """Solve a problem through the pipeline."""
        return await self._pipeline_orchestrator.execute_pipeline(
            user_query=user_query,
            domain=domain,
            constraints=constraints,
            available_resources=available_resources,
            priority=priority
        )
    
    async def acquire_skill(self, gene_key: str) -> Result[Any]:
        """Acquire a skill by gene key."""
        from ..domain.entities import GeneKey, SkillCategory
        gk = GeneKey.from_string(gene_key)
        return await self._skill_acquisition.acquire_skill(gk)
    
    async def start_research(self, problem_id: str, context: Dict[str, Any]) -> Result[Any]:
        """Start a research project."""
        from ..domain.entities import ProblemId, ProblemContext
        pid = ProblemId(UUID(problem_id))
        pctx = ProblemContext(
            problem_id=pid,
            user_query=context.get("user_query", ""),
            domain=context.get("domain", "general"),
            constraints=context.get("constraints", {}),
            available_resources=context.get("available_resources", {}),
            priority=context.get("priority", 1)
        )
        return await self._research_management.start_research(pid, pctx)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "running": self._running,
            "memory_stats": self._memory_store.get_stats() if self._memory_store else {},
            "skill_count": len(self._skill_registry.get_all_genes()) if self._skill_registry else 0,
            "uptime_seconds": 0  # Would track actual uptime
        }


# FastAPI Application
def create_app() -> FastAPI:
    """Create FastAPI application."""
    app_config = AppConfig()
    application = Application(app_config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await application.initialize()
        # Start background tasks
        asyncio.create_task(application.start())
        yield
        await application.stop()
    
    app = FastAPI(
        title="Multi-Agent Autonomous Problem Solving System",
        description="A multi-AI core autonomous problem solving system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check
    @app.get("/health")
    async def health_check():
        health_registry = get_health_registry()
        results = await health_registry.run_all()
        overall_status = "healthy" if all(r.status == "healthy" for r in results.values()) else "degraded"
        return {"status": overall_status, "components": {k: v.__dict__ for k, v in results.items()}}
    
    # Problem solving endpoint
    class ProblemRequest(BaseModel):
        user_query: str
        domain: str = "general"
        constraints: Dict[str, Any] = {}
        available_resources: Dict[str, Any] = {}
        priority: int = 1
    
    @app.post("/solve")
    async def solve_problem(request: ProblemRequest):
        result = await application.solve_problem(
            user_query=request.user_query,
            domain=request.domain,
            constraints=request.constraints,
            available_resources=request.available_resources,
            priority=request.priority
        )
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result.unwrap_err()))
        return {"answer": result.unwrap()}
    
    # Skill acquisition endpoint
    @app.post("/skills/acquire")
    async def acquire_skill(gene_key: str):
        result = await application.acquire_skill(gene_key)
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result.unwrap_err()))
        return result.unwrap()
    
    # Research endpoint
    class ResearchRequest(BaseModel):
        problem_id: str
        context: Dict[str, Any]
    
    @app.post("/research/start")
    async def start_research(request: ResearchRequest):
        result = await application.start_research(request.problem_id, request.context)
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result.unwrap_err()))
        return result.unwrap()
    
    # System status
    @app.get("/status")
    async def system_status():
        return application.get_system_status()
    
    # Deployment config
    @app.get("/deployment/{target}")
    async def deployment_config(target: str):
        try:
            dt = DeploymentTarget(target)
            config = get_deployment_config(dt)
            return config
        except ValueError:
            raise HTTPException(status_code=404, detail="Unknown deployment target")
    
    @app.get("/deployment/{target}/docker-compose")
    async def docker_compose(target: str):
        try:
            dt = DeploymentTarget(target)
            compose = generate_docker_compose(dt)
            return {"docker_compose": compose}
        except ValueError:
            raise HTTPException(status_code=404, detail="Unknown deployment target")
    
    return app


# CLI Entry Point
async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Autonomous Problem Solving System")
    parser.add_argument("--target", choices=["gcp", "kaggle", "local"], default="local",
                       help="Deployment target")
    parser.add_argument("--mode", choices=["api", "cli", "worker"], default="api",
                       help="Run mode")
    parser.add_argument("--problem", type=str, help="Problem to solve (cli mode)")
    parser.add_argument("--domain", type=str, default="general", help="Problem domain")
    
    args = parser.parse_args()
    
    # Set deployment target
    target_map = {
        "gcp": DeploymentTarget.GCP_FREE_TIER,
        "kaggle": DeploymentTarget.KAGGLE_2xT4,
        "local": DeploymentTarget.LOCAL_DEV
    }
    
    app_config = AppConfig()
    app_config.deployment_target = target_map[args.target]
    app_config.config = get_deployment_config(app_config.deployment_target)
    
    application = Application(app_config)
    await application.initialize()
    
    if args.mode == "api":
        import uvicorn
        app = create_app()
        # Override the app's application instance
        # In real implementation, would use proper dependency injection
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    elif args.mode == "cli":
        if not args.problem:
            print("Error: --problem required for CLI mode")
            return 1
        
        print(f"Solving problem: {args.problem}")
        result = await application.solve_problem(
            user_query=args.problem,
            domain=args.domain
        )
        
        if result.is_success():
            print(f"Answer: {result.unwrap()}")
        else:
            print(f"Error: {result.unwrap_err()}")
            return 1
    
    elif args.mode == "worker":
        # Run background workers
        await application.start()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))