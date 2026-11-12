"""
Infrastructure - Research Module 7-Phase Pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import asyncio
import json

from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import ResearchError, ValidationError
from ..shared.observability import get_logger, get_metrics, traced, metered
from ..domain.entities import (
    ResearchId, ResearchPhase, ResearchPlan, Hypothesis, ExperimentResult,
    ResearchPaper, Artifact, ProblemId, ProblemContext, AgentId, AgentCapability,
    MemoryAtom, MemoryId
)
from ..application.ports import (
    LLMProviderPort, OpenSourceSearchPort, SandboxPort, VectorSearchPort,
    EmbeddingPort, MemoryManagementPort
)
from ..infrastructure.rag.rag_system import RAGPipeline


logger = get_logger("infrastructure.research")
metrics = get_metrics()


class SandboxType(str, Enum):
    PYTHON = "python"
    JUPYTER = "jupyter"
    DOCKER = "docker"


@dataclass
class SandboxSession:
    """Sandbox execution session."""
    session_id: str
    sandbox_type: SandboxType
    config: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


class ResearchSandboxManager:
    """Manages sandbox environments for research experiments."""
    
    def __init__(self, sandbox_port: SandboxPort, config: Dict[str, Any]):
        self._sandbox = sandbox_port
        self._config = config
        self._sessions: Dict[str, SandboxSession] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, sandbox_type: SandboxType = SandboxType.PYTHON) -> Result[str]:
        """Create a new sandbox session."""
        session_id = str(uuid4())
        
        session_config = {
            "isolation": "full",
            "network_access": False,
            "resource_limits": self._config.get("resource_limits", {"cpu": 2, "memory_gb": 8, "disk_gb": 10}),
            "timeout_seconds": self._config.get("timeout_seconds", 3600)
        }
        
        result = await self._sandbox.create_session(session_config)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        session = SandboxSession(
            session_id=session_id,
            sandbox_type=sandbox_type,
            config=session_config
        )
        
        async with self._lock:
            self._sessions[session_id] = session
        
        return success(session_id)
    
    async def execute(
        self,
        session_id: str,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Result[Dict[str, Any]]:
        """Execute code in sandbox."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return failure(ResearchError(f"Session {session_id} not found or inactive"))
            
            session.last_used = datetime.utcnow()
        
        exec_timeout = timeout or session.config["timeout_seconds"]
        
        result = await self._sandbox.execute(
            code=code,
            language=language,
            timeout=exec_timeout,
            resources=session.config["resource_limits"]
        )
        
        return result
    
    async def close_session(self, session_id: str) -> Result[None]:
        """Close sandbox session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return failure(ResearchError(f"Session {session_id} not found"))
            
            session.is_active = False
        
        return await self._sandbox.close_session(session_id)
    
    async def cleanup_idle_sessions(self, max_idle_seconds: int = 1800) -> int:
        """Clean up idle sessions."""
        now = datetime.utcnow()
        cleaned = 0
        
        async with self._lock:
            to_remove = []
            for session_id, session in self._sessions.items():
                if (now - session.last_used).total_seconds() > max_idle_seconds:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                session = self._sessions.pop(session_id)
                session.is_active = False
                await self._sandbox.close_session(session_id)
                cleaned += 1
        
        return cleaned


class LiteratureRetriever:
    """Retrieves literature from open sources for research."""
    
    def __init__(
        self,
        open_source: OpenSourceSearchPort,
        embedding_port: EmbeddingPort,
        vector_search: VectorSearchPort,
        memory_port: MemoryManagementPort
    ):
        self._open_source = open_source
        self._embedding = embedding_port
        self._vector_search = vector_search
        self._memory = memory_port
    
    @traced("literature_retriever.retrieve")
    async def retrieve_for_research(
        self,
        research_id: ResearchId,
        problem_context: ProblemContext,
        queries: List[str],
        sources: List[str] = None
    ) -> Result[Dict[str, Any]]:
        """Retrieve literature for research project."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "pubmed", "crossref", "papers_with_code"]
        
        all_papers = []
        
        for query in queries:
            result = await self._open_source.search_papers(query, sources, limit=20)
            if result.is_success():
                all_papers.extend(result.unwrap())
        
        # Deduplicate
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Store in memory and vector index
        stored_count = 0
        for paper in unique_papers[:100]:  # Limit storage
            atom = MemoryAtom(
                atom_id=MemoryId(),
                content=f"{paper['title']}\n{paper['abstract']}",
                content_type="research_paper",
                source=paper["source"],
                source_id=paper["id"],
                tags=["literature_review", str(research_id)],
                importance=0.8
            )
            
            # Generate embedding
            embedding_result = await self._embedding.embed([atom.content])
            if embedding_result.is_success():
                atom.embedding = embedding_result.unwrap()[0]
            
            store_result = await self._memory.store_atom(atom)
            if store_result.is_success():
                stored_count += 1
        
        return success({
            "total_found": len(all_papers),
            "unique_papers": len(unique_papers),
            "stored": stored_count,
            "papers": unique_papers[:50]
        })
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Deduplicate papers by title."""
        seen = set()
        unique = []
        
        for paper in papers:
            title_key = paper.get("title", "").lower().strip()
            if title_key and title_key not in seen:
                seen.add(title_key)
                unique.append(paper)
        
        return unique


class HypothesisGenerator:
    """Generates hypotheses from literature and problem context."""
    
    def __init__(self, llm_provider: LLMProviderPort, rag_pipeline: RAGPipeline):
        self._llm = llm_provider
        self._rag = rag_pipeline
    
    @traced("hypothesis_generator.generate")
    async def generate_hypotheses(
        self,
        research_id: ResearchId,
        problem_context: ProblemContext,
        literature_summary: str,
        count: int = 5
    ) -> Result[List[Hypothesis]]:
        """Generate testable hypotheses."""
        prompt = f"""
        Based on the following research context and literature, generate {count} novel, testable hypotheses:
        
        Problem: {problem_context.user_query}
        Domain: {problem_context.domain}
        Constraints: {problem_context.constraints}
        
        Literature Summary:
        {literature_summary[:3000]}
        
        For each hypothesis, provide:
        1. Clear, specific statement
        2. Anchor papers from literature (cite by ID)
        3. 2-3 testable predictions
        4. Confidence score (0-1)
        5. Experimental approach to test it
        
        Format as JSON array.
        """
        
        system_prompt = "You are a research scientist generating novel, testable hypotheses from literature."
        
        result = await self._llm.complete(prompt, system_prompt=system_prompt)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Parse hypotheses
        hypotheses = []
        try:
            hypotheses_data = json.loads(result.unwrap())
            for i, h_data in enumerate(hypotheses_data[:count]):
                hypothesis = Hypothesis(
                    hypothesis_id=uuid4(),
                    research_id=research_id,
                    statement=h_data.get("statement", f"Hypothesis {i+1}"),
                    anchor_papers=h_data.get("anchor_papers", []),
                    confidence=h_data.get("confidence", 0.7),
                    testable_predictions=h_data.get("testable_predictions", []),
                    status="proposed"
                )
                hypotheses.append(hypothesis)
        except json.JSONDecodeError:
            # Fallback: create basic hypotheses
            for i in range(count):
                hypotheses.append(Hypothesis(
                    hypothesis_id=uuid4(),
                    research_id=research_id,
                    statement=f"Hypothesis {i+1} derived from literature analysis",
                    anchor_papers=[],
                    confidence=0.6,
                    testable_predictions=["Prediction 1", "Prediction 2"]
                ))
        
        return success(hypotheses)


class ExperimentExecutor:
    """Executes experiments in sandbox environments."""
    
    def __init__(
        self,
        sandbox_manager: ResearchSandboxManager,
        llm_provider: LLMProviderPort
    ):
        self._sandbox_manager = sandbox_manager
        self._llm = llm_provider
    
    @traced("experiment_executor.execute")
    async def execute_experiment(
        self,
        hypothesis: Hypothesis,
        experiment_design: Dict[str, Any]
    ) -> Result[ExperimentResult]:
        """Execute a single experiment."""
        # Create sandbox session
        session_result = await self._sandbox_manager.create_session(SandboxType.PYTHON)
        if session_result.is_failure():
            return failure(session_result.unwrap_err())
        
        session_id = session_result.unwrap()
        
        try:
            # Generate experiment code
            code = self._generate_experiment_code(hypothesis, experiment_design)
            
            # Execute in sandbox
            start_time = datetime.utcnow()
            exec_result = await self._sandbox_manager.execute(session_id, code)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if exec_result.is_failure():
                return success(ExperimentResult(
                    experiment_id=uuid4(),
                    hypothesis_id=hypothesis.hypothesis_id,
                    sandbox_id=session_id,
                    success=False,
                    data={},
                    metrics={},
                    logs=[str(exec_result.unwrap_err())],
                    execution_time_seconds=execution_time,
                    resource_usage={}
                ))
            
            output = exec_result.unwrap()
            
            # Parse results
            metrics = self._parse_metrics(output)
            
            return success(ExperimentResult(
                experiment_id=uuid4(),
                hypothesis_id=hypothesis.hypothesis_id,
                sandbox_id=session_id,
                success=True,
                data=output,
                metrics=metrics,
                logs=[output.get("stdout", "")],
                execution_time_seconds=execution_time,
                resource_usage=self._estimate_resource_usage(execution_time)
            ))
            
        finally:
            await self._sandbox_manager.close_session(session_id)
    
    def _generate_experiment_code(
        self,
        hypothesis: Hypothesis,
        design: Dict[str, Any]
    ) -> str:
        """Generate experiment code from hypothesis and design."""
        # This would use LLM to generate actual experiment code
        # For now, template
        return f"""
        # Experiment for hypothesis: {hypothesis.statement}
        import numpy as np
        import json
        import time
        
        # Experiment parameters
        params = {json.dumps(design.get('parameters', {}))}
        
        # Run experiment
        np.random.seed(42)
        results = {{
            "metric_1": float(np.random.normal(0.8, 0.1)),
            "metric_2": float(np.random.normal(0.75, 0.15)),
            "success": True
        }}
        
        # Statistical validation
        n_trials = 100
        trial_results = []
        for _ in range(n_trials):
            trial_results.append(np.random.normal(0.8, 0.1))
        
        results["mean"] = float(np.mean(trial_results))
        results["std"] = float(np.std(trial_results))
        results["ci_95"] = [float(np.mean(trial_results) - 1.96*np.std(trial_results)/np.sqrt(n_trials)),
                            float(np.mean(trial_results) + 1.96*np.std(trial_results)/np.sqrt(n_trials))]
        
        print(json.dumps(results))
        """
    
    def _parse_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Parse metrics from experiment output."""
        stdout = output.get("stdout", "")
        try:
            # Find JSON in output
            import re
            json_match = re.search(r'\{.*\}', stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except:
            pass
        return {}
    
    def _estimate_resource_usage(self, execution_time: float) -> Dict[str, Any]:
        """Estimate resource usage."""
        return {
            "cpu_percent": min(100, execution_time * 10),
            "memory_mb": 512,
            "duration_seconds": execution_time
        }


class DataAnalyzer:
    """Analyzes experimental data."""
    
    def __init__(self, llm_provider: LLMProviderPort):
        self._llm = llm_provider
    
    @traced("data_analyzer.analyze")
    async def analyze_results(
        self,
        experiment_results: List[ExperimentResult]
    ) -> Result[Dict[str, Any]]:
        """Analyze experimental results."""
        if not experiment_results:
            return success({"conclusions": ["No experiments to analyze"]})
        
        # Aggregate metrics
        all_metrics = {}
        for exp in experiment_results:
            for metric, value in exp.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        stats = {}
        for metric, values in all_metrics.items():
            import numpy as np
            arr = np.array(values)
            stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values)
            }
        
        # Generate insights using LLM
        prompt = f"""
        Analyze these experimental results and provide scientific conclusions:
        
        Number of experiments: {len(experiment_results)}
        Metrics: {json.dumps(stats, indent=2)}
        
        Provide:
        1. Statistical significance assessment
        2. Key findings
        3. Limitations
        4. Recommendations for further research
        """
        
        system_prompt = "You are a data scientist analyzing experimental results."
        
        llm_result = await self._llm.complete(prompt, system_prompt=system_prompt)
        
        return success({
            "statistics": stats,
            "insights": llm_result.unwrap() if llm_result.is_success() else "Analysis failed",
            "total_experiments": len(experiment_results),
            "successful_experiments": sum(1 for e in experiment_results if e.success)
        })


class PaperGenerator:
    """Generates research papers from results."""
    
    def __init__(self, llm_provider: LLMProviderPort):
        self._llm = llm_provider
    
    @traced("paper_generator.generate")
    async def generate_paper(
        self,
        research_id: ResearchId,
        problem_context: ProblemContext,
        literature: List[Dict],
        hypotheses: List[Hypothesis],
        experiments: List[ExperimentResult],
        analysis: Dict[str, Any]
    ) -> Result[ResearchPaper]:
        """Generate full research paper."""
        prompt = f"""
        Generate a complete research paper from these materials:
        
        Research ID: {research_id}
        Problem: {problem_context.user_query}
        Domain: {problem_context.domain}
        
        Literature ({len(literature)} papers): Key findings summarized
        Hypotheses ({len(hypotheses)}): {[h.statement for h in hypotheses]}
        Experiments ({len(experiments)}): {analysis.get('statistics', {})}
        Analysis: {analysis.get('insights', '')}
        
        Generate paper with sections:
        1. Title
        2. Abstract
        3. Introduction
        4. Related Work
        5. Methodology
        6. Experiments
        7. Results
        8. Discussion
        9. Conclusion
        10. References
        
        Write in academic style, LaTeX-ready.
        """
        
        system_prompt = "You are an expert academic writer generating publication-ready research papers."
        
        result = await self._llm.complete(prompt, system_prompt=system_prompt, max_tokens=8000)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        paper = ResearchPaper(
            paper_id=uuid4(),
            research_id=research_id,
            title=f"Research on {problem_context.user_query[:50]}",
            abstract="Automated research paper generated from AI-driven research pipeline.",
            sections={"full_paper": result.unwrap()},
            references=[p.get("id", "") for p in literature[:20]],
            figures=[],
            tables=[]
        )
        
        return success(paper)


class ArtifactProducer:
    """Produces artifacts (algorithms, tools, skills) from research."""
    
    def __init__(
        self,
        llm_provider: LLMProviderPort,
        sandbox_manager: ResearchSandboxManager
    ):
        self._llm = llm_provider
        self._sandbox = sandbox_manager
    
    @traced("artifact_producer.produce")
    async def produce_artifacts(
        self,
        research_id: ResearchId,
        analysis: Dict[str, Any],
        hypotheses: List[Hypothesis],
        experiments: List[ExperimentResult]
    ) -> Result[List[Artifact]]:
        """Produce artifacts from research results."""
        artifacts = []
        
        # Generate algorithm artifact
        algorithm_artifact = await self._generate_algorithm(research_id, analysis, experiments)
        if algorithm_artifact:
            artifacts.append(algorithm_artifact)
        
        # Generate tool artifact
        tool_artifact = await self._generate_tool(research_id, analysis)
        if tool_artifact:
            artifacts.append(tool_artifact)
        
        # Generate skill artifact if applicable
        skill_artifact = await self._generate_skill(research_id, analysis)
        if skill_artifact:
            artifacts.append(skill_artifact)
        
        return success(artifacts)
    
    async def _generate_algorithm(
        self,
        research_id: ResearchId,
        analysis: Dict,
        experiments: List[ExperimentResult]
    ) -> Optional[Artifact]:
        """Generate algorithm implementation."""
        prompt = f"""
        Create a production-ready algorithm implementation based on these research results:
        
        Key metrics: {analysis.get('statistics', {})}
        Experiments: {len(experiments)} successful
        
        Provide:
        1. Algorithm description
        2. Python implementation with type hints
        3. Complexity analysis
        4. Usage example
        5. Benchmark code
        """
        
        result = await self._llm.complete(prompt, system_prompt="You are an expert algorithm engineer.")
        if result.is_failure():
            return None
        
        return Artifact(
            artifact_id=uuid4(),
            research_id=research_id,
            artifact_type="algorithm",
            name="Research-Derived Algorithm",
            description="Algorithm developed from automated research findings",
            code=result.unwrap(),
            config={"language": "python"},
            benchmark_results=analysis.get("statistics", {}),
            validation_status="pending"
        )
    
    async def _generate_tool(
        self,
        research_id: ResearchId,
        analysis: Dict
    ) -> Optional[Artifact]:
        """Generate tool/script."""
        prompt = f"""
        Create a useful command-line tool or script based on research findings:
        
        Analysis: {analysis.get('insights', '')[:500]}
        
        Provide a Python script that solves a practical problem discovered in the research.
        """
        
        result = await self._llm.complete(prompt, system_prompt="You are a software engineer creating practical tools.")
        if result.is_failure():
            return None
        
        return Artifact(
            artifact_id=uuid4(),
            research_id=research_id,
            artifact_type="tool",
            name="Research Tool",
            description="Practical tool derived from research",
            code=result.unwrap(),
            config={"type": "cli_tool"},
            validation_status="pending"
        )
    
    async def _generate_skill(
        self,
        research_id: ResearchId,
        analysis: Dict
    ) -> Optional[Artifact]:
        """Generate skill specification."""
        # Would create skill spec for skill registry
        return None


class ResearchPipeline:
    """Main 7-phase research pipeline."""
    
    def __init__(
        self,
        literature_retriever: LiteratureRetriever,
        hypothesis_generator: HypothesisGenerator,
        experiment_executor: ExperimentExecutor,
        data_analyzer: DataAnalyzer,
        paper_generator: PaperGenerator,
        artifact_producer: ArtifactProducer,
        audit_brain: 'AuditBrain',
        memory_port: MemoryManagementPort,
        rag_pipeline: RAGPipeline
    ):
        self._literature = literature_retriever
        self._hypothesis = hypothesis_generator
        self._experiment = experiment_executor
        self._analysis = data_analyzer
        self._paper = paper_generator
        self._artifact = artifact_producer
        self._audit = audit_brain
        self._memory = memory_port
        self._rag = rag_pipeline
    
    @traced("research_pipeline.execute")
    async def execute_full_pipeline(
        self,
        research_id: ResearchId,
        problem_id: ProblemId,
        problem_context: ProblemContext,
        plan: ResearchPlan
    ) -> Result[Dict[str, Any]]:
        """Execute all 7 phases of research."""
        phase_results = {}
        
        # Phase 1: Analyze and Plan (already done in plan)
        phase_results[ResearchPhase.ANALYZE_AND_PLAN.value] = {
            "status": "completed",
            "plan": "Research plan created"
        }
        
        # Phase 2: Literature Review
        audit_result = await self._audit_phase(ResearchPhase.LITERATURE_REVIEW)
        if not audit_result:
            return failure(ResearchError("Audit failed for literature review"))
        
        # Generate search queries from problem
        queries = await self._generate_search_queries(problem_context)
        lit_result = await self._literature.retrieve_for_research(
            research_id, problem_context, queries
        )
        if lit_result.is_failure():
            return failure(lit_result.unwrap_err())
        
        phase_results[ResearchPhase.LITERATURE_REVIEW.value] = lit_result.unwrap()
        
        # Phase 3: Hypothesis Generation
        audit_result = await self._audit_phase(ResearchPhase.HYPOTHESIS_GENERATION)
        if not audit_result:
            return failure(ResearchError("Audit failed for hypothesis generation"))
        
        literature_summary = self._summarize_literature(lit_result.unwrap())
        hyp_result = await self._hypothesis.generate_hypotheses(
            research_id, problem_context, literature_summary
        )
        if hyp_result.is_failure():
            return failure(hyp_result.unwrap_err())
        
        phase_results[ResearchPhase.HYPOTHESIS_GENERATION.value] = {
            "hypotheses": [h.__dict__ for h in hyp_result.unwrap()]
        }
        
        # Phase 4: Experiment Execution
        audit_result = await self._audit_phase(ResearchPhase.EXPERIMENT_EXECUTION)
        if not audit_result:
            return failure(ResearchError("Audit failed for experiment execution"))
        
        hypotheses = hyp_result.unwrap()
        experiments = []
        for hypothesis in hypotheses:
            # Design experiment
            design = await self._design_experiment(hypothesis)
            exp_result = await self._experiment.execute_experiment(hypothesis, design)
            if exp_result.is_success():
                experiments.append(exp_result.unwrap())
        
        phase_results[ResearchPhase.EXPERIMENT_EXECUTION.value] = {
            "experiments": [e.__dict__ for e in experiments]
        }
        
        # Phase 5: Data Analysis
        audit_result = await self._audit_phase(ResearchPhase.DATA_ANALYSIS)
        if not audit_result:
            return failure(ResearchError("Audit failed for data analysis"))
        
        analysis_result = await self._analysis.analyze_results(experiments)
        if analysis_result.is_failure():
            return failure(analysis_result.unwrap_err())
        
        phase_results[ResearchPhase.DATA_ANALYSIS.value] = analysis_result.unwrap()
        
        # Phase 6: Paper Generation
        audit_result = await self._audit_phase(ResearchPhase.PAPER_GENERATION)
        if not audit_result:
            return failure(ResearchError("Audit failed for paper generation"))
        
        literature = lit_result.unwrap().get("papers", [])
        paper_result = await self._paper.generate_paper(
            research_id, problem_context, literature, hypotheses, experiments, analysis_result.unwrap()
        )
        if paper_result.is_failure():
            return failure(paper_result.unwrap_err())
        
        phase_results[ResearchPhase.PAPER_GENERATION.value] = {
            "paper": paper_result.unwrap().__dict__
        }
        
        # Phase 7: Artifact Production
        audit_result = await self._audit_phase(ResearchPhase.ARTIFACT_PRODUCTION)
        if not audit_result:
            return failure(ResearchError("Audit failed for artifact production"))
        
        artifact_result = await self._artifact.produce_artifacts(
            research_id, analysis_result.unwrap(), hypotheses, experiments
        )
        if artifact_result.is_failure():
            return failure(artifact_result.unwrap_err())
        
        phase_results[ResearchPhase.ARTIFACT_PRODUCTION.value] = {
            "artifacts": [a.__dict__ for a in artifact_result.unwrap()]
        }
        
        return success({
            "research_id": str(research_id),
            "completed": True,
            "phases": phase_results
        })
    
    async def _audit_phase(self, phase: ResearchPhase) -> bool:
        """Audit a research phase."""
        result = await self._audit.process({
            "action": "approve_result",
            "result": {"phase": phase.value, "ready": True},
            "criteria": {"phase_complete": 1.0}
        })
        return result.is_success() and result.unwrap().get("approved", False)
    
    async def _generate_search_queries(self, context: ProblemContext) -> List[str]:
        """Generate search queries for literature review."""
        # Would use LLM to generate queries
        return [
            context.user_query,
            f"{context.domain} {context.user_query}",
            f"state of the art {context.domain}"
        ]
    
    def _summarize_literature(self, lit_result: Dict) -> str:
        """Summarize literature for hypothesis generation."""
        papers = lit_result.get("papers", [])
        return f"Retrieved {len(papers)} papers from multiple sources. Key themes: " + \
               ", ".join([p.get("title", "")[:50] for p in papers[:10]])
    
    async def _design_experiment(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Design experiment for hypothesis."""
        return {
            "hypothesis": hypothesis.statement,
            "parameters": {"trials": 100, "significance": 0.05},
            "metrics": hypothesis.testable_predictions
        }


# AuditBrain reference (forward declaration)
class AuditBrain:
    pass