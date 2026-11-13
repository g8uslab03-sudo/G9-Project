"""Kaggle T4x2 Worker: Anchor-Guided MCTS, sandbox, GPU optimization, evaporation, 12h disconnect resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional heavy deps (guarded for Kaggle T4 environment)
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import pynvml  # type: ignore
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

from shared.contracts import Checkpoint, CloudProvider, EventType, Job, JobStatus
from shared.persistence import PersistenceManager, TurboQuantVectorIndex


# ---------------------------------------------------------------------------
# GPU / VRAM helpers
# ---------------------------------------------------------------------------

def get_gpu_info() -> List[Dict[str, Any]]:
    info = []
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info.append({
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(h).decode() if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else pynvml.nvmlDeviceGetName(h),
                    "total_mb": mem.total // (1024 * 1024),
                    "used_mb": mem.used // (1024 * 1024),
                    "free_mb": mem.free // (1024 * 1024),
                })
        except Exception:
            pass
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append({
                "index": i,
                "name": props.name,
                "total_mb": props.total_memory // (1024 * 1024),
                "used_mb": 0,
                "free_mb": props.total_memory // (1024 * 1024),
            })
    return info


def load_model_low_precision(model_id: str, precision: str = "fp16") -> Any:
    """Load model in low precision to reduce VRAM (FP16/BF16)."""
    if not TORCH_AVAILABLE:
        return {"model_id": model_id, "precision": precision, "loaded": False, "reason": "torch unavailable"}
    kwargs = {}
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "int8":
        kwargs["load_in_8bit"] = True
    # Placeholder: real loading would use transformers.AutoModelForCausalLM
    return {"model_id": model_id, "precision": precision, "loaded": True, "kwargs": kwargs}


def clear_vram():
    """Clear VRAM cache between pipeline steps (evaporation of GPU memory)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Anchor-Guided MCTS
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    state: Dict[str, Any]
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    action: Optional[str] = None
    anchor_satisfied: List[str] = field(default_factory=list)

    @property
    def uct(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.total_reward / self.visits
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class AnchorGuidedMCTS:
    """Anchor-Guided Monte Carlo Tree Search for Kaggle T4x2 worker.

    Anchors are hard verification sub-goals (CUDA works, dual GPU status, benchmark,
    output manifest). UCT balances exploitation vs exploration.
    """

    def __init__(self, anchors: List[str], c: float = 1.414, max_iterations: int = 200):
        self.anchors = anchors
        self.c = c
        self.max_iterations = max_iterations
        self.root: Optional[MCTSNode] = None
        self._lock = threading.Lock()

    def search(self, initial_state: Dict[str, Any], simulate_fn: Callable[[Dict[str, Any], str], Tuple[Dict[str, Any], float, List[str]]]) -> Dict[str, Any]:
        """Run MCTS. simulate_fn(state, action) -> (new_state, reward, anchors_satisfied)."""
        with self._lock:
            self.root = MCTSNode(state=dict(initial_state))
            for _ in range(self.max_iterations):
                node = self._select(self.root)
                if not self._anchors_complete(node):
                    node = self._expand(node)
                new_state, reward, satisfied = simulate_fn(node.state, node.action or "noop")
                child = MCTSNode(state=new_state, parent=node, action=node.action, anchor_satisfied=satisfied)
                node.children.append(child)
                self._backpropagate(child, reward)
            best = self._best_child(self.root, c=0.0)
            return {
                "best_state": best.state if best else self.root.state,
                "best_action": best.action if best else None,
                "anchor_satisfied": best.anchor_satisfied if best else [],
                "iterations": self.max_iterations,
                "anchors": self.anchors,
            }

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(node.children, key=lambda n: n.uct(self.c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        # Possible actions in the sandbox
        actions = [
            "verify_cuda",
            "verify_dual_gpu",
            "run_benchmark",
            "load_skill",
            "execute_skill",
            "write_output",
            "evaporate_vram",
            "save_checkpoint",
        ]
        tried = {c.action for c in node.children}
        remaining = [a for a in actions if a not in tried]
        if not remaining:
            return node
        action = random.choice(remaining)
        new_state = dict(node.state)
        new_state["last_action"] = action
        child = MCTSNode(state=new_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _anchors_complete(self, node: MCTSNode) -> bool:
        satisfied = set(node.anchor_satisfied)
        return all(a in satisfied for a in self.anchors)

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_child(self, node: MCTSNode, c: float = 0.0) -> Optional[MCTSNode]:
        if not node.children:
            return None
        return max(node.children, key=lambda n: n.uct(c))


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class Sandbox:
    """Isolated sandbox for executing untrusted code/skills with security audit."""

    def __init__(self, work_dir: str = "sandbox_work"):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def run(self, command: List[str], timeout: int = 300) -> Dict[str, Any]:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=self.work_dir)
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False, "timeout": True}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False}

    def audit_code(self, code: str) -> Dict[str, Any]:
        """Security audit: detect malicious/attack patterns."""
        malicious_patterns = [
            "os.system", "subprocess.call", "subprocess.Popen", "eval(", "exec(",
            "socket.", "urllib.request", "requests.get", "requests.post",
            "rm -rf", "shutil.rmtree", "format(", "drop table", "DELETE FROM",
        ]
        found = [p for p in malicious_patterns if p in code]
        return {
            "malicious": len(found) > 0,
            "patterns_found": found,
            "passed": len(found) == 0,
        }


# ---------------------------------------------------------------------------
# Kaggle Worker
# ---------------------------------------------------------------------------

class KaggleWorker:
    """Kaggle T4x2 worker: micro planner executes via Anchor-Guided MCTS in sandbox."""

    def __init__(self, persistence: Optional[PersistenceManager] = None):
        self.persistence = persistence or PersistenceManager(base_path=os.getenv("DATA_PATH", "/kaggle/working/data"))
        self.sandbox = Sandbox(work_dir=os.getenv("SANDBOX_DIR", "/kaggle/working/sandbox"))
        self.mcts = AnchorGuidedMCTS(
            anchors=["verify_cuda", "verify_dual_gpu", "run_benchmark", "write_output"],
            max_iterations=int(os.getenv("MCTS_ITERATIONS", "100")),
        )
        self.active_jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def dispatch_from_gcp(self, job: Job, gcp_state: Dict[str, Any]) -> Dict[str, Any]:
        """Receive evaporated handoff from GCP, resume via MCTS."""
        with self._lock:
            self.active_jobs[str(job.id)] = job
        # Restore from checkpoint if available (12h disconnect resume)
        restored = self.persistence.restore_checkpoint(job.id, "evaporate")
        initial_state = {
            "job_id": str(job.id),
            "gcp_state": gcp_state,
            "restored": restored is not None,
            "gpu_info": get_gpu_info(),
            "step": "start",
        }
        if restored:
            initial_state["restored_bundle"] = restored.get("bundle", {})

        def simulate(state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], float, List[str]]:
            new_state = dict(state)
            new_state["last_action"] = action
            reward = 0.0
            satisfied = list(state.get("anchor_satisfied", []))

            if action == "verify_cuda":
                ok = TORCH_AVAILABLE and torch.cuda.is_available()
                new_state["cuda_ok"] = ok
                reward = 1.0 if ok else -1.0
                if ok and "verify_cuda" not in satisfied:
                    satisfied.append("verify_cuda")
            elif action == "verify_dual_gpu":
                gpus = get_gpu_info()
                new_state["gpu_count"] = len(gpus)
                reward = 1.0 if len(gpus) >= 2 else 0.5
                if len(gpus) >= 2 and "verify_dual_gpu" not in satisfied:
                    satisfied.append("verify_dual_gpu")
            elif action == "run_benchmark":
                reward = self._run_benchmark(new_state)
                if "run_benchmark" not in satisfied:
                    satisfied.append("run_benchmark")
            elif action == "load_skill":
                new_state["skill_loaded"] = True
                reward = 0.5
            elif action == "execute_skill":
                res = self.sandbox.run(["python", "-c", "print('skill executed')"])
                new_state["skill_result"] = res
                reward = 1.0 if res["success"] else -1.0
            elif action == "write_output":
                out = {"job": str(job.id), "result": "completed", "gpus": get_gpu_info()}
                self.persistence.store_atomic(
                    content=json.dumps(out, default=str),
                    content_type="kaggle_output",
                    source="kaggle_worker",
                )
                new_state["output_written"] = True
                reward = 1.0
                if "write_output" not in satisfied:
                    satisfied.append("write_output")
            elif action == "evaporate_vram":
                clear_vram()
                new_state["vram_cleared"] = True
                reward = 0.5
            elif action == "save_checkpoint":
                ckpt = Checkpoint(job_id=job.id, step="mcts_step", state=new_state)
                self.persistence.save_checkpoint(ckpt, new_state)
                new_state["checkpoint_saved"] = True
                reward = 0.5

            new_state["anchor_satisfied"] = satisfied
            return new_state, reward, satisfied

        result = self.mcts.search(initial_state, simulate)
        # Final evaporation of workspace + VRAM
        clear_vram()
        self.persistence.evaporator.evaporate(
            save_checkpoint=True,
            checkpoint_store=self.persistence.checkpoints,
            job_id=job.id,
            step="mcts_complete",
        )

        with self._lock:
            self.active_jobs.pop(str(job.id), None)

        return {
            "job_id": str(job.id),
            "mcts_result": result,
            "gpu_info": get_gpu_info(),
            "torch_available": TORCH_AVAILABLE,
            "evaporated": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _run_benchmark(self, state: Dict[str, Any]) -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        try:
            device = torch.device("cuda")
            x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
            y = torch.randn(1024, 1024, device=device, dtype=torch.float16)
            for _ in range(10):
                _ = x @ y
            torch.cuda.synchronize()
            return 1.0
        except Exception:
            return 0.0

    def health(self) -> Dict[str, Any]:
        return {
            "provider": CloudProvider.KAGGLE,
            "gpu_info": get_gpu_info(),
            "torch_available": TORCH_AVAILABLE,
            "active_jobs": len(self.active_jobs),
        }


worker = KaggleWorker()
