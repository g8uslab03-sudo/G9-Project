"""
Infrastructure - Compression and Optimization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager

from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import CompressionError, ResourceExhaustedError
from ..shared.observability import get_logger, get_metrics, traced, metered


logger = get_logger("infrastructure.compression")
metrics = get_metrics()


class PrecisionMode(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class QuantizationMethod(str, Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    GPTQ = "gptq"
    AWQ = "awq"
    TURBOQUANT = "turboquant"


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    precision: PrecisionMode = PrecisionMode.FP16
    quantization: QuantizationMethod = QuantizationMethod.DYNAMIC
    calibration_data: Optional[List[Any]] = None
    target_modules: Optional[List[str]] = None
    exclude_modules: List[str] = field(default_factory=list)
    bits: int = 8
    group_size: int = 128


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression."""
    method: str = "1bit_adam"  # 1bit_adam, top_k, random_k, sign_sgd
    compression_ratio: float = 0.01  # For top_k/random_k
    momentum_correction: bool = True
    error_feedback: bool = True


class ModelCompressor:
    """Handles model quantization and compression."""
    
    def __init__(self, config: CompressionConfig):
        self._config = config
        self._original_model: Optional[nn.Module] = None
        self._compressed_model: Optional[nn.Module] = None
    
    @traced("model_compressor.compress")
    def compress_model(self, model: nn.Module) -> Result[nn.Module]:
        """Compress model according to configuration."""
        self._original_model = model
        
        try:
            if self._config.precision == PrecisionMode.FP16:
                self._compressed_model = self._to_fp16(model)
            elif self._config.precision == PrecisionMode.BF16:
                self._compressed_model = self._to_bf16(model)
            elif self._config.precision in [PrecisionMode.INT8, PrecisionMode.INT4]:
                self._compressed_model = self._quantize_model(model)
            else:
                self._compressed_model = model
            
            metrics.gauge("model_compression_ratio", self._get_compression_ratio())
            return success(self._compressed_model)
            
        except Exception as e:
            return failure(CompressionError(f"Model compression failed: {e}"))
    
    def _to_fp16(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        model = model.half()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.float()  # Keep normalization in FP32
        return model
    
    def _to_bf16(self, model: nn.Module) -> nn.Module:
        """Convert model to BF16."""
        if not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported, falling back to FP16")
            return self._to_fp16(model)
        
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.float()
        return model
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT8/INT4."""
        if self._config.quantization == QuantizationMethod.DYNAMIC:
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d, nn.Embedding},
                dtype=torch.qint8 if self._config.bits == 8 else torch.quint4x2
            )
        elif self._config.quantization == QuantizationMethod.STATIC:
            if not self._config.calibration_data:
                raise CompressionError("Calibration data required for static quantization")
            
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate
            with torch.no_grad():
                for data in self._config.calibration_data:
                    model(data)
            
            return torch.quantization.convert(model, inplace=True)
        
        elif self._config.quantization == QuantizationMethod.TURBOQUANT:
            return self._turboquant_quantize(model)
        
        else:
            raise CompressionError(f"Quantization method {self._config.quantization} not implemented")
    
    def _turboquant_quantize(self, model: nn.Module) -> nn.Module:
        """TurboQuant quantization - optimized for vector indices."""
        # TurboQuant is specifically for vector quantization
        # This is a placeholder for the actual implementation
        logger.info("Applying TurboQuant quantization")
        return self._quantize_model(model)
    
    def _get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self._original_model is None or self._compressed_model is None:
            return 1.0
        
        orig_size = sum(p.numel() * p.element_size() for p in self._original_model.parameters())
        comp_size = sum(p.numel() * p.element_size() for p in self._compressed_model.parameters())
        
        return orig_size / comp_size if comp_size > 0 else 1.0
    
    def get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB."""
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


class GradientCompressor:
    """Compresses gradients for distributed training."""
    
    def __init__(self, config: GradientCompressionConfig):
        self._config = config
        self._error_buffers: Dict[str, torch.Tensor] = {}
        self._momentum_buffers: Dict[str, torch.Tensor] = {}
    
    @traced("gradient_compressor.compress")
    def compress_gradient(self, name: str, gradient: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress gradient according to method."""
        if self._config.method == "1bit_adam":
            return self._compress_1bit_adam(name, gradient)
        elif self._config.method == "top_k":
            return self._compress_top_k(name, gradient)
        elif self._config.method == "random_k":
            return self._compress_random_k(name, gradient)
        elif self._config.method == "sign_sgd":
            return self._compress_sign_sgd(name, gradient)
        else:
            return gradient, {"method": "none"}
    
    def _compress_1bit_adam(self, name: str, gradient: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """1-bit Adam gradient compression."""
        # Compute sign
        sign = torch.sign(gradient)
        
        # Compute momentum if enabled
        if self._config.momentum_correction:
            if name not in self._momentum_buffers:
                self._momentum_buffers[name] = torch.zeros_like(gradient)
            
            momentum = self._momentum_buffers[name]
            momentum.mul_(0.9).add_(gradient, alpha=0.1)
            
            # Use momentum sign
            sign = torch.sign(momentum)
        
        # Error feedback
        if self._config.error_feedback:
            if name not in self._error_buffers:
                self._error_buffers[name] = torch.zeros_like(gradient)
            
            error = self._error_buffers[name]
            corrected = gradient + error
            sign = torch.sign(corrected)
            error.add_(gradient).sub_(sign * gradient.abs().mean())
        
        return sign, {"method": "1bit_adam", "original_shape": gradient.shape}
    
    def _compress_top_k(self, name: str, gradient: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Top-K gradient compression."""
        k = int(gradient.numel() * self._config.compression_ratio)
        flat_grad = gradient.flatten()
        
        # Get top-k indices
        _, indices = torch.topk(flat_grad.abs(), k)
        
        # Create sparse representation
        values = flat_grad[indices]
        
        # Error feedback
        if self._config.error_feedback:
            if name not in self._error_buffers:
                self._error_buffers[name] = torch.zeros_like(gradient)
            
            error = self._error_buffers[name]
            error.add_(gradient)
            flat_grad = error.flatten()
            
            _, indices = torch.topk(flat_grad.abs(), k)
            values = flat_grad[indices]
            error.zero_()
            error.flatten()[indices] = values
        
        compressed = torch.sparse_coo_tensor(
            indices.unsqueeze(0), values, gradient.shape
        )
        
        return compressed, {"method": "top_k", "k": k, "indices": indices}
    
    def _compress_random_k(self, name: str, gradient: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Random-K gradient compression."""
        k = int(gradient.numel() * self._config.compression_ratio)
        flat_grad = gradient.flatten()
        
        # Random sampling
        indices = torch.randperm(flat_grad.numel())[:k]
        values = flat_grad[indices]
        
        # Scale to preserve expectation
        values = values * (flat_grad.numel() / k)
        
        compressed = torch.sparse_coo_tensor(
            indices.unsqueeze(0), values, gradient.shape
        )
        
        return compressed, {"method": "random_k", "k": k, "indices": indices}
    
    def _compress_sign_sgd(self, name: str, gradient: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """SignSGD gradient compression."""
        sign = torch.sign(gradient)
        # Scale by mean absolute value
        scale = gradient.abs().mean()
        
        return sign, {"method": "sign_sgd", "scale": scale.item()}
    
    @traced("gradient_compressor.decompress")
    def decompress_gradient(
        self,
        name: str,
        compressed: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """Decompress gradient."""
        method = metadata.get("method", "none")
        
        if method == "1bit_adam":
            return compressed.to(torch.float32)
        elif method == "top_k":
            return compressed.to_dense()
        elif method == "random_k":
            return compressed.to_dense()
        elif method == "sign_sgd":
            scale = metadata.get("scale", 1.0)
            return compressed.to(torch.float32) * scale
        else:
            return compressed


class VectorIndexCompressor:
    """Compresses vector indices using RaBitQ and other methods."""
    
    def __init__(self, config: CompressionConfig):
        self._config = config
        self._codebooks: Dict[str, np.ndarray] = {}
    
    @traced("vector_compressor.compress")
    def compress_vectors(
        self,
        vectors: np.ndarray,
        method: str = "RaBitQ"
    ) -> Result[Dict[str, Any]]:
        """Compress vector array."""
        try:
            if method == "RaBitQ":
                return self._compress_rabitq(vectors)
            elif method == "PQ":  # Product Quantization
                return self._compress_pq(vectors)
            elif method == "SQ":  # Scalar Quantization
                return self._compress_sq(vectors)
            else:
                return failure(CompressionError(f"Unknown compression method: {method}"))
        except Exception as e:
            return failure(CompressionError(f"Vector compression failed: {e}"))
    
    def _compress_rabitq(self, vectors: np.ndarray) -> Result[Dict[str, Any]]:
        """RaBitQ compression - 1-bit per dimension with rotation."""
        n_vectors, dim = vectors.shape
        
        # Learn rotation matrix (PCA-based)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(dim, 256))
        rotated = pca.fit_transform(vectors)
        rotation_matrix = pca.components_
        
        # 1-bit quantization per dimension
        thresholds = np.median(rotated, axis=0)
        codes = (rotated > thresholds).astype(np.uint8)
        
        # Pack bits
        packed = np.packbits(codes, axis=1)
        
        return success({
            "compressed_data": packed.tobytes(),
            "rotation_matrix": rotation_matrix.astype(np.float32).tobytes(),
            "thresholds": thresholds.astype(np.float32).tobytes(),
            "shape": vectors.shape,
            "method": "RaBitQ",
            "compression_ratio": (vectors.nbytes) / (packed.nbytes + rotation_matrix.nbytes + thresholds.nbytes)
        })
    
    def _compress_pq(self, vectors: np.ndarray) -> Result[Dict[str, Any]]:
        """Product Quantization compression."""
        n_vectors, dim = vectors.shape
        n_subvectors = min(8, dim // 4)
        sub_dim = dim // n_subvectors
        
        from sklearn.cluster import KMeans
        
        codebooks = []
        codes = []
        
        for i in range(n_subvectors):
            start = i * sub_dim
            end = start + sub_dim
            subvecs = vectors[:, start:end]
            
            # Train codebook
            n_centroids = 256
            kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
            kmeans.fit(subvecs)
            
            codebooks.append(kmeans.cluster_centers_.astype(np.float32))
            codes.append(kmeans.predict(subvecs).astype(np.uint8))
        
        codes_array = np.column_stack(codes)
        
        return success({
            "compressed_data": codes_array.tobytes(),
            "codebooks": [cb.tobytes() for cb in codebooks],
            "shape": vectors.shape,
            "method": "PQ",
            "n_subvectors": n_subvectors,
            "compression_ratio": (vectors.nbytes) / (codes_array.nbytes + sum(cb.nbytes for cb in codebooks))
        })
    
    def _compress_sq(self, vectors: np.ndarray) -> Result[Dict[str, Any]]:
        """Scalar Quantization compression."""
        # Per-dimension quantization
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        scales = (maxs - mins) / 255
        scales[scales == 0] = 1
        
        quantized = ((vectors - mins) / scales).astype(np.uint8)
        
        return success({
            "compressed_data": quantized.tobytes(),
            "mins": mins.astype(np.float32).tobytes(),
            "scales": scales.astype(np.float32).tobytes(),
            "shape": vectors.shape,
            "method": "SQ",
            "compression_ratio": vectors.nbytes / quantized.nbytes
        })
    
    @traced("vector_compressor.decompress")
    def decompress_vectors(self, compressed: Dict[str, Any]) -> Result[np.ndarray]:
        """Decompress vectors."""
        try:
            method = compressed.get("method", "RaBitQ")
            shape = compressed["shape"]
            
            if method == "RaBitQ":
                return self._decompress_rabitq(compressed, shape)
            elif method == "PQ":
                return self._decompress_pq(compressed, shape)
            elif method == "SQ":
                return self._decompress_sq(compressed, shape)
            else:
                return failure(CompressionError(f"Unknown decompression method: {method}"))
        except Exception as e:
            return failure(CompressionError(f"Decompression failed: {e}"))
    
    def _decompress_rabitq(self, compressed: Dict, shape: Tuple[int, int]) -> Result[np.ndarray]:
        n_vectors, dim = shape
        packed = np.frombuffer(compressed["compressed_data"], dtype=np.uint8)
        codes = np.unpackbits(packed).reshape(n_vectors, -1)[:, :dim]
        
        rotation_matrix = np.frombuffer(compressed["rotation_matrix"], dtype=np.float32).reshape(-1, dim)
        thresholds = np.frombuffer(compressed["thresholds"], dtype=np.float32)
        
        # Reconstruct
        reconstructed = codes.astype(np.float32) * 2 - 1  # -1, 1
        reconstructed = reconstructed * thresholds + thresholds  # Approximate
        reconstructed = reconstructed @ rotation_matrix
        
        return success(reconstructed)
    
    def _decompress_pq(self, compressed: Dict, shape: Tuple[int, int]) -> Result[np.ndarray]:
        n_vectors, dim = shape
        n_subvectors = compressed["n_subvectors"]
        sub_dim = dim // n_subvectors
        
        codes = np.frombuffer(compressed["compressed_data"], dtype=np.uint8).reshape(n_vectors, n_subvectors)
        
        reconstructed = np.zeros(shape, dtype=np.float32)
        
        for i in range(n_subvectors):
            codebook = np.frombuffer(compressed["codebooks"][i], dtype=np.float32).reshape(256, sub_dim)
            reconstructed[:, i*sub_dim:(i+1)*sub_dim] = codebook[codes[:, i]]
        
        return success(reconstructed)
    
    def _decompress_sq(self, compressed: Dict, shape: Tuple[int, int]) -> Result[np.ndarray]:
        quantized = np.frombuffer(compressed["compressed_data"], dtype=np.uint8).reshape(shape)
        mins = np.frombuffer(compressed["mins"], dtype=np.float32)
        scales = np.frombuffer(compressed["scales"], dtype=np.float32)
        
        reconstructed = quantized.astype(np.float32) * scales + mins
        return success(reconstructed)


class MixedPrecisionTrainer:
    """Handles mixed precision training with gradient scaling."""
    
    def __init__(self, config: CompressionConfig):
        self._config = config
        self._scaler = torch.cuda.amp.GradScaler() if config.precision in [PrecisionMode.FP16, PrecisionMode.BF16] else None
        self._enabled = config.precision in [PrecisionMode.FP16, PrecisionMode.BF16]
    
    @contextmanager
    def autocast(self):
        """Autocast context for mixed precision."""
        if self._enabled:
            dtype = torch.float16 if self._config.precision == PrecisionMode.FP16 else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient scaling."""
        if self._enabled and self._scaler:
            return self._scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with gradient scaling."""
        if self._enabled and self._scaler:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient scaling."""
        if self._enabled and self._scaler:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()


class MemoryOptimizer:
    """Optimizes memory usage for training and inference."""
    
    def __init__(self, config: CompressionConfig):
        self._config = config
    
    def apply_gradient_checkpointing(self, model: nn.Module, segments: int = 4) -> nn.Module:
        """Apply gradient checkpointing to model."""
        from torch.utils.checkpoint import checkpoint_sequential
        
        # This is a simplified version
        # Real implementation would wrap specific modules
        return model
    
    def offload_to_cpu(self, model: nn.Module, modules: List[str]) -> None:
        """Offload specific modules to CPU."""
        for name, module in model.named_modules():
            if any(m in name for m in modules):
                module.cpu()
    
    def enable_activation_offloading(self, model: nn.Module) -> None:
        """Enable activation offloading to CPU."""
        # Would use DeepSpeed or similar
        pass
    
    def estimate_memory_usage(self, model: nn.Module, batch_size: int, seq_length: int) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        
        # Estimate activation memory
        hidden_size = 4096  # Typical
        num_layers = 32
        activation_memory = batch_size * seq_length * hidden_size * num_layers * 4 / (1024**3)  # FP32
        
        if self._config.precision == PrecisionMode.FP16:
            activation_memory /= 2
        elif self._config.precision == PrecisionMode.BF16:
            activation_memory /= 2
        
        # Gradient memory (same as params)
        gradient_memory = param_memory
        
        # Optimizer states (Adam: 2x params)
        optimizer_memory = param_memory * 2
        
        total = param_memory + activation_memory + gradient_memory + optimizer_memory
        
        return {
            "parameters_gb": param_memory,
            "activations_gb": activation_memory,
            "gradients_gb": gradient_memory,
            "optimizer_gb": optimizer_memory,
            "total_gb": total,
            "with_gradient_checkpointing_gb": total - activation_memory * 0.75
        }


# TurboQuant Integration
class TurboQuantBackend:
    """TurboQuant integration for vector index quantization."""
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self, dimension: int, n_subvectors: int = 8) -> None:
        """Initialize TurboQuant."""
        # Would load TurboQuant library
        self._dimension = dimension
        self._n_subvectors = n_subvectors
        self._initialized = True
    
    def quantize_index(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Quantize vector index using TurboQuant."""
        if not self._initialized:
            raise RuntimeError("TurboQuant not initialized")
        
        # Placeholder for actual TurboQuant implementation
        # Would call TurboQuant C++ backend
        return {
            "quantized": vectors.astype(np.uint8),
            "metadata": {"method": "turboquant"}
        }
    
    def search(self, query: np.ndarray, index: Dict, top_k: int) -> np.ndarray:
        """Search using TurboQuant index."""
        # Placeholder
        return np.zeros(top_k, dtype=np.int32)