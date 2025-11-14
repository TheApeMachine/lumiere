"""
Local LLM Agent System - Replacement for OpenAI Agents SDK
Uses llama-cpp-python for efficient local inference with GGUF models
Supports models like Llama 3.1, Mistral, etc.
"""

import json
import logging
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

# Fallback to transformers if llama-cpp not available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# For automatic model downloading
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface-hub")

T = TypeVar('T')


@dataclass
class AgentContext:
    """Context shared between agents."""
    data: Dict[str, Any] = field(default_factory=dict)


class LocalLLM:
    """Wrapper for local LLM inference with automatic model downloading."""
    
    # Default model mappings for automatic download
    DEFAULT_MODELS = {
        "mistral-7b": {
            "gguf": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "gguf_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "transformers": "mistralai/Mistral-7B-Instruct-v0.2"
        },
        "mixtral-8x7b": {
            "gguf": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            "gguf_file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            "transformers": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        },
        "llama-3.1-70b": {
            "gguf": "TheBloke/Llama-3.1-70B-Instruct-GGUF",
            "gguf_file": "llama-3.1-70b-instruct.Q4_K_M.gguf",
            "transformers": "meta-llama/Llama-3.1-70B-Instruct"
        },
        "llama-3.1-8b": {
            "gguf": "TheBloke/Llama-3.1-8B-Instruct-GGUF",
            "gguf_file": "llama-3.1-8b-instruct.Q4_K_M.gguf",
            "transformers": "meta-llama/Llama-3.1-8B-Instruct"
        }
    }
    
    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None, use_gguf: bool = False, auto_download: bool = True):
        """
        Initialize local LLM with automatic model downloading.
        
        Args:
            model_path: Path to GGUF model file, HuggingFace repo ID, or model alias (optional)
            model_name: HuggingFace model name (e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1") - default
            use_gguf: If True, use GGUF models with llama-cpp-python (default: False, use transformers)
            auto_download: If True, automatically download models if not found locally
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.model_name = model_name
        self.use_gguf = use_gguf and LLAMA_CPP_AVAILABLE
        self.auto_download = auto_download
        
        # Use transformers by default (more reliable)
        if not self.use_gguf:
            # Resolve model name from alias if needed
            if self.model_path and self.model_path in self.DEFAULT_MODELS:
                self.model_name = self.DEFAULT_MODELS[self.model_path]["transformers"]
            elif not self.model_name:
                self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            
            if TRANSFORMERS_AVAILABLE:
                self._load_transformers(None)
            else:
                raise RuntimeError("transformers library not available. Install with: pip install transformers torch")
        else:
            # GGUF path (optional, for advanced users)
            resolved_path = self._resolve_model_path()
            if resolved_path and resolved_path.endswith('.gguf'):
                try:
                    self._load_gguf(resolved_path)
                except Exception as e:
                    logger.error(f"Failed to load GGUF model: {e}")
                    logger.warning("Falling back to transformers library...")
                    if TRANSFORMERS_AVAILABLE:
                        if self.model_path and self.model_path in self.DEFAULT_MODELS:
                            self.model_name = self.DEFAULT_MODELS[self.model_path]["transformers"]
                        elif not self.model_name:
                            self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                        self.use_gguf = False
                        self._load_transformers(None)
                    else:
                        raise RuntimeError(f"GGUF loading failed and transformers not available: {e}")
            else:
                raise RuntimeError("GGUF model path required when use_gguf=True")
    
    def _resolve_model_path(self) -> Optional[str]:
        """Resolve model path, downloading if necessary."""
        # If model_path is provided and exists, use it
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        
        # Check if model_path is a model alias
        if self.model_path and self.model_path in self.DEFAULT_MODELS:
            model_config = self.DEFAULT_MODELS[self.model_path]
            if self.use_gguf:
                return self._download_gguf_model(model_config["gguf"], model_config["gguf_file"])
            else:
                self.model_name = model_config["transformers"]
                return None
        
        # If model_path looks like a HuggingFace repo ID, try to download
        if self.model_path and "/" in self.model_path and not os.path.exists(self.model_path):
            if self.use_gguf:
                # Try to find GGUF file in the repo
                return self._download_gguf_from_repo(self.model_path)
            else:
                self.model_name = self.model_path
                return None
        
        # If model_name is provided, use transformers
        if self.model_name:
            return None
        
        # Default fallback - use transformers (handled in __init__)
        # This should not be reached if model_name is set
        return None
    
    def _download_gguf_model(self, repo_id: str, filename: str) -> str:
        """Download GGUF model from HuggingFace."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub not available. Install with: pip install huggingface-hub")
        
        # Use cache directory or local models directory
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        models_dir = os.path.join(cache_dir, "hub", "models--" + repo_id.replace("/", "--"))
        
        # Check if model already exists
        potential_paths = [
            os.path.join(models_dir, "snapshots", "*", filename),
            os.path.join(os.path.dirname(__file__), "models", filename),
            os.path.join(".", "models", filename),
        ]
        
        for pattern in potential_paths:
            import glob
            matches = glob.glob(pattern)
            if matches:
                logger.info(f"Found existing model at: {matches[0]}")
                return matches[0]
        
        # Download the model
        logger.info(f"Downloading GGUF model: {repo_id}/{filename}")
        logger.info("This may take several minutes depending on your internet connection...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded successfully to: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download model {repo_id}/{filename}. Error: {e}")
    
    def _download_gguf_from_repo(self, repo_id: str) -> Optional[str]:
        """Try to find and download GGUF file from a HuggingFace repo."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            return None
        
        # List files in the repo to find GGUF files
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                logger.warning(f"No GGUF files found in {repo_id}")
                return None
            
            # Prefer Q4_K_M quantization (good balance)
            preferred = [f for f in gguf_files if 'Q4_K_M' in f]
            if preferred:
                filename = preferred[0]
            else:
                # Fall back to any Q4 quantization
                preferred = [f for f in gguf_files if 'Q4' in f]
                filename = preferred[0] if preferred else gguf_files[0]
            
            logger.info(f"Found GGUF file: {filename}")
            return self._download_gguf_model(repo_id, filename)
        except Exception as e:
            logger.warning(f"Could not list files in {repo_id}: {e}")
            return None
    
    def _load_gguf(self, model_path: str):
        """Load GGUF model with llama-cpp-python."""
        logger.info(f"Loading GGUF model from {model_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size / (1024**3):.2f} GB")
        
        # Check if file seems valid (at least 1GB for a quantized model)
        if file_size < 1024 * 1024 * 1024:  # Less than 1GB
            raise RuntimeError(f"Model file seems too small ({file_size} bytes). Download may be incomplete.")
        
        try:
            # Use n_ctx for context window, n_threads for CPU cores
            # n_gpu_layers=-1 uses all GPU layers (Metal on Apple Silicon)
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_threads=None,  # Auto-detect CPU cores
                n_gpu_layers=-1,  # Use GPU/Metal if available
                verbose=False
            )
            logger.info("GGUF model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            logger.error(f"Model path: {model_path}")
            logger.error(f"File exists: {os.path.exists(model_path)}")
            logger.error(f"File size: {file_size} bytes")
            logger.error("This might indicate:")
            logger.error("  1. Model file is corrupted - try deleting and re-downloading")
            logger.error("  2. llama-cpp-python wasn't built with Metal support")
            logger.error("  3. Model format is incompatible with this llama-cpp-python version")
            raise
    
    def _load_transformers(self, model_path: Optional[str] = None):
        """Load model with transformers library."""
        model_name = self.model_name or "mistralai/Mistral-7B-Instruct-v0.2"
        logger.info(f"Loading transformers model: {model_name}")
        
        # Transformers will automatically download if not cached
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        if device != "auto":
            self.model = self.model.to(device)
        logger.info(f"Transformers model loaded on {device}")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt."""
        if self.use_gguf and isinstance(self.model, Llama):
            # GGUF inference
            stop_sequences = stop or []
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                echo=False
            )
            return response['choices'][0]['text'].strip()
        else:
            # Transformers inference
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text.strip()


class Agent:
    """Local agent compatible with OpenAI Agents SDK interface."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Callable]] = None,
        handoffs: Optional[List] = None,
        llm: Optional[LocalLLM] = None
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def _build_prompt(self, user_input: str, context: Optional[AgentContext] = None) -> str:
        """Build the full prompt for the agent."""
        prompt_parts = [
            self.instructions,
            "",
            "User Request:",
            user_input,
            "",
        ]
        
        if context and context.data:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context.data, indent=2))
            prompt_parts.append("")
        
        prompt_parts.append("Please provide your response in JSON format matching the requested structure.")
        
        return "\n".join(prompt_parts)
    
    async def run(self, user_input: str, context: Optional[AgentContext] = None) -> str:
        """Run the agent (async for compatibility)."""
        prompt = self._build_prompt(user_input, context)
        
        if not self.llm:
            raise RuntimeError(f"Agent {self.name} has no LLM configured")
        
        self.logger.info(f"Agent {self.name} generating response...")
        response = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
        self.logger.info(f"Agent {self.name} completed")
        
        return response


class Runner:
    """Runner for executing agents (compatible with OpenAI Agents SDK)."""
    
    @staticmethod
    async def run(agent: Agent, user_input: str, context: Optional[AgentContext] = None) -> 'RunResult':
        """Run an agent and return result."""
        response = await agent.run(user_input, context)
        return RunResult(final_output=response, agent_name=agent.name)


@dataclass
class RunResult:
    """Result from running an agent."""
    final_output: Any
    agent_name: str = ""


def function_tool(func: Callable) -> Callable:
    """Decorator for function tools (compatible with OpenAI Agents SDK)."""
    func._is_tool = True
    return func


def handoff(target_agent: Agent, tool_name_override: Optional[str] = None):
    """Create a handoff configuration (compatible with OpenAI Agents SDK)."""
    return {
        "target": target_agent,
        "tool_name": tool_name_override or f"handoff_to_{target_agent.name.lower().replace(' ', '_')}"
    }


# Recommended prompt prefix (compatible with OpenAI Agents SDK)
RECOMMENDED_PROMPT_PREFIX = """You are a helpful AI assistant. Follow instructions carefully and provide detailed, creative responses."""

