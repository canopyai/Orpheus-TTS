"""
Enhanced Orpheus-TTS Engine with FP8 Support
Extends the original OrpheusModel to support FP8 quantized models via vLLM
"""

import asyncio
import os
import threading
import queue
from typing import Generator, Iterable, List, Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams  # type: ignore
except Exception:
    AsyncLLMEngine = None  # type: ignore
    AsyncEngineArgs = None  # type: ignore
    SamplingParams = None  # type: ignore

import requests
import json

# Import the original decoder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../orpheus_tts_pypi/orpheus_tts'))
from decoder import tokens_decoder_sync


class OrpheusModelFP8:
    """
    Enhanced OrpheusModel with FP8 quantization support.
    
    This class extends the original OrpheusModel to support loading and running
    FP8 quantized models through vLLM, enabling significant performance improvements
    while maintaining audio quality.
    """
    
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        tokenizer: str = 'canopylabs/orpheus-3b-0.1-pretrained',
        backend: str = 'vllm',
        fp8_enabled: bool = False,
        fp8_config: Optional[Dict[str, Any]] = None,
        **engine_kwargs,
    ):
        """
        Initialize the FP8-enhanced Orpheus model.
        
        Args:
            model_name: Model name or path (can be local FP8 checkpoint)
            dtype: Default dtype for non-quantized models
            tokenizer: Tokenizer path
            backend: Backend to use ('vllm' or 'sglang_server')
            fp8_enabled: Whether to enable FP8 quantization
            fp8_config: FP8-specific configuration options
            **engine_kwargs: Additional arguments for the engine
        """
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.backend = backend
        self.fp8_enabled = fp8_enabled
        self.fp8_config = fp8_config or {}
        self.engine_kwargs = engine_kwargs
        
        # Check if this is a local FP8 model
        self._check_fp8_model()
        
        self.engine = self._setup_engine()
        self.available_voices = ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

        # Tokenizer: provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else self.model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)
    
    def _check_fp8_model(self):
        """Check if the model is an FP8 quantized model and auto-detect settings."""
        if os.path.isdir(self.model_name):
            config_path = Path(self.model_name) / "quantization_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    quant_config = json.load(f)
                if quant_config.get("quant_method") == "fp8":
                    print(f"Detected FP8 quantized model at {self.model_name}")
                    self.fp8_enabled = True
                    self.fp8_config.update(quant_config)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
    
    def _map_model_params(self, model_name: str) -> str:
        """Map model size names to repository IDs or paths."""
        # If it's a local path, return as-is
        if os.path.isdir(model_name):
            return model_name
            
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
            "medium-3b-fp8": {
                "repo_id": "./orpheus-3b-fp8",  # Default local FP8 model path
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name
        
    def _setup_engine(self):
        """Set up the inference engine with FP8 support."""
        if self.backend == 'vllm':
            if AsyncEngineArgs is None or AsyncLLMEngine is None:
                raise ImportError("vLLM is not installed but backend='vllm' was requested.")
            
            # Prepare engine arguments
            engine_args_dict = {
                "model": self.model_name,
                "dtype": self.dtype,
                **self.engine_kwargs,
            }
            
            # Add FP8-specific arguments if enabled
            if self.fp8_enabled:
                print("Configuring vLLM for FP8 inference...")
                
                # Set quantization method
                engine_args_dict["quantization"] = "fp8"
                
                # Enable FP8 KV cache if specified
                if self.fp8_config.get("quantize_kv_cache", True):
                    engine_args_dict["kv_cache_dtype"] = "fp8"
                    print("  - FP8 KV cache enabled")
                
                # Override dtype for FP8 models (vLLM handles this internally)
                engine_args_dict["dtype"] = "auto"
                
                # Set compute capability check
                if "enforce_eager" not in engine_args_dict:
                    # Check GPU compute capability
                    if torch.cuda.is_available():
                        capability = torch.cuda.get_device_capability()
                        if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
                            print(f"WARNING: GPU compute capability {capability[0]}.{capability[1]} detected.")
                            print("FP8 requires Hopper (8.9+), Ada (8.9+), or Blackwell GPUs.")
                            print("Falling back to eager mode, which may be slower.")
                            engine_args_dict["enforce_eager"] = True
            
            engine_args = AsyncEngineArgs(**engine_args_dict)
            return AsyncLLMEngine.from_engine_args(engine_args)
            
        elif self.backend == 'sglang_server':
            # SGLang server configuration (same as original)
            self._sg_base_url: str = self.engine_kwargs.get('sglang_base_url', 'http://127.0.0.1:30000')
            self._sg_model_name: str = self.engine_kwargs.get('sglang_model', 'default')
            self._sg_api_key: Optional[str] = self.engine_kwargs.get('sglang_api_key', None)
            self._sg_extra_headers: dict = self.engine_kwargs.get('sglang_extra_headers', {})
            
            if not self._sg_base_url.startswith('http'):
                raise ValueError("sglang_base_url must be an absolute URL, e.g., http://host:port")
            return None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def validate_voice(self, voice):
        """Validate that the requested voice is available."""
        if voice:
            if voice not in self.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt: str, voice: Optional[str] = "tara", model_type: str = "larger") -> str:
        """Format the prompt with voice and special tokens."""
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def _decode_stop_strings(self, stop_token_ids: Optional[List[int]]) -> Optional[List[str]]:
        """Decode stop token IDs to strings."""
        if not stop_token_ids:
            return None
        stops: List[str] = []
        for tid in stop_token_ids:
            try:
                stops.append(self.tokenizer.decode([tid]))
            except Exception:
                # Skip if decoding fails
                continue
        return stops or None

    def _sglang_stream_completions(self, prompt_string: str, temperature: float, top_p: float, 
                                   max_tokens: int, stop_token_ids: Optional[List[int]]) -> Iterable[str]:
        """Stream completions from SGLang server (same as original)."""
        base = self._sg_base_url.rstrip('/')
        url = f"{base}/v1/completions"

        headers = {
            'Content-Type': 'application/json',
            **self._sg_extra_headers,
        }
        if self._sg_api_key and 'Authorization' not in headers:
            headers['Authorization'] = f"Bearer {self._sg_api_key}"

        stop = self._decode_stop_strings(stop_token_ids)

        payload = {
            "model": self._sg_model_name,
            "prompt": prompt_string,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        accumulated_text = ""

        with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    line = raw_line.decode('utf-8', errors='ignore')
                else:
                    line = raw_line
                if not line.startswith('data: '):
                    continue
                data = line[6:].strip()
                if data == '[DONE]':
                    break
                try:
                    import json as _json
                    chunk = _json.loads(data)
                except Exception:
                    continue
                choices = chunk.get('choices') or []
                if not choices:
                    continue
                text_piece = choices[0].get('text')
                if text_piece:
                    accumulated_text += text_piece
                    yield accumulated_text

    def generate_tokens_sync(
        self,
        prompt: str,
        voice: Optional[str] = None,
        request_id: str = "req-001",
        temperature: float = 0.6,
        top_p: float = 0.8,
        max_tokens: int = 1200,
        stop_token_ids: List[int] = [49158],
        repetition_penalty: float = 1.3,
    ) -> Generator[str, None, None]:
        """Generate tokens synchronously with FP8 support."""
        prompt_string = self._format_prompt(prompt, voice)
        print(f"Generating with {'FP8' if self.fp8_enabled else 'BF16'} model: {prompt}")

        if self.backend == 'vllm':
            if SamplingParams is None:
                raise ImportError("vLLM is not installed but backend='vllm' was requested.")
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty,
            )

            token_queue: "queue.Queue[Optional[str]]" = queue.Queue()

            async def async_producer():
                async for result in self.engine.generate(
                    prompt=prompt_string,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    token_queue.put(result.outputs[0].text)
                token_queue.put(None)  # Sentinel

            def run_async():
                asyncio.run(async_producer())

            thread = threading.Thread(target=run_async)
            thread.start()

            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token

            thread.join()

        elif self.backend == 'sglang_server':
            for acc_text in self._sglang_stream_completions(
                prompt_string=prompt_string,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
            ):
                yield acc_text
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def generate_speech(self, **kwargs):
        """Generate speech audio from text."""
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "backend": self.backend,
            "fp8_enabled": self.fp8_enabled,
            "dtype": str(self.dtype) if not self.fp8_enabled else "fp8",
        }
        
        if self.fp8_enabled:
            info["fp8_config"] = self.fp8_config
            
        if self.backend == 'vllm' and hasattr(self.engine, 'model_config'):
            info["model_config"] = {
                "max_model_len": self.engine.model_config.max_model_len,
                "quantization": self.engine.model_config.quantization,
            }
        
        return info


# Convenience function to create an FP8-enabled model
def create_fp8_model(
    model_path: str = "./orpheus-3b-fp8",
    **kwargs
) -> OrpheusModelFP8:
    """
    Create an FP8-enabled Orpheus model.
    
    Args:
        model_path: Path to the FP8 quantized model
        **kwargs: Additional arguments for OrpheusModelFP8
    
    Returns:
        OrpheusModelFP8 instance configured for FP8 inference
    """
    return OrpheusModelFP8(
        model_name=model_path,
        fp8_enabled=True,
        **kwargs
    )
