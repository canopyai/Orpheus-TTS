import asyncio
import os
import threading
import queue
from typing import Generator, Iterable, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams  # type: ignore
except Exception:
    AsyncLLMEngine = None  # type: ignore
    AsyncEngineArgs = None  # type: ignore
    SamplingParams = None  # type: ignore

import requests

from .decoder import tokens_decoder_sync

class OrpheusModel:
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        tokenizer: str = 'canopylabs/orpheus-3b-0.1-pretrained',
        backend: str = 'vllm',
        **engine_kwargs,
    ):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.backend = backend  # 'vllm' | 'sglang_server'
        self.engine_kwargs = engine_kwargs
        self.engine = self._setup_engine()
        self.available_voices = ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

        # Tokenizer: provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)

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
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b":{
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name  in unsupported_models):
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name
        
    def _setup_engine(self):
        if self.backend == 'vllm':
            if AsyncEngineArgs is None or AsyncLLMEngine is None:
                raise ImportError("vLLM is not installed but backend='vllm' was requested.")
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                dtype=self.dtype,
                **self.engine_kwargs,
            )
            return AsyncLLMEngine.from_engine_args(engine_args)
        elif self.backend == 'sglang_server':
            # No in-process engine; we'll talk to SGLang OpenAI-compatible server.
            # Required keys (with defaults):
            self._sg_base_url: str = self.engine_kwargs.get('sglang_base_url', 'http://127.0.0.1:30000')
            self._sg_model_name: str = self.engine_kwargs.get('sglang_model', 'default')
            self._sg_api_key: Optional[str] = self.engine_kwargs.get('sglang_api_key', None)
            # Allow caller to pass extra headers, e.g., {'Authorization': 'Api-Key ...'}
            self._sg_extra_headers: dict = self.engine_kwargs.get('sglang_extra_headers', {})
            # validate URL
            if not self._sg_base_url.startswith('http'):
                raise ValueError("sglang_base_url must be an absolute URL, e.g., http://host:port")
            return None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt: str, voice: Optional[str] = "tara", model_type: str = "larger") -> str:
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def _decode_stop_strings(self, stop_token_ids: Optional[List[int]]) -> Optional[List[str]]:
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

    def _sglang_stream_completions(self, prompt_string: str, temperature: float, top_p: float, max_tokens: int, stop_token_ids: Optional[List[int]]) -> Iterable[str]:
        base = self._sg_base_url.rstrip('/')
        url = f"{base}/v1/completions"

        headers = {
            'Content-Type': 'application/json',
            **self._sg_extra_headers,
        }
        if self._sg_api_key and 'Authorization' not in headers:
            # Prefer Bearer; many SGLang deployments accept either
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

        # Accumulate text so the decoder sees the full sequence; this mirrors vLLM behavior.
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
                # OpenAI Completions streaming delta
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
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)

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
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))


