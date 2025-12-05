# Orpheus-TTS FP8 Integration Guide

This directory contains the implementation of FP8 (8-bit floating point) precision support for Orpheus-TTS, enabling significant performance improvements while maintaining audio quality.

## Overview

The FP8 integration is implemented in two phases:

1. **Phase 1: FP8 Inference Pipeline** (Implemented) - Low risk, high reward
2. **Phase 2: FP8 Finetuning Pipeline** (Planned) - High risk, research-oriented

## Hardware Requirements

FP8 requires specific NVIDIA GPU architectures for hardware acceleration:
- **Hopper** (H100, H200) - Compute Capability 9.0
- **Ada Lovelace** (RTX 4090, RTX 4080, L40S) - Compute Capability 8.9
- **Blackwell** (B100, B200) - Compute Capability 9.0+

**Note**: FP8 will not provide acceleration benefits on older GPUs (Ampere, Turing, etc.)

## Directory Structure

```
fp8_integration/
├── quantization/          # (removed) FP8 PTQ scripts – vLLM FP8 is runtime-only
├── inference/            # FP8 inference implementation
│   └── orpheus_fp8_engine.py
├── validation/           # Audio quality validation tools
│   └── audio_quality_validator.py
├── finetuning/          # FP8 finetuning scripts (Phase 2)
└── README.md
```

## Phase 1: FP8 Inference Pipeline

### Installation

1. Install required dependencies:
```bash
pip install llmcompressor vllm>=0.5.0 transformers torch whisper librosa soundfile matplotlib seaborn
```

2. Ensure you have the correct CUDA version (12.1+):
```bash
nvcc --version
```

### Step 1: Enable FP8 in vLLM (runtime)

vLLM’s FP8 is enabled at load time; no offline FP8 checkpoint is needed. Example:

```python
from vllm import LLM

llm = LLM(
    model="canopylabs/orpheus-3b-0.1-ft",
    quantization="fp8",
    kv_cache_dtype="fp8",
    dtype="auto",
    max_model_len=4096,
)
```

### Step 2: Run FP8 Inference

Use the FP8-enhanced engine for inference:

```python
from fp8_integration.inference.orpheus_fp8_engine import create_fp8_model

# Load FP8 model
model = create_fp8_model(
    model_path="./orpheus-3b-fp8",
    max_model_len=4096,
)

# Generate speech
audio_generator = model.generate_speech(
    prompt="Hello! This is running on an FP8 quantized model.",
    voice="tara",
    temperature=0.6,
    top_p=0.8,
)

# Save audio
import soundfile as sf
import numpy as np

audio_chunks = []
for chunk in audio_generator:
    audio_chunks.append(chunk)

audio_bytes = b''.join(audio_chunks)
audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
audio_float = audio_array.astype(np.float32) / 32767.0
sf.write("output.wav", audio_float, 24000)
```

### Step 3: Validate Audio Quality

Use the validator to compare FP8 vs BF16 outputs for your prompts:

```python
from fp8_integration.validation.audio_quality_validator import AudioQualityValidator
from fp8_integration.inference.orpheus_fp8_engine import create_fp8_model
from orpheus_tts_pypi.orpheus_tts.engine_class import OrpheusModel

validator = AudioQualityValidator(whisper_model="tiny")
fp8_model = create_fp8_model(model_path="./orpheus-3b-fp8")
bf16_model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")

# Generate and save two audio samples externally, then compare spectrograms
comparison = validator.compare_spectrograms("fp8_sample.wav", "bf16_sample.wav", save_path="spectrogram_comparison.png")
print(comparison)
```

## Performance Expectations

Based on the FP8 format and hardware:

| Metric | Expected Improvement |
|--------|---------------------|
| Inference Speed | 1.5-2x faster |
| Memory Usage | ~50% reduction |
| TTFB (Time to First Byte) | < 200ms |
| Real-Time Factor | < 0.1x |

## Audio Quality Guidelines

### Acceptable Quality Thresholds

- **WER Increase**: < 5% relative to BF16
- **SNR Decrease**: < 3dB
- **MOS Drop**: < 0.2 points
- **Spectral Correlation**: > 0.95

### Known Limitations

1. **Emotion Tags**: Subtle emotional expressions may be slightly less pronounced
2. **Voice Consistency**: Minor variations in voice characteristics possible
3. **Long Sequences**: Quality may degrade slightly on very long generations

## Troubleshooting

### Common Issues

1. **"FP8 not supported on this GPU"**
   - Check GPU compute capability: `torch.cuda.get_device_capability()`
   - FP8 requires CC 8.9 or higher

2. **Poor Audio Quality**
   - Increase calibration samples
   - Try E4M3 format instead of E5M2
   - Keep lm_head and embeddings in BF16

3. **Memory Errors**
   - Reduce `max_model_len`
   - Enable CPU offloading in vLLM
   - Use smaller batch sizes

### Debugging Tools

Check model configuration:
```python
model = create_fp8_model("./orpheus-3b-fp8")
print(model.get_model_info())
```

## Phase 2: FP8 Finetuning (Planned)

Phase 2 will implement FP8 support for model finetuning using:
- NVIDIA Transformer Engine
- FlashAttention-3 with FP8 support
- Mixed precision training with E4M3/E5M2

### Prerequisites (Phase 2)

- NVIDIA Transformer Engine
- FlashAttention-3 (compiled from source)
- PyTorch 2.2+
- CUDA 12.1+

## Best Practices

1. **Calibration Dataset**: Use diverse, representative TTS prompts
2. **Validation**: Always validate audio quality after quantization
3. **Incremental Rollout**: Test with small batches before full deployment
4. **Monitoring**: Track inference metrics and audio quality in production

## Contributing

When contributing to FP8 integration:
1. Ensure backward compatibility with BF16 models
2. Add comprehensive tests for new features
3. Document any hardware-specific optimizations
4. Include audio quality validation results

## References

- [NVIDIA FP8 Formats](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/fp8.html)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691)
- [Transformer Engine Documentation](https://github.com/NVIDIA/TransformerEngine)
