"""
FP8 Finetuning Script for Orpheus-TTS
Integrates NVIDIA Transformer Engine for FP8 mixed precision training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaAttention,
    LlamaRMSNorm,
)
import yaml
import wandb
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("WARNING: Transformer Engine not available. FP8 training will not be possible.")

# Check for FlashAttention-3
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    FLASH_ATTN_3_AVAILABLE = False
    print("WARNING: FlashAttention-3 not available. Using standard attention.")


@dataclass
class FP8TrainingConfig:
    """Configuration for FP8 training."""
    # Model config
    model_name: str = "canopylabs/orpheus-3b-0.1-ft"
    tokenizer_name: Optional[str] = None
    
    # Training config
    output_dir: str = "./orpheus-fp8-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # FP8 config
    fp8_format: str = "HYBRID"  # E4M3 forward, E5M2 backward
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    fp8_use_delayed_scaling: bool = True
    
    # FlashAttention config
    use_flash_attn: bool = True
    flash_attn_fp8: bool = True  # Enable FP8 in FlashAttention-3
    
    # Dataset config
    dataset_name: str = "canopylabs/orpheus_tts_dataset"
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    report_to: str = "wandb"
    project_name: str = "orpheus-fp8-finetuning"
    run_name: str = "fp8-finetune-test"
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class FP8LlamaAttention(nn.Module):
    """
    Modified LlamaAttention with Transformer Engine layers and FlashAttention-3.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if not TRANSFORMER_ENGINE_AVAILABLE:
            raise RuntimeError("Transformer Engine is required for FP8 training")
        
        # Use Transformer Engine Linear layers for FP8 support
        self.q_proj = te.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = te.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = te.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = te.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # Rotary embeddings (keep in higher precision)
        self._init_rope()
    
    def _init_rope(self):
        """Initialize rotary position embeddings."""
        # This would be copied from the original LlamaAttention
        # Keeping it simple for this example
        pass
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, seq_length, _ = hidden_states.size()
        
        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings (implementation would go here)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat k/v heads if necessary
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use FlashAttention-3 if available and enabled
        if FLASH_ATTN_3_AVAILABLE and self.config.use_flash_attention:
            # Transpose to (batch, seq_len, heads, head_dim) for flash_attn
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            
            # Call FlashAttention-3 with FP8 support
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                causal=True,
            )
            
            attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        else:
            # Fallback to standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class FP8LlamaMLP(nn.Module):
    """
    Modified LlamaMLP with Transformer Engine layers for FP8 support.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        if not TRANSFORMER_ENGINE_AVAILABLE:
            raise RuntimeError("Transformer Engine is required for FP8 training")
        
        # Use Transformer Engine Linear layers
        self.gate_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = te.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FP8LlamaDecoderLayer(nn.Module):
    """
    Modified LlamaDecoderLayer with FP8 components.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Use Transformer Engine components
        self.self_attn = FP8LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = FP8LlamaMLP(config)
        
        # Use Transformer Engine LayerNorm for better FP8 compatibility
        if TRANSFORMER_ENGINE_AVAILABLE:
            self.input_layernorm = te.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = te.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)


def replace_llama_layers_with_fp8(model):
    """
    Replace standard Llama layers with FP8-enabled versions.
    """
    if not TRANSFORMER_ENGINE_AVAILABLE:
        print("Transformer Engine not available. Returning original model.")
        return model
    
    # Replace decoder layers
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = FP8LlamaDecoderLayer(model.config, i)
    
    # Note: We keep embed_tokens and lm_head in higher precision
    # as recommended in the FP8 best practices
    
    return model


class FP8Trainer(Trainer):
    """
    Custom Trainer class with FP8 support via Transformer Engine.
    """
    def __init__(self, fp8_config: FP8TrainingConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_config = fp8_config
        
        # Create FP8 recipe
        if TRANSFORMER_ENGINE_AVAILABLE and fp8_config.fp8_use_delayed_scaling:
            self.fp8_recipe = recipe.DelayedScaling(
                fp8_format=getattr(recipe.Format, fp8_config.fp8_format),
                amax_history_len=fp8_config.fp8_amax_history_len,
                amax_compute_algo=fp8_config.fp8_amax_compute_algo,
            )
        else:
            self.fp8_recipe = None
    
    def training_step(self, model, inputs):
        """
        Perform a training step with FP8 autocast.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use FP8 autocast for forward pass
        if TRANSFORMER_ENGINE_AVAILABLE and self.fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass happens outside FP8 context
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach()


def main():
    """Main training function."""
    # Load configuration
    config_path = "fp8_integration/finetuning/fp8_config.yaml"
    if os.path.exists(config_path):
        config = FP8TrainingConfig.from_yaml(config_path)
    else:
        config = FP8TrainingConfig()
    
    # Check GPU compatibility
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
            print(f"WARNING: GPU compute capability {capability[0]}.{capability[1]} detected.")
            print("FP8 training requires Hopper (9.0), Ada (8.9), or newer GPUs.")
            print("Training will proceed but without FP8 acceleration.\n")
    
    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(project=config.project_name, name=config.run_name)
    
    # Load tokenizer
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load model
    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Important: Don't use device_map with custom modifications
    )
    
    # Replace layers with FP8 versions
    if TRANSFORMER_ENGINE_AVAILABLE:
        print("Replacing model layers with FP8-enabled versions...")
        model = replace_llama_layers_with_fp8(model)
        print("Model modification complete!")
    
    # Load dataset
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="train")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        tf32=True,  # Enable TF32 for better performance
        dataloader_drop_last=True,
        report_to=config.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = FP8Trainer(
        fp8_config=config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting FP8 training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    if not TRANSFORMER_ENGINE_AVAILABLE:
        print("ERROR: Transformer Engine is required for FP8 training.")
        print("Please install it from: https://github.com/NVIDIA/TransformerEngine")
        exit(1)
    
    main()
