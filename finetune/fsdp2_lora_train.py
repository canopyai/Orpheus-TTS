"""
FSDP2 + LoRA Training Script for Orpheus TTS
Combines PyTorch's FSDP2 (Fully Sharded Data Parallel) with LoRA for efficient distributed fine-tuning
"""

import os
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import yaml
import wandb
import numpy as np
from functools import partial
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_distributed():
    """Initialize distributed training environment"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    """Clean up distributed training"""
    destroy_process_group()

def get_fsdp_config(config):
    """Configure FSDP2 settings"""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        ShardedStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.wrap import activation_checkpointing_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    # Auto wrap policy for transformer blocks
    llama_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Activation checkpointing
    if config.get("activation_checkpointing", False):
        from torch.distributed.fsdp.wrap import checkpoint_wrapper
        auto_wrap_policy = partial(
            activation_checkpointing_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
    else:
        auto_wrap_policy = llama_auto_wrap_policy
    
    # Mixed precision configuration
    mixed_precision_config = None
    if config.get("use_mixed_precision", True):
        from torch.distributed.fsdp import MixedPrecision
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # CPU offload configuration
    cpu_offload_config = None
    if config.get("cpu_offload", False):
        from torch.distributed.fsdp import CPUOffload
        cpu_offload_config = CPUOffload(offload_params=True)
    
    # Sharding strategy
    sharding_strategy = getattr(ShardingStrategy, config.get("sharding_strategy", "FULL_SHARD"))
    
    return {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mixed_precision_config,
        "cpu_offload": cpu_offload_config,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "use_orig_params": True,  # Required for LoRA
        "sync_module_states": True,
        "device_id": torch.cuda.current_device(),
    }

def create_peft_model(model, config):
    """Apply LoRA configuration to model"""
    lora_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 64),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=config.get("modules_to_save", ["lm_head", "embed_tokens"]),
        use_rslora=config.get("use_rslora", True),
    )
    
    model = get_peft_model(model, lora_config)
    return model

def load_orpheus_dataset(config):
    """Load Orpheus TTS dataset."""
    # The dataset is assumed to be pre-tokenized with 'input_ids' and 'labels' columns.
    dataset = load_dataset(config["dataset_path"], split="train")
    return dataset

def train_epoch(model, dataloader, optimizer, scheduler, accelerator, config, epoch):
    """Train for one epoch with FSDP2 + LoRA"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping
        if config.get("max_grad_norm", 1.0) > 0:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        if step % config.get("logging_steps", 10) == 0:
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            if accelerator.is_main_process and config.get("use_wandb", True):
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": step + epoch * len(dataloader),
                })
    
    return total_loss / len(dataloader)

def save_model_checkpoint(model, tokenizer, output_dir, epoch, accelerator):
    """Save model checkpoint with FSDP2 state dict"""
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Save FSDP2 model state
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Use FSDP2's state dict handling
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
        
        if accelerator.is_main_process:
            # Save LoRA weights
            unwrapped_model.save_pretrained(
                os.path.join(output_dir, f"checkpoint-epoch-{epoch}"),
                state_dict=state_dict,
                safe_serialization=True
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch}"))

def main():
    # Load configuration
    config_file = "fsdp2_lora_config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Initialize distributed training
    setup_distributed()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        mixed_precision="bf16" if config.get("use_mixed_precision", True) else "no",
    )
    
    # Initialize wandb
    if accelerator.is_main_process and config.get("use_wandb", True):
        wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config
        )
    
    # Load model and tokenizer
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with bfloat16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config.get("use_mixed_precision", True) else torch.float32,
        device_map=None,  # Important: Don't use device_map with FSDP
    )
    
    # Apply LoRA
    model = create_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Get FSDP configuration
    fsdp_config = get_fsdp_config(config)

    # Enable gradient checkpointing if configured
    if config.get("activation_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Wrap model with FSDP2
    model = FSDP(model, **fsdp_config)
    
    # Load dataset
    train_dataset = load_orpheus_dataset(config)
    
    # Create distributed sampler and dataloader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        sampler=train_sampler,
        collate_fn=default_data_collator,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    # Prepare scheduler
    num_training_steps = len(train_dataloader) * config["num_epochs"]
    num_warmup_steps = int(num_training_steps * config.get("warmup_ratio", 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        avg_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            config=config,
            epoch=epoch,
        )
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get("save_epochs", 1) == 0:
            save_model_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=config["output_dir"],
                epoch=epoch + 1,
                accelerator=accelerator,
            )
    
    # Final save
    save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=config["output_dir"],
        epoch="final",
        accelerator=accelerator,
    )
    
    # Cleanup
    cleanup_distributed()
    
    if accelerator.is_main_process:
        print("Training completed!")
        if config.get("use_wandb", True):
            wandb.finish()

if __name__ == "__main__":
    main()
