import unittest
import os
import torch
import yaml
from unittest.mock import patch, MagicMock

# To run this test, you need to have a distributed environment set up.
# You can use torchrun for this:
# torchrun --nproc_per_node=2 finetune/test_fsdp2_lora.py

# Add the parent directory to the Python path to allow importing the training script
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finetune.fsdp2_lora_train import (
    setup_distributed,
    cleanup_distributed,
    get_fsdp_config,
    create_peft_model,
    train_epoch
)

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from datasets import Dataset

class TestFSDP2LoRATraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment and mock configurations."""
        if "LOCAL_RANK" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
        
        setup_distributed()
        
        # Mock configuration
        cls.config = {
            "model_name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "dataset_path": "dummy-dataset",
            "max_length": 128,
            "seed": 42,
            "per_device_train_batch_size": 1,
            "num_epochs": 1,
            "learning_rate": 1e-4,
            "use_mixed_precision": False,
            "sharding_strategy": "NO_SHARD",
            "activation_checkpointing": False,
            "lora_r": 4,
            "lora_alpha": 8,
            "project_name": "test-project",
            "run_name": "test-run",
            "use_wandb": False,
            "gradient_accumulation_steps": 1,
        }

        # Mock dataset
        data = {
            "input_ids": [torch.randint(0, 1000, (cls.config["max_length"],)).tolist()],
            "labels": [torch.randint(0, 1000, (cls.config["max_length"],)).tolist()],
            "attention_mask": [[1] * cls.config["max_length"]],
        }
        cls.mock_dataset = Dataset.from_dict(data)

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment."""
        cleanup_distributed()

    def test_single_training_step(self):
        """Test a single training step with a mock model and dataset."""
        accelerator = Accelerator()

        # Load a tiny model for testing
        model = AutoModelForCausalLM.from_pretrained(self.config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

        # Apply LoRA
        model = create_peft_model(model, self.config)

        # FSDP wrapping
        fsdp_config = get_fsdp_config(self.config)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model, **fsdp_config)

        # Dataloader
        sampler = DistributedSampler(self.mock_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
        dataloader = DataLoader(self.mock_dataset, batch_size=1, sampler=sampler, collate_fn=default_data_collator)

        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # Prepare with accelerator
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters() if p.requires_grad]

        # Run one training epoch (which is one step in this case)
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, accelerator, self.config, 0)
        
        # Get updated parameters
        updated_params = [p.clone() for p in model.parameters() if p.requires_grad]

        # Assertions
        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0)

        # Check if parameters have been updated
        params_changed = False
        for initial_param, updated_param in zip(initial_params, updated_params):
            if not torch.equal(initial_param.cpu(), updated_param.cpu()):
                params_changed = True
                break
        
        # In a single-GPU test with NO_SHARD, parameters should update.
        # In a multi-GPU test, we can't guarantee which shard gets updated on which rank,
        # but the loss calculation should still work.
        if accelerator.num_processes == 1:
            self.assertTrue(params_changed, "Model parameters were not updated after one training step.")

if __name__ == "__main__":
    unittest.main()

