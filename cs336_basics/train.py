#!/usr/bin/env python3
"""
Training script for Transformer Language Models
Following CS336 Assignment 1 Section 5.3 Training loop guidance

This script uses the components implemented in models.py and utils.py
"""

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import wandb

from .models import TransformerLM
from .utils import (
    AdamW, 
    get_lr_cosine_schedule, 
    gradient_clipping, 
    get_batch, 
    cross_entropy,
    save_checkpoint,
    load_checkpoint
)


@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344  # Approximately 8/3 * d_model, multiple of 64
    rope_theta: float = 10000.0


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model config
    model_config: ModelConfig
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Learning rate schedule
    warmup_steps: int = 1000
    max_steps: int = 10000
    min_lr_ratio: float = 0.1
    
    # Training settings
    grad_clip: float = 1.0
    eval_interval: int = 500
    log_interval: int = 100
    checkpoint_interval: int = 1000
    
    # Data paths
    train_data_path: str = "data/tinystories_train_tokens.npy"
    val_data_path: str = "data/tinystories_val_tokens.npy"
    
    # Output settings
    output_dir: str = "outputs/experiment"
    experiment_name: str = "transformer_lm"
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    compile_model: bool = True
    mixed_precision: bool = True
    
    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = "cs336-transformer-lm"
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_tags: Optional[list] = None
    wandb_notes: Optional[str] = None


def get_device(device_str: str) -> str:
    """Get the appropriate device following low-resource tips."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    """Main trainer class following section 5.3 Training loop guidance."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup device
        self.device = get_device(config.device)
        print(f"Using device: {self.device}")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Setup logging
        self.log_file = self.output_dir / "training_log.jsonl"
        self.metrics = []
        
        # Load datasets with memory mapping as recommended in PDF
        print("Loading datasets...")
        self.train_data = np.load(config.train_data_path, mmap_mode='r')
        self.val_data = np.load(config.val_data_path, mmap_mode='r')
        print(f"Train data: {len(self.train_data):,} tokens")
        print(f"Val data: {len(self.val_data):,} tokens")
        
        # Initialize model
        print("Initializing model...")
        self.model = TransformerLM(
            d_model=config.model_config.d_model,
            num_heads=config.model_config.num_heads,
            d_ff=config.model_config.d_ff,
            theta=config.model_config.rope_theta,
            vocab_size=config.model_config.vocab_size,
            context_length=config.model_config.context_length,
            num_layers=config.model_config.num_layers,
            device=self.device
        )
        self.model.to(self.device)
        
        # Print model info
        param_count = count_parameters(self.model)
        print(f"Model parameters: {param_count:,}")
        
        # Compile model if requested (following low-resource tips)
        if config.compile_model:
            try:
                if self.device == "cpu":
                    self.model = torch.compile(self.model)
                elif self.device == "mps":
                    self.model = torch.compile(self.model, backend="aot_eager")
                else:  # cuda
                    self.model = torch.compile(self.model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )

        # self.optimizer = AdamW(
        #     self.model.parameters(),
        #     lr=config.learning_rate,
        #     betas=(config.beta1, config.beta2),
        #     eps=config.eps,
        #     weight_decay=config.weight_decay
        # )
        
        # Initialize mixed precision scaler if using CUDA
        self.scaler = None
        if config.mixed_precision and self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.step = 0
        self.start_time = time.time()
        
        # Initialize Weights & Biases
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            self.init_wandb()
        
        print("Trainer initialized successfully!")
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            # Create wandb config from training config
            wandb_config = {
                # Model config
                "vocab_size": self.config.model_config.vocab_size,
                "context_length": self.config.model_config.context_length,
                "d_model": self.config.model_config.d_model,
                "num_layers": self.config.model_config.num_layers,
                "num_heads": self.config.model_config.num_heads,
                "d_ff": self.config.model_config.d_ff,
                "rope_theta": self.config.model_config.rope_theta,
                
                # Training hyperparameters
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "beta1": self.config.beta1,
                "beta2": self.config.beta2,
                "eps": self.config.eps,
                
                # Learning rate schedule
                "warmup_steps": self.config.warmup_steps,
                "max_steps": self.config.max_steps,
                "min_lr_ratio": self.config.min_lr_ratio,
                
                # Training settings
                "grad_clip": self.config.grad_clip,
                "eval_interval": self.config.eval_interval,
                "log_interval": self.config.log_interval,
                "checkpoint_interval": self.config.checkpoint_interval,
                
                # Model info
                "model_parameters": count_parameters(self.model),
                "device": self.device,
                "compile_model": self.config.compile_model,
                "mixed_precision": self.config.mixed_precision,
                
                # Data info
                "train_tokens": len(self.train_data),
                "val_tokens": len(self.val_data),
                "target_tokens": self.config.batch_size * self.config.max_steps * self.config.model_config.context_length,
            }
            
            # Initialize wandb run
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config=wandb_config,
                tags=self.config.wandb_tags,
                notes=self.config.wandb_notes,
                dir=str(self.output_dir),
                resume="allow",
                id=f"{self.config.experiment_name}_{int(time.time())}"
            )
            
            # Watch model for gradients and parameters
            wandb.watch(self.model, log="all", log_freq=self.config.log_interval)
            
            print(f"Weights & Biases initialized: {wandb.run.url}")
            
        except Exception as e:
            print(f"Failed to initialize Weights & Biases: {e}")
            print("Continuing without wandb logging...")
            self.use_wandb = False
    
    def get_lr(self) -> float:
        """Get current learning rate based on schedule."""
        min_lr = self.config.learning_rate * self.config.min_lr_ratio
        return get_lr_cosine_schedule(
            self.step,
            self.config.learning_rate,
            min_lr,
            self.config.warmup_steps,
            self.config.max_steps
        )
    
    def update_lr(self):
        """Update optimizer learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()
        
        # Get batch
        x, y = get_batch(
            self.train_data, 
            self.config.batch_size, 
            self.config.model_config.context_length, 
            self.device
        )
        
        # Forward pass with mixed precision if available
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.model(x)
                # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                loss = cross_entropy(logits_flat, targets_flat)
        else:
            logits = self.model(x)
            # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            loss = cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            gradient_clipping(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            gradient_clipping(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'lr': self.get_lr(),
            'perplexity': calculate_perplexity(loss.item())
        }
    
    def eval_step(self) -> Dict[str, float]:
        """Perform evaluation on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = min(100, len(self.val_data) // (self.config.batch_size * self.config.model_config.context_length))
        
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = get_batch(
                    self.val_data, 
                    self.config.batch_size, 
                    self.config.model_config.context_length, 
                    self.device
                )
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        logits_flat = logits.view(-1, logits.size(-1))
                        targets_flat = y.view(-1)
                        loss = cross_entropy(logits_flat, targets_flat)
                else:
                    logits = self.model(x)
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = y.view(-1)
                    loss = cross_entropy(logits_flat, targets_flat)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {
            'val_loss': avg_loss,
            'val_perplexity': calculate_perplexity(avg_loss)
        }
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to file, console, and wandb."""
        # Add timestamp and step
        metrics['step'] = self.step
        metrics['elapsed_time'] = time.time() - self.start_time
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Store in memory
        self.metrics.append(metrics)
        
        # Log to wandb
        if self.use_wandb:
            try:
                # Create wandb log dict with proper prefixes
                wandb_metrics = {}
                for key, value in metrics.items():
                    if key == 'step':
                        continue  # wandb handles step automatically
                    elif key.startswith('val_'):
                        wandb_metrics[f"validation/{key[4:]}"] = value
                    elif key in ['loss', 'lr', 'perplexity']:
                        wandb_metrics[f"train/{key}"] = value
                    else:
                        wandb_metrics[key] = value
                
                wandb.log(wandb_metrics, step=self.step)
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
        
        # Print to console
        if 'loss' in metrics:
            print(f"Step {self.step:6d} | Loss: {metrics['loss']:.4f} | "
                  f"PPL: {metrics['perplexity']:.2f} | LR: {metrics['lr']:.2e} | "
                  f"Time: {metrics['elapsed_time']:.1f}s")
        
        if 'val_loss' in metrics:
            print(f"Step {self.step:6d} | Val Loss: {metrics['val_loss']:.4f} | "
                  f"Val PPL: {metrics['val_perplexity']:.2f}")
    
    def save_checkpoint_wrapper(self):
        """Save checkpoint using the utils function."""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.step}.pt"
        save_checkpoint(
            self.model, 
            self.optimizer, 
            self.step, 
            checkpoint_path
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        save_checkpoint(
            self.model, 
            self.optimizer, 
            self.step, 
            latest_path
        )
    
    def load_checkpoint_wrapper(self, checkpoint_path: str):
        """Load checkpoint using the utils function."""
        self.step = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer
        )
        print(f"Resumed from step {self.step}")
    
    def train(self):
        """Main training loop following section 5.3 guidance."""
        print(f"\nStarting training for {self.config.max_steps:,} steps...")
        print(f"Target tokens: {self.config.batch_size * self.config.max_steps * self.config.model_config.context_length:,}")
        print("=" * 60)
        
        while self.step < self.config.max_steps:
            # Update learning rate
            self.update_lr()
            
            # Training step
            train_metrics = self.train_step()
            
            # Logging
            if self.step % self.config.log_interval == 0:
                self.log_metrics(train_metrics)
            
            # Evaluation
            if self.step % self.config.eval_interval == 0:
                eval_metrics = self.eval_step()
                self.log_metrics(eval_metrics)
            
            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0 and self.step > 0:
                self.save_checkpoint_wrapper()
            
            self.step += 1
        
        # Final checkpoint and evaluation
        print("\nTraining completed!")
        self.save_checkpoint_wrapper()
        
        final_eval = self.eval_step()
        self.log_metrics(final_eval)
        
        print(f"Final validation loss: {final_eval['val_loss']:.4f}")
        print(f"Final validation perplexity: {final_eval['val_perplexity']:.2f}")
        print(f"Total training time: {time.time() - self.start_time:.1f}s")
        
        # Finish wandb run
        if self.use_wandb:
            try:
                wandb.finish()
                print("Weights & Biases run finished successfully")
            except Exception as e:
                print(f"Failed to finish wandb run: {e}")


def create_tinystories_config() -> TrainingConfig:
    """Create default configuration for TinyStories following section 7.2."""
    model_config = ModelConfig(
        vocab_size=10000,        # As specified in section 7.2
        context_length=256,      # As specified in section 7.2
        d_model=512,            # As specified in section 7.2
        num_layers=4,           # As specified in section 7.2
        num_heads=16,           # As specified in section 7.2
        d_ff=1344,              # Approximately 8/3 * d_model, multiple of 64
        rope_theta=10000.0,     # As specified in section 7.2
    )
    
    # Calculate steps for ~327M tokens (as specified in PDF)
    batch_size = 64
    context_length = 256
    target_tokens = 327_680_000
    max_steps = target_tokens // (batch_size * context_length)

    # Create experiment name with dataset name and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "tinystories"  # Extract from data path or use default
    experiment_name = f"{dataset_name}_{timestamp}_wd0.1"
    
    return TrainingConfig(
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=6e-3,      # This needs to be tuned as per problem (learning_rate)
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,             # LLaMA-style as mentioned in PDF
        eps=1e-8,
        warmup_steps=1000,
        max_steps=max_steps,
        min_lr_ratio=0.1,
        grad_clip=1.0,
        eval_interval=500,
        log_interval=100,
        checkpoint_interval=5000,
        train_data_path="tokenized_datasets/tinystories_train_tokens_0.npy",
        val_data_path="tokenized_datasets/tinystories_valid_tokens_0.npy",
        output_dir=f"outputs/{experiment_name}",
        experiment_name=experiment_name,
        device="auto",
        compile_model=True,
        mixed_precision=False
    )



def create_owt_config() -> TrainingConfig:
    """Create default configuration for TinyStories following section 7.2."""
    model_config = ModelConfig(
        vocab_size=32000,        # As specified in section 7.2
        context_length=256,      # As specified in section 7.2
        d_model=512,            # As specified in section 7.2
        num_layers=4,           # As specified in section 7.2
        num_heads=16,           # As specified in section 7.2
        d_ff=1344,              # Approximately 8/3 * d_model, multiple of 64
        rope_theta=10000.0,     # As specified in section 7.2
    )
    
    # Calculate steps for ~327M tokens (as specified in PDF)
    batch_size = 64
    # context_length = 256
    # target_tokens = 327_680_000
    # max_steps = target_tokens // (batch_size * context_length)
    max_steps = 20000

    # Create experiment name with dataset name and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "owt"  # Extract from data path or use default
    experiment_name = f"{dataset_name}_{timestamp}_wd0.1_lr1e-2"
    
    return TrainingConfig(
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=1e-2,      # This needs to be tuned as per problem (learning_rate)
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,             # LLaMA-style as mentioned in PDF
        eps=1e-8,
        warmup_steps=1000,
        max_steps=max_steps,
        min_lr_ratio=0.1,
        grad_clip=1.0,
        eval_interval=500,
        log_interval=100,
        checkpoint_interval=5000,
        train_data_path="tokenized_datasets/owt_train_tokens.npy",
        val_data_path="tokenized_datasets/owt_valid_tokens.npy",
        output_dir=f"outputs/{experiment_name}",
        experiment_name=experiment_name,
        device="auto",
        compile_model=True,
        mixed_precision=False
    )


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    # Weights & Biases arguments
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cs336-transformer-lm", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity (username/team)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", help="Wandb tags for the run")
    parser.add_argument("--wandb-notes", type=str, help="Wandb notes for the run")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config == 'owt':
        config = create_owt_config()
        print('Using preset OWT config')
    elif args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Convert to dataclass (simplified - you might want more robust conversion)
        config = TrainingConfig(**config_dict)
    else:
        config = create_tinystories_config()
        print('Using preset TinyStories config')
        
        # Override with command line args
        if args.output_dir != "outputs/experiment":
            config.output_dir = args.output_dir
        if args.batch_size != 32:
            config.batch_size = args.batch_size
        if args.learning_rate != 3e-4:
            config.learning_rate = args.learning_rate
        if args.max_steps != 10000:
            config.max_steps = args.max_steps
        if args.device != "auto":
            config.device = args.device
    
    # Override wandb settings with command line args
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project != "cs336-transformer-lm":
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.wandb_tags:
        config.wandb_tags = args.wandb_tags
    if args.wandb_notes:
        config.wandb_notes = args.wandb_notes
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint_wrapper(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
