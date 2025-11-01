
from abc import ABC
from dataclasses import dataclass
from typing import Iterable, Iterator
import regex as re
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch
import math
import numpy.typing as npt
import json
import base64
import pickle



class Tokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    
    # Pre-tokenization pattern from GPT-2 (from the PDF)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = set(merges)
        # self.merges = merges
        self.special_tokens = special_tokens or []
        # self.special_tokens = special_tokens if special_tokens else []
        
        # Create reverse vocabulary mapping (bytes -> token_id)
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # Add special tokens to vocabulary if not already present
        for special_token in self.special_tokens:
            special_bytes = special_token.encode('utf-8')
            if special_bytes not in self.bytes_to_id:
                # Find next available ID
                max_id = max(self.vocab.keys()) if self.vocab else -1
                new_id = max_id + 1
                self.vocab[new_id] = special_bytes
                self.bytes_to_id[special_bytes] = new_id
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Load tokenizer from saved pickle files.
        
        Args:
            vocab_filepath: Path to pickle file containing vocabulary (dict[int, bytes])
            merges_filepath: Path to pickle file containing merges (list[tuple[bytes, bytes]])
            special_tokens: List of special tokens to add
            
        Returns:
            Tokenizer instance
        """
        
        # Load vocabulary from pickle file
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        
        # Load merges from pickle file
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        
    
    def encode(self, string: str) -> list[int]:
        """Encode a string into a list of token IDs using BPE."""
        if not string:
            return []
        
        # Handle special tokens by splitting on them first
        text_parts = self._split_on_special_tokens(string)
        
        # # Print the length of each part in text_parts
        # print(f"text_parts lengths: {[(len(part), is_special) for part, is_special in text_parts]}")

        result = []
        
        for part, is_special in text_parts:
            if is_special:
                # Special token - encode directly
                special_bytes = part.encode('utf-8')
                if special_bytes in self.bytes_to_id:
                    result.append(self.bytes_to_id[special_bytes])
            else:
                # Regular text - apply BPE encoding
                result.extend(self._encode_text(part))
        
        return result
    
    def _split_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text on special tokens, returning (text, is_special_token) pairs."""
        if not self.special_tokens:
            return [(text, False)]
        
        # Sort special tokens by length (descending) to prioritize longer matches
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        # Create pattern to match any special token, with longer tokens first
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        pattern = '|'.join(escaped_tokens)
        
        parts = []
        last_end = 0
        
        for match in re.finditer(pattern, text):
            # Add text before the special token
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            
            # Add the special token
            parts.append((match.group(), True))
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            parts.append((text[last_end:], False))
        
        return parts
    
    def _encode_text(self, text: str) -> list[int]:
        """Encode regular text (non-special tokens) using BPE."""
        if not text:
            return []
        
        result = []
        for match in re.finditer(self.PAT, text):
            pre_token = match.group()  # Extract the matched text

            # Convert to UTF-8 bytes
            token_bytes = pre_token.encode('utf-8')
            
            # Apply BPE merges to this pre-token
            merged_tokens = self._apply_bpe_merges(token_bytes)
            
            # Convert to token IDs
            for token in merged_tokens:
                if token in self.bytes_to_id:
                    result.append(self.bytes_to_id[token])
                else:
                    # This shouldn't happen if vocabulary is complete
                    # Fall back to individual bytes
                    for byte_val in token:
                        byte_token = bytes([byte_val])
                        if byte_token in self.bytes_to_id:
                            result.append(self.bytes_to_id[byte_token])
        
        return result
    
    # def _apply_bpe_merges(self, token_bytes: bytes) -> list[bytes]:
    #     """Apply BPE merges to a sequence of bytes."""
    #     if len(token_bytes) <= 1:
    #         return [token_bytes]
        
    #     # Start with individual bytes
    #     tokens = [bytes([b]) for b in token_bytes]
        
    #     # Apply merges in order
    #     for merge_pair in self.merges:
    #         left, right = merge_pair
            
    #         # Look for consecutive occurrences of this merge pair
    #         i = 0
    #         while i < len(tokens) - 1:
    #             if tokens[i] == left and tokens[i + 1] == right:
    #                 # Merge these two tokens
    #                 merged = left + right
    #                 tokens = tokens[:i] + [merged] + tokens[i + 2:]
    #                 # Don't increment i, check the same position again
    #             else:
    #                 i += 1
        
    #     return tokens
    

    def _apply_bpe_merges(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a sequence of bytes."""
        if len(token_bytes) <= 1:
            return [token_bytes]
        
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]

        while True:
            min_token_id = float('inf')
            best_pair_idx = -1
            merged = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    combined = pair[0] + pair[1]
                    token_id = self.bytes_to_id.get(combined)
                    if token_id is not None and token_id < min_token_id:
                        min_token_id = token_id
                        best_pair_idx = i
                        merged = combined
            
            if best_pair_idx == -1:
                break
            
            # Apply best merge
            tokens = (
                tokens[:best_pair_idx]
                + [merged]
                + tokens[best_pair_idx + 2:]
            )
        
        return tokens


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs lazily."""
        for string in iterable:
            yield from self.encode(string)
    
    def decode(self, indices: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        if not indices:
            return ""
        
        # Convert token IDs to bytes
        byte_sequences = []
        for token_id in indices:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_sequences)
        
        # Decode to string, replacing malformed bytes with Unicode replacement character
        try:
            return all_bytes.decode('utf-8', errors='replace')
        except Exception:
            return all_bytes.decode('utf-8', errors='replace')


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # Compute log_softmax directly for numerical stability
    # This is more stable than log(softmax(x)) for large inputs
    max_vals = torch.max(inputs, dim=-1, keepdim=True)[0]
    shifted = inputs - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    log_probs = shifted - log_sum_exp

    # Gather the log probabilities for the correct classes
    # targets contains the indices of the correct classes for each example
    batch_size = inputs.shape[0]
    correct_log_probs = log_probs[torch.arange(batch_size), targets]

    # Cross-entropy loss is the negative log probability of the correct class
    # Return the average loss across all examples
    return -torch.mean(correct_log_probs)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Apply weight decay (AdamW style - decoupled weight decay)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size
                step_size = group['lr'] / bias_correction1

                # Compute bias corrected second moment estimate
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    
    # Input validation
    if it < 0:
        raise ValueError(f"Iteration number must be non-negative, got {it}")
    if max_learning_rate < 0:
        raise ValueError(f"Max learning rate must be non-negative, got {max_learning_rate}")
    if min_learning_rate < 0:
        raise ValueError(f"Min learning rate must be non-negative, got {min_learning_rate}")
    if min_learning_rate > max_learning_rate:
        raise ValueError(f"Min learning rate ({min_learning_rate}) must be <= max learning rate ({max_learning_rate})")
    if warmup_iters < 0:
        raise ValueError(f"Warmup iterations must be non-negative, got {warmup_iters}")
    if cosine_cycle_iters <= 0:
        raise ValueError(f"Cosine cycle iterations must be positive, got {cosine_cycle_iters}")
    
    # Handle edge case where warmup_iters is 0
    if warmup_iters == 0:
        # Skip warmup phase, go directly to cosine annealing
        if it < cosine_cycle_iters:
            # Cosine annealing phase
            cosine_progress = it / cosine_cycle_iters
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
            return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor
        else:
            # Post-cycle phase
            return min_learning_rate
    else:
        # Phase 1: Linear warmup
        if it < warmup_iters:
            # Linear interpolation from 0 to max_learning_rate
            return (it / warmup_iters) * max_learning_rate

        # Phase 2: Cosine annealing
        elif it < cosine_cycle_iters:
            # Calculate progress through the cosine cycle (0 to 1)
            # The cosine phase goes from warmup_iters to cosine_cycle_iters-1
            cosine_iterations = cosine_cycle_iters - warmup_iters
            cosine_progress = (it - warmup_iters) / cosine_iterations
            
            # Apply cosine annealing formula
            # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
            return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor
    
        # Phase 3: Post-cycle (constant minimum learning rate)
        else:
            return min_learning_rate
        
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> Iterable[torch.nn.Parameter]:

    # Collect all parameters that have gradients
    params_with_grads = [p for p in parameters if p.grad is not None]

    if len(params_with_grads) == 0:
        return
    
    # Compute the total L2 norm of all gradients
    total_norm = 0.0
    for param in params_with_grads:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # If the total norm exceeds the maximum, scale down all gradients
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for param in params_with_grads:
            param.grad.data.mul_(clip_coef)

    return parameters


def get_batch(
    dataset: npt.NDArray, batch_size: int, 
    context_length: int, device: str   
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # Calculate the maximum valid starting index
    # We need context_length + 1 tokens to get both input and target sequences
    max_start_idx = len(dataset) - context_length

    # Sample random starting indices for each batch element
    start_indices = torch.randint(0, max_start_idx, (batch_size,))

    # Create input sequences (x) and target sequences (y)
    x = torch.zeros((batch_size, context_length), dtype=torch.long)
    y = torch.zeros((batch_size, context_length), dtype=torch.long)

    for i in range(batch_size):
        start_idx = start_indices[i].item()
        # Input sequence: tokens from start_idx to start_idx + context_length
        x[i] = torch.from_numpy(dataset[start_idx:start_idx + context_length]).long()
        # Target sequence: tokens from start_idx + 1 to start_idx + context_length + 1
        y[i] = torch.from_numpy(dataset[start_idx + 1:start_idx + context_length + 1]).long()
    
    # Move tensors to the specified device
    x = x.to(device)
    y = y.to(device)
    
    return x, y


def save_checkpoint(model, optimizer, iteration, out):

    # Create checkpoint dictionary containing model state, optimizer state, and iteration
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    # Save the checkpoint using torch.save
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    # Load the checkpoint using torch.load
    checkpoint = torch.load(src)

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the iteration number
    return checkpoint['iteration']
