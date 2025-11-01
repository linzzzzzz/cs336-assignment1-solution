import time
from collections import defaultdict
import os
import pickle
import json
from pathlib import Path



class BPEStepProfiler:
    """Profiler that instruments each step of the BPE training algorithm."""
    
    def __init__(self):
        self.step_times = {}
        self.step_start_time = None
        self.total_start_time = None
        self.merge_times = []  # Track individual merge times
        
    def start_total_timing(self):
        """Start timing the entire BPE training process."""
        self.total_start_time = time.perf_counter()
        
    def start_step(self, step_name: str):
        """Start timing a specific step."""
        self.step_start_time = time.perf_counter()
        
    def end_step(self, step_name: str):
        """End timing a specific step and record the duration."""
        if self.step_start_time is None:
            raise ValueError(f"start_step() must be called before end_step() for {step_name}")
        
        duration = time.perf_counter() - self.step_start_time
        self.step_times[step_name] = duration
        self.step_start_time = None
        
    def record_merge_time(self, merge_idx: int, duration: float):
        """Record the time for an individual merge operation."""
        self.merge_times.append((merge_idx, duration))
        
    def get_total_time(self) -> float:
        """Get the total time elapsed since start_total_timing()."""
        if self.total_start_time is None:
            return 0.0
        return time.perf_counter() - self.total_start_time
        
    def print_report(self):
        """Print a detailed timing report."""
        total_time = self.get_total_time()
        
        print("\n" + "="*80)
        print("BPE TRAINING STEP-BY-STEP PROFILING REPORT")
        print("="*80)
        
        print(f"\nTotal Training Time: {total_time:.3f} seconds")
        
        # Separate skipped and executed steps
        executed_steps = {k: v for k, v in self.step_times.items() if not k.endswith("(SKIPPED)")}
        skipped_steps = {k: v for k, v in self.step_times.items() if k.endswith("(SKIPPED)")}
        
        executed_time = sum(executed_steps.values())
        print(f"Total Executed Steps Time: {executed_time:.3f} seconds")
        
        if skipped_steps:
            print(f"Skipped Steps: {len(skipped_steps)} (resumed from checkpoint)")
        
        print("\nDetailed Step Breakdown:")
        print("-" * 70)
        
        # Sort executed steps by time (descending)
        sorted_executed = sorted(executed_steps.items(), key=lambda x: x[1], reverse=True)
        
        for step_name, duration in sorted_executed:
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{step_name:<50} {duration:>8.3f}s ({percentage:>5.1f}%)")
        
        # Show skipped steps if any
        if skipped_steps:
            print("\nSkipped Steps (Resumed from Checkpoint):")
            for step_name in sorted(skipped_steps.keys()):
                clean_name = step_name.replace(" (SKIPPED)", "")
                print(f"{clean_name:<50} {'SKIPPED':>8s} {'':>8s}")
            
        print("-" * 70)
        
        # Merge timing analysis
        if self.merge_times:
            print(f"\nMerge Operations Analysis ({len(self.merge_times)} merges):")
            total_merge_time = sum(duration for _, duration in self.merge_times)
            avg_merge_time = total_merge_time / len(self.merge_times)
            print(f"• Total merge time: {total_merge_time:.3f}s")
            print(f"• Average time per merge: {avg_merge_time:.6f}s")
            
            # Show slowest merges
            slowest_merges = sorted(self.merge_times, key=lambda x: x[1], reverse=True)[:5]
            print("• Slowest merges:")
            for merge_idx, duration in slowest_merges:
                print(f"  - Merge {merge_idx}: {duration:.6f}s")
        
        # Identify bottlenecks
        print("\nBottleneck Analysis:")
        if sorted_executed:
            slowest_step, slowest_time = sorted_executed[0]
            print(f"• Slowest step: {slowest_step} ({slowest_time:.3f}s, {(slowest_time/total_time)*100:.1f}%)")
            
            if len(sorted_executed) > 1:
                second_slowest_step, second_slowest_time = sorted_executed[1]
                print(f"• Second slowest: {second_slowest_step} ({second_slowest_time:.3f}s, {(second_slowest_time/total_time)*100:.1f}%)")
        
        print("\nOptimization Recommendations:")
        if sorted_executed:
            slowest_step, slowest_time = sorted_executed[0]
            slowest_pct = (slowest_time / total_time) * 100
            
            if "Pre-tokenization" in slowest_step and slowest_pct > 30:
                print("• Consider using multiprocessing for pre-tokenization")
                print("• Optimize regex operations or use compiled patterns")
            elif "BPE Merging" in slowest_step and slowest_pct > 40:
                print("• The incremental pair counting optimization is working")
                print("• Consider further optimizations like caching or C++ extensions")
            elif "Initial Pair Counting" in slowest_step and slowest_pct > 20:
                print("• Consider optimizing the initial pair counting algorithm")
                print("• Use more efficient data structures for pair tracking")
        
        print("="*80)



# Global helper function for multiprocessing (must be at module level to be pickable)
def _process_text_chunk_for_bpe(args):
    """Process a chunk of text and return word frequencies. Module-level function for multiprocessing."""
    import regex as re
    from tests.common import gpt2_bytes_to_unicode
    import time
    import os
    
    # Unpack arguments - now supports optional progress tracking
    if len(args) == 3:
        text_chunk, special_tokens, chunk_info = args
        chunk_id, total_chunks, enable_progress = chunk_info
    else:
        # Backward compatibility
        text_chunk, special_tokens = args
        chunk_id, total_chunks, enable_progress = None, None, False
    
    start_time = time.perf_counter() if enable_progress else None
    
    # GPT-2 regex pattern for pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Get GPT-2 byte to unicode mapping
    byte_encoder = gpt2_bytes_to_unicode()
    
    chunk_word_freqs = defaultdict(int)
    tokens_processed = 0
    chars_processed = 0
    total_chars = len(text_chunk)
    
    if enable_progress and chunk_id is not None:
        chunk_size_mb = len(text_chunk.encode('utf-8')) / (1024 * 1024)
        print(f"[Process {os.getpid()}] Starting chunk {chunk_id + 1}/{total_chunks} "
              f"({chunk_size_mb:.2f} MB)")
    
    # Handle special tokens in this chunk
    if special_tokens:
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        text_parts = re.split(f'({special_pattern})', text_chunk)
        text_parts = [part for part in text_parts if part and part not in special_tokens]
        
        # Pre-tokenize each text part separately
        part_offset = 0
        for part_idx, text_part in enumerate(text_parts):
            for match in re.finditer(PAT, text_part):
                token = match.group()
                token_bytes = token.encode('utf-8')
                gpt2_chars = tuple(byte_encoder[b] for b in token_bytes)
                chunk_word_freqs[gpt2_chars] += 1
                tokens_processed += 1
                
                # Update character position based on match end position
                chars_processed = part_offset + match.end()
                
                # Progress update every 2000000 tokens
                if enable_progress and tokens_processed % 2000000 == 0:
                    elapsed = time.perf_counter() - start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    completion_pct = (chars_processed / total_chars) * 100
                    print(f"[Process {os.getpid()}] Chunk {chunk_id + 1}: "
                          f"{tokens_processed:,} tokens processed "
                          f"({tokens_per_sec:.0f} tokens/sec, {completion_pct:.1f}% complete)")
            
            # Update part offset for next text part
            part_offset += len(text_part)
    else:
        # Pre-tokenize the entire chunk if no special tokens
        for match in re.finditer(PAT, text_chunk):
            token = match.group()
            token_bytes = token.encode('utf-8')
            gpt2_chars = tuple(byte_encoder[b] for b in token_bytes)
            chunk_word_freqs[gpt2_chars] += 1
            tokens_processed += 1
            
            # Update character position based on match end position
            chars_processed = match.end()
            
            # Progress update every 1000000 tokens
            if enable_progress and tokens_processed % 1000000 == 0:
                elapsed = time.perf_counter() - start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                completion_pct = (chars_processed / total_chars) * 100
                print(f"[Process {os.getpid()}] Chunk {chunk_id + 1}: "
                      f"{tokens_processed:,} tokens processed "
                      f"({tokens_per_sec:.0f} tokens/sec, {completion_pct:.1f}% complete)")
    
    if enable_progress and chunk_id is not None:
        elapsed = time.perf_counter() - start_time
        unique_tokens = len(chunk_word_freqs)
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        print(f"[Process {os.getpid()}] Completed chunk {chunk_id + 1}/{total_chunks}: "
              f"{tokens_processed:,} total tokens, {unique_tokens:,} unique tokens "
              f"in {elapsed:.2f}s ({tokens_per_sec:.0f} tokens/sec)")
    
    return dict(chunk_word_freqs)


def save_checkpoint(checkpoint_path: str, merge_idx: int, vocab: dict, merges_list: list, 
                   word_freqs: dict, pair_counts: dict, training_params: dict):
    """Save BPE training checkpoint to disk."""
    checkpoint_data = {
        'merge_idx': merge_idx,
        'vocab': vocab,
        'merges_list': merges_list,
        'word_freqs': dict(word_freqs),  # Convert defaultdict to regular dict
        'pair_counts': dict(pair_counts),  # Convert defaultdict to regular dict
        'training_params': training_params,
        'timestamp': time.time()
    }
    
    # Use atomic write: write to temp file then rename
    temp_path = checkpoint_path + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        os.rename(temp_path, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} (merge {merge_idx})")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load BPE training checkpoint from disk."""
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Convert back to defaultdicts
        word_freqs = defaultdict(int, checkpoint_data['word_freqs'])
        pair_counts = defaultdict(int, checkpoint_data['pair_counts'])
        
        print(f"Checkpoint loaded: {checkpoint_path} (merge {checkpoint_data['merge_idx']})")
        
        return (
            checkpoint_data['merge_idx'],
            checkpoint_data['vocab'],
            checkpoint_data['merges_list'],
            word_freqs,
            pair_counts,
            checkpoint_data['training_params']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file in the given directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoint_files:
        return None
    
    # Sort by merge index (extracted from filename)
    def extract_merge_idx(path):
        try:
            # Extract merge index from filename like "checkpoint_1000.pkl"
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return -1
    
    latest_checkpoint = max(checkpoint_files, key=extract_merge_idx)
    return str(latest_checkpoint)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    enable_profiling: bool = True,
    checkpoint_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
    checkpoint_interval: int = 1000,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    import multiprocessing
    import os
    import regex as re
    from tests.common import gpt2_bytes_to_unicode
    from cs336_basics.pretokenization_example import find_chunk_boundaries
    
    # Initialize profiler first to capture all timing
    profiler = BPEStepProfiler() if enable_profiling else None
    if profiler:
        profiler.start_total_timing()
        profiler.start_step("Step 0: Setup and Initialization")
    
    # Store training parameters for checkpointing
    training_params = {
        'input_path': str(input_path),
        'vocab_size': vocab_size,
        'special_tokens': special_tokens,
        'checkpoint_interval': checkpoint_interval
    }
    
    # Check for resume from checkpoint
    checkpoint_to_resume = None
    resumed_from_checkpoint = False
    if resume_from_checkpoint:
        checkpoint_to_resume = resume_from_checkpoint
        if enable_profiling:
            print(f"Resuming from specified checkpoint: {checkpoint_to_resume}")
    elif checkpoint_dir:
        # Auto-detect latest checkpoint
        checkpoint_to_resume = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_to_resume and enable_profiling:
            print(f"Auto-detected checkpoint to resume: {checkpoint_to_resume}")
    
    # Initialize variables for both fresh and resumed training
    start_merge_idx = 0
    vocab = None
    merges_list = []
    word_freqs = None
    pair_counts = None
    
    # If resuming from checkpoint, load state and skip pre-processing
    if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
        try:
            start_merge_idx, vocab, merges_list, word_freqs, pair_counts, saved_params = load_checkpoint(checkpoint_to_resume)
            
            # Validate checkpoint compatibility
            if (saved_params['vocab_size'] != vocab_size or 
                saved_params['special_tokens'] != special_tokens):
                raise ValueError("Checkpoint parameters don't match current training parameters")
            
            resumed_from_checkpoint = True
            if enable_profiling:
                print(f"Resumed from checkpoint at merge {start_merge_idx}")
            
        except Exception as e:
            if enable_profiling:
                print(f"Failed to resume from checkpoint: {e}")
                print("Starting fresh training...")
            # Reset to fresh training
            start_merge_idx = 0
            vocab = None
            merges_list = []
            word_freqs = None
            pair_counts = None
            resumed_from_checkpoint = False
    
    # Get GPT-2 byte to unicode mapping
    byte_encoder = gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    if profiler:
        profiler.end_step("Step 0: Setup and Initialization")
    
    # Handle profiling for both fresh and resumed training scenarios
    if word_freqs is None:
        # Fresh training - execute all pre-processing steps
        if profiler:
            profiler.start_step("Step 1: Parallel Pre-tokenization Setup")
    else:
        # Resumed training - record skipped steps
        if profiler:
            profiler.step_times["Step 1: Parallel Pre-tokenization Setup (SKIPPED)"] = 0.0
            profiler.step_times["Step 2: Parallel Pre-tokenization Execution (SKIPPED)"] = 0.0
            profiler.step_times["Step 3: Vocabulary Initialization (SKIPPED)"] = 0.0
            profiler.step_times["Step 4: Initial Pair Counting (SKIPPED)"] = 0.0
            if enable_profiling:
                print("Pre-processing steps skipped (resumed from checkpoint)")
    
    # Only do pre-processing if not resuming from checkpoint
    if word_freqs is None:
        if profiler:
            # Already started in the condition above
            pass
        
        # Determine number of processes to use
        num_processes = min(multiprocessing.cpu_count(), 16)  # Cap at 16 processes

        # Find chunk boundaries using the provided helper function
        # Use the first special token as the split token (typically <|endoftext|>)
        split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
        split_token_bytes = split_token.encode('utf-8')

        with open(input_path, 'rb') as f:
            chunk_boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)
        
        if enable_profiling and profiler:
            print(f"Using {num_processes} processes with {len(chunk_boundaries)-1} chunks")
            print(f"Chunk boundaries: {chunk_boundaries}")
        
        if profiler:
            profiler.end_step("Step 1: Parallel Pre-tokenization Setup")
            profiler.start_step("Step 2: Parallel Pre-tokenization Execution")

        # Prepare arguments for parallel processing with progress tracking
        chunk_args = []
        total_chunks = len(chunk_boundaries) - 1
        
        with open(input_path, 'rb') as f:
            for chunk_id, (start, end) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:])):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
                
                # Include progress tracking information
                chunk_info = (chunk_id, total_chunks, enable_profiling)
                chunk_args.append((chunk_text, special_tokens, chunk_info))
        
        # Process chunks in parallel
        word_freqs = defaultdict(int)

        if len(chunk_args) > 1:
            # Use multiprocessing for multiple chunks
            if enable_profiling:
                print(f"\nStarting parallel pre-tokenization with {num_processes} processes...")
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                chunk_results = pool.map(_process_text_chunk_for_bpe, chunk_args)
            
            if enable_profiling:
                print("All chunks completed. Combining results...")
            
            # Combine results from all chunks
            for chunk_word_freqs in chunk_results:
                for word_tuple, freq in chunk_word_freqs.items():
                    word_freqs[word_tuple] += freq
        else:
            # Single chunk - process directly without multiprocessing overhead
            if enable_profiling:
                print("Processing single chunk...")
            chunk_word_freqs = _process_text_chunk_for_bpe(chunk_args[0])
            word_freqs.update(chunk_word_freqs)
        
        total_tokens = sum(word_freqs.values())

        if enable_profiling and profiler:
            print(f"Pre-tokenized into {len(word_freqs)} unique tokens, {total_tokens} total tokens")
        
        if profiler:
            profiler.end_step("Step 2: Parallel Pre-tokenization Execution")
            profiler.start_step("Step 3: Vocabulary Initialization")
        
        # Initialize vocabulary
        vocab: dict[int, bytes] = {}

        # Add special tokens first
        for i, special_token in enumerate(special_tokens):
            vocab[i] = special_token.encode('utf-8')
        
        # Add base byte vocabulary (256 bytes) starting after special tokens
        special_token_count = len(special_tokens)
        for i in range(256):
            vocab[special_token_count + i] = bytes([i])
        
        if profiler:
            profiler.end_step("Step 3: Vocabulary Initialization")
            profiler.start_step("Step 4: Initial Pair Counting")
        
        # Initialize data structures for efficient pair tracking
        pair_counts = defaultdict(int)

        # Build initial pair counts - O(total_chars) operation
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] += freq
        
        if enable_profiling and profiler:
            print(f"Found {len(pair_counts)} unique character pairs")
        
        if profiler:
            profiler.end_step("Step 4: Initial Pair Counting")
    
    # Calculate number of merges needed
    special_token_count = len(special_tokens)
    num_merges = vocab_size - special_token_count - 256
    if enable_profiling and profiler:
        remaining_merges = num_merges - start_merge_idx
        print(f"Will perform {remaining_merges} merge operations (starting from merge {start_merge_idx})")
    
    if num_merges <= start_merge_idx:
        if profiler:
            profiler.print_report()
        return vocab, merges_list
    
    # Helper function to convert GPT-2 unicode characters back to bytes
    def gpt2_char_to_bytes(gpt2_char):
        if len(gpt2_char) == 1 and gpt2_char in byte_decoder:
            # Single GPT-2 unicode character -> single byte
            return bytes([byte_decoder[gpt2_char]])
        else:
            # Merged character -> convert each component back to bytes
            result = b''
            for char in gpt2_char:
                if char in byte_decoder:
                    result += bytes([byte_decoder[char]])
                else:
                    # This shouldn't happen if our algorithm is correct
                    raise ValueError(f"Unknown GPT-2 character: {char}")
            return result
    
    if profiler:
        step_name = "Step 5: BPE Merging Loop (Resumed)" if start_merge_idx > 0 else "Step 5: BPE Merging Loop"
        profiler.start_step(step_name)
    

    # Perform merges using optimized incremental updates
    for merge_idx in range(start_merge_idx, num_merges):
        merge_start_time = time.perf_counter() if profiler else None
        
        # If no pairs found, break early
        if not pair_counts:
            break
        
        # Find the most frequent pair with deterministic tie-breaking
        max_freq = max(pair_counts.values())
        candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        
        # Among candidates with max frequency, choose lexicographically greatest
        # Convert to bytes for proper tie-breaking (matching reference implementation)
        def pair_to_bytes(pair):
            return (gpt2_char_to_bytes(pair[0]), gpt2_char_to_bytes(pair[1]))
        
        best_pair = max(candidates, key=pair_to_bytes)
        
        # Convert to bytes for the merge list
        char1_bytes = gpt2_char_to_bytes(best_pair[0])
        char2_bytes = gpt2_char_to_bytes(best_pair[1])
        merges_list.append((char1_bytes, char2_bytes))
        
        # Add the merged token to the vocabulary
        merged_token_bytes = char1_bytes + char2_bytes
        new_token_id = special_token_count + 256 + merge_idx
        vocab[new_token_id] = merged_token_bytes
        
        # Create new merged character (concatenate the GPT-2 unicode chars)
        new_char = best_pair[0] + best_pair[1]
        
        # INCREMENTAL UPDATE: Only update pair counts that are affected by this merge
        # Track changes to pair counts for incremental update
        pair_count_changes = defaultdict(int)
        
        # Apply merge to all words and track pair count changes
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            # Check if this word contains the pair to merge
            contains_merge_pair = False
            for i in range(len(word_tuple) - 1):
                if word_tuple[i] == best_pair[0] and word_tuple[i + 1] == best_pair[1]:
                    contains_merge_pair = True
                    break
            
            if not contains_merge_pair:
                # Word doesn't contain the merge pair, keep it unchanged
                new_word_freqs[word_tuple] = freq
            else:
                # Word contains the merge pair, apply merge and track changes
                
                # First, subtract old pair counts for this word
                for i in range(len(word_tuple) - 1):
                    old_pair = (word_tuple[i], word_tuple[i + 1])
                    pair_count_changes[old_pair] -= freq
                
                # Apply merge to create new word
                new_word_list = []
                i = 0
                while i < len(word_tuple):
                    # Check if we have the pair to merge at this position
                    if (i < len(word_tuple) - 1 and 
                        word_tuple[i] == best_pair[0] and 
                        word_tuple[i + 1] == best_pair[1]):
                        # Replace the pair with the merged character
                        new_word_list.append(new_char)
                        i += 2  # Skip both characters of the pair
                    else:
                        # Keep the original character
                        new_word_list.append(word_tuple[i])
                        i += 1
                
                new_word_tuple = tuple(new_word_list)
                new_word_freqs[new_word_tuple] = freq
                
                # Add new pair counts for the transformed word
                for i in range(len(new_word_tuple) - 1):
                    new_pair = (new_word_tuple[i], new_word_tuple[i + 1])
                    pair_count_changes[new_pair] += freq
        
        # Update word frequencies
        word_freqs = new_word_freqs
        
        # Apply incremental updates to pair_counts
        for pair, change in pair_count_changes.items():
            pair_counts[pair] += change
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
        
        # Record timing for this merge
        if profiler and merge_start_time is not None:
            merge_duration = time.perf_counter() - merge_start_time
            profiler.record_merge_time(merge_idx, merge_duration)
        
        # Checkpoint every checkpoint_interval merges
        if checkpoint_dir and (merge_idx + 1) % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{merge_idx + 1}.pkl")
            save_checkpoint(checkpoint_path, merge_idx + 1, vocab, merges_list, 
                          word_freqs, pair_counts, training_params)
        
        # Progress reporting
        if enable_profiling and ((merge_idx + 1) % 100 == 0 or merge_idx < 10):
            merge_duration = time.perf_counter() - merge_start_time if merge_start_time else 0
            print(f"Completed merge {merge_idx + 1}/{num_merges} "
                  f"(merged '{char1_bytes.decode('utf-8', errors='replace')}' + "
                  f"'{char2_bytes.decode('utf-8', errors='replace')}', "
                  f"freq={max_freq}, time={merge_duration:.6f}s)")
        
    
    # Save final checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pkl")
        save_checkpoint(final_checkpoint_path, len(merges_list), vocab, merges_list, 
                      word_freqs, pair_counts, training_params)
    
    if profiler:
        profiler.end_step("Step 5: BPE Merging Loop")
    
    # Print profiling report if enabled
    if profiler:
        print(f"\nBPE training completed successfully!")
        print(f"Final vocabulary size: {len(vocab)}")
        print(f"Number of merges: {len(merges_list)}")
        profiler.print_report()
    
    return vocab, merges_list
