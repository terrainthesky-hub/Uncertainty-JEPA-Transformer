# train_uncertainty_jepa_llm.py
import os
import math
import time
import queue
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

# Environment optimizations for RTX 5080 / Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# =====================================================================
# 1. ROTARY POSITIONAL EMBEDDINGS (RoPE)
# =====================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# =====================================================================
# 2. CORE MODULES (RMSNorm, SwiGLU, GQA with QK-Norm)
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttentionQKNorm(nn.Module):
    def __init__(self, d_model=896, n_heads=14, n_kv_heads=7, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.head_dim)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, intermediate_dim):
        super().__init__()
        self.attn = GroupedQueryAttentionQKNorm(d_model, n_heads, n_kv_heads, head_dim)
        self.mlp = SwiGLU(d_model, intermediate_dim)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# =====================================================================
# 3. FULL ARCHITECTURE WITH INITIALIZATION
# =====================================================================
class UncertaintyJEPALM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=896, n_layers=16, n_heads=14, 
                 n_kv_heads=7, head_dim=64, max_seq_len=1024, logit_cap=30.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.logit_cap = logit_cap
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

        intermediate_dim = int(2 * (4 * d_model) / 3)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, head_dim, intermediate_dim)
            for _ in range(n_layers)
        ])

        self.jepa_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(x, L)

        for block in self.blocks:
            x = block(x, cos, sin)

        x_norm = self.norm_final(x)
        logits = self.lm_head(x_norm)

        if self.logit_cap > 0.0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)

        s_pred_future = self.jepa_predictor(x_norm[:, :-1, :])
        s_target_future = x_norm[:, 1:, :].detach()

        return logits, s_pred_future, s_target_future

    @torch.no_grad()
    def generate(self, tokenizer, prompt="The fundamental law of physics states that", max_new_tokens=40, temperature=0.7):
        self.eval()
        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        self.train()
        return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

# =====================================================================
# 4. LEARNED MULTI-TASK UNCERTAINTY LOSS (Kendall et al.)
# =====================================================================
class MultiTaskUncertaintyLoss(nn.Module):
    """
    Learns task log-variances automatically via gradient descent.
    """
    def __init__(self, init_s_lm=0.0, init_s_jepa=1.8):
        super().__init__()
        # s = log(sigma^2). Initialized so exp(s_lm - s_jepa) ≈ 0.165
        self.s_lm = nn.Parameter(torch.tensor(init_s_lm, dtype=torch.float32))
        self.s_jepa = nn.Parameter(torch.tensor(init_s_jepa, dtype=torch.float32))

    def forward(self, lm_loss, jepa_loss):
        precision_lm = torch.exp(-self.s_lm)
        precision_jepa = torch.exp(-self.s_jepa)
        
        weighted_lm = 0.5 * precision_lm * lm_loss + 0.5 * self.s_lm
        weighted_jepa = 0.5 * precision_jepa * jepa_loss + 0.5 * self.s_jepa
        
        return weighted_lm + weighted_jepa

    @property
    def effective_jepa_multiplier(self):
        """Calculates effective relative weight ratio (W_jepa / W_lm)."""
        return torch.exp(self.s_lm - self.s_jepa).item()

    @property
    def uncertainties(self):
        """Returns (sigma_lm, sigma_jepa)"""
        sigma_lm = torch.exp(0.5 * self.s_lm).item()
        sigma_jepa = torch.exp(0.5 * self.s_jepa).item()
        return sigma_lm, sigma_jepa

# =====================================================================
# MULTI-STREAM ASYNC PREFETCHER (FineWeb, PG19, OpenWebText, WikiText)
# =====================================================================
import time
import queue
import random
import threading
import torch
from datasets import load_dataset

class MultiStreamDataPrefetcher:
    """
    Asynchronously streams, mixes, tokenizes, and buffers batches from 
    multiple Hugging Face datasets with automatic failover and stream recycling.
    
    Default Dataset Mixture:
      • 50% FineWeb-Edu   (High-quality educational reasoning)
      • 20% PG19          (Classic literature, long-range narrative rhythm)
      • 20% OpenWebText   (Modern conversational prose & articles)
      • 10% WikiText-103  (Encyclopedic facts & definitions)
    """
    def __init__(
        self, 
        tokenizer, 
        max_seq_len=1024, 
        batch_size=6, 
        queue_size=25,
        dataset_weights=None
    ):
        self.tokenizer = tokenizer
        self.chunk_size = max_seq_len + 1  # Input tokens + shifted target token
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

        # Configurable sampling probability weights
        self.weights = dataset_weights or {
            "fineweb": 0.50,
            "pg19": 0.20,
            "openwebtext": 0.20,
            "wikitext": 0.10
        }
        
        self.dataset_keys = list(self.weights.keys())
        self.sampling_probs = list(self.weights.values())

        # Start dedicated background worker thread
        self.worker = threading.Thread(target=self._producer_loop, daemon=True)
        self.worker.start()

    def _init_stream(self, key):
        """Initializes or resets a specific streaming dataset iterator with retries."""
        for attempt in range(5):
            try:
                if key == "fineweb":
                    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                elif key == "pg19":
                    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
                elif key == "openwebtext":
                    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
                elif key == "wikitext":
                    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
                else:
                    raise ValueError(f"Unknown dataset key: {key}")
                return iter(ds)
            except Exception as e:
                time.sleep(2 ** attempt)
        return None

    def _producer_loop(self):
        """Background thread that samples, tokenizes, and feeds the queue."""
        print("[+] Connecting to Multi-Stream Datasets (FineWeb, PG19, OpenWebText, WikiText)...")
        
        # Instantiate active iterators for all 4 datasets
        iterators = {key: self._init_stream(key) for key in self.dataset_keys}
        
        buffer = []
        batch_accum = []

        while not self.stop_event.is_set():
            # 1. Probabilistically select which dataset stream to pull from
            chosen_key = random.choices(self.dataset_keys, weights=self.sampling_probs, k=1)[0]
            curr_iter = iterators[chosen_key]

            # 2. Fetch sample with automatic stream recycling on exhaustion/socket glitch
            try:
                sample = next(curr_iter)
            except (StopIteration, Exception):
                # Reset stream seamlessly if interrupted or end of split reached
                iterators[chosen_key] = self._init_stream(chosen_key)
                continue

            # 3. Extract text content
            text = sample.get("text", "")
            if not text or len(text.strip()) < 80:
                continue

            # 4. Tokenize & append EOS token
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            # 5. Pack into exact chunk sizes (Continuous Token Packing)
            while len(buffer) >= self.chunk_size:
                batch_accum.append(buffer[:self.chunk_size])
                buffer = buffer[self.chunk_size:]

                # 6. Once a micro-batch is assembled, push to GPU prefetch queue
                if len(batch_accum) == self.batch_size:
                    tensor_batch = torch.tensor(batch_accum, dtype=torch.long)
                    self.queue.put(tensor_batch, block=True)
                    batch_accum = []

    def next_batch(self, device):
        """Fetches the next pre-assembled batch and ships it non-blocking to CUDA."""
        batch = self.queue.get(block=True)
        return batch.to(device, non_blocking=True)

    def close(self):
        """Clean shutdown trigger."""
        self.stop_event.set()

# =====================================================================
# 6. ENGINE
# =====================================================================
def get_lr(step, total_steps, max_lr=5.5e-4, min_lr=1.0e-5, warmup_steps=1000):
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    MAX_SEQ_LEN = 1024
    BATCH_SIZE = 6          # Fits comfortably at ~13.6 GB on RTX 5080
    GRAD_ACCUM_STEPS = 6    # Effective batch = 36 sequences (36,864 tokens/step)
    D_MODEL = 896
    N_HEADS = 14
    N_KV_HEADS = 7
    N_LAYERS = 16
    TOTAL_STEPS = 80_000
    CKPT_PATH = "uncertainty_jepa_5080_ckpt.pt"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = UncertaintyJEPALM(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=64,
        max_seq_len=MAX_SEQ_LEN,
        logit_cap=30.0
    ).to(device)

    # Uncertainty Loss module
    criterion = MultiTaskUncertaintyLoss(init_s_lm=0.0, init_s_jepa=1.8).to(device)

    # Optimizer with decoupled weight decay (No decay on uncertainty parameters)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': 0.05},
        {'params': criterion.parameters(), 'weight_decay': 0.0}
    ], lr=5.5e-4, betas=(0.9, 0.95))

    start_step = 0
    if os.path.exists(CKPT_PATH):
        checkpoint_data = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint_data['model'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        if 'criterion' in checkpoint_data:
            criterion.load_state_dict(checkpoint_data['criterion'])
        start_step = checkpoint_data['step']
        print(f"[+] Resuming from Checkpoint at Step {start_step:,}/{TOTAL_STEPS:,}...")
    else:
        print(f"[+] Initialized fresh weights. Starting from Step 0...")


    streamer = MultiStreamDataPrefetcher(
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        queue_size=25,
        dataset_weights={
            "fineweb": 0.50,      # 50% Educational knowledge
            "pg19": 0.20,         # 20% Literature & narrative cadence
            "openwebtext": 0.20,  # 20% General Web prose
            "wikitext": 0.10      # 10% Encyclopedic facts
        }
    )
    print(f"\n[+] Uncertainty-Weighted Training Commenced...\n")

    start_time = time.time()
    interval_lm_loss = 0.0
    interval_jepa_loss = 0.0
    interval_steps = 0

    for step in range(start_step, TOTAL_STEPS):
        lr = get_lr(step, TOTAL_STEPS, max_lr=5.5e-4, min_lr=1.0e-5, warmup_steps=1000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        step_lm = 0.0
        step_jepa = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            batch = streamer.next_batch(device)
            inputs = batch[:, :MAX_SEQ_LEN]
            targets = batch[:, 1:MAX_SEQ_LEN + 1]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, s_pred_future, s_target_future = model(inputs)
                
                raw_lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                jepa_l1 = F.smooth_l1_loss(s_pred_future, s_target_future)
                jepa_cos = 1.0 - F.cosine_similarity(s_pred_future, s_target_future, dim=-1).mean()
                raw_jepa_loss = jepa_l1 + 0.5 * jepa_cos

                # Homoscedastic uncertainty optimization
                total_loss = criterion(raw_lm_loss, raw_jepa_loss) / GRAD_ACCUM_STEPS

            total_loss.backward()
            step_lm += raw_lm_loss.item()
            step_jepa += raw_jepa_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        interval_lm_loss += (step_lm / GRAD_ACCUM_STEPS)
        interval_jepa_loss += (step_jepa / GRAD_ACCUM_STEPS)
        interval_steps += 1

        if (step + 1) % 25 == 0:
            elapsed = time.time() - start_time
            tokens_processed = (step + 1 - start_step) * BATCH_SIZE * GRAD_ACCUM_STEPS * MAX_SEQ_LEN
            tok_sec = tokens_processed / max(1e-5, elapsed)
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            avg_lm = interval_lm_loss / interval_steps
            avg_jepa = interval_jepa_loss / interval_steps
            eff_jepa_w = criterion.effective_jepa_multiplier
            sigma_lm, sigma_jepa = criterion.uncertainties
            
            print(f"Step [{step+1:5d}/{TOTAL_STEPS}] | LM: {avg_lm:.4f} | JEPA: {avg_jepa:.4f} | "
                  f"Eff-W: {eff_jepa_w:.3f} (σ_lm:{sigma_lm:.2f}, σ_jepa:{sigma_jepa:.2f}) | "
                  f"VRAM: {vram_used:.2f}/16.0 GB | Speed: {tok_sec:.0f} tok/s")
            
            interval_lm_loss = 0.0
            interval_jepa_loss = 0.0
            interval_steps = 0

        if (step + 1) % 500 == 0:
            print("\n" + "="*60)
            sample = model.generate(tokenizer, prompt="The fundamental law of physics states that", max_new_tokens=50)
            print(f"[Generation @ Step {step+1}]:\n\"{sample}\"")
            print("="*60 + "\n")

        if (step + 1) % 2000 == 0:
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, CKPT_PATH)

    torch.save(model.state_dict(), "uncertainty_jepa_5080_final.pt")
    print("\n[+] Training Complete! Final model saved as 'uncertainty_jepa_5080_final.pt'.")

if __name__ == "__main__":
    main()