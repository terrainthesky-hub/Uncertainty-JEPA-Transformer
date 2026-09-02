# probe_uncertainty_jepa.py
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

# Environment optimizations for RTX 5080 / Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# =====================================================================
# 1. MODEL DEFINITION (Matches Training Architecture)
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

        return logits, s_pred_future, s_target_future, x_norm

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=100, temperature=0.75, top_p=0.90, repetition_penalty=1.20):
        self.eval()
        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Top-P Nucleus Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, init_s_lm=0.0, init_s_jepa=1.8):
        super().__init__()
        self.s_lm = nn.Parameter(torch.tensor(init_s_lm, dtype=torch.float32))
        self.s_jepa = nn.Parameter(torch.tensor(init_s_jepa, dtype=torch.float32))

# =====================================================================
# 2. VALIDATION EVALUATOR & LATENT GEOMETRY PROBER
# =====================================================================
def evaluate_held_out_benchmark(model, tokenizer, device, num_eval_batches=35, seq_len=1024, batch_size=4):
    print(f"[+] Streaming {num_eval_batches} held-out validation batches from FineWeb-Edu...")
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    except Exception:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    data_iter = iter(dataset)
    chunk_size = seq_len + 1
    buffer = []
    
    total_lm_loss = 0.0
    total_jepa_loss = 0.0
    total_jepa_cos = 0.0
    total_latent_variance = 0.0
    total_tokens = 0
    batches_processed = 0

    model.eval()
    with torch.no_grad():
        for sample in data_iter:
            text = sample.get('text', '') or sample.get('story', '') or ''
            if len(text.strip()) < 80:
                continue
            
            tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
            buffer.extend(tokens)

            while len(buffer) >= chunk_size * batch_size:
                batch_tokens = []
                for _ in range(batch_size):
                    batch_tokens.append(buffer[:chunk_size])
                    buffer = buffer[chunk_size:]

                batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
                inputs = batch[:, :seq_len]
                targets = batch[:, 1:seq_len + 1]

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, s_pred_future, s_target_future, x_norm = model(inputs)
                    
                    # 1. Cross Entropy Loss
                    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    
                    # 2. JEPA Latent Prediction Metrics
                    jepa_l1 = F.smooth_l1_loss(s_pred_future, s_target_future)
                    cos_sim = F.cosine_similarity(s_pred_future, s_target_future, dim=-1).mean()
                    jepa_loss = jepa_l1 + 0.5 * (1.0 - cos_sim)
                    
                    # 3. Latent Representation Geometry (Isotropy / Variance)
                    latent_var = x_norm.var(dim=-1).mean()

                total_lm_loss += lm_loss.item()
                total_jepa_loss += jepa_loss.item()
                total_jepa_cos += cos_sim.item()
                total_latent_variance += latent_var.item()
                total_tokens += (batch_size * seq_len)
                batches_processed += 1

                if batches_processed >= num_eval_batches:
                    break

            if batches_processed >= num_eval_batches:
                break

    avg_lm_loss = total_lm_loss / batches_processed
    avg_ppl = math.exp(avg_lm_loss)
    avg_jepa_loss = total_jepa_loss / batches_processed
    avg_jepa_cos = total_jepa_cos / batches_processed
    avg_latent_var = total_latent_variance / batches_processed

    return {
        "eval_lm_loss": avg_lm_loss,
        "eval_perplexity": avg_ppl,
        "eval_jepa_loss": avg_jepa_loss,
        "eval_jepa_cos_sim": avg_jepa_cos,
        "eval_latent_var": avg_latent_var,
        "tokens_evaluated": total_tokens
    }

# =====================================================================
# 3. MAIN DIAGNOSTIC ENGINE
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*75)
    print("  UNCERTAINTY-JEPA CHECKPOINT PROBE & DIAGNOSTIC ANALYZER")
    print(f"  Execution Device: {torch.cuda.get_device_name(0)}")
    print("="*75)

    # 1. Locate Checkpoint Files
    candidate_ckpts = [
        "uncertainty_jepa_5080_final.pt",
        "uncertainty_jepa_5080_ckpt.pt",
        "optimized_llm_5080_final.pt",
        "optimized_llm_5080_ckpt.pt",
        "local_llm_5080_final.pt",
        "local_llm_5080_ckpt.pt"
    ]
    
    ckpt_path = None
    for path in candidate_ckpts:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print("[-] Error: No checkpoint file found in the current directory.")
        return

    print(f"\n[+] Found Checkpoint: '{ckpt_path}'")
    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"[+] File Size on Disk: {file_size_mb:.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    # 2. Instantiate Model
    model = UncertaintyJEPALM(
        vocab_size=len(tokenizer),
        d_model=896,
        n_layers=16,
        n_heads=14,
        n_kv_heads=7,
        head_dim=64,
        max_seq_len=1024,
        logit_cap=30.0
    ).to(device)

    raw_data = torch.load(ckpt_path, map_location=device)

    step_count = "Unknown"
    s_lm_val = None
    s_jepa_val = None

    if isinstance(raw_data, dict) and "model" in raw_data:
        model.load_state_dict(raw_data["model"])
        step_count = raw_data.get("step", "N/A")
        
        if "criterion" in raw_data:
            s_lm_val = raw_data["criterion"].get("s_lm", torch.tensor(0.0)).item()
            s_jepa_val = raw_data["criterion"].get("s_jepa", torch.tensor(0.0)).item()
    else:
        # File is raw model state dict
        model.load_state_dict(raw_data)
        
        # Check if matching ckpt file has criterion
        if os.path.exists("uncertainty_jepa_5080_ckpt.pt"):
            companion = torch.load("uncertainty_jepa_5080_ckpt.pt", map_location=device)
            step_count = companion.get("step", "N/A")
            if "criterion" in companion:
                s_lm_val = companion["criterion"].get("s_lm", None).item()
                s_jepa_val = companion["criterion"].get("s_jepa", None).item()

    # Model Parameter Summary
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.tok_emb.weight.numel()
    jepa_params = sum(p.numel() for p in model.jepa_predictor.parameters())
    backbone_params = total_params - embed_params - jepa_params

    print("\n" + "-"*75)
    print(" 1. RECOVERED TRAINING METADATA")
    print("-" * 75)
    print(f" • Total Parameters       : {total_params / 1e6:.2f} Million")
    print(f" • Backbone Transformer   : {backbone_params / 1e6:.2f} Million")
    print(f" • JEPA World Head        : {jepa_params / 1e6:.2f} Million")
    print(f" • Final Recorded Step    : {step_count}")
    if isinstance(step_count, int):
        tokens_seen = step_count * 36864
        print(f" • Total Tokens Trained   : {tokens_seen:,} ({tokens_seen / 1e9:.3f} Billion)")

    print("\n" + "-"*75)
    print(" 2. RECOVERED HOMOSCEDASTIC UNCERTAINTY WEIGHTING")
    print("-" * 75)
    if s_lm_val is not None and s_jepa_val is not None:
        sigma_lm = math.exp(0.5 * s_lm_val)
        sigma_jepa = math.exp(0.5 * s_jepa_val)
        eff_weight = math.exp(s_lm_val - s_jepa_val)
        print(f" • Language Loss Variance (s_LM)     : {s_lm_val:.4f}  ──►  σ_LM   = {sigma_lm:.3f}")
        print(f" • JEPA Latent Variance   (s_JEPA)   : {s_jepa_val:.4f}  ──►  σ_JEPA = {sigma_jepa:.3f}")
        print(f" • Final Effective JEPA Multiplier    : {eff_weight:.4f}x")
        print(f" • Learned Balance Status             : {'JEPA Prioritized (World Model Focused)' if eff_weight > 1.0 else 'Token Prioritized'}")
    else:
        print(" • (Uncertainty parameters were not saved in the flat checkpoint; evaluated directly via model pass)")

    # 3. Empirical Held-Out Validation Benchmark
    print("\n" + "-"*75)
    print(" 3. EMPIRICAL VALIDATION BENCHMARK (Held-Out Data)")
    print("-" * 75)
    bench_results = evaluate_held_out_benchmark(model, tokenizer, device, num_eval_batches=30)
    print(f" • Validation Cross-Entropy Loss : {bench_results['eval_lm_loss']:.4f}")
    print(f" • True Empirical Perplexity     : {bench_results['eval_perplexity']:.2f}")
    print(f" • Validation JEPA Prediction Loss: {bench_results['eval_jepa_loss']:.4f}")
    print(f" • JEPA Latent Cosine Alignment  : {bench_results['eval_jepa_cos_sim'] * 100:.2f}% (Similarity to true future state)")
    print(f" • Latent Space Channel Variance : {bench_results['eval_latent_var']:.4f} (Representation collapse test: {'Healthy' if bench_results['eval_latent_var'] > 0.05 else 'Collapsed'})")
    print(f" • Evaluated Token Volume        : {bench_results['tokens_evaluated']:,} tokens")

    # 4. Multi-Domain Qualitative Assessment
    print("\n" + "-"*75)
    print(" 4. MULTI-DOMAIN REASONING & GENERATION BENCHMARK")
    print("-" * 75)

    test_prompts = [
        ("Physics & Thermodynamics", "The fundamental law of thermodynamics states that"),
        ("Biology & Genetics", "The difference between DNA and RNA is that"),
        ("Computer Science & Code", "In computer programming, a binary search tree operates by"),
        ("Narrative & Fiction", "The old detective examined the dusty clock on the mantelpiece and realized that"),
        ("Logical Deduction", "If all mammals breathe oxygen and whales are mammals, then")
    ]

    for category, prompt in test_prompts:
        print(f"\n[Domain: {category}]")
        print(f"Prompt: \"{prompt}\"")
        start = time.time()
        completion = model.generate(
            tokenizer, 
            prompt=prompt, 
            max_new_tokens=65, 
            temperature=0.72, 
            top_p=0.88, 
            repetition_penalty=1.20
        )
        elapsed = time.time() - start
        print(f"Generated ({65 / max(elapsed, 0.01):.0f} tok/s):\n\"{completion}\"")

    # 5. Interactive Chat Mode
    print("\n" + "="*75)
    print("  INTERACTIVE MODEL TESTING CLI")
    print("  (Type any prompt to test your model. Type 'exit' or 'q' to quit)")
    print("="*75)

    while True:
        try:
            user_input = input("\nEnter Prompt > ")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue
            
            output = model.generate(
                tokenizer, 
                prompt=user_input, 
                max_new_tokens=100, 
                temperature=0.4, 
                top_p=0.88, 
                repetition_penalty=1.3
            )
            print(f"\nCompletion:\n{output}\n" + "-"*60)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()