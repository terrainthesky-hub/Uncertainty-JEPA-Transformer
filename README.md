# Technical Architecture Report: Uncertainty-Weighted Causal JEPA-LLM

**Model Scale:** 189.56M Parameters  
**Compute Target:** Single NVIDIA RTX 5080 (16GB GDDR7)  
**Primary Paradigm:** Hybrid Autoregressive-Self-Supervised Causal Language Modeling with Dual-Objective World-State Regularization  

---

## 1. Executive Summary

The **Uncertainty-Weighted Causal JEPA-LLM** is a dense, decoder-only transformer architecture engineered to address sample-inefficiency and surface-level memorization in sub-billion-parameter language models. 

Standard autoregressive language models (LLMs) optimize solely for discrete next-token classification via cross-entropy. In small-parameter regimes (<= 300M parameters), this often leads to shallow n-gram memorization, brittle out-of-distribution reasoning, and attractor loops (repetitive tautologies).

This architecture pairs a modern **Rotary Grouped-Query Attention (RoPE-GQA)** transformer backbone with a **Joint-Embedding Predictive Architecture (JEPA)** auxiliary head. The network simultaneously optimizes:
1. **Discrete Token Prediction:** Negative Log-Likelihood over the discrete vocabulary manifold.
2. **Continuous World-Model Prediction:** Latent trajectory forecasting in continuous embedding space.
3. **Homoscedastic Uncertainty Auto-Balancing:** A learned Bayesian weighting mechanism that dynamically balances gradient allocation between discrete language prediction and continuous representation forecasting.

---

## 2. Core Transformer Backbone Architecture

### 2.1. Grouped-Query Attention with QK-Normalization (GQA-QKNorm)
To maintain high throughput while preserving multi-head representational capacity, the model uses a **2:1 Grouped-Query Attention** configuration (H_Q = 14, H_KV = 7) with head dimension d_k = 64.

To prevent attention entropy collapse:
- q_norm = RMSNorm(W_q x)
- k_norm = RMSNorm(W_k x)
- q_rot, k_rot = RoPE(q_norm), RoPE(k_norm)

### 2.2. Rotary Positional Embeddings (RoPE)
Relative token distances are encoded via complex rotational coordinate transformations across the 64-dimensional head subspace.

### 2.3. SwiGLU Gated Feed-Forward Layers
The standard MLP block is replaced with a Swish-Gated Linear Unit (SwiGLU) with intermediate dimension scaled to 2,389 (floor(8/3 * d_model)).

### 2.4. Logit Soft-Capping
Output logits are constrained via a hyperbolic tangent soft-cap (cap = 30.0):
- Logits_capped = 30.0 * tanh(Logits / 30.0)

---

## 3. The Continuous JEPA Latent World-Model Head

The JEPA Latent Predictor g_phi is parameterized as a 2-layer non-linear MLP with SiLU and RMSNorm.
- Forecast: s_pred = g_phi(s_t)
- Target: s_target = stop_gradient(s_{t+1})
- Loss: Smooth L1 + 0.5 * (1.0 - CosineSimilarity(s_pred, s_target))

---

## 4. Multi-Task Homoscedastic Uncertainty Loss

Parameterizing task variances as learnable parameters s_LM = ln(sigma_LM^2) and s_JEPA = ln(sigma_JEPA^2):
- L_total = 0.5 * exp(-s_LM) * L_LM + 0.5 * exp(-s_JEPA) * L_JEPA + 0.5 * s_LM + 0.5 * s_JEPA
- Effective Ratio: W_eff = exp(s_LM - s_JEPA) = 4.1723x (Learned equilibrium)

---

## 5. Inference Engine: JEPA-Guided Latent Trajectory Steering

During generation, top candidate token embeddings are evaluated against the continuous JEPA forecast:
- Alignment Score: A_k = CosineSimilarity(tok_emb(w_k), s_pred)
- Guided Logits: Logits_k = Logits_k + (0.45 * A_k * |Logits_k|)

---

## 6. Structural Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **189,563,392** | Full network footprint (723 MB FP32 / 379 MB BF16) |
| **Backbone Parameters** | **141,310,464** | Core transformer layers |
| **Embedding Parameters** | **45,030,272** | GPT-2 Tokenizer (V = 50,257, d = 896), Weight-Tied |
| **JEPA Head Parameters** | **3,222,656** | 2-Layer Predictive World-State MLP |
| **Number of Layers (L)** | **16** | Identical Transformer Blocks |
| **Model Dimension (d_model)** | **896** | Embedding / Hidden dimension |
| **Attention Heads (H_Q / H_KV)**| **14 / 7** | 2:1 Grouped-Query Attention |
| **Head Dimension (d_k)** | **64** | Dimension per attention head |
| **FFN Hidden Dimension** | **2,389** | SwiGLU Intermediate Dimension |
| **Maximum Context Length** | **1,024** | Sequence window |

---

## 7. Empirical Validation Profile

- **Training Tokens Trained**: 2,949,120,000 (2.95 Billion)
- **Validation Cross-Entropy**: 3.9498 nats
- **Empirical Perplexity**: 51.92
- **JEPA Cosine Alignment**: 76.36%
- **Latent Channel Variance**: 7.6617 (Healthy, zero collapse)
- **Learned Task Variances**: sigma_LM = 1.774 | sigma_JEPA = 0.868 (Eff-W = 4.17x)
