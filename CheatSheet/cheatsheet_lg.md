# Modern NLP

---

## 1. Word Embeddings

**Key idea.** Replace atomic one-hot $\mathbf{e}_w \in \{0,1\}^{|V|}$ (always orthogonal, $|V|\approx 10^5$) with dense $\mathbf{e}_w \in \mathbb{R}^d$, $d \ll |V|$, so that _geometry encodes meaning_. Foundation: **distributional hypothesis** — words appearing in similar contexts have similar meanings.

---

### Word2Vec

- **CBOW** — predict center $w_t$ from averaged context $\{w_{t-k},\ldots,w_{t+k}\}$.
- **Skip-gram** — predict each context word from center $w_t$; generates $k$ training pairs per center ⇒ more updates for rare words than CBOW.
- **Negative sampling** — replace the $|V|$-softmax with binary classification over $k$ noise samples ⇒ $\mathcal{O}(k)$ instead of $\mathcal{O}(|V|)$.

---

### GloVe

Combines global co-occurrence statistics $X_{ij}$ (LSA-style) with dense prediction (Word2Vec-style):

$$J = \sum_{i,j} f(X_{ij}) \left( w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij} \right)^2$$

$$f(x) = (x/x_{\max})^\alpha \ \text{if}\ x < x_{\max},\ \text{else}\ 1$$

GloVe _explicitly_ factorizes the log co-occurrence matrix; Word2Vec _implicitly_ factorizes the shifted PMI matrix — equivalent in theory, different inductive biases in practice.

---

### Issues left open

- Both W2V and GloVe are **static**: one vector per word-type ⇒ _polysemy_
- CBOW averages context embeddings ⇒ destroys word order and individual identity (bag-of-words over window).

---

_If two words never co-occur but appear in identical contexts, can one-hot capture their similarity?_ No — they remain orthogonal regardless of any evidence. _Can smoothing fix this?_ Also no — smoothing redistributes probability mass; it cannot create similarity structure over atomic symbols.

---

## 2. Fixed-Context LMs: n-grams

**Key idea.** Markov assumption truncates history to the last $n-1$ words:

$$P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-n+1:t-1})$$

**MLE estimate:** $\hat P(w_t \mid \text{ctx}) = C(\text{ctx}, w_t) / C(\text{ctx})$.

**Perplexity:** $PP(W) = P(W)^{-1/N}$ = exponentiated average NLL.

---

### Smoothing

- **Laplace / add-$\alpha$** — pretend every event was seen $\alpha$ extra times.
- **Kneser–Ney** — uses _continuation probability_: how many distinct contexts a word appears in, not raw frequency.

---

### Issues

- **Sparsity** of high-order n-grams: most combinations unseen.
- **No cross-word generalization**: cat and dog are atoms even if synonyms.
- **Hard context cap** at $n-1$ tokens.

**Next step →** Neural LMs share _embedding parameters_ across contexts, enabling generalization across similar words.

---

_Why can't even perfect smoothing generalize across synonyms?_ Because there is no **distributed representation**: each word is an atomic symbol, and smoothing only redistributes mass — it never creates similarity structure between different symbols.

---

## 3. Neural LMs: RNN & LSTM

### RNN

Same $W_h$, $W_x$ reused at every timestep:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
P(w_{t+1}) = softmax(W_out · h_t)

x_t:  (B, d)    # token embedding
h_t:  (B, h)    # hidden state
```

**Key property.** The hidden state $h_t$ theoretically encodes _all_ previous tokens ⇒ unlimited context vs n-gram's $n-1$. Parameter sharing across time = translation invariance.

---

### Vanishing gradient

$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} \to 0 \quad \text{when } \|W_h\|_2 < 1$$

Early context is forgotten. Also: strictly sequential computation, no parallelism.

---

### LSTM — gated additive cell

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    [forget gate]
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    [input gate]
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
c_t  = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t      [cell — ADDITIVE!]
o_t  = σ(W_o · [h_{t-1}, x_t] + b_o)   [output gate]
h_t  = o_t ⊙ tanh(c_t)
```

**Why it works.** Cell state uses **addition**, not multiplication ⇒ gradient highway:

$$\frac{\partial c_T}{\partial c_t} \approx \prod_\tau f_\tau \approx \text{const when } f \approx 1$$

Solves the vanishing-gradient problem of the vanilla RNN. Still sequential ⇒ no parallelism, and still has the encoder-decoder bottleneck in seq2seq.

---

_If the forget gate always outputs 1, what does LSTM reduce to?_ An unbounded accumulator that never forgets — the cell state grows without limit. The forget gate is what enables **selective, bounded** memory.

---

## 4. Seq2Seq

**Key idea.** Encoder–decoder for variable-length → variable-length mapping (machine translation, summarization). Encoder RNN compresses the source into a _single_ context vector $c$; decoder RNN generates the target autoregressively conditioned on $c$.

```
Encoder: RNN over source x_1..x_n
         c = h_n                    # (B, h) — FIXED SIZE
Decoder: P(y_t | y_1..y_{t-1}, c)
         → autoregressively generates y_1..y_m
```

**Training — teacher forcing:** feed the _gold_ $y_{t-1}$, not the model's prediction, so each step is an independent classification on a clean prefix.

---

### Issue — the bottleneck

The entire source sequence must fit into one fixed-size vector $c = h_n$. Capacity is bounded ⇒ long sequences suffer catastrophic forgetting _inside_ the encoder.

**Next step →** Attention: compute a separate, _dynamic_ context per decoder step, pulling from _all_ encoder hidden states rather than just the last.

---

_Doesn't making $h$ much larger fix the bottleneck?_ No — it is **information-theoretic**. A fixed-size vector has bounded capacity regardless of its dimensionality; the problem is forced compression, not vector width.

---

## 5. Attention

**Key idea.** At every decoder step $t$, build a _fresh_ context as a weighted sum of _all_ encoder states. Weights come from a learned compatibility score between the current decoder state and each encoder position.

```
e_{t,s} = vᵀ tanh(W_h · h^d_{t-1} + W_s · h^e_s)   [additive score]
α_{t,s} = softmax_s(e_{t,s})                       [attention weights]
c_t     = Σ_s α_{t,s} · h^e_s                      [dynamic context]

e_t:  (B, n)     # energy over n source positions
α_t:  (B, n)     # weights, Σ_s α_{t,s} = 1
c_t:  (B, h)     # weighted sum of encoder states
```

---

### What it gives

- **No bottleneck** — capacity scales with source length.
- **Interpretable soft alignment** — $\alpha_{t,s}$ shows how much decoder step $t$ attends to source position $s$ (diagonal for monotonic MT).
- **Direct gradient path** — decoder step $t$ connects to every encoder position, mitigating long-range forgetting.

---

### Still limited

Both encoder and decoder remain RNN-based ⇒ $O(n)$ strictly sequential steps. Next leap: drop recurrence entirely, keep only attention → **Transformer**.

---

_Cross-attention is $O(n \cdot m)$. What is the complexity of decoder self-attention at inference?_ $O(n^2)$ as the sequence grows — which motivates **KV caching**: store past $K$ and $V$, only compute the new query.

---

## 6. Transformer

**Key idea.** Drop recurrence entirely. Every token attends to every other token _in parallel_. Sequential depth $O(1)$, compute $O(n^2)$.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

- $Q$ (query) — "what I'm looking for"
- $K$ (key) — "what I advertise"
- $V$ (value) — "what I actually give"

**Why 3 projections if $Q$, $K$, $V$ all start from the same $x$?** Without them the model can't learn the three distinct roles — and a token's dot product with itself always dominates, so every token would mostly attend to itself.

**Why $\div \sqrt{d_k}$?** In high dimensions dot products grow large ⇒ softmax saturates to near one-hot ⇒ gradients vanish. Scaling keeps magnitudes stable.

---

### Shape flow — single head

```
input x       [B, S, H]
query_proj    [B, S, H]
key_proj      [B, S, H]
value_proj    [B, S, H]
attn_scores   [B, S, S]   Q · Kᵀ
attn_weights  [B, S, S]   softmax
output        [B, S, H]   weights · V
```

B = batch, S = seq_len, H = hidden_dim

---

### Shape flow — multi-head

```
[B, S, H]
  → .view(B, -1, nH, hD)    [B, S, nH, hD]
  → .transpose(1, 2)         [B, nH, S, hD]

Q · Kᵀ                       [B, nH, S, S]
softmax(dim=-1)              [B, nH, S, S]
weights · V                  [B, nH, S, hD]

  → .transpose(1, 2)         [B, S, nH, hD]
  → .contiguous()
  → .view(B, -1, H)          [B, S, H]
out_proj                     [B, S, H]
```

nH = num_heads, hD = head_dim = H / nH

---

### Multi-head attention

$$\text{head}_i = \text{Attn}(X W^Q_i,\ X W^K_i,\ X W^V_i), \quad i = 1 \ldots h$$
$$\text{MHA} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

$h$ heads at $d/h$ dims ≈ 1 head at $d$ dims (same FLOPs). Each head _can_ attend to different relationship types (syntax, coreference, position), but specialization is **emergent** not enforced.

---

<!-- ### Implementation

- **`.view(-1, ...)`** — `-1` infers a dim; passing seq_len directly also works.
- **`.transpose(a, b)`** — symmetric: `(1, 2) ≡ (2, 1)`; `(-2, -1) ≡ (-1, -2)`.
- **`.contiguous()`** — transpose only changes strides, not memory. `.view()` requires contiguous memory — omitting this crashes.
- **`softmax(dim=-1)`** — normalizes over $K$ positions for each $Q$. `dim=-2` would normalize over $Q$ — meaningless.
- **`head_dim = hidden_dim // num_heads`** — must divide evenly (assert in `__init__`).

--- -->

### Masks — shapes & purposes

| Mask               | Raw shape     | Expanded shape          | Blocks           | Where used     |
| ------------------ | ------------- | ----------------------- | ---------------- | -------------- |
| padding (src)      | `[B, S]`      | `[B, 1, S, S]`          | PAD in source    | enc self-attn  |
| causal             | `[S, S]` tril | `[B, nH, S, S]`         | future positions | dec self-attn  |
| decoder (combined) | —             | `[B, nH, S_tgt, S_tgt]` | future + PAD     | dec self-attn  |
| cross-attn         | `[B, S_src]`  | `[B, 1, S_tgt, S_src]`  | PAD in source    | dec cross-attn |

- **Cross-attn mask is rectangular** $[S_{\text{tgt}} \times S_{\text{src}}]$ because $Q$ comes from the decoder (tgt) and $K$, $V$ from the encoder (src). Self-attn masks are always square.
- **Decoder mask = causal AND pad.** Only attend if the position is _both_ a real token _and_ not in the future.

---

### Self-attn vs cross-attn

- **self (encoder)** — $Q$, $K$, $V$ all from $x$; every token sees all others.
- **self (decoder)** — $Q$, $K$, $V$ from $x$, but causal mask hides future.
- **cross (decoder)** — $Q$ from decoder $x$; $K$, $V$ from `enc_output` — decoder queries attend to encoder states.

_Why does the same model work in training (full target at once) and inference (one token at a time)?_ The causal mask simulates sequential generation during parallel training — the two are **identical computationally**.

---

### Encoder layer forward

```
x = embedding(src) * √H              [B, S, H]   scaling balances emb vs pos_enc
x = x + pos_encoding(0..S-1)         [B, S, H]   broadcast over batch
for layer in layers:
    attn_out, _ = self_attn(x, x, x) [B, S, H]   x,x,x = self-attention
    x = norm1(x + attn_out)          [B, S, H]   residual + layernorm
    ff = linear2(relu(linear1(x)))   [B, S, H]   FFN: H → intermediate → H
    x = norm2(x + ff)                [B, S, H]   residual + layernorm
```

---

### Decoder — final projection

```
logits = out_proj(x)    [B, S_tgt, vocab_size]
# training:  cross_entropy(logits, target_ids)
# inference: argmax(logits[:, -1, :]) or sample → next token id
```

Only the _last_ decoder layer's attention weights survive — each layer overwrites them. Useful for visualization, not needed for the forward pass.

---

### Residuals & LayerNorm

- **Residual connections.** Without them, each layer must reconstruct the original signal from scratch _in addition_ to learning — very hard in deep networks, and gradients vanish. Residuals let each layer learn only the _delta_.
- **LayerNorm.** Normalizes activations across the hidden dim _per token_, independently of other tokens. Stabilizes training in deep stacks. Applied after the residual add.
- **FFN role.** Attention _mixes_ information across positions; FFN ($d \to 4d \to d$ with GeLU) transforms each position _independently_ — the per-token nonlinear "memory / computation." It is the thing attention cannot do.

---

### Positional encoding

Self-attention is permutation-equivariant — without PE the model has no sense of order.

$$\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \quad \text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

**Add**, don't concatenate — keeps $d$ constant.

<!-- - _Self-attention has no recurrence and no convolution — how does it know token order?_ It doesn't. Positional encodings must be injected; the architecture is permutation-equivariant by default.
- _Shuffle tokens AND their positional encodings together ⇒_ output is **identical**. The model only knows _relative_ structure, not absolute index. -->

---

## 7. Pretraining & Transfer Learning

**Key idea.** Learn a strong **prior over language** from unlabeled text, then shift it toward a task by fine-tuning. Same weights; same contextual embeddings — and _contextual_ embeddings are what finally fixes the **polysemy** problem of Word2Vec/GloVe (each token gets a representation conditioned on its sentence).

---

### BERT — Masked LM

Mask ~15% of tokens, predict originals using **bidirectional** context:

$$\mathcal{L}_{\text{MLM}} = -\sum \log P(w_{\text{mask}} \mid \text{full bidirectional ctx})$$

Encoder-only. Great for classification / QA; not generative.

---

### GPT — Causal LM

Predict $w_t$ from $w_{<t}$ only:

$$\mathcal{L}_{\text{CLM}} = -\sum_t \log P(w_t \mid w_1, \ldots, w_{t-1})$$

Decoder-only. Natively generative, left-to-right.

---

_Why can't you pretrain BERT with a causal LM objective?_ Its attention is bidirectional, so the model would _see_ the token it is supposed to predict — trivially solved by copying, no useful representation learned. The masking forces genuine contextual inference precisely because the target is hidden.

---

## 8. Text Generation: Training & Decoding

### Training — teacher forcing

```
for t in 1..T:
    input  = gold y_{t-1}         # NOT model's prediction
    target = y_t
    loss   = −log P(y_t | gold prefix)
```

**Exposure bias.** At training time the model always sees a gold prefix; at inference it conditions on its own (possibly wrong) outputs ⇒ train/test distribution mismatch, and errors compound.

---

### Decoding strategies

```
Greedy:    ŷ_t = argmax P(y_t | y_{<t})      fast, myopic
Beam(k):   top-k hypotheses per step         ↑BLEU, repetitive
Sampling:  y_t ~ P(· | y_{<t})               diverse, incoherent
Top-k:     sample from top-k tokens only
Top-p:     sample from smallest set Σp ≥ p   (nucleus)
Temp τ:    P'(y) ∝ P(y)^{1/τ}
           τ → 0 = greedy, τ → ∞ = uniform
```

---

_Why does beam search score higher in BLEU than sampling but produce worse text by human judgment?_ Beam maximizes $\log P(Y)$, which favors short, generic, high-probability completions. Human preference rewards fluency and informativeness — properties not captured by likelihood alone.

---

## 9. REINFORCE, RLHF & DPO

**Why RL at all?** CLM optimizes next-token likelihood; human preferences (helpful, harmless, honest) are not predictable from MLE alone, and are not differentiable through discrete token samples ⇒ use policy gradient.

---

### REINFORCE

Treat the LM as a policy $\pi_\theta(y \mid x)$. For a scalar reward $r(x, y)$:

$$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[r] = \mathbb{E}_{y \sim \pi_\theta}\!\left[ r(x, y) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \right]$$

High variance ⇒ subtract a baseline $b$ (e.g. value estimate) to get $r - b$ without changing the expected gradient.

---

### RLHF pipeline

1. **SFT** — fine-tune on human demonstrations.
2. **Reward model** $r_\phi$ on preference pairs $(y^w \succ y^l)$, Bradley–Terry:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}\!\left[ \log \sigma\!\left( r_\phi(x, y^w) - r_\phi(x, y^l) \right) \right]$$

3. **PPO** — maximize reward while staying close to the SFT reference:

$$\max_\theta\ \mathbb{E}_{\pi_\theta}\!\left[ r_\phi(x, y) \right] - \beta \cdot \text{KL}\!\left[ \pi_\theta \,\|\, \pi_{\text{ref}} \right]$$

---

### Why the KL term?

Without it, $\pi_\theta$ exploits the blind spots of $r_\phi$ — **reward hacking** — drifting to OOD text that scores high under $r_\phi$ but is actually bad.

---

### DPO

Closed-form reparameterization removes the explicit RM and RL loop:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y^w)}{\pi_{\text{ref}}(y^w)} - \beta \log \frac{\pi_\theta(y^l)}{\pi_{\text{ref}}(y^l)} \right) \right]$$

Same optimum in theory, simpler in practice.

---

_Why can't you skip SFT and run PPO directly on a pretrained model?_ It doesn't follow instructions; the RM was trained on instruction-following preferences; and a near-random policy provides essentially no useful gradient signal — it just wastes compute. SFT puts the policy in the neighborhood where the RM is informative.

---

## 10. LLMs: ICL, Instruction Tuning, Post-Training

### In-Context Learning (ICL)

Task specified entirely via prompt. **Zero gradient updates** to weights.

```
Prompt = [demo_1, demo_2, ..., demo_k, query]
Model completes query — no parameter change.
```

**Mechanistic views:**

- _Meta-learning_ — attention over demos behaves like **implicit gradient descent** in activation space.
- _Task vector_ — demos shift the residual stream into a task-relevant subspace.

---

### Instruction tuning

Fine-tune on $\{(\text{instruction}, \text{response})\}$ pairs ⇒ better zero-shot generalization across _unseen_ task types. Unlike ICL, the capability is baked into the weights, not re-invoked each prompt.

**Instruction tuning ≠ ICL.** IT updates weights and generalizes to new task types; ICL uses context and stays within pretrained capabilities. Both expose the model to task format, but via different mechanisms.

---

- _ICL accuracy is robust even when labels in the demonstrations are **wrong** — what does this reveal?_ The model uses demos mainly to identify task **format** and input **distribution**, not to learn the input → output mapping. The pretraining prior dominates the label signal.
- _Why does RLHF often make models less calibrated (sycophantic, overconfident)?_ The RM rewards confident, agreeable answers; the KL penalty limits but cannot prevent distributional drift; and human raters often prefer a confident wrong answer over an uncertain correct one. Calibration is not in the loss.

---

## Full Concept Chain

```
n-gram          fixed context, sparse counts, no generalization
  ↓ shared embeddings + recurrence
RNN             unlimited context in theory; vanishing gradients
  ↓ gated additive cell state
LSTM            stable gradients; still sequential, still bottlenecked
  ↓ dynamic context per decoder step
Seq2Seq+Attn    no bottleneck; still O(n) sequential steps
  ↓ drop recurrence entirely
Transformer     O(1) depth, parallel, scales to huge data
  ↓ scale + unlabeled pretraining
BERT / GPT      contextual embeddings ⇒ fix polysemy
  ↓ demos in prompt, no weight update
ICL             task adaptation at inference time
  ↓ supervised instruction fine-tuning
Instruction-tuned LM
  ↓ human preference signal
RLHF / DPO      aligned, instruction-following LLM
```

**Meta-pattern.** Each arrow removes one hard constraint of the previous stage: atomicity → distribution, fixed window → recurrence, gradient decay → gating, fixed context → dynamic attention, sequential computation → parallelism, labeled data → unlabeled pretraining, likelihood → human preferences.

---

## 11. Derivatives

| Name         | $f$                          | $df/dx$                    |
| ------------ | ---------------------------- | -------------------------- |
| Power rule   | $x^n$                        | $n \cdot x^{n-1}$          |
| General exp  | $a^x$                        | $a^x \cdot \ln a$          |
| Log base $a$ | $\log_a x$                   | $1 / (x \ln a)$            |
| Sigmoid      | $\sigma(x) = 1/(1 + e^{-x})$ | $\sigma(x)(1 - \sigma(x))$ |
| Tanh         | $\tanh(x)$                   | $1 - \tanh^2(x)$           |
| ReLU         | $\max(0, x)$                 | $1$ if $x > 0$ else $0$    |

<!--
| GELU         | $x \cdot \Phi(x)$            | $\Phi(x) + x \cdot \phi(x)$       |
| SiLU / Swish | $x \cdot \sigma(x)$          | $\sigma(x)(1 + x(1 - \sigma(x)))$ |
-->

| Softmax | $s_i = e^{x_i} / \sum_j e^{x_j}$ | $s_i (\delta_{ij} - s_j)$ |
| Log-Softmax | $\log s_i$ | $\delta_{ij} - s_j$ |
| **XEnt + Softmax ★** | $-\log s_y$ | $p_i - y_i$ |
| KL Divergence | $\sum_i p_i \log(p_i / q_i)$ | $-p_i / q_i$ |

**Why the ★ matters.** The softmax Jacobian and the log cancel — the gradient collapses to **predicted − target**. That's why cross-entropy with softmax is the numerically stable default for classification.

---

### Layers & Normalization

| Name                  | $f$                                         | $df/dx$                                                                                                                                           |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear (weights)      | $Wx + b$                                    | $\delta \cdot x^\top$                                                                                                                             |
| Linear (input)        | $Wx + b$                                    | $W^\top \cdot \delta$                                                                                                                             |
| Linear (bias)         | $Wx + b$                                    | $\delta$                                                                                                                                          |
| Layer Norm            | $\gamma \cdot (x - \mu)/\sigma + \beta$     | $(\gamma/\sigma)\left[\partial_{\hat y} L - \text{mean}(\partial_{\hat y} L) - \hat y \cdot \text{mean}(\partial_{\hat y} L \cdot \hat y)\right]$ |
| Attention (V)         | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $S^\top \cdot \partial_O L$                                                                                                                       |
| Attention ($QK^\top$) | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $J_{\text{softmax}}(\partial_S L) / \sqrt d$                                                                                                      |

---
