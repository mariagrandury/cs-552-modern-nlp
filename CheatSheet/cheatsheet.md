# Modern NLP Cheatsheet

---

## 1. Word Embeddings

One-hot $\mathbf{e}_w \in \{0,1\}^{|V|}$: orthogonal ⇒ no similarity. Dense $\mathbf{e}_w \in \mathbb{R}^d$, $d \ll |V|$: geometry encodes meaning (**distributional hypothesis**).

- **Skip-gram** — predict context from center $w_t$. Captures **contextual associations**. Dynamic window: sample $i \in [1,N]$, closer words seen more often.
- **CBOW** — predict center from summed context ⇒ removes word order (bag-of-words). Captures **substitutable words** (synonyms). Projection $U \in \mathbb{R}^{V \times d}$ maps context vector to one score per vocab word, dot product $U \cdot h_t$ gives scores $\in \mathbb{R}^V$.
- **Negative sampling** — replace $|V|$-softmax with $O(k)$ binary classification. **Hierarchical softmax** — tree-structured approximation of full $|V|$-softmax, used in practice for compute reasons.
- **GloVe** — $J = \sum f(X_{ij})(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$. **Global** co-occurrence matrix; W2V uses local windows. Combines matrix factorization efficiency with W2V linear substructures.
- **FastText** — splits words into character n-grams ⇒ handles morphology, typos, OOV.

**Static** embeddings: one vector per type ⇒ polysemy unsolved. One-hot can't capture similarity even with smoothing (it redistributes mass but never creates similarity structure).

**Pipeline:** (1) Tokenization → vectors, (2) Model → representations, (3) Head (classifier) → prediction, (4) Backprop.
**Logits** = raw unnormalized scores from last linear layer, no probabilistic interpretation. **Probabilities** = softmax(logits): exponentiate all scores (→ positive), divide by sum (→ sum to 1).
**Weight tying:** input embedding matrix and output projection $W_o$ are transposes (vocab→$d$ vs $d$→vocab) ⇒ optimized jointly. Embeddings shared across all instances of same word.

---

## 2. N-gram LMs

Markov: $P(w_t|w_{1:t-1}) \approx P(w_t|w_{t-n+1:t-1})$. MLE: $\hat P = C(\text{ctx},w)/C(\text{ctx})$. **Perplexity** $= P(W)^{-1/N}$ = exponentiated avg NLL. Uniform baseline: PPL = $|V|$.

- **Smoothing:** redistributes prob from seen to unseen patterns. **Laplace** (add $\alpha$ to every count; games PPL w/o improving LM), **back-off** (fall back to lower n-gram when count is 0), **linear interpolation,** weighted mix of n-gram probs: ($P = \sum_i \lambda_i P_{\text{i-gram}}$), $\sum \lambda_i = 1$. **Kneser-Ney** continuation probability: how many distinct contexts a word appears in, not raw freq.
- **Zipf's law:** word freq ~2× the next ⇒ long tail drives sparsity. Even Google only used 5-grams.
- **Limits:** sparsity, can't generalize across synonyms, no distributed representation, no cross-word generalization (each word atomic), hard context cap at $n-1$.
- **PPL issues:** Can't compare PPL across different vocab sizes (larger vocab ⇒ higher PPL). Domain mismatch inflates PPL. PPL > $|V|$ = worse than random (sanity check).
- **Fixed-context neural LM (Bengio):** represent n-gram as NN. Concatenate $n$ embeddings → hidden layer → softmax over $V$. No sparsity (softmax ≠ 0), smaller model. But fixed window, no weight sharing across positions ⇒ enlarging window explodes $W$.

---

## 3. RNN & LSTM

**RNN:** $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$, shapes: $x_t$`(B,d)`, $h_t$`(B,h)`. Same $W_h, W_x$ reused ⇒ parameter sharing = translation invariance. Unlimited context in theory.

**Vanishing gradient:** $\partial\mathcal{L}/\partial h_0 = \prod_{t} \partial h_t/\partial h_{t-1} \to 0$ when product of **activation derivative** (usually <1, esp. sigmoid) **times** $W_{hh}$ (small due to regularization) is repeatedly multiplied. Early context forgotten. **Exploding gradients:** dominant singular value >1 ⇒ gradients grow exponentially. Fix: **gradient clipping** — rescale all gradients proportionally when global norm exceeds threshold (preserves direction). Also: sequential ⇒ no parallelism.

**LSTM** — cell uses **addition** ⇒ gradient highway: $\partial c_T/\partial c_t \approx \prod f_\tau \approx \text{const when } f \approx 1$.

```
f_t = σ(W_f·[h_{t-1},x_t])   i_t = σ(W_i·[h_{t-1},x_t])
c̃_t = tanh(W_c·[h_{t-1},x_t])
c_t = f_t⊙c_{t-1} + i_t⊙c̃_t     [ADDITIVE cell update]
o_t = σ(W_o·[h_{t-1},x_t])   h_t = o_t⊙tanh(c_t)
```

_If forget gate = 1 always?_ Unbounded accumulator — never forgets. The forget gate enables **selective, bounded** memory. Cell state uses **addition** (not matrix multiply) ⇒ gradient highway: gradients flow through additive path without full weight matrix at each step. When $f \approx 1$: constant error carousel.

**GRU:** simpler gated RNN. $z_t$ = update gate, $r_t$ = reset gate. $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. Uses **weighted average** ⇒ values stay bounded. $z=1$: forget old state entirely. $z=0$: keep old state. Less powerful than LSTM (fewer gates), but fewer params.

**BiRNN:** two independent RNNs — forward (l→r) and backward (r→l). Output = concatenation of both hidden states ⇒ each token sees both past and future context. Separate params for each direction. Use for: classification, sequence labeling. **Not** for generation (can't see future).

**Multiple layers:** cascade RNN outputs through layers, each with different $W$. Typically 3–4 layers, up to 8–12. Each layer has its own hidden state.

**BPTT (Backpropagation Through Time):** unroll the dynamic computation graph into a static one (length known from forward pass), then apply standard backprop. Same matrix $W_{hh}$ appears at every step ⇒ chain of multiplications. Key insight: gradient computations at each layer reuse ("cache") results from the next layer ⇒ efficient.

**Dropout in LSTMs:** standard dropout disrupts cell state gradient flow. Solutions: **variational dropout** (same mask for entire sequence), dropout only on non-recurrent connections, **zoneout** (randomly preserve hidden state units instead of zeroing).

**Training detail:** loss alignment — `outputs[:, :-1, :]` predicts `labels[:, 1:]` (shifted by 1). Use `ignore_index=pad_idx` in CrossEntropyLoss.

---

## 4. Seq2Seq & Attention

**Seq2Seq:** encoder compresses source into fixed $c = h_n$ `(B,h)`. Decoder generates autoregressively conditioned on $c$. Decoder **cannot be bidirectional** — would make generation trivial at training, impossible at inference. **Teacher forcing:** train on gold prefix $y^*_{<t}$, not model predictions. **Temporal bottleneck:** single fixed-size vector must represent arbitrarily long input — capacity doesn't scale with source length.

**Attention:** dynamic context per decoder step. $e_{t,s} = v^\top\tanh(W_h h^d_{t-1} + W_s h^e_s)$, $\alpha_{t,s} = \text{softmax}_s(e_{t,s})$, $c_t = \sum_s \alpha_{t,s} h^e_s$. Shapes: $e_t$`(B,n)`, $\alpha_t$`(B,n)`, $c_t$`(B,h)`.

- **Query-key mechanism** (like database lookup): key = encoder hidden state, query = decoder hidden state. Similarity score determines how much to attend. Scaled dot product most common today.
- Fixes bottleneck (capacity scales with source length), provides soft alignment, direct gradient path.
- Still $O(n)$ sequential (RNN-based). Cross-attention: $O(n \cdot m)$. Decoder self-attention at inference: $O(n^2)$ ⇒ motivates **KV caching** (store past K,V; only compute new query).

**Exposure bias:** teacher forcing trains on gold prefix ≠ inference on own outputs ⇒ errors compound. **Scheduled sampling:** gradually replace gold with model predictions during training.

---

## 5. Transformer

Drop recurrence. $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. Sequential depth $O(1)$, compute $O(n^2)$.

**Why 3 projections from same $x$?** Without them, self-dot-product dominates ⇒ each token attends mostly to itself. **Why $/\sqrt{d_k}$?** High-dim dots grow large ⇒ softmax saturates to one-hot ⇒ vanishing gradients. Scale by $\sqrt{d_\text{head}}$, NOT $\sqrt{d_\text{model}}$.

**Multi-head:** $\text{head}_i = \text{Attn}(XW^Q_i, XW^K_i, XW^V_i)$, $\text{MHA} = \text{Concat}(\text{heads})W_O$. $h$ heads at $d/h$ dims ≈ same FLOPs as 1 head at $d$. Specialization is **emergent**, not enforced. `hidden_dim` must be divisible by `num_heads`.

**Shapes:** `[B,S,H]` → `.view(B,S,nH,hD)` → `.transpose(1,2)` → `[B,nH,S,hD]` → attn scores `[B,nH,S,S]` → output `[B,nH,S,hD]` → `.transpose(1,2).contiguous().view(B,S,H)`. **`.contiguous()` is required** after transpose before view (memory layout).

**Masks:** Padding `[B,S]`→`[B,1,S,S]` blocks PAD. Causal: `torch.tril` lower-triangle blocks future. Cross-attn is **rectangular** `[S_tgt×S_src]` ($Q$ from decoder, $K$/$V$ from encoder). Decoder mask = causal AND pad (bitwise). Applied via `masked_fill(mask==0, -inf)` ⇒ softmax gives 0. **Why mask before softmax (not multiply scores by 0)?** If we zero out scores directly, remaining scores won't sum to 1 after softmax. Setting to $-\infty$ before softmax ensures proper probability distribution. Causal mask makes training (parallel) and inference (sequential) **computationally identical**.

**Self vs cross:** Encoder self-attn: $Q,K,V$ all from $x$, bidirectional. Decoder self-attn: causal masked. Cross-attn: $Q$ from decoder, $K,V$ from encoder output. Encoder = 2 sub-layers (self-attn + FFN). Decoder = 3 sub-layers (masked self-attn + cross-attn + FFN), each with residual + LayerNorm.

**Residuals** let layers learn deltas. **LayerNorm** normalizes across hidden dim per token. **Pre-norm** (modern): LayerNorm before attention/FFN. **Post-norm** (original): LayerNorm after. Pre-norm trains more stably. **FFN** ($d→4d→d$, GeLU) transforms each position independently — attention mixes across positions, FFN provides per-token nonlinearity. `intermediate_size` is usually 4× `hidden_dim`.

**PE:** attention is permutation-equivariant ⇒ PE required. Sinusoidal $\sin(p/10000^{2i/d})$ or learned. **Add** (don't concat). Learned can't generalize beyond training length. Embedding scaled by $\sqrt{H}$. to balance magnitude vs PE.
**Cross-attn:** `attn_weights` shape `[B,nH,S_tgt,S_src]

---

## 6. Tokenization

**Subword** = sweet spot between word-level (OOV) and char-level (too long seq).

| Method            | Used by    | Merge criterion                                         | Key difference    |
| ----------------- | ---------- | ------------------------------------------------------- | ----------------- |
| **BPE**           | GPT, Llama | most frequent pair                                      | raw frequency     |
| **WordPiece**     | BERT       | $\text{freq(pair)}/(\text{freq(a)}\cdot\text{freq(b)})$ | mutual info       |
| **SentencePiece** | mT5        | no pre-tokenization                                     | language-agnostic |
| **Byte-level**    | ByT5       | UTF-8 bytes (vocab=256)                                 | no OOV, long seqs |

<!-- _Shortcoming of BPE?_ Greedy merge by raw frequency ignores how informative each token is. _Why does ByT5 use a heavy encoder / light decoder?_ To compensate for much longer byte sequences. -->

Special tokens: `<pad>` (uniform batch length for computation graph; unnecessary if batch_s=1), `<unk>`, `<s>` (BOS), `</s>` (EOS, learn when to stop). BPE needs (whitespace) pre-tokenization ⇒ fails without spaces (Chinese, Thai) unless SentencePiece.

---

## 7. Pretrained LMs

**Contextual** embeddings fix polysemy. Pretraining: self-supervised on large corpus (easy objectives, naturally occurring data), then fine-tune.

**ELMo (2018).** Two separate unidirectional LSTMs, concatenated. "Bidirectional" is **fake**, no shared params between directions. Shared: input embeddings + vocab projection layer between both LSTMs. Embedding = $\gamma \sum_j s_j h_j$ (task-weighted sum across all layers; learn $\gamma, s_j$ per task, don't update pretrained params). Lower layers ≈ syntax, upper ≈ semantics. _Why not just use the last layer?_ Different tasks benefit from different layers.

**BERT (2019).** Encoder-only, truly bidirectional. Diff w/ Transformer: **learned** position embeddings + **segment** embeddings (distinguish sentence A/B). MLM: mask 15% (80% `[MASK]`, 10% random, 10% unchanged — prevents train/test mismatch). `[CLS]` at front (convention; bidirectional ⇒ any position works). **Whole-word masking** helps named entities (`[MASK]bama`). Cannot generate text. BERT: better to fully fine-tune (vs ELMo: adapt only some params). **Learning**: Heads learn diverse concepts, emergently.

**ELECTRA.** Discriminator classifies **every** token as real/corrupted ⇒ full-signal training (vs BERT's 15%).

**GPT.** Decoder-only, causal masking, no cross-attention. `[CLS]` at **end** (only position with full context). **GPT2:** same arch, larger, strong zero-shot.

**BART.** BERT encoder + GPT decoder. Best corruption: text infilling + sentence permutation. Classification: input to both encoder AND decoder. Handles BERT + generation (autoregressive decoder).
**T5:** all tasks as text-to-text.
**DistilBERT,** 3 losses: MLM + distillation (soft probs from teacher) + embedding cosine.
**RoBERTa:** BERT trained longer on more data.

**FT details:** When loading pretrained weights, UNEXPECTED keys (e.g. MLM head) are safe to ignore; MISSING keys (classifier head) are freshly initialized - those are what you fine-tune.
**Sentence similarity** (siamese arch): 2 models with shared params, encoder → CLS token; decoder-only → last token or mean/sum-pool.

---

## 8. Text Generation

**Teacher forcing** (= SFT): train on gold prefix ⇒ **exposure bias** at inference (errors compound).

| Decoding    | Formula                                       | Tradeoff                                              |
| ----------- | --------------------------------------------- | ----------------------------------------------------- |
| Greedy      | $\hat y_t = \arg\max P(y_t \mid y_{<t})$      | fast, repetitive                                      |
| Beam($b$)   | keep $b$ top hypotheses, b=1→greedy           | ↑BLEU but generic; $O(b^2)$                           |
| Top-$k$     | sample from top-$k$ tokens                    | fixed $k$: too loose when peaky, aggressive when flat |
| Top-$p$     | sample from smallest set with $\sum p \geq p$ | adapts to distribution shape                          |
| Temp $\tau$ | $P' \propto \exp(S_w/\tau)$                   | $\tau→0$: argmax, $\tau→\infty$: uniform              |

Temperature applied **before** top-k/top-p. Top-p: include token that makes cumsum **exceed** $p$ (implementation: **shift mask right by 1**), then renormalize.

<!--
**Beam search extras:** `no_repeat_ngram_size=n` sets prob of repeated n-gram to 0 (but breaks proper nouns like "New York"). `repetition_penalty` penalizes all repeating tokens (problematic with BPE — penalizes stopwords). `length_penalty` = score / $\text{len}^\alpha$ (positive $\alpha$ ⇒ longer outputs).
_Why does beam score higher BLEU but worse human judgment?_ Beam maximizes $\log P(Y)$ ⇒ short, generic, high-prob completions. Humans reward informativeness, not likelihood.
-->

**Repetition trap:** greedy/beam NLL decreases with repetition ⇒ self-reinforcing loops.

**Re-ranking:** generate multiple sequences, rerank by score. Recalibrate: k-NN, combine with 2nd model (MT).
**KV Cache:** reuse `past_key_values` ⇒ avoid recomputing hidden states at each step.

**Eval metrics.** BLEU: n-gram precision, MT, no semantics. ROUGE: n-gram recall, summarization, no semantics. BERTScore: contextual sim, depends on BERT. BLEURT: BERT regression, grammar + meaning, needs training. COMET: neural, human correlation, requires source+hyp+ref. LLM-as-judge: rubric-based, flexible, position bias, self-preference.
Also, Pyramid (summ), SPICE (captioning), SPIDEr (SPICE+CIDEr), Word Mover's Distance (embedding sim). N-gram metrics degrade as tasks become more open-ended. PPL of generated text measures model calibration, not generation quality (repetition scores well). Humans: never compare across studies, clear guidelines, calibration examples.

---

## 9. RLHF, DPO & Beyond

**Why RL?** MLE optimizes next-token likelihood; human preferences aren't differentiable through discrete samples ⇒ policy gradient.

**REINFORCE:** $\nabla_\theta \mathbb{E}[r] = \mathbb{E}[r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)]$. Reward **scales the loss**: high reward → larger loss → learn to reproduce; low reward → loss near 0 → don't update much. High variance ⇒ subtract baseline $b$: $(r - b)$ without changing expected gradient. **Credit assignment:** reward applied at sequence level (hard to assign per-token). **Variance reduction** via baseline (e.g. BLEU 0–100 range). Stabilize: **joint optimization** $\mathcal{L} = \mathcal{L}_\text{MLE} + \alpha\mathcal{L}_\text{RL}$ (MLE term promotes fluency since RL alone doesn't always generate readable text). **Reward gaming:** RL can optimize metrics (higher BLEU/ROUGE) without improving human judgment. Start RL only after model is already somewhat calibrated (SFT first).

**RLHF pipeline:** (1) **SFT** on demonstrations → (2) **RM** on preference pairs: $\mathcal{L}_\text{RM} = -\log\sigma(r_\phi(y^w) - r_\phi(y^l))$ (Bradley-Terry) → (3) **PPO**: $\max \mathbb{E}[r_\phi] - \beta\text{KL}[\pi_\theta \| \pi_\text{ref}]$. PPO clips $\pi_\theta/\pi_\text{ref}$ to $[1-\epsilon, 1+\epsilon]$ (REINFORCE updates are unbounded).

**KL term** prevents **reward hacking** — without it, $\pi_\theta$ drifts to OOD text that exploits RM blind spots. _Using a politeness classifier as reward?_ Model learns to add "please" 20× — reward hacking.

**DPO:** $\mathcal{L} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y^w)}{\pi_\text{ref}(y^w)} - \beta\log\frac{\pi_\theta(y^l)}{\pi_\text{ref}(y^l)}\right)$. Eliminates explicit RM and RL loop. Same optimum, simpler.

**DPO implementation:** log-probs via shifted logits: `log_softmax(logits[:,:-1,:])` gathered by `labels[:,1:]`, mask out `-100` positions (prompt tokens). Initial loss ≈ $\log 2 \approx 0.693$ because $\pi_\theta = \pi_\text{ref}$ ⇒ ratios = 0 ⇒ $\sigma(0)=0.5$. Implicit reward = $\beta \log(\pi_\theta / \pi_\text{ref})$. Freeze reference model; disable dropout in both. **Reward accuracy > 0.5** = model correctly ranks chosen over rejected.

**GRPO:** sample $G$ outputs, advantage = $(R_i - \mu)/\sigma$, PPO-style clipping ($\epsilon=0.2$) + KL penalty. KL estimator: $\exp(\delta)-\delta-1$ where $\delta = \log\pi_\text{ref} - \log\pi_\theta$ (always $\geq 0$). **Fails when all $G$ correct OR all wrong** — $\sigma=0$ ⇒ no gradient. GRPO replaces PPO's value network (critic) ⇒ halves memory.

**RLVR:** binary reward from programmatic check — no RM needed (⇒ no RM noise or reward hacking).

_Why not skip SFT?_ Near-random policy gives no useful RL signal.

<!-- Model doesn't follow instructions; RM trained on instruction-following; near-random policy gives no useful signal. SFT puts the policy where the RM is informative. -->

---

## 10. ICL, Instruction Tuning & Scaling

**Emergence:** quantitative changes → qualitative changes. ICL emergent ~175B params.

**ICL:** prompt-based, **zero weight updates**. Uses demos for task format, not input→output mapping — works with wrong labels. Sensitive to example selection/order (worse in SLM). Better for tasks w/ terms frequent in pretraining.

**Cloze prompting (PET):** classification as fill-mask. **Verbalizer** maps labels→words; choice strongly affects accuracy. RoBERTa tokenizes with leading space: use `"\u0120"+word` for token lookup.

**Few-shot can hurt:** context label imbalance or lexical cues bias predictions.

**Instruction tuning:** fine-tune on (instruction, response) ⇒ zero-shot generalization to unseen tasks. Updates weights.

**CoT:** reasoning steps before answer, needs scale. **Zero-shot CoT:** "Let's think step by step." **Self-Consistency:** sample $N$ responses w/ T>0, majority vote, $N\times$ compute.

**AutoPrompt:** gradient-guided search for optimal tokens. Best prompt sometimes random-looking ⇒ foundation of **jailbreaking**. **Soft prompts/prompt-tuning:** instead of discrete tokens, learn continuous prompt representations. Model frozen, only prompt vectors trained.

**Efficient FT:** **Adapters** (FFN inserts between layers), **LoRA** (low rank $\Delta W = AB$, $r \ll d$, frozen base).

**Test-time scaling:** more reasoning tokens (compute) ⇒ higher accuracy.

<!--
_Why does RLHF make models sycophantic?_ RM rewards confident answers; KL limits but can't prevent drift; human raters prefer confident wrong over uncertain correct. Calibration is not in the loss.
-->

---

## 11. Dataset Artifacts

**Mitigation:** contrast sets (more examples), adversarial filtering (weaker model finds spurious examples), bias-only ensemble (update params only for samples where bias model failed, e.g. hypothesis-only for NLI), data augmentation, annotation guidelines.

**Pretraining data matters most.** Quality heuristics can be flawed (e.g. filtering "sex" from C4). Good benchmarks: monotonic, low variance (CommonsenseQA, HellaSwag,OpenBookQA, PIQA). Bad: SocialIQA, TruthfulQA. Benchmarks are aggregations — one problem cascades.

**Inter-annotator agreement:** Cohen's κ (2 raters), Fleiss' κ (>2), Krippendorff's α. But filtering by agreement can eliminate legitimate ambiguity.

---

## Derivatives

| Name                | $f$                              | $df/dx$                    |
| ------------------- | -------------------------------- | -------------------------- |
| Power rule          | $x^n$                            | $n \cdot x^{n-1}$          |
| General exp         | $a^x$                            | $a^x \cdot \ln a$          |
| Log base $a$        | $\log_a x$                       | $1 / (x \ln a)$            |
| Sigmoid             | $\sigma(x) = 1/(1 + e^{-x})$     | $\sigma(x)(1 - \sigma(x))$ |
| Tanh                | $\tanh(x)$                       | $1 - \tanh^2(x)$           |
| ReLU                | $\max(0, x)$                     | $1$ if $x > 0$ else $0$    |
| Softmax             | $s_i = e^{x_i} / \sum_j e^{x_j}$ | $s_i (\delta_{ij} - s_j)$  |
| **XEnt + Softmax"** | $-\log s_y$                      | $p_i - y_i$                |
| KL Divergence       | $\sum_i p_i \log(p_i / q_i)$     | $-p_i / q_i$               |

"Softmax Jacobian and log cancel → gradient = **predicted − target**. That's why cross-entropy with softmax is the numerically stable default for classification.

### Layers & Normalization

| Name                  | $f$                                         | $df/dx$                                                                                                                                           |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear (weights)      | $Wx + b$                                    | $\delta \cdot x^\top$                                                                                                                             |
| Linear (input)        | $Wx + b$                                    | $W^\top \cdot \delta$                                                                                                                             |
| Linear (bias)         | $Wx + b$                                    | $\delta$                                                                                                                                          |
| Layer Norm            | $\gamma \cdot (x - \mu)/\sigma + \beta$     | $(\gamma/\sigma)\left[\partial_{\hat y} L - \text{mean}(\partial_{\hat y} L) - \hat y \cdot \text{mean}(\partial_{\hat y} L \cdot \hat y)\right]$ |
| Attention (V)         | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $S^\top \cdot \partial_O$                                                                                                                         |
| Attention ($QK^\top$) | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $J_{\text{softmax}}(\partial_S L) / \sqrt d$                                                                                                      |

---
