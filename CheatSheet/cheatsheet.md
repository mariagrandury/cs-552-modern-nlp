# Modern NLP Cheatsheet

---

## 1. Word Embeddings

One-hot $\mathbf{e}_w \in \{0,1\}^{|V|}$ are always orthogonal ⇒ no similarity structure even for synonyms. Dense $\mathbf{e}_w \in \mathbb{R}^d$, $d \ll |V|$: geometry encodes meaning (**distributional hypothesis**).

- **Skip-gram** — predict context from center $w_t$; more updates for rare words than CBOW.
- **CBOW** — predict center from averaged context ⇒ destroys word order (bag-of-words).
- **Negative sampling** — replace $|V|$-softmax with $O(k)$ binary classification.
- **GloVe** — $J = \sum f(X_{ij})(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$. Explicitly factorizes log co-occurrence; W2V implicitly factorizes shifted PMI. GloVe uses **global** statistics ⇒ better semantic similarity than W2V (which uses local windows only).

**Issues:** both are **static** (one vector per type ⇒ polysemy unsolved). One-hot can't capture similarity even with smoothing — smoothing redistributes mass but never creates similarity structure.

**Sentence embeddings from word embeddings:** average-pool or max-pool token vectors ⇒ fixed-size `(emb_dim,)`. OOV tokens are skipped. Simplest classifier: `Linear(emb_dim, 1) + Sigmoid + BCELoss`

---

## 2. N-gram LMs

Markov: $P(w_t|w_{1:t-1}) \approx P(w_t|w_{t-n+1:t-1})$. MLE: $\hat P = C(\text{ctx},w)/C(\text{ctx})$. **Perplexity** $= P(W)^{-1/N}$ = exponentiated avg NLL. **Uniform LM baseline:** PPL = $|V|$.

- **Laplace** — add $\alpha$ to every count. **Kneser-Ney** — continuation probability: how many distinct contexts a word appears in, not raw freq.
- **Limits:** sparsity, no cross-word generalization (cat≠dog), hard context cap at $n-1$.
- **Trigram can be WORSE than bigram** — higher-order n-grams overfit more with sparse data. Replacing rare words with `<unk>` always helps (reduces sparsity).
- _Why can't perfect smoothing generalize across synonyms?_ No distributed representation — each word is atomic.
- Cannot compare perplexity across models with **different vocab sizes** (larger vocab ⇒ higher PPL mechanically).

---

## 3. RNN & LSTM

**RNN:** $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$, shapes: $x_t$`(B,d)`, $h_t$`(B,h)`. Same $W_h, W_x$ reused ⇒ parameter sharing = translation invariance. Unlimited context in theory.

**Vanishing gradient:** $\partial\mathcal{L}/\partial h_0 = \prod_{t} \partial h_t/\partial h_{t-1} \to 0$ when $\|W_h\| < 1$. Early context forgotten. Also: sequential ⇒ no parallelism.

**LSTM** — cell uses **addition** ⇒ gradient highway: $\partial c_T/\partial c_t \approx \prod f_\tau \approx \text{const when } f \approx 1$.

```
f_t = σ(W_f·[h_{t-1},x_t])   i_t = σ(W_i·[h_{t-1},x_t])
c̃_t = tanh(W_c·[h_{t-1},x_t])
c_t = f_t⊙c_{t-1} + i_t⊙c̃_t     [ADDITIVE cell update]
o_t = σ(W_o·[h_{t-1},x_t])   h_t = o_t⊙tanh(c_t)
```

_If forget gate = 1 always?_ Unbounded accumulator — never forgets. The forget gate enables **selective, bounded** memory. Still sequential, still bottlenecked in seq2seq.

**Fixed-window MLP vs RNN:** MLP concatenates last $n$ embeddings `(B, n*d)` → Linear → tanh → Linear → LogSoftmax. Fixed context. RNN handles variable length; same weights at each step. Neural LMs (PPL~200-250) far outperform n-gram (PPL~300-900) even with small architectures.

**Training detail:** loss alignment — `outputs[:, :-1, :]` predicts `labels[:, 1:]` (shifted by 1). Use `ignore_index=pad_idx` in CrossEntropyLoss.

---

## 4. Seq2Seq & Attention

**Seq2Seq:** encoder compresses source into fixed $c = h_n$ `(B,h)`. Decoder generates autoregressively conditioned on $c$. **Teacher forcing:** train on gold prefix $y^*_{<t}$, not model predictions. **Bottleneck:** fixed-size $c$ has bounded capacity _regardless of dimensionality_ — information-theoretic, not fixable by making $h$ larger.

**Attention:** dynamic context per decoder step. $e_{t,s} = v^\top\tanh(W_h h^d_{t-1} + W_s h^e_s)$, $\alpha_{t,s} = \text{softmax}_s(e_{t,s})$, $c_t = \sum_s \alpha_{t,s} h^e_s$. Shapes: $e_t$`(B,n)`, $\alpha_t$`(B,n)`, $c_t$`(B,h)`.

- Fixes bottleneck (capacity scales with source length), provides soft alignment, direct gradient path.
- Still $O(n)$ sequential (RNN-based). Cross-attention: $O(n \cdot m)$. Decoder self-attention at inference: $O(n^2)$ ⇒ motivates **KV caching** (store past K,V; only compute new query).

---

## 5. Transformer

Drop recurrence. $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. Sequential depth $O(1)$, compute $O(n^2)$.

**Why 3 projections from same $x$?** Without them, self-dot-product dominates ⇒ each token attends mostly to itself. **Why $/\sqrt{d_k}$?** High-dim dots grow large ⇒ softmax saturates to one-hot ⇒ vanishing gradients. Scale by $\sqrt{d_\text{head}}$, NOT $\sqrt{d_\text{model}}$.

**Multi-head:** $\text{head}_i = \text{Attn}(XW^Q_i, XW^K_i, XW^V_i)$, $\text{MHA} = \text{Concat}(\text{heads})W_O$. $h$ heads at $d/h$ dims ≈ same FLOPs as 1 head at $d$. Specialization is **emergent**, not enforced. `hidden_dim` must be divisible by `num_heads`.

**Shapes:** `[B,S,H]` → `.view(B,S,nH,hD)` → `.transpose(1,2)` → `[B,nH,S,hD]` → attn scores `[B,nH,S,S]` → output `[B,nH,S,hD]` → `.transpose(1,2).contiguous().view(B,S,H)`. **`.contiguous()` is required** after transpose before view (memory layout).

**Masks:** Padding `[B,S]`→`[B,1,S,S]` blocks PAD. Causal: `torch.tril` lower-triangle blocks future. Cross-attn is **rectangular** `[S_tgt×S_src]` ($Q$ from decoder, $K$/$V$ from encoder). Decoder mask = causal AND pad (bitwise). Applied via `masked_fill(mask==0, -inf)` ⇒ softmax gives 0. Causal mask makes training (parallel) and inference (sequential) **computationally identical**.

**Self vs cross:** Encoder self-attn: $Q,K,V$ all from $x$, bidirectional. Decoder self-attn: causal masked. Cross-attn: $Q$ from decoder, $K,V$ from encoder output. Encoder = 2 sub-layers (self-attn + FFN). Decoder = 3 sub-layers (masked self-attn + cross-attn + FFN), each with residual + LayerNorm.

**Residuals** let layers learn deltas. **LayerNorm** normalizes across hidden dim per token. **FFN** ($d→4d→d$, GeLU) transforms each position independently — attention mixes across positions, FFN provides per-token nonlinearity.

**PE:** self-attention is permutation-equivariant ⇒ PE required. Sinusoidal: $\sin(p/10000^{2i/d})$, $\cos(p/10000^{2i/d})$. **Add** (don't concat). Learned PE can't generalize beyond training length. Embedding scaled by $\sqrt{H}$ to balance magnitude vs PE.

**Cross-attn visualization:** `attn_weights` shape `[B,nH,S_tgt,S_src]` — row $i$ col $j$ = how much target token $i$ attends to source token $j$. Different heads show different alignment patterns.

---

## 6. Tokenization

Word-level: OOV, huge vocab, ignores morphology. Char-level: no OOV but sequences too long ⇒ hard to learn word meaning. **Subword** = sweet spot: frequent words stay whole, rare words decompose.

| Method            | Used by    | How it picks merges                                     | Key difference         |
| ----------------- | ---------- | ------------------------------------------------------- | ---------------------- |
| **BPE**           | GPT, Llama | most frequent adjacent pair                             | raw frequency only     |
| **WordPiece**     | BERT       | $\text{freq(pair)}/(\text{freq(a)}\cdot\text{freq(b)})$ | mutual information     |
| **SentencePiece** | mT5        | no pre-tokenization                                     | language-agnostic      |
| **Byte-level**    | ByT5       | UTF-8 bytes (vocab=256)                                 | no OOV ever, long seqs |

_Shortcoming of BPE?_ Greedy merge by raw frequency ignores how informative each token is. _Why does ByT5 use a heavy encoder / light decoder?_ To compensate for much longer byte sequences.

Special tokens: `<pad>`, `<unk>`, `<s>` (BOS), `</s>` (EOS). BPE requires pre-tokenization (whitespace split) ⇒ fails for languages without spaces (Chinese, Thai) unless SentencePiece is used.

---

## 7. Pretrained LMs

**Key idea:** learn a language prior from unlabeled text, then fine-tune. **Contextual** embeddings finally fix polysemy (each token conditioned on sentence).

**ELMo.** "Bidirectional" is **fake**: 2 separate unidirectional LSTMs, concatenated. Embedding = $\gamma \sum_j s_j h_j$ (task-weighted sum across all layers). Lower layers ≈ syntax, upper ≈ semantics. _Why not just use the last layer?_ Different tasks benefit from different layers.

**BERT.** Encoder-only, truly bidirectional. MLM: mask 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged). _Why 10% random + 10% unchanged?_ If model only saw `[MASK]` during pretraining, it's surprised by real tokens at fine-tuning (train/test mismatch). `[CLS]` at **front** (bidirectional ⇒ attends to all equally). 30k WordPiece. **Whole-word masking** prevents trivially guessing from remaining subwords (e.g. `[MASK]bama`). BERT **cannot generate** text — no autoregressive masking. _Why can't BERT use causal LM?_ It would see the target ⇒ trivial copy.

**ELECTRA.** BERT wastes 85% of tokens (learns only from 15% masked). Small generator produces corruptions; discriminator classifies **every** token as real/replaced ⇒ full-signal training, drastically faster.

**GPT.** Decoder-only, causal masking, no cross-attention. `[CLS]` at **end** (left-to-right ⇒ last position has full context). Natively generative.

**BART.** BERT encoder + GPT decoder. Best corruption: text infilling + sentence permutation. For classification: input to both encoder AND decoder. **T5:** all tasks as text-to-text; multi-task pretraining ⇒ better transfer. For translation: prefix `"translate SRC to TGT:"`. **DistilBERT:** 3 losses: MLM + distillation (soft probs from teacher) + embedding cosine. ~97% BERT at 40% fewer params.

**Fine-tuning details:** `lr=2e-5`, `weight_decay=0.01`. When loading pretrained weights: UNEXPECTED keys (e.g. MLM head) are safe to ignore; MISSING keys (classifier head) are freshly initialized — those are what you fine-tune. For sentence pairs: `tokenizer(premise, hypothesis, truncation=True)`.

---

## 8. Text Generation

**Teacher forcing:** train on gold prefix; at inference, model conditions on own outputs ⇒ **exposure bias** (train/test mismatch, errors compound).

**Decoding:**

| Method      | Formula                                       | Tradeoff                                                  |
| ----------- | --------------------------------------------- | --------------------------------------------------------- |
| Greedy      | $\hat y_t = \arg\max P(y_t \mid y_{<t})$      | fast, repetitive                                          |
| Beam($b$)   | keep $b$ top hypotheses                       | ↑BLEU but generic, repetitive                             |
| Sampling    | $\hat y_t \sim P(y_t \mid y_{<t})$            | diverse but incoherent (long tail of ~50k tokens)         |
| Top-$k$     | sample from top-$k$ tokens                    | fixed $k$: too aggressive when flat, too loose when peaky |
| Top-$p$     | sample from smallest set with $\sum p \geq p$ | adapts to distribution shape                              |
| Temp $\tau$ | $P' \propto \exp(S_w/\tau)$                   | $\tau\to0$: argmax, $\tau\to\infty$: uniform              |

**Implementation details:** Top-k: `torch.topk(logits/τ, k)` → softmax → `multinomial`. Top-p: sort descending, cumsum softmax, mask where cumsum > $p$, **shift mask right by 1** so first token exceeding $p$ is kept. Temperature is applied **before** top-k/top-p filtering. `beam=1` = greedy.

**Beam search extras:** `no_repeat_ngram_size=n` sets prob of repeated n-gram to 0 (but breaks proper nouns like "New York"). `repetition_penalty` penalizes all repeating tokens (problematic with BPE — penalizes stopwords). `length_penalty` = score / $\text{len}^\alpha$ (positive $\alpha$ ⇒ longer outputs).

**Repetition trap:** greedy/beam NLL _decreases_ with repetition (Holtzman 2020) ⇒ self-reinforcing loops. Worse for transformers than LSTMs (sharper distributions).

_Why does beam score higher BLEU but worse human judgment?_ Beam maximizes $\log P(Y)$ ⇒ short, generic, high-prob completions. Humans reward informativeness, not likelihood.

**KV Cache:** `use_cache=True` + recycle `past_key_values` ⇒ avoid recomputing all hidden states at each step.

**Evaluation metrics:**

| Metric           | Type                  | What it measures         | Fatal flaw                                           |
| ---------------- | --------------------- | ------------------------ | ---------------------------------------------------- |
| **BLEU**         | n-gram precision      | MT quality               | no semantics ("Yup"=0, "Heck no"=0.67 vs "Heck yes") |
| **ROUGE**        | n-gram recall         | summarization            | same                                                 |
| **BERTScore**    | contextual cosine sim | semantic similarity      | depends on underlying model                          |
| **BLEURT**       | BERT regression       | grammaticality + meaning | needs training data                                  |
| **COMET**        | neural, trained on DA | best human correlation   | requires source+hyp+ref                              |
| **LLM-as-judge** | rubric-based          | flexible                 | position bias, self-preference                       |
| **Human**        | gold standard         | everything               | slow, expensive, inconsistent                        |

_Why not perplexity of generated text?_ Greedy decoders that produce repetitive garbage would score best. Perplexity of _reference_ text measures model calibration, not generation quality.

N-gram metrics get **progressively worse** as tasks become more open-ended: MT → summarization → dialogue → story generation. BLEU needs ≥1 matching 4-gram to be >0. Reordered words get high BLEU-1 but 0 BLEU-4. **Metric-human correlation** measured via **Kendall's Tau** (rank concordance).

---

## 9. RLHF, DPO & Beyond

**Why RL?** MLE optimizes next-token likelihood; human preferences aren't differentiable through discrete samples ⇒ policy gradient.

**REINFORCE:** $\nabla_\theta \mathbb{E}[r] = \mathbb{E}[r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)]$. High variance ⇒ subtract baseline $b$: $(r - b)$ without changing expected gradient. Stabilize: $\mathcal{L} = \mathcal{L}_\text{MLE} + \alpha\mathcal{L}_\text{RL}$.

**RLHF pipeline:** (1) **SFT** on demonstrations → (2) **RM** on preference pairs: $\mathcal{L}_\text{RM} = -\log\sigma(r_\phi(y^w) - r_\phi(y^l))$ (Bradley-Terry) → (3) **PPO**: $\max \mathbb{E}[r_\phi] - \beta\text{KL}[\pi_\theta \| \pi_\text{ref}]$. PPO clips $\pi_\theta/\pi_\text{ref}$ to $[1-\epsilon, 1+\epsilon]$ (REINFORCE updates are unbounded).

**KL term** prevents **reward hacking** — without it, $\pi_\theta$ drifts to OOD text that exploits RM blind spots. _Using a politeness classifier as reward?_ Model learns to add "please" 20× — reward hacking.

**DPO:** $\mathcal{L} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y^w)}{\pi_\text{ref}(y^w)} - \beta\log\frac{\pi_\theta(y^l)}{\pi_\text{ref}(y^l)}\right)$. Eliminates explicit RM and RL loop. Same optimum, simpler.

**DPO implementation:** log-probs via shifted logits: `log_softmax(logits[:,:-1,:])` gathered by `labels[:,1:]`, mask out `-100` positions (prompt tokens). Initial loss ≈ $\log 2 \approx 0.693$ because $\pi_\theta = \pi_\text{ref}$ ⇒ ratios = 0 ⇒ $\sigma(0)=0.5$. Implicit reward = $\beta \log(\pi_\theta / \pi_\text{ref})$. Freeze reference model; disable dropout in both. **Reward accuracy > 0.5** = model correctly ranks chosen over rejected.

**GRPO:** sample $G$ outputs, advantage = $(R_i - \mu)/\sigma$, PPO-style clipping ($\epsilon=0.2$) + KL penalty. KL estimator: $\exp(\delta)-\delta-1$ where $\delta = \log\pi_\text{ref} - \log\pi_\theta$ (always $\geq 0$). **Fails when all $G$ correct OR all wrong** — $\sigma=0$ ⇒ no gradient. GRPO replaces PPO's value network (critic) ⇒ halves memory.

**RLVR (Reinforcement Learning with Verifiable Rewards):** binary reward from programmatic check (math, coding) — no reward model needed. Avoids RM noise and reward hacking.

_Why can't you skip SFT?_ Model doesn't follow instructions; RM trained on instruction-following; near-random policy gives no useful signal. SFT puts the policy where the RM is informative.

---

## 10. ICL, Instruction Tuning & Scaling

**ICL:** task specified via prompt, **zero weight updates**. Model uses demos for task **format/distribution**, not input→output mapping — works even with **wrong labels**. Sensitive to example order.

**Prompting as cloze (PET):** reformulate classification as fill-mask. **Verbalizer** maps labels to words (e.g. `great→positive`). **Template** wraps input with `<mask>`. Verbalizer choice **strongly affects accuracy** (e.g. `great/horrible`=85.8% vs `great/terrible`=88.6% on IMDB). RoBERTa tokenizes with leading space: use `"\u0120"+word` for token lookup.

**Few-shot prompting can hurt:** context label imbalance or dominant lexical cues bias the mask prediction. 10 examples degraded negative accuracy to ~7% in one experiment.

**Zero-shot prompting works** because MLM pretraining already captures sentiment associations ("great" follows positive text in pretraining data). _Why does untrained classifier head give ~33% on NLI?_ Randomly initialized weights ⇒ random predictions.

**Instruction tuning:** fine-tune on (instruction, response) pairs ⇒ zero-shot generalization to _unseen_ task types. Updates weights (unlike ICL).

**CoT:** include reasoning steps in examples ⇒ model generates reasoning before answer. Requires sufficient scale. **Zero-shot CoT:** just append "Let's think step by step." ICL addresses "what to do"; CoT addresses "how to reason" — they solve different bottlenecks.

**Self-Consistency:** sample $N$ responses with $\tau > 0$, extract answer, **majority vote**. Errors are random (spread across wrong answers); correct reasoning converges. Expected: Zero-shot < Zero-shot CoT < Few-shot CoT < Self-Consistency. Cost: $N\times$ compute.

**Efficient FT:** **Adapters** — small FFN inserts between layers. **LoRA** — low-rank $\Delta W = AB$, $r \ll d$, frozen base. **Prompt tuning** — only prompt vectors trained, model frozen.

**Test-time scaling:** trade inference compute for accuracy. More reasoning tokens ⇒ higher accuracy. Thinking budget controls max reasoning; can force-end early.

_Why does RLHF make models sycophantic?_ RM rewards confident answers; KL limits but can't prevent drift; human raters prefer confident wrong over uncertain correct. Calibration is not in the loss.

---

## 11. Dataset Artifacts

**Annotation artifacts:** annotators systematically add negation for contradiction, extra info for neutral. **Hypothesis-only baseline:** 69% on SNLI (vs 33% random), ~57% even with simpler models ⇒ model learns annotator patterns, not reasoning. Few annotators (380 for 402k MNLI examples) ⇒ individual writing style is a learnable signal.

**OOD test (McCoy 2019):** ~100% when OOD labels align with training bias, ~0% otherwise ⇒ pure shortcut exploitation.

**Mitigation:** contrast sets, adversarial filtering, bias-only ensemble models, data augmentation, balanced annotation guidelines.

**Inter-annotator agreement:** Cohen's Kappa (2 raters), Fleiss' Kappa (>2), Krippendorff's Alpha. But filtering by agreement can eliminate legitimate ambiguity.

---

## Concept Chain

```
n-gram (fixed ctx, sparse, no generalization)
  → RNN (shared embeddings, unlimited ctx; vanishing gradients)
  → LSTM (gating, additive cell; still sequential)
  → Seq2Seq+Attn (dynamic ctx, no bottleneck; still O(n) sequential)
  → Transformer (parallel, O(n²); scales to huge data)
  → BERT/GPT (pretraining ⇒ contextual embeddings, fixes polysemy)
  → ICL (inference-time task adaptation, no weight update)
  → Instruction tuning (weight update ⇒ zero-shot generalization)
  → RLHF/DPO (human preference alignment)
```

Each step removes one hard constraint of the previous stage.

---

## Derivatives

| Name                 | $f$                              | $df/dx$                    |
| -------------------- | -------------------------------- | -------------------------- |
| Power rule           | $x^n$                            | $n \cdot x^{n-1}$          |
| General exp          | $a^x$                            | $a^x \cdot \ln a$          |
| Log base $a$         | $\log_a x$                       | $1 / (x \ln a)$            |
| Sigmoid              | $\sigma(x) = 1/(1 + e^{-x})$     | $\sigma(x)(1 - \sigma(x))$ |
| Tanh                 | $\tanh(x)$                       | $1 - \tanh^2(x)$           |
| ReLU                 | $\max(0, x)$                     | $1$ if $x > 0$ else $0$    |
| Softmax              | $s_i = e^{x_i} / \sum_j e^{x_j}$ | $s_i (\delta_{ij} - s_j)$  |
| Log-Softmax          | $\log s_i$                       | $\delta_{ij} - s_j$        |
| **XEnt + Softmax ★** | $-\log s_y$                      | $p_i - y_i$                |
| KL Divergence        | $\sum_i p_i \log(p_i / q_i)$     | $-p_i / q_i$               |

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
