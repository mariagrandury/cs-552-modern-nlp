## 1. Word Embeddings

One-hot $e_w \in \{0,1\}^{|V|}$: orthogonal ⇒ no similarity. Dense $e_w \in R^d$, $d \ll |V|$: geometry encodes meaning (**distributional hypothesis**).
**Skip-gram:** predict context from center $w_t$. Captures **contextual associations**. Dynamic (but fixed) window: sample $i \in [1,N]$, closer words seen more often.
**CBOW:** predict center from summed context ⇒ removes word order (bag-of-words). Captures **substitutable words** (synonyms). Projection $U \in R^{V \times d}$ maps context vector to one score per vocab word, dot product $U \cdot h_t$ gives scores $\in R^V$.
**Negative sampling:** replace $|V|$-softmax with $O(k)$ binary classification.
**GloVe:** $J = \sum f(X_{ij})(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$.
Word2Vec iteratively updates embeds on local contexts, GloVe efficiently leverage global stats once the c-matrix built.

<!-- **FastTex:t** splits words into character n-grams ⇒ handles morphology, typos, OOV. -->

**Static** embeddings: one vector per type ⇒ polysemy unsolved.
Dense (vs sparse): encode similarity, easier to include as ML features, generalize better to rare words. Both no representation for unseen words.

**Backprop**: compute gradients of loss wrt weights through layers, improves time efficiency but increases memory requirements (stores computation map for every partial gradient).
**Logits** = raw unnormalized scores from last linear layer, no prob interpret. **Probs** = softmax(logits): exponentiate all scores (→ positive), divide by sum (→ sum to 1).
**Weight tying:** input embedding matrix and output projection $W_o$ are transposes (vocab→$d$ vs $d$→vocab) ⇒ optimized jointly. Embeddings shared across all instances of same word.

---

## 2. N-gram LMs

Markov: $P(w_t|w_{1:t-1}) \approx P(w_t|w_{t-n+1:t-1})$. Max likelihood estimation (**MLE**): $\hat P (w|c) = C(w,c)/ \sum_i C(w_i,c)$. NB: $P(+|X) = (P(X|+)·P(+))/P(X), P(X)=P(X|-)P(-)+P(X|+)P(+), P(X|+)=\prod P(n-gram|+).

**Perplexity** $= P(W)^{-1/N}$ = exponentiated avg negative loglikelihood (NLL). Uniform baseline: PPL = $|V|$. PPL = k, model being confused among k tokens in the vocab on avg. Easy to compute, correlates with fluency, fitting scaling laws.

<!-- Prob sequence in 1-gram, n-gram -->

**Smoothing:** redistributes prob from seen to unseen patterns (but does not create similarity structure). **Laplace** (add $\alpha$ to every count; games PPL w/o improving LM), **back-off** (fall back to lower n-gram when count is 0), **linear interpolation,** weighted mix of n-gram probs: ($P = \sum_i \lambda_i P_{\text{i-gram}}$), $\sum \lambda_i = 1$. **Kneser-Ney** continuation probability: how many distinct contexts a word appears in, not raw freq.

<!-- **Zipf's law:** word freq ~2× the next ⇒ long tail drives sparsity. -->

**Limits:** sparsity, can't generalize across synonyms, no distributed representation, no cross-word generalization (each word atomic), hard context cap at $n-1$.
**PPL issues:** Can't compare PPL across different vocab sizes (larger vocab ⇒ higher PPL). Domain mismatch inflates PPL. PPL > $|V|$ = worse than random (sanity check).
**Fixed-context neural LM (Bengio):** represent n-gram as NN. Concatenate $n$ embeddings + hidden layer + softmax over $V$. Dims: embed dim, embed dim · input tokens, hidden dim. No sparsity (softmax ≠ 0). Too small for long deps. Fixed window, no weight sharing across positions ⇒ enlarging window explodes $W$.

---

## 3. RNN & LSTM

**RNN:** $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$, shapes: $x_t$`(B,d)`, $h_t$`(B,h)`. Same $W_h, W_x$ reused ⇒ weight sharing = translation invariance. Unlimited context in theory.

**Vanishing gradient:** $\partial\mathcal{L}/\partial h_0 = \prod_{t} \partial h_t/\partial h_{t-1} \to 0$ when product of **activation derivative** (<1, esp. sigmoid) x $W_{hh}$ (small due to regularization) repeated ⇒ early context forgotten.
Exploding: dominant singular value >1 ⇒ fix with **gradient clipping** (rescale when norm > threshold, preserves direction). Sequential ⇒ no parallelism.

**LSTM** — cell uses **addition** ⇒ gradient highway: $\partial c_T/\partial c_t \approx \prod f_\tau \approx \text{const when } f \approx 1$.

```
f_t = σ(W_f·[h_{t-1},x_t])   i_t = σ(W_i·[h_{t-1},x_t])
c̃_t = tanh(W_c·[h_{t-1},x_t])
c_t = f_t⊙c_{t-1} + i_t⊙c̃_t     [ADDITIVE cell update]
o_t = σ(W_o·[h_{t-1},x_t])   h_t = o_t⊙tanh(c_t)
```

$f=1$ always ⇒ unbounded accumulator, never forgets. Forget gate enables selective memory. Addition (not matrix multiply) ⇒ gradients flow without full weight matrix. If $f \approx 1$: constant error carousel.

**GRU:** $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ (z: update, r: reset). Weighted average ⇒ values stay bounded. $z=1$: forget old state, $z=0$: keep old state. Simpler gated RNN, fewer params than LSTM.

**BiRNN:** 2 indep RNNs: forward + backward, output: concatenation of both hidden states. Separate params. Use for classification/labeling, **not** generation.

<!-- each token sees both past and future context -->

**BPTT:** unroll dynamic graph into static (length known from forward pass), apply standard backprop. Same $W_{hh}$ at every step ⇒ chain of multiplications. Gradient computations reuse ("cache") prior layer results.

**LSTM dropout:** standard dropout disrupts cell gradient. Fixes: **variational dropout** (same mask per sequence), dropout only on non-recurrent connections, or **zoneout** (randomly preserve hidden state instead of zeroing).

<!-- **Training:** `outputs[:, :-1, :]` predicts `labels[:, 1:]` (shifted by 1). Use `ignore_index=pad_idx` in CrossEntropyLoss. -->

---

## 4. Seq2Seq & Attention

**Seq2Seq:** encoder compresses source into fixed $c = h_n$ `(B,h)`. Decoder autoregressively conditions on $c$, **cannot be bidirectional**.
**Temporal bottleneck:** single fixed-size state vector for arbitrary-length input.

**Attention:** $e_{t,s} = v^\top\tanh(W_h h^d_{t-1} + W_s h^e_s)$, $\alpha_{t,s} = \text{softmax}_s(e_{t,s})$, $c_t = \sum_s \alpha_{t,s} h^e_s$.
Shapes: $e_t$`(B,n)`, $\alpha_t$`(B,n)`, $c_t$`(B,h)`.
Query-key mechanism: key = encoder hidden state, query = final decoder hidden state, attention = sim score. Fixes bottleneck, soft alignment, direct gradient path enc-dec.
Still $O(n)$ sequential (RNN-based). Cross-attention: $O(n \cdot m)$. Decoder self-attention at inference: $O(n^2)$ ⇒ motivates **KV caching** (only compute new Q).

**Exposure bias:** teacher forcing trains on gold prefix ≠ inference on own outputs ⇒ errors compound. **Scheduled sampling:** gradually replace gold with model predictions during training.

---

## 5. Transformer

$\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. Sequential depth $O(1)$, compute $O(n^2)$.
$/\sqrt{d_k}$: High-dim dots grow large ⇒ softmax saturates to one-hot ⇒ vanishing gradients. Scale by $\sqrt{d_\text{head}}$, NOT $\sqrt{d_\text{model}}$.

**Multi-head:** $\text{head}_i = \text{Attn}(XW^Q_i, XW^K_i, XW^V_i)$, $\text{MHA} = \text{Concat}(\text{heads})W_O$, $h$ heads at $d/h$ dims ≈ same FLOPs as 1 head at $d$. Specialization is **emergent**. `hidden_dim` divisible by `num_heads`. d_model = n_heads \* d_heads

**Shapes:** `[B,S,H]` → `.view(B,S,nH,hD).transpose(1,2)` → `[B,nH,S,hD]` → attn `[B,nH,S,S]` → output `[B,nH,S,hD]` → `.transpose(1,2).contiguous().view(B,S,H)`.

**Masks:**
Padding: `[B,S]`→`[B,1,S,S]`.
Causal: `torch.tril`.
Cross-attn: rectangular `[S_tgt×S_src]` ($Q$ dec, $K$/$V$ enc).
Decoder = causal AND pad.
`masked_fill(mask==0, -inf)` ⇒ softmax → 0. Why not multiply by 0? Remaining scores won't sum to 1.
Causal mask makes training (parallel) and inference (sequential) **computationally identical**.

**Self vs cross:** Encoder self-attn: $Q,K,V$ all from $x$, bidirectional. Decoder self-attn: causal masked. Cross-attn: $Q$ from decoder, $K,V$ from encoder output. Encoder = 2 sub-layers (self-attn + FFN). Decoder = 3 sub-layers (masked self-attn + cross-attn + FFN), each with residual + LayerNorm.
**Cross-attn:** `attn_weights` shape `[B,nH,S_tgt,S_src]

<!-- **Why 3 projections from same $x$?** Without them, self-dot-product dominates ⇒ each token attends mostly to itself. -->

**Residuals** let layers learn deltas. **LayerNorm** normalizes across hidden dim per token. Pre-norm (modern) trains more stably. **FFN** ($d→4d→d$, GeLU) transforms each position independently — attention mixes across positions, FFN provides per-token nonlinearity. `intermediate_size` is usually 4× `hidden_dim`.

**PE:** required bc attention is permutation-equivariant. Sinusoidal $\sin(p/10000^{2i/d})$ or learned. **Add** (don't concat). Learned can't generalize beyond training length. Embedding scaled by $\sqrt{H}$. to balance magnitude vs PE.
**ROPE:** relative positional info by rotating Q&K, generalize better to longer seq than absolute embed.

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

Special tokens: `<pad>` (uniform batch length for GPU processing; unnecessary if batch_s=1, sorted-length seq avoid compute waste), `<unk>`, `<s>` (BOS), `</s>` (EOS, learn when to stop). BPE needs (whitespace) pre-tokenization ⇒ fails without spaces (Chinese, Thai) unless SentencePiece.

---

## 7. Pretrained LMs

**Contextual** embeddings fix polysemy. Pretraining: self-supervised on large corpus (easy objectives, naturally occurring data), then fine-tune.

**ELMo (2018).** Two separate unidirectional LSTMs, hidden states from both dirs concatenated at each pos. "Bidirectional" is **fake**, no shared params between directions. Shared: input embeddings + vocab projection layer between both LSTMs. Generates contextualized embeddings. Embedding = task-weighted sum across all layers; learn params per task, don't update pretrained params. Lower layers ≈ syntax, upper ≈ semantics.

**BERT (2019).** Encoder-only, truly bidirectional. Diff w/ Transformer: **learned** position embeddings + **segment** embeddings (distinguish sentence A/B). Pretraining: Masked Language Modeling (MLM) and next-sentence prediction. mask 15% (80% `[MASK]`, 10% random, 10% unchanged — prevents train/test mismatch). `[CLS]` at front (convention; bidirectional ⇒ any position works). **Whole-word masking** helps named entities (`[MASK]bama`). Cannot generate text. BERT: better to fully fine-tune (vs ELMo: adapt only some params). **Learning**: Heads learn diverse concepts, emergently.
**ELECTRA.** Discriminator classifies **every** token as real/corrupted ⇒ full-signal training (vs BERT's 15%).

**GPT.** Decoder-only, causal masking, no cross-attention. `[CLS]` at **end** (only position with full context). **GPT2:** same arch, larger, strong zero-shot.
**BART.** BERT encoder + GPT decoder. Best corruption: text infilling + sentence permutation. Classification: input to both encoder AND decoder. Handles BERT + generation (autoregressive decoder).
**T5:** seq2seq, read corrupted input bidir + decode masked spans autoregressively, all tasks as text-to-text (prefix).
**DistilBERT,** 3 losses. 1. Masked Language Modeling, MLM: student-predicted masked tokens (predicted labels) vs true labels. 2. Distillation: student vs teacher soft probs (prob distrib over V). 3. Embedding cosine: cos dist between student-teacher sentence embeddings (from the last hidden layer).
**RoBERTa:** BERT trained longer on more data.
**SBERT**: On top of a transformer layer, there is a pooling layer, which pools the token level embeddings into a single sentence level embedding. The network uses a siamese architecture, i.e. 2 sentences are embedded separately from each other (the BERT/pooling layers on either side of SBERT is the same network).
encoder → CLS token; decoder-only → last token or mean/sum-pool.

Inference: `with torch.no_grad(): pred = model(**inputs).logits.argmax(dim=-1)`
do not compute gradients, more efficient.

<!--
**FT details:** When loading pretrained weights, UNEXPECTED keys (e.g. MLM head) are safe to ignore; MISSING keys (classifier head) are freshly initialized - those are what you fine-tune.
Bias: classification head is randomly initialized (before FT), the bias term in the output layer could by chance heavily favor the negative class logit. (DistilBERT exercise)
-->

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

**Beam search:**
`no_repeat_ngram_size=n` sets prob of repeated n-gram to 0
`repetition_penalty` penalizes all repeating tokens (can break named entities, problematic with BPE: penalizes stopwords, plural "s").

<!-- Deterministic search algorithm (prob of a seq being generated is 0 or 1) -->

<!--
_Why does beam score higher BLEU but worse human judgment?_ Beam maximizes $\log P(Y)$ ⇒ short, generic, high-prob completions. Humans reward informativeness, not likelihood.
-->

**Repetition trap:** NLL **decreases** w/ seq length in greedy/beam, NLL decreases w/ repetition ⇒ self-reinforcing loops.

**Re-ranking:** generate multiple sequences, rerank by score. Recalibrate: k-NN, combine with 2nd model (MT).
**KV Cache:** reuse `past_key_values` ⇒ avoid recomputing hidden states at each step. `self.model(**inputs, use_cache=True)`

**Eval metrics.** BLEU (0-1 or 100): n-gram overlap counts (n=1-4, min 1 4-gram match for BLEU>0), MT, no semantics, bad for corpus w/ variable length seq, dep on tokenizer. ROUGE: n-gram recall, summarization, no semantics. METEOR: synonym matching. BERTScore: contextual sim, depends on BERT. BLEURT: BERT regression, grammar + meaning, needs training. COMET: neural, human correlation, requires source+hyp+ref. LLM-as-judge: rubric-based, flexible, position bias, self-preference.
Also, Pyramid (summ), SPICE (captioning), SPIDEr (SPICE+CIDEr), Word Mover's Distance (embedding sim). N-gram metrics degrade as tasks become more open-ended. PPL of generated text measures model calibration, not generation quality (repetition scores well). Humans: never compare across studies, clear guidelines, calibration examples.
Challenge: link evals back to training decisions!

---

## 9. ICL & Prompting

**Emergence:** quantitative changes → qualitative changes. ICL emergent ~175B params.

**ICL:** prompt-based, **zero weight updates**. Uses demos for task format, not input→output mapping — works with wrong labels. Sensitive to example selection/order (worse in SLM). Better for tasks w/ terms frequent in pretraining. Examples required, no 0-shot. **Few-shot can hurt:** context label imbalance or lexical cues bias predictions.

**Cloze prompting, Pattern Exploiting Training (PET):** classification as fill-mask, few-shot learning. Tune linear class head (mostly an MLP layer, attached on the top of original pretrained model) instead of entire model. **Verbalizer** maps labels→words; choice affects accuracy.

**CoT:** reasoning steps before answer, needs scale. Each new token attends to prev gen reasoning, better than all reasoning in 1 forward pass within its hidden states. **Zero-shot CoT:** "Let's think step by step." Few-shot CoT: task + reasoning format. **Self-Consistency:** sample $N$ responses w/ T>0, majority vote, $N\times$ compute, why: errors random but correct reasoning converges.

<!--
Why CoT works: When a model is asked to directly produce an answer to a multi-step problem, it must perform all the reasoning in a single forward pass (i.e., within its hidden states). By generating intermediate steps as tokens, the model gets additional "computation" — each new token can attend to the previously generated reasoning, effectively allowing the model to decompose complex problems into simpler sub-problems.
Why this helps beyond zero-shot CoT: When the model generates its own reasoning structure from scratch (zero-shot CoT), it often makes formatting errors, skips steps, or loses track of intermediate results. The demonstrations provide a template for clean reasoning, reducing these errors.
Self-consistency: This is an ensemble method over reasoning paths. Each sample explores a different trajectory through the reasoning space. Errors tend to be random (spread across wrong answers), while correct reasoning converges (clusters on the right answer).
-->

**AutoPrompt:** gradient-guided search for optimal tokens. Best prompt sometimes random-looking ⇒ foundation of **jailbreaking**. **Soft prompts/prompt-tuning:** instead of discrete tokens, learn continuous prompt representations. Model frozen, only prompt vectors trained.

**Soft prompts or prompt tuning:** init special prompt vector(s), prepend to task example, gradient of loss wrt prompt params, update prompt params (rest frozen). More efficient than full ft, multi-task. Less interpretable than discrete prompts.

**Efficient FT:** keep pretrained params frozen, init new FFN layers and adapt only those. Keep FNN limited in #params (= 2·d·r), r = rank = FNN hidden dim.
**Adapters**: FFN between transformer blocks.
**LoRA**: FFN alongside.

---

## 10. RLHF, DPO, GRPO

IT/RLHF: adapt models to new unseen tasks described in natural language.
Pipeline: 1. train RM from human pref, 2. use RL to optimize a LM against that learned reward. Issues: RM noise (R diff human), RHacking (policy exploits RM weaknesses), RM data costly.

**REINFORCE:** $L_{RL}$ = -R( $\hat Y$) $\sum_t$ log P ( \hat $y_t$ | {x^\*}; { \hat y}{<t} ). Feed into it the LM-generated tokens.
Reward **scales the loss**: high reward → larger loss → learn to reproduce; low reward → loss near 0 → don't update much. High variance ⇒ subtract baseline $b$: $(r - b)$ without changing expected gradient. **Credit assignment:** reward applied at sequence level (hard to assign per-token). **Variance reduction** via baseline (e.g. BLEU 0–100 range). Stabilize: **joint optimization** $L = L*\text{MLE} + \alpha L\_\text{RL}$ (MLE term promotes fluency since RL alone doesn't always generate readable text).

**RLHF pipeline:** (1) **SFT** on demonstrations (instruction, response) (needed bc near-random policy gives no useful RL signal). (2) **RM** on preference pairs: $L_\text{RM} = -\log\sigma(R(Y_+) - R(yY-))$, (train to max difference between rewards = good/bad demonstration) diff → $\inf$, $\sigma$→1, $\log$→0. (3) **PPO**: limitation of deviation between current & base policy by optimizing diff (log div = diff) $\max \mathbb{E}[R] - \beta\text{KL}[\pi_\theta \| \pi_\text{ref}]$. min(diff, clipping f), PPO clips $\pi_\theta/\pi_\text{ref}$ to $[1-\epsilon, 1+\epsilon]$ (REINFORCE updates are unbounded).
L = - min (R $\sum_t$ log (Pcurr / Pbase), clipping f), clipping f = (1-eps)R if R>=0, (1-eps)R if R<0.
**KL term** prevents RH avoiding $\pi_\theta$ drift to OOD text that exploits RM blind spots.

**DPO:** $\mathcal{L} = -\log\sigma \left(\beta\log\frac{\pi_\theta(y^w)}{\pi_\text{ref}(y^w)} - \beta\log\frac{\pi_\theta(y^l)}{\pi_\text{ref}(y^l)}\right)$. Optimize directly on preference data. Learn to do more deviation if positive demo, less if negative. Eliminates explicit RM and RL loop. Less flexible than RLHF in theory, very useful in practice.
**Implementation:** log-probs via shifted logits: `log_softmax(logits[:,:-1,:])` gathered by `labels[:,1:]`, mask out `-100` positions (prompt tokens). Initial loss ≈ $\log 2 \approx 0.693$ because $\pi_\theta = \pi_\text{ref}$ ⇒ ratios = 0 ⇒ $\sigma(0)=0.5$. Implicit reward = $\beta \log(\pi_\theta / \pi_\text{ref})$. Freeze reference model, disable dropout in both. Reward accuracy > 0.5 = model correctly ranks chosen over rejected.

**RLVR:** binary reward from programmatic check, no RM needed ⇒ no RM noise/RH.
**GRPO:** 1. sample $G$ group of outputs (demos), 2. verify demos, 3. reward group computation = (indiv reward - mean R group)/std R group = $(R_i - \mu)/\sigma$ = "advantage". L optimizes ratio of current/base policies $\times$ reward, PPO-style epsilon-based clipping factor to limit deviation of current policy from base (start of step), average over every element in the Group. + KL divergence term limit current/ref (ref=SFT model). KL estimator: $\exp(\delta)-\delta-1$ where $\delta = \log\pi_\text{ref} - \log\pi_\theta$ (always $\geq 0$). Fails when all $G$ correct OR all wrong: $\sigma=0$ ⇒ no gradient. GRPO replaces PPO's value network (critic) ⇒ halves memory. **Differences:** L GRPO = L PPO - beta KL (current/ref). Baseline: PPO learns value function "critic", GRPO group mean reward, GRPO 1 model less in memory. KL constraint: PPO reward penalty, GRPO loss term.
**Test-time scaling:** (RLVR) more reasoning tokens (compute) ⇒ higher accuracy.

<!-- Model doesn't follow instructions; RM trained on instruction-following; near-random policy gives no useful signal. SFT puts the policy where the RM is informative. -->

<!--
_Why does RLHF make models sycophantic?_ RM rewards confident answers; KL limits but can't prevent drift; human raters prefer confident wrong over uncertain correct. Calibration is not in the loss.
-->

---

## 11. Dataset Artifacts

**Mitigation:** contrast sets (more examples), adversarial filtering (weaker model finds spurious examples), bias-only ensemble (update params only for samples where bias model failed, e.g. hypothesis-only for NLI), data augmentation, annotation guidelines.

**Pretraining data matters most.** Quality heuristics can be flawed (e.g. filtering "sex" from C4). Good benchmarks: monotonic, low variance (CommonsenseQA, HellaSwag,OpenBookQA, PIQA). Bad: SocialIQA, TruthfulQA. Benchmarks are aggregations: one problem cascades.

**Inter-annotator agreement:** Cohen's κ (2 raters), Fleiss' κ (>2), Krippendorff's α. But filtering by agreement can eliminate legitimate ambiguity.

<!-- TODO review guest lectures--- -->

## Derivatives

| Name                  | $f$                                         | $df/dx$                                      |
| --------------------- | ------------------------------------------- | -------------------------------------------- |
| Power rule            | $x^n$                                       | $n \cdot x^{n-1}$                            |
| General exp           | $a^x$                                       | $a^x \cdot \ln a$                            |
| Log base $a$          | $\log_a x$                                  | $1 / (x \ln a)$                              |
| Sigmoid               | $\sigma(x) = 1/(1 + e^{-x})$                | $\sigma(x)(1 - \sigma(x))$                   |
| Tanh                  | $\tanh(x)$                                  | $1 - \tanh^2(x)$                             |
| ReLU                  | $\max(0, x)$                                | $1$ if $x > 0$ else $0$                      |
| Softmax               | $s_i = e^{x_i} / \sum_j e^{x_j}$            | $s_i (\delta_{ij} - s_j)$                    |
| **XEnt + Softmax"**   | $-\log s_y$                                 | $p_i - y_i$                                  |
| KL Divergence         | $\sum_i p_i \log(p_i / q_i)$                | $-p_i / q_i$                                 |
| Attention (V)         | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $S^\top \cdot \partial_O L$                  |
| Attention ($QK^\top$) | $\text{softmax}(QK^\top / \sqrt d) \cdot V$ | $J_{\text{softmax}}(\partial_S L) / \sqrt d$ |

"Softmax Jacobian and log cancel → gradient = **predicted − target**. That's why cross-entropy with softmax is the numerically stable default for classification.

---
