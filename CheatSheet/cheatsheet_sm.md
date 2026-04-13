# Modern NLP Cheatsheet

**Embeddings.** One-hot vectors are orthogonal ⇒ no similarity structure. Dense embeddings $\mathbf{e} \in \mathbb{R}^d$ encode meaning via geometry (distributional hypothesis). **W2V Skip-gram:** predict context from center, negative sampling replaces $|V|$-softmax with $O(k)$ binary classification. **CBOW:** predict center from averaged context. **GloVe:** explicitly factorizes log co-occurrence; W2V implicitly factorizes shifted PMI. Both are **static** (one vector per type ⇒ no polysemy).

**N-gram LMs.** $P(w_t|w_{1:t-1}) \approx P(w_t|w_{t-n+1:t-1})$. MLE: $\hat P = C(\text{ctx},w)/C(\text{ctx})$. **Perplexity** $= P(W)^{-1/N}$. **Kneser-Ney:** uses continuation probability (how many _distinct_ contexts). Fundamental limit: atomic symbols ⇒ no generalization across synonyms; smoothing redistributes mass but cannot create similarity.

**RNN.** $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$. Shapes: $x_t$`(B,d)`, $h_t$`(B,h)`. Unlimited context in theory, but **vanishing gradient**: $\partial\mathcal{L}/\partial h_0 = \prod_t \partial h_t/\partial h_{t-1} \to 0$ when $\|W_h\| < 1$. Sequential ⇒ no parallelism.

**LSTM.** Cell uses **addition** not multiplication ⇒ gradient highway: $\partial c_T/\partial c_t \approx \prod f_\tau \approx \text{const}$. Forget gate $f_t$, input gate $i_t$, output gate $o_t$, cell $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, $h_t = o_t \odot \tanh(c_t)$. If $f=1$ always ⇒ unbounded accumulator (forget gate enables selective memory). Still sequential.

**Seq2Seq.** Encoder compresses source into fixed $c = h_n$. Decoder generates autoregressively conditioned on $c$. **Teacher forcing:** train on gold prefix $y^*_{<t}$, not model predictions. **Bottleneck:** fixed-size $c$ has bounded capacity regardless of dim — information-theoretic, not fixable by making $h$ larger.

**Attention.** Dynamic context per decoder step: $\alpha_{t,s} = \text{softmax}(e_{t,s})$, $c_t = \sum_s \alpha_{t,s} h^e_s$. Fixes bottleneck, provides soft alignment, direct gradient path. Still $O(n)$ sequential (RNN-based).

---

**Transformer.** Drop recurrence. $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. Why 3 projections from same $x$? Without them, self-dot-product dominates ⇒ each token attends to itself. Why $/\sqrt{d_k}$? High-dim dots grow large ⇒ softmax saturates ⇒ vanishing gradients.

**Multi-head:** $h$ heads at $d/h$ dims ≈ same FLOPs as 1 head at $d$. Specialization is emergent. Shape: `[B,S,H]` → split `[B,nH,S,hD]` → attn `[B,nH,S,S]` → merge back `[B,S,H]`.

**Masks.** Padding mask `[B,S]` → `[B,1,S,S]` blocks PAD. Causal mask: upper-triangle `[S,S]` blocks future. Cross-attn mask is **rectangular** `[S_tgt × S_src]` (Q from decoder, K/V from encoder). Decoder = causal AND pad. Causal mask makes training (parallel) and inference (sequential) **computationally identical**.

**Residuals** let layers learn deltas (not reconstruct signal). **LayerNorm** normalizes across hidden dim per token. **FFN** ($d→4d→d$, GeLU) transforms each position independently — attention mixes across positions, FFN adds per-token nonlinearity.

**Positional encoding.** Self-attention is permutation-equivariant ⇒ PE required. Sinusoidal: $\sin(p/10000^{2i/d})$. **Add** to embeddings (don't concatenate). Learned PE can't generalize beyond training length. Scaling embeddings by $\sqrt{H}$ balances magnitude vs PE.

---

**Tokenization.** Word-level: OOV + huge vocab. Char-level: no OOV but sequences too long. **Subword** = sweet spot. **BPE** (GPT, Llama): greedily merge most frequent pair — ignores informativeness. **WordPiece** (BERT): scores by $\text{freq(pair)} / (\text{freq(a)} \cdot \text{freq(b)})$ ⇒ mutual information. **SentencePiece**: language-agnostic, no pre-tokenization. **ByT5**: 256 byte vocab, no OOV, but much longer sequences.

---

**ELMo.** "Bidirectional" is **fake**: 2 separate unidirectional LSTMs concatenated. Embedding = task-weighted sum across all layers ($\gamma \sum s_j h_j$). Lower layers ≈ syntax, upper ≈ semantics.

**BERT.** Encoder-only, bidirectional. MLM: mask 15% (80% `[MASK]`, 10% random, 10% unchanged — avoids train/test mismatch). `[CLS]` at front. 30k WordPiece. Whole-word masking prevents trivial subword completion. Cannot generate text (no autoregressive masking). Why can't you train BERT with causal LM? It would see the target token ⇒ trivial copy.

**ELECTRA.** BERT wastes 85% of tokens. Generator corrupts, discriminator classifies every token as real/replaced ⇒ full-signal training, much faster.

**GPT.** Decoder-only (causal masking, no cross-attention). `[CLS]` at end (last position has full context). Natively generative.

**BART.** BERT encoder + GPT decoder. Best corruption: text infilling + sentence permutation. **T5.** All tasks as text-to-text. **DistilBERT:** $\mathcal{L} = \gamma_1 \mathcal{L}_\text{distil} + \gamma_2 \mathcal{L}_\text{mlm}$, ~97% BERT performance at 40% fewer params.

---

**Decoding.** Greedy (argmax): fast, repetitive — NLL _decreases_ with repetition ⇒ self-reinforcing loops (worse for transformers). Beam($k$): top-$k$ hypotheses; ↑BLEU but repetitive, generic. Sampling: diverse but incoherent. **Top-$k$**: fixed $k$ cuts too aggressively when flat, too loosely when peaky. **Top-$p$ (nucleus)**: adapts — sample from smallest set with cumulative prob $\geq p$. **Temperature $\tau$**: $P' \propto P^{1/\tau}$; $\tau \to 0$ = argmax, $\tau \to \infty$ = uniform.

**Exposure bias.** Training sees gold prefix; inference sees own predictions ⇒ error compounding.

**Eval metrics.** BLEU = n-gram precision (MT). ROUGE = n-gram recall (summarization). Both have **no semantic awareness** ("Yup" scores 0 vs "Heck yes", "Heck no" scores 0.67). BERTScore = cosine sim of contextual embeddings. LLM-as-judge: position bias, self-preference. Human eval = gold standard but slow/inconsistent. Perplexity of generated text is circular — greedy decoders would always win.

---

**REINFORCE.** $\nabla_\theta \mathbb{E}[r] = \mathbb{E}[r \cdot \nabla_\theta \log \pi_\theta(y|x)]$. Subtract baseline $b$ for variance reduction. Mix with MLE: $\mathcal{L} = \mathcal{L}_\text{MLE} + \alpha\mathcal{L}_\text{RL}$.

**RLHF.** (1) SFT on demonstrations → (2) RM on preference pairs: $\mathcal{L}_\text{RM} = -\log\sigma(r(y^w) - r(y^l))$ → (3) PPO: maximize $r_\phi$ with KL penalty to $\pi_\text{ref}$. PPO clips ratio $\pi_\theta/\pi_\text{ref}$ to $[1-\epsilon,1+\epsilon]$ (REINFORCE is unbounded). **KL term** prevents reward hacking (exploiting RM blind spots).

**DPO.** $\mathcal{L} = -\log\sigma(\beta\log\frac{\pi_\theta(y^w)}{\pi_\text{ref}(y^w)} - \beta\log\frac{\pi_\theta(y^l)}{\pi_\text{ref}(y^l)})$. No explicit RM or RL loop. Same optimum, simpler.

**GRPO.** Sample $G$ outputs, advantage = $(R_i - \mu)/\sigma$. **Fails when all $G$ correct OR all wrong** (zero variance ⇒ no gradient).

Why skip SFT → PPO fails? Model doesn't follow instructions; RM trained on instruction-following; near-random policy gives no useful signal.

---

**ICL.** Task via prompt, no weight update. Demos specify format/distribution, not input→output mapping (works even with wrong labels!). Sensitive to example order.

**Instruction tuning.** Fine-tune on (instruction, response) pairs ⇒ zero-shot generalization to unseen tasks. Updates weights (unlike ICL).

**CoT.** Reasoning steps in examples ⇒ model generates reasoning before answer. Requires sufficient scale.

**Efficient FT.** Adapters: small FFN inserts. **LoRA**: low-rank $\Delta W = AB$, $r \ll d$. Prompt tuning: only prompt vectors trained.

---

**Dataset artifacts.** NLI hypothesis-only baseline: 69% on SNLI (vs 33% random) ⇒ models learn annotator patterns (negation→contradiction), not reasoning. Mitigation: contrast sets, adversarial filtering, data augmentation. Few annotators ⇒ individual style becomes learnable signal.

---

## Derivatives

| $f$                          | $df/dx$                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| $\sigma(x)$                  | $\sigma(x)(1-\sigma(x))$                                                                                    |
| $\tanh(x)$                   | $1-\tanh^2(x)$                                                                                              |
| ReLU                         | $1$ if $x>0$, else $0$                                                                                      |
| Softmax $s_i$                | $s_i(\delta_{ij}-s_j)$                                                                                      |
| **XEnt+Softmax** $-\log s_y$ | $p_i - y_i$ (predicted − target)                                                                            |
| KL $\sum p_i\log(p_i/q_i)$   | $-p_i/q_i$                                                                                                  |
| Linear (weights) $Wx+b$      | $\delta \cdot x^\top$                                                                                       |
| Linear (input)               | $W^\top \cdot \delta$                                                                                       |
| LayerNorm                    | $(\gamma/\sigma)[\partial L - \text{mean}(\partial L) - \hat{y}\cdot\text{mean}(\partial L \cdot \hat{y})]$ |
| Attn wrt $V$                 | $S^\top \cdot \partial_O L$                                                                                 |

---
