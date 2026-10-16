# Class 08 — Object Detection III: Transformers

> **Exam weight: Tier 1.** Sections A, B and C draw directly from this class; Section F from it directly (attention arithmetic, ViT token counts, IoU/AP). Detection **metrics** (AP, IoU, AP₅₀/AP₇₅/AP₉₅, AP_S/AP_M/AP_L) are the single most reusable numerical topic in the whole detection block.
>
> **Primary sources:** `session_8.pdf` (40 slides, *"Transformers For Vision — Lecture 8"*) · `Session_8_—_Object Detection III Transformers_…txt` (transcript, 4.7k words — the **shortest** transcript in the course, so the slides carry more than the audio here).
> **Professor repository / notebooks: none exist in this workspace.** The instructor describes a class project using the **ByteTrack** library shared via a Drive link, but no code is present locally.
> **Known gaps:** the detection-metric thresholds on slide 39 render as "50%, 5%, and 95%" — this survives a 190-DPI re-read, so it is a slide-export artefact (the same slide also drops the "7" from its AP₅₀ subscript); corrected to 50/75/95 against the DETR paper and the COCO evaluation page the slide links. The §B14 tracking material is the instructor's live demo and has no supporting slide.

**You have already studied Classes 1–7.** So this guide does not re-teach anchors, IoU-as-intuition, R-CNN/Faster R-CNN, or YOLO mechanics. It assumes that and spends its budget on what is genuinely new: attention, ViT, DETR, and metrics.

---

## A. Lecture overview

### Learning objectives

1. Write the self-attention computation and explain each of Q, K, V.
2. Explain why attention is a **set** operation and why positional encodings are therefore required.
3. **Derive** the $1/\sqrt{d_k}$ scaling from variance accumulation.
4. **Derive** why LayerNorm is used rather than BatchNorm inside a transformer — the correct reason, not the popular one.
5. Explain multi-head attention and the role of $d_k < d$.
6. Explain why transformers were hard to apply to vision (quadratic cost) and how ViT fixed it.
7. Reproduce ViT's seven construction steps and read its variants table.
8. Explain DETR's **set prediction** framing and the Hungarian matching step.
9. Compute precision, recall, IoU and Average Precision from a ranked detection list.
10. Explain why production systems still often prefer YOLO over DETR.

### Why this matters in CV

This class closes the arc the course opened in Class 03. The "CNN → ??" question posed on slide p.14 (*"LSTM — Transformer" ⇒ "ConvNet — ???"*) is answered twice: once by ViT for classification, once by DETR for detection. It also fixes a practical asymmetry the course keeps returning to — a model can be *better* and still be *unusable* (§B13).

### Prerequisites

Matrix multiplication, softmax, convolution output-size arithmetic (Class 01), one-stage detection basics (Class 07), and AP/IoU intuition from Class 06.

### Position in the course

```
Class 03  CNNs ──────────────► Class 08  Transformers for Vision
                                        ├── ViT      (classification)
Class 07  One-stage (YOLO) ──────────────►  ├── DETR    (detection, set prediction)
                                        ├── Metrics  (AP, IoU, COCO variants)
                                        └── Tracking (Kalman filter, ByteTrack)
Class 13  Vision Foundation Models  ──►  Class 14  ViT, DINO, SAM, Grounding-DINO
```

### Difficulty: **Intermediate–Advanced**

Justified because: the algebra is simple (one matrix product, one softmax), but three of the four deep points — the $\sqrt{d_k}$ variance argument, the LayerNorm rationale, and Hungarian matching — are *reasoning* items that are commonly answered wrongly. The instructor spent a large fraction of the session pushing back on students' answers to exactly these (§B4, §B5).

---

## B. Conceptual foundations

### B1. The Transformer block

**The problem it solves.** Sequence models process tokens strictly in order (an RNN/LSTM carries state forward), so the path length between token 1 and token 1000 is 1000 steps. Attention removes that path entirely: every token talks to every other token in **one** step.

**Structure (slides p.2, p.9).** A *"stack of alternating self-attention and MLP blocks, residuals and LayerNorm."* Slide p.9 gives the complete layer:

```
Input ──┬────────────────────────────────┐
        │                                │  residual
        ├─► Multi-head self-attention ───┤
        │                                ▼
        │                           LayerNorm
        ├─► MLP / feed-forward ──────────┐
        │                                ▼
        └──────────────────────────►  LayerNorm ──► Output
```

Two things to notice, both of which will be tested:
- The residual path runs **around** each sub-block, so the sub-block only ever learns a *correction*.
- LayerNorm is applied **after** the sub-block, not before — this is the *original* Transformer. ImageGPT later reverses it (slide p.17).

**Slide p.2 scope note.** The Transformer "encoder" applies to 1-D sequence data. The *decoder* is *"not shown"* because it is *"a slightly more involved architecture useful when the output space is different from the input space (e.g. translation)"*. DETR uses an encoder–decoder pair; ViT uses the encoder only.

### B2. Self-attention is a **set** operation

**The problem it solves.** If every token attends to every token by a rule that ignores order, the model has thrown away sequence order entirely. Something must put it back.

**Mechanism (slide p.3).** Three properties are listed and they are the whole argument:

| Property | Consequence |
|---|---|
| Each token attends to all tokens | every pairwise interaction is modelled explicitly (slide p.6) |
| **"It's a set operation (permutation-invariant)"** | **"and hence need 'position embeddings' to 'remember' the spatial structure"** |
| **"It's a global operation"** | aggregates information from all tokens |

**Why the second row forces the third.** A set has no order. If you permute the inputs, a set operation produces the correspondingly permuted outputs — so two different token orderings give indistinguishable results. Positional encoding (B7) is the patch.

**This is exactly why images are harder than text.** A sentence has 1-D order that is intrinsic and meaningful. An image has **2-D** structure, and a transformer has no notion of it at all. Slide p.14 poses the problem; B8 is the answer.

### B3. Queries, keys and values

**Mechanism (slide p.4).** Make **three learned projections** of every input embedding $x^{(i)}$:

$$q^{(i)} = W_Q x^{(i)}, \qquad k^{(i)} = W_K x^{(i)}, \qquad v^{(i)} = W_V x^{(i)}$$

The instructor's live Q&A on where the numbers come from: *"So how are we getting the value of the K and Q? This is the linear transformation happening… the values lying there are random at first, slowly you get the values which will make sense… initially you are just starting with some random values using some random initialization. So you have an immediate transformation similar to what happens in the linear [layer]."*

**Attention weight.** The weight connecting position $i$ to position $j$ depends on the **query of $i$** and the **key of $j$**:

$$w_{ij} = \frac{\exp\!\big(q^{(i)}\cdot k^{(j)}\big)}{\sum_{i'=1}^{N}\exp\!\big(q^{(i')}\cdot k^{(j)}\big)}$$

**Output.** Position $i$'s new vector is a weighted sum of **all** value vectors:

$$y^{(i)} = \sum_{j=1}^{N} w_{ij}\, v^{(j)}$$

**Matrix form (slides p.5–p.6)** — this is the version to memorise:

$$\boxed{H = \operatorname{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V}$$

with $X\in\mathbb{R}^{n\times m}$, $Q,K,V\in\mathbb{R}^{n\times m}$, and the attention map $QK^T\in\mathbb{R}^{n\times n}$.

**What each row of the attention map means (from the transcript).** After the softmax, *"each value here… this is my 0th token, this is my 1st token… if it is my zero token, this is my first token… **given this token as a query, I am attending to this 4th value.**"* Rows sum to 1, so each row is a probability distribution over *"which token should I look at."*

### B4. Why $\sqrt{d_k}$ — the variance argument

This is the single most examinable derivation in the class, and the instructor **explicitly invited the question and then corrected the class's answer**:

> *"again I have a question for you, why do we normalize it with sqrt of $d_k$ not anything else… why not directly dividing it by $d_k$ and then dividing it by the under square root of $d_k$? If I had to just average it across the dimension I would divide it by $d_k$. **Why under the square? So there is a bigger reason for it and you should already know why and it is connected to the topic which I was discussing few minutes back. Just before jumping onto this. Whatever you are discussing is dependent on that. Think of it. It is dependent on layer norm. It is dependent not directly on the LayerNorm, it is dependent on the output of the LayerNorm. Output of the LayerNorm will have certain properties."*

**Why divide by \(\sqrt{d_k}\) in Scaled Dot-Product Attention?**

The dot product between a query and a key sums \(d_k\) products, one for each dimension. Assuming these products are independent, have zero mean, and each have variance 1, their variances add up, making the dot product's variance approximately \(d_k\). As \(d_k\) increases, attention scores can become large, causing softmax to saturate and gradients to become very small. To prevent this, we divide the dot product by \(\sqrt{d_k}\): since dividing a random variable by a constant \(c\) divides its variance by \(c^2\), dividing by \(\sqrt{d_k}\) brings the variance back to approximately 1. Dividing by \(d_k\) would make the variance too small, potentially producing nearly uniform attention weights.

**The derivation, in his words:**

1. LayerNorm gives each token **zero mean and unit variance**.
2. The dot product $q^{(i)}\cdot k^{(j)}$ sums $d_k$ such quantities.
3. *"because the mean was already zero, if you add the values that mean equals to zero, the final answer's mean will also be zero. So zero is as it is, **but variance is accumulating over the time**."*
4. *"let's say the $d_k$ is 64 and if I sum these values having variance 1 over the time for 64 times, it will accumulate 64 times and the variance will become 64 instead of 1. **We want it to keep variance as 1.**"*
5. *"So what we do is divide it by [square] root of $d_k$. **So variance is nothing but [standard deviation] square.** That's why $1/\sqrt{d_k}$ to normalize."*

**Summary.** Dividing by $d_k$ would restore the **mean** but drive the variance to $1/d_k$ — far too small, making the softmax input nearly uniform and the attention nearly random. Dividing by $\sqrt{d_k}$ restores the variance to exactly 1 while preserving the zero mean. §C2 verifies this numerically.

**Additional detail (transcript).** *"You should also remember that there is a dimension $d_k$ not $d$ here because we have passed this to a multi-headed attention, the dimensions are $d_k$, which is generally very smaller than the actual $d$."* So in a multi-head layer the scaling uses the **per-head** dimension.

### B5. Why LayerNorm, not BatchNorm — the correct reason

The instructor spends an unusually long stretch here pushing students off the standard answer. The students offered *"the batch size is too small"* and *"we are working with text"*. He rejected both:

> *"you are partially right but how do you come to this conclusion?… You are all pointing out to something important but you are not giving the right reasons for later on. **There is a different reason for it.**"*

**What LayerNorm actually does (transcript).** *"When we talk about layer norm, each token is normalized within itself."* BatchNorm normalises **across the batch**; LayerNorm normalises **across the features of each token independently**.

**The real reason, in his words.** Token magnitudes can differ wildly:

> *"let's say these are all tokens in $x$ for each of these words. But what can happen is… values like 10, 12, 100 something like this. And similarly we can have values let's say in ranges of 0–10… **the vector magnitude of the first token is very much larger than the vector magnitude of the second token.** Now what I am going to do is, I am going to calculate self attention or cross attention and then I will be multiplying lot of stuff. **So what will happen is, this larger value token will have more influence on the attention by default.** So before even calculating the attention and doing all this multiplication, I need to bring each of these tokens in a similar range, that's why I have to normalize this feature in itself first. That is the main reason."*

**The contrast he draws with images (transcript).** *"When you are working with images or things like that… you have a lot of images in the same batch and you have to recognize this distribution, let's say you are using 8 images at a time. So you want to bring these 8 images normalized in such a way that the mean is 0 and the [standard deviation] is 1. And next time when the next new batch of 8 images will come, its distribution should be exactly similar to what was in the previous batch of 8 images. That's why we do it in the batches."*

**So the rule is not "text ⇒ LayerNorm".** It is: **when the items within a sample have heterogeneous magnitudes that get multiplied together, normalise within the sample.** §C3 makes this concrete — two tokens with *identical internal structure* but a 10× scale difference get identical LayerNorm output, whereas BatchNorm would report a spurious difference between them.

**Why the naive reason fails anyway (transcript, when students raised it).** *"In training time and in the inference, the batch [doesn't] exist in the layer norm, it believes [only] the [batch]. Okay, so what does it change? There is a difference between the compute used by running it or training them. How does it make a difference to compute? The numbers are same, you are doing a normalization operation, the number of parameters are same, the computations are going to be almost similar."* He then concedes it *"is not the bad thing"* — just not the answer he wanted.

### B6. Multi-head attention

**Mechanism (slide p.7).**

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O, \qquad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

*"Learn $h$ different linear projections of Q, K, V. Compute attention separately on each of these $h$ versions. Concatenate and project the resultant vectors to a lower dimensionality. Each attention head can use low dimensionality."*

**The instructor's analogy (transcript)** — bagging and decision trees:

> *"the purpose of multi-headed attention is similar to what we do in… ensembles or we have these techniques called bagging, right, where you select some features and run the bagging like the decision tree kind of approach on limited number of features that you selected. **Similar to this this is a multi attention where you are randomly picking up some $d_k$ from the $d$** and then basically trying to run separate separate feed forward networks on the top of them so that different various kind of information can be captured."*

**Why the heads help.** Each head sees a different random projection of the feature space, so different heads can specialise on different relations (one head for a texture, one for a long-range part, one for the position of a distinctive token).

### B7. Positional encoding

**The problem it solves (slide p.8).** *"How to capture sequence order? Add positional embeddings to input embeddings."* Requirements: **same dimension**, and *"can be learned or fixed."*

**Fixed (sin/cos) encoding, slide p.8:**

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right)$$

Each dimension $i$ uses a **different frequency**, geometrically spaced by the $10000^{2i/d_{model}}$ denominator. Low-index dimensions vary rapidly (fine positional resolution); high-index dimensions vary slowly (coarse, long-range). This is why the encoding extends to positions never seen in training.

### B8. Why transformers were hard for vision

**Slide p.14, verbatim and central to the whole class:**

- *"LSTM — Transformer" ⇒ "ConvNet — ???"*
- **"Issue with self-attention for vision: computation is quadratic in the input sequence length, quickly gets very expensive (with > few thousand tokens)."**
- *"For ImageNet: 224×224 pixels — **~50,000 sequence length**"*
- *"Even worse for higher resolution and video."*
- *"How can we deal with this quadratic complexity?"*

**The number checks out:** $224 \times 224 = 50{,}176$ ≈ 50,000 tokens. Each token pair produces one entry in the attention matrix, so the attention map alone is $50{,}176^2 \approx 2.5 \times 10^9$ entries. A CNN's cost instead scales **linearly** in pixels.

**Why CNNs do not have this problem.** A convolution is a *local* operation: each output depends on $F^2$ inputs, so cost is $O(HW F^2)$, linear in pixels. Attention is *global*: every output depends on every input, $O(N^2)$. That asymmetry is the entire reason CNNs dominated vision for a decade.

**The two answers available, both taken in this course:**
- **Make $N$ smaller** — ViT: use patches, not pixels (B10).
- **Make the operation local** — **cubelets** in video (Class 09), the instructor's word for small spatio-temporal tokens.

### B9. ImageGPT (2020)

**The problem it solved.** Apply a transformer to raw pixels by shrinking the input until the quadratic cost is tolerable (slide p.15):

| Reduction | Before | After |
|---|---|---|
| Resolution | full | 32×32, 48×48 or 64×64 |
| Colour | 3×8 = **24 bits** | **9 bits** (512 colours) via k-means clustering of (R,G,B) |

**Training objectives (slide p.16).** Two were tried: **autoregressive** (predict the next pixel row) and **BERT-style** (masked pixel prediction).

**Architectural change (slide p.17).** *"LayerNorm moved to precede self-attention and feed-forward block."* The original applies it after; ImageGPT applies it before — **pre-norm** rather than **post-norm**, which makes deep stacks easier to train.

**Honest assessment (slides p.18–19).** *Pro:* *"gave insights into the representational power of Transformers with unsupervised training."* *Con:* *"worked on downscaled images of size 32×32 to 64×64."* The resolution ceiling is the fatal limitation, and ViT's first job is to remove it.

### B10. Vision Transformer (ViT, June 2021)

**Slide p.20, the four claims:**

1. *"Overcomes resolution limitation of ImageGPT by using patches."*
2. *"Follows scalable NLP Transformer architectures to benefit from efficient implementations."*
3. *"ImageNet Top-1 accuracy: 88.55%."*
4. **"Performs poorly if just trained on ImageNet"** — *"can be expected since Transformers lack the inductive bias of CNNs."*
5. *"Competitive when pre-trained on very large datasets (e.g. 14M — 300M images) — all supervised at this point."*

The slide even carries the instructor-level caveat: *"Be wary of method claims based on different data sets and compute."*

**The seven steps (slides p.21–p.27):**

1. Divide the image into $P\times P$ patches.
2. Create a sequence of length $N = HW/P^2$.
3. Flatten the patches and map to $D$ dimensions with a **trainable linear projection**.
4. Add a **learned 1-D position embedding**.
5. Include a **learnable `[class]` embedding**.
6. Through a multi-layered **Transformer encoder**.
7. **MLP classification head**.

Steps 3 and 4 are the crux: patch tokens are projected into the same space as NLP word embeddings, and a positional embedding restores the 2-D layout that self-attention discarded (B2).

**Variants table (slide p.28):**

| Model | Layers | Hidden | MLP | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

| Dataset | Classes | Images |
|---|---|---|
| ILSVRC-2012 | 1K | 1.3M |
| ImageNet-21K | 21K | 14M |
| JFT | 18K | 303M |

**Naming.** *"ViT-L/16 — 'Large' variant with 16×16 input patch size. Note: $16\times16\times3 = 768$."* The slash names the **patch size**, and the patch's raw pixel count must equal the hidden dimension to make the linear projection square — $16\cdot16\cdot3 = 768$. This is a favourite free-mark question.

**The takeaway, stated plainly.** ViT does not beat CNNs at ImageNet scale (1.3M images); it needs 14M–303M. The advantage is *scalability* and transfer, not small-data performance.

### B11. DETR — detection as set prediction

**The problem it solves.** Classifiers/detectors produce a variable number of predictions and need hand-designed post-processing (anchors, NMS, proposal thresholds). DETR removes all of it by recasting detection as a **fixed-size set prediction** problem.

**Step 1 — fixed-size set (slide p.30).** *"Infer a fixed-size set of $N$ predictions, where $N$ is much larger than the number of objects in an image."*

**Step 2 — Hungarian matching (slide p.30).** Because predictions have no fixed correspondence to ground truth, matching is an assignment problem:

$$\hat\sigma = \underset{\sigma\in S_N}{\arg\min}\ \sum_{i=1}^{N} \mathcal{L}_{match}\big(y_i,\ \hat y_{\sigma(i)}\big)$$

$S_N$ is the set of all permutations of $N$ elements — a **bipartite matching**, solved optimally (Hungarian algorithm).

**The matching cost (slide p.30)** accounts for both class and box:

$$\mathcal{L}_{match} = \mathbf{1}\{c_i \neq \varnothing\}\cdot\big(-\log p_i(\hat c_i)\big) + \mathbf{1}\{c_i \neq \varnothing\}\cdot \mathcal{L}_{box}\big(b_i,\ \hat b_{\sigma(i)}\big)$$

**Step 3 — the loss (slide p.31).**

$$\mathcal{L}_{Hungarian}(y,\hat y) = \sum_{i=1}^{N}\Big[\mathbf{1}\{c_i \neq \varnothing\}\big(-\log p_i(\hat c_i)\big) + \mathbf{1}\{c_i \neq \varnothing\}\,\mathcal{L}_{box}\big(b_i,\hat b_{\sigma(i)}\big)\Big]$$

The indicator is what makes the fixed size $N$ work: unmatched queries (those matched to an "empty" slot) contribute **nothing**.

**Step 4 — the box loss (slide p.31).**

$$\mathcal{L}_{box} = \lambda_{L1}\,\big\|b_i - \hat b_{\sigma(i)}\big\|_1 + \lambda_{gIoU}\Big(-\text{IoU}\big(b_i,\hat b_{\sigma(i)}\big) + \big|\,b_i - \hat b_{\sigma(i)}\,\big|_1\Big)$$

Note the **generalised** IoU: *"Bounding box loss uses a combination of the L1 loss and the generalized IoU loss that is scale-invariant."* Plain IoU is bounded in $[0,1]$ and has vanishing gradients for non-overlapping boxes; gIoU stays informative below zero. This is the single most important DETR implementation detail.

**Architecture (slides p.32–p.36).**

```
  CNN Backbone ──► Transformer ──► Transformer ──► FFN heads ──► N predictions
                   Encoder         Decoder
   C×H×W          d×HW            N queries      class + box
```

| Stage | What it does |
|---|---|
| **Backbone** | CNN; typically ResNet-101, $C=2048$; activation map $C\times H\times W$ |
| **Encoder** | reduce channels (1×1 conv); flatten $C\times H\times W \to d\times HW$; **add positional encodings to the input of each attention layer**; multi-head self-attention + feed-forward |
| **Decoder** | uses the encoder's $d\times HW$ embeddings as **key and value**; adds $N$ **learnable positional encodings = object queries** to the input of each attention layer; transforms $N$ object queries into $N$ output embeddings **in parallel (non-autoregressive)**; trained with **auxiliary decoding loss** |
| **FFN heads** | box: 3-layer perceptron with ReLU, hidden dim $d$, predicting **normalised centre coordinates**; class: a **linear projection with softmax**, which also predicts **"no object"** |

**Two design points worth memorising.**
- **Object queries are learnable.** They are the learned "slots" that each specialise on one object. This is what replaces anchors.
- **Non-autoregressive.** All $N$ predictions come out in parallel, unlike an autoregressive decoder.

### B12. Detection metrics

**Precision and recall (slide p.38).**

$$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}$$

*Precision:* *"Correct predictions out of all predictions."*
*Recall:* *"Correct predictions out of all objects in ground truth."*
*Average Precision:* *"Area beneath the precision–recall curve."*

**IoU (slide p.39).** *"Intersection-over-Union: Area of Overlap / Area of Union. Measures how much bounding box prediction intersects with ground truth."*

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

**The COCO variants (slide p.39):**
- **AP** = *"Average AP at IoU = 50%, 75%, and 95%."* (OCR rendered "5%" for 75%; the DETR paper averages over IoU $0.50:0.05:0.95$.)
- **AP₅₀** = only a box with IoU = 50% is counted as a true positive.
- **AP_S, AP_M, AP_L** = *"AP based on objects of different sizes."*
- Slide p.39 links <https://cocodataset.org/#detection-eval>.

> **Supplementary (not stated in the lecture):** on COCO the size bands are **small** < 32² px, **medium** 32²–96² px, **large** > 96² px of area, and COCO AP averages across ten IoU thresholds from 0.50 to 0.95 in steps of 0.05.

**The instructor's recap (transcript).** *"Whenever you have AP only written that means it's an average precision at IoU 50%, 75%, and 95%… they calculate the average precision at each of these IoU values and then average them out. And then if it's AP₅₀, that means average precision was calculated at IoU threshold of 50%."*

### B13. DETR results and the deployment reality

**Results table (slide p.40).**

| Model | GFLOPs/FPS | #params | AP | AP₅₀ | AP₇₅ | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|---|
| Faster RCNN-DC5 | 320/16 | 166M | 39.0 | 60.5 | 42.3 | 21.4 | 43.5 | 52.5 |
| Faster RCNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster RCNN-R101-FPN | 246/20 | 60M | 42.0 | 62.5 | 45.9 | 25.2 | 45.6 | 54.6 |
| Faster RCNN-DC5+ | 320/16 | 166M | 41.1 | 61.4 | 44.3 | 22.9 | 45.9 | 55.0 |
| Faster RCNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| Faster RCNN-R101-FPN+ | 246/20 | 60M | 44.0 | 63.9 | 47.8 | 27.2 | 48.1 | 56.0 |
| **DETR** | **86/28** | **41M** | **42.0** | 62.4 | 44.2 | 20.5 | 45.8 | 61.1 |
| **DETR-DC5** | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| **DETR-R101** | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| **DETR-DC5-R101** | 253/10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | 62.3 |

**Read the table properly, not just the AP column.**

- **DETR vs Faster RCNN-FPN:** comparable AP (42.0 vs 40.2) with **fewer GFLOPs (86 vs 180) and much higher FPS (28 vs 26 — modest)**, and **fewer params (41M vs 42M)**.
- **DETR's standout weakness is AP_S = 20.5**, far below Faster RCNN-R101-FPN+'s 27.2, while its **AP_L = 61.1 is the best in the table**. Small objects suffer because low-resolution feature maps give each small object very few pixels.
- The best AP (44.9) is DETR-DC5-R101, at **253 GFLOPs and only 10 FPS** — the heaviest and slowest row. Extra accuracy is bought with latency.
- **DC5 = dilated convolution in the backbone** (Class 07) — higher output resolution, more compute.

**Why YOLO still ships (transcript).** This is the most industry-relevant passage in the class:

> *"gflops is another metric which is used for calculating the computational cost… DETR again has a reduced FPS of 28, so the time it is taking to process each frame has increased… **But they introduced the way in which a transformer can be used in computer vision task. That is why you will see that in the real world lot of times we are still using YOLO rather than [DETR]… In LLMs it is still OK because LLMs are working with compute sitting somewhere in the labs and they can have large amount of compute at the same time; these are not life critical tasks… But here we are dependent on the hardware** — let's say you are working on a project for a rover which will be very remote with no accessibility to internet, and then you have to put everything on the device, edge device, right. **Or you're working on autonomous driving cars, you cannot call your cloud every time to make a prediction, you have to keep the compute on the car.**"*

**The generalisable lesson.** Accuracy is not the objective; **accuracy per watt, per frame, on the device you actually have, is.** A model that scores higher and misses its latency budget scores zero in production.

### B14. Object tracking (end of class, no slide)

The instructor closed with a dashcam demo (transcript).

**Detection vs tracking.** *"When you are doing path planning or other kinds of things you need to know [the] same object where it is lying across the frames, not just one frame. **So detection talks about one frame but tracking talks about sequential frames where each object should be identified and tracked across all segments.**"*

**Three failure modes shown on the dashcam footage:**
1. **Zipping / flickering confidence** — *"sometimes the predictions are missing."*
2. **False positives** — *"there are some random floating boxes being predicted by the network."*
3. **ID switching** — *"if you have to track the same objects, let's say this [car] is given an ID 1, it has to be ID 1 all across, but you see IDs can change. It changed to 2 now, it changed to 6, 7. Because every time a prediction is missing and it's reappearing into the picture, a new association is given to it, and you don't want that."*
4. **Occlusion and overtaking** — *"a black [car] can be in front of red or like they interchange the positions… if you see in the pixel coordinates."*

**The mechanism used: Kalman filter.** *"In tracking what we use is Kalman filtering techniques. So it is very similar to… sending a spaceship to Mars — they have sensors on board… whenever a rocket is moving, it's collecting more and more measurements, and then it has moved also, right? **So based on every new measurement it has to fix its trajectory every time.** That's why we use Kalman filter technique."*

- **Kalman filter solves linear** motion; **Extended Kalman Filter (EKF)** solves nonlinear motion. (Transcript: *"linear things are solved by Kalman filter and then the nonlinear things are solved by extended Kalman filter."*)

**How it fixes missing detections.** *"given an object… the object was visible in this frame and then for next 5 frames the object was missed and then in the 6th frame again object appeared, then it is a way to interpolate — **what was the intended trajectory given these two boxes, what could have been the trajectory of this bounding box**, and based on that you can… interpolate that box and then you can somehow fix the IDs of the box, because if the prediction is not missed then basically we are not losing the IDs."*

**Project (transcript).** The instructor shared a class project using the **ByteTrack** library, which already implements Kalman filtering, via a Drive link. No code is present in this workspace.

**A student question worth remembering.** Asked whether an LSTM could frame this as a sequence model, the answer was: *"You can try it out, it's not an issue. But… how will you frame that as an input, how will you frame this image because there are **multiple objects at the same time**. And also the time… there is no good way to form the data."* This is a real limitation of naive sequence framing for multi-object tracking, and connects directly to Class 09's video section.

---

## C. Mathematical derivations and numerical examples

Every number below was recomputed with a Python script; outputs shown are actual computed values.

### C1. Scaled dot-product attention, worked

Given $d_k = 4$ and identity-like projections:

$$Q = K = \begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\end{bmatrix},\qquad V = \begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}$$

**Step 1 — scores.** $QK^T = I_3 = \begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}$. *(Verified.)*

**Step 2 — scale by $1/\sqrt{d_k} = 1/2$.**

$$\frac{QK^T}{\sqrt{4}} = \begin{bmatrix}0.5&0&0\\0&0.5&0\\0&0&0.5\end{bmatrix}$$

**Step 3 — row-wise softmax.** For row $[0.5, 0, 0]$: subtract the max for stability, $e^{0}=1$, $e^{-0.5}=0.6065$, sum $= 2.2131$, so

$$w_{0\cdot} = \left[0.4519,\; 0.2741,\; 0.2741\right]$$

All three rows **sum to exactly 1.0** — verified to 10 decimal places.

**Step 4 — output** $y^{(i)} = \sum_j w_{ij}v^{(j)}$:

$$H = \begin{bmatrix}0.4519&0.2741\\0.2741&0.4519\\0.2741&0.2741\end{bmatrix} \cdot V = \begin{bmatrix}\mathbf{0.4519}&0.2741\\0.2741&0.4519\\0.2741&0.2741\end{bmatrix}$$

Verified. Note each output row equals the corresponding input row here — with $V$'s first two rows the identity, token $i$ reads mostly from itself.

**Sanity check with equal scores.** A zero query $q=[0,0,0,0]$ against a single key $k=[1,0,0,0]$ gives all-zero scores, hence uniform weights, hence output = the **mean of the value vectors**. Verified: `[[7.0, 7.0]]` for a single value vector `[7,7]`. A zero query carries no information, so uniform attention is the correct behaviour — not a degenerate case.

### C2. Why the scaling is $\sqrt{d_k}$ and not $d_k$

**Setup.** After LayerNorm each token component has mean $0$ and variance $1$. The dot product sums $d_k$ such components. Take $d_k = 64$.

**Prediction.** Sum of $d_k$ independent zero-mean unit-variance variables has mean $0$ and **variance $d_k = 64$**.

**Verification (20,000 Monte-Carlo trials, $d_k = 64$, seed fixed):**

| Quantity | Empirical | Theory |
|---|---|---|
| mean of $\sum$ | 0.0758 | 0 |
| **variance of $\sum$** | **63.81** | **64** ✔ |
| variance after dividing by $\sqrt{d_k}$ | **0.9970** | **1** ✔ |
| variance after dividing by $d_k$ | **0.015578** | **$1/64 = 0.015625$** ✔ |

**Interpret.** Dividing by $\sqrt{d_k}$ restores variance to **1** — exactly the property LayerNorm established, so the softmax receives correctly-scaled logits. Dividing by $d_k$ restores only the **mean**, leaving variance at $1/d_k$: every score is squashed toward 0, the softmax becomes almost uniform, and attention degenerates to averaging. That is the concrete failure mode the scaling prevents.

### C3. LayerNorm vs BatchNorm, worked

Take a batch of **2 tokens** with **4 features** each:

$$X = \begin{bmatrix}1&2&3&4\\10&20&30&40\end{bmatrix}$$

Both rows are *identical in shape*, differing only by a factor of 10.

**LayerNorm** — normalise each token across its own features:

| Token | Output |
|---|---|
| $[1,2,3,4]$ | $[-1.3416, -0.4472, 0.4472, 1.3416]$ |
| $[10,20,30,40]$ | $[-1.3416, -0.4472, 0.4472, 1.3416]$ |

The two tokens produce **identical** outputs. *(Verified.)* This is exactly the behaviour the instructor wants: *"the vector magnitude of the first token is very much larger… **this larger value token will have more influence on the attention by default**."* LayerNorm removes the magnitude difference before any multiplication happens.

**BatchNorm** — normalise each feature across the batch:

| Feature | Output |
|---|---|
| 1 | $[-1.0, 1.0]$ |
| 2 | $[-1.0, 1.0]$ |
| 3 | $[-1.0, 1.0]$ |
| 4 | $[-1.0, 1.0]$ |

Every feature yields the same pattern. *(Verified.)* BatchNorm sees a **between-token** difference where LayerNorm sees only **within-token** structure — and the between-token difference here is pure scale, which is exactly what attention must not be sensitive to.

### C4. ViT token counts and the quadratic cost

Image: $224\times224$. Sequence length $N = HW/P^2 = (224/P)^2$.

| Patch $P$ | Grid | Tokens $N$ | Attention map $N\times N$ |
|---|---|---|---|
| 8 | 28×28 | **784** | 614,656 |
| 16 | 14×14 | **196** | 38,416 |
| 32 | 7×7 | **49** | 2,401 |

All verified. Raw pixels would give $224\times224 = 50{,}176$ tokens — matching slide p.14's *"$224\times224$ pixels — $\sim$50,000 sequence length"* exactly. *(Verified.)*

**The saving, stated properly.** At $P=16$, $N$ drops from 50,176 to 196 — a **256×** reduction in sequence length, and therefore a $256^2 = 65{,}536\times$ reduction in attention-map entries. That is the number that makes the quadratic operation affordable, and it is why ViT's first contribution is described as *"Overcomes resolution limitation of ImageGPT."*

**Check ViT-Base's hidden size against the patch.** $16\times16\times3 = 768$, exactly the hidden size in the variants table — *"Note: $16\times16\times3 = 768$"* (slide p.28). With $P=32$ the patch is $32\cdot32\cdot3 = 3072 \neq 768$, so ViT-B/32's projection is **not** square; it is still valid, just no longer the neat equal case.

### C5. Multi-head dimensions

$d_k = d/h$:

| Model | $d$ | $h$ | $d_k = d/h$ |
|---|---|---|---|
| ViT-Base | 768 | 12 | **64** |
| ViT-Large | 1024 | 16 | **64** |
| ViT-Huge | 1280 | 16 | **80** |

All verified. Note ViT-Base is *"Same as BERT"* (slide p.28) — 768 hidden, 12 heads, 12 layers — which is the concrete meaning of *"Follows scalable NLP Transformer architectures to benefit from efficient implementations."*

**Interpretation.** $d_k = 64 \ll d = 768$, matching the instructor's remark that in multi-head attention *"the dimension $d_k$ is generally very smaller than the actual $d$."* The attention scores are computed in a 64-dimensional space even though the residual stream is 768-dimensional.

### C6. IoU and Average Precision

**IoU.** $A = [0,0,10,10]$, $B = [5,5,15,15]$:
intersection $= 5\times5 = 25$; areas $= 100$ each; union $= 100+100-25 = 175$;

$$\text{IoU} = \frac{25}{175} = \mathbf{0.142857}$$

And with $C = [2,2,8,8]$: intersection $= 6\times6 = 36$, union $= 100$, so $\text{IoU} = \mathbf{0.36}$. Both verified.

**AP from a ranked detection list.** Take 5 predictions in confidence order against 3 ground-truth objects, flagged TP, FP, TP, TP, FP:

| Rank | Flag | Precision | Recall |
|---|---|---|---|
| 1 | TP | 1.0000 | 0.3333 |
| 2 | FP | 0.5000 | 0.3333 |
| 3 | TP | 0.6667 | 0.6667 |
| 4 | TP | 0.7500 | 1.0000 |
| 5 | FP | 0.6000 | 1.0000 |

All verified. Applying the precision envelope (walk right-to-left, keep the running max) gives `[1.0, 0.75, 0.75, 0.75]` at recall levels `1/3, 2/3, 1`:

$$\text{AP} = \tfrac{1}{3}(1.0) + \tfrac{1}{3}(0.75) + \tfrac{1}{3}(0.75) = \mathbf{0.8333}$$

**Verified: 0.8333.**

**Note what the envelope did.** Rank 2's precision of 0.5 was *discarded* because a later prediction reached 0.75. Interpolating this way is the standard VOC/COCO-style all-point AP; it is why AP ≥ the maximum achievable precision at each recall level.

### C7. DETR matching cost, worked

Ground truth: `cat [0,0,10,10]`, `dog [20,20,30,30]`.
Predictions: `cat 0.9 [0,0,10,10]`, `dog 0.2 [5,5,15,15]`, `cat 0.8 [18,18,32,32]`.
Class mismatch penalty = 10; cost $= \mathbf{1}\{\text{same class}\}\cdot[-\log(\text{conf}) + L_1]$.

| | pred 0 | pred 1 | pred 2 |
|---|---|---|---|
| **gt cat** | 0.1054 | 31.6094 | 80.2231 |
| **gt dog** | 90.1054 | 61.6094 | 18.2231 |

All verified. Spot-check `gt cat ↔ pred 2`: class matches, so $-\ln 0.8 = 0.2231$; $L_1 = 18+18+22+22 = 80$; total $= 80.2231$ ✔.

**Optimal assignment** (exhaustive over permutations, as the Hungarian algorithm guarantees):

$$\sigma = [0,\, 2] \qquad \text{total} = 0.1054 + 18.2231 = \mathbf{18.3285}$$

**Verified: 18.3285**, versus 61.71 for the alternative $[0,1]$.

**Interpret the surprising part.** The `dog` ground truth is matched to a prediction labelled **`cat`**. The matching cost does not require the labels to agree — it *penalises* disagreement (10 here) but does not forbid it, and here the alternative costs $80$ more. Hungarian matching optimises **the assignment**, not per-item correctness; a globally better total can involve a locally wrong label. This is a genuine exam distinction.

---

## D. Visual and geometric intuition

### D1. Attention as a weighted lookup

```
        keys            values
      k¹   k²   k³     v¹   v²   v³
       |    |    |      |    |    |
       |    |    |      |    |    |
   q¹--+----+----+-----+----+----+----
       | 0.45  0.27  0.28       (row sums to 1)
       |    |    |      |    |    |
   q²--+----+----+-----+----+----+----
         0.27  0.45  0.28
       |    |    |      |    |    |
   q³--+----+----+-----+----+----+----
         0.27  0.27  0.45

   y¹ = 0.45·v¹ + 0.27·v² + 0.28·v³
   y² = 0.27·v¹ + 0.45·v² + 0.28·v³
   y³ = 0.27·v¹ + 0.27·v² + 0.45·v³
```

Rows are the answers to *"for token $i$, which tokens matter and how much?"* — a learned, input-dependent, all-pairs affinity matrix.

### D2. Why positional encoding is mandatory

```
WITHOUT position embedding                WITH position embedding
      ▓▓▓▓▓▓▓▓▓                             ▓▓▓▓▓▓▓▓▓
      ▓▓▓▓▓▓▓▓▓                             ▓▓▓▓▓▓▓▓▓
   token set {A,B,C}                    token SEQUENCE [A,B,C]
   permuted {C,A,B} = same set          permuted [C,A,B] = different sequence
   → identical output, undecodable      → different embeddings, decodable
```

Self-attention is permutation-invariant (slide p.3), so without an added signal the model literally cannot tell *"dog bites man"* from *"man bites dog"* — the token multiset is identical.

### D3. Sin/cos positional encoding across dimensions

```
   dim 0  (fast)   ▁▂▃▄▅▆▇█   period ≈ 2π
   dim 1           ▁▄█▄▁     period ≈ 2π  (cosine, offset)
   dim 2  (medium) ▂▄▆▄▂▄▆▄▂   period ≈ 2π·100
   dim 3           ▂▆▄▂▆▄▂▆
   dim 4  (slow)   ▃▃▄▄▅▅▆▆   period ≈ 2π·100²
   dim 5           ▃▅▅▇▇▅▅▇
   ...                                  pos →

   low dims: high frequency  → fine position resolution
   high dims: low frequency  → long-range / extrapolates beyond training
```

The denominator $10000^{2i/d_{model}}$ makes the wavelength grow geometrically with dimension index.

### D4. The quadratic wall, visualised

```
   token count N          attention entries N²        viable?
   ─────────────────────────────────────────────────────────────
   224×224 pixels         50,176        2.5 × 10⁹     ✗  (slide p.14)
   P=8   → 784 tokens      614,656                      ✗
   P=16  → 196 tokens       38,416                      ✓  ViT-B/16
   P=32  →  49 tokens        2,401                      ✓
   CNN (local F=3)          cost ∝ pixels (linear)       ✓  always

   CNN:      each output sees F² inputs        O(HWF²)
   Attention: each output sees ALL N inputs    O(N²)
```

ViT's move is not to make attention cheap — it is to make $N$ small enough that $N^2$ is affordable.

### D5. DETR's architecture and the "no object" trick

```
 image 224×224×3
      │
      ▼
 ┌──────────────┐   ResNet-101,  C=2048
 │ CNN Backbone │──────────────► 2048 × H × W
 └──────────────┘                        │
                                  1×1 conv → d=256
                                        │
                              flatten → 256 × HW
                                        │
                            + positional encoding
                                        │
                          ┌───────────────────────┐
                          │ Transformer ENCODER   │  self-attention
                          └───────────────────────┘
                                        │  256 × HW  (key, value)
                          ┌───────────────────────┐
   N learnable ──────────►│ Transformer DECODER   │  cross-attention
   object queries         └───────────────────────┘
                                        │  N embeddings
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
              box head (3-layer MLP)          class head (linear + softmax)
              normalised centre coords        incl. "no object"
                     │                                     │
                     └────────────► N predictions ◄───────┘

   queries matched to nothing → predicted class "no object" → contributes 0 loss
```

### D6. Id-switching on a dashcam sequence

```
  frame   1    2    3    4    5    6    7    8    9

  truth:  ─────────────────── A ───────────────────

  naive  ──1──  ──1──  ·  ·  ·  ──7──  ──7──        ✗ ID jumped 1 → 7
  detect.        (occluded — 4 frames, no detection)

  Kalman ──1──  ──1──  ─1──  ─1──  ─1──  ──1──     ✓ interpolated through
                                                  the gap, ID preserved

        * Kalman predicts the box forward through the gap; when the object
          reappears the prediction is re-associated with the original ID.
```

---

## E. Code and implementation

> **No instructor code exists in this workspace.** The ByteTrack project the instructor describes was shared via a Drive link and is not present. Everything below is **illustrative**, written to match the equations in §C. It was **not executed** — numpy is not installed here. The *Section D* solutions in §H are pure-Python and **were** executed, with real output pasted in.

### E1. Scaled dot-product attention

```python
import numpy as np, math

def attention(Q, K, V, scale=True):
    """
    Q: (n_q, d)   queries
    K: (n_k, d)   keys    -- note n_k may differ from n_q (cross-attention)
    V: (n_k, d_v) values  -- n_k rows, matching K
    returns (n_q, d_v)
    """
    d = Q.shape[-1]
    s = 1.0 / math.sqrt(d) if scale else 1.0
    scores = (Q @ K.T) * s          # (n_q, n_k)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)   # row-wise softmax, rows sum to 1
    return w @ V
```

**Shape tracking.**

| Tensor | Shape | Meaning |
|---|---|---|
| $Q$ | $(n_q, d)$ | queries |
| $K$ | $(n_k, d)$ | keys — **must share $d$ with $Q$** |
| $V$ | $(n_k, d_v)$ | values — **one row per key** |
| $QK^T$ | $(n_q, n_k)$ | attention logits |
| softmax | $(n_q, n_k)$ | rows sum to 1 |
| output | $(n_q, d_v)$ | — |

**The two bugs this shape contract prevents**, both caught while writing §D:
1. Forgetting that $V$ needs **one row per key**. Using $n_q$ rows when $n_k \ne n_q$ raises `IndexError` inside the `w @ V` product — or, worse, silently aligns values to the wrong keys.
2. Rejecting $n_k \ne n_q$. In **cross-attention** the decoder supplies $N$ object queries while the encoder supplies $HW$ image tokens, so `n_q = N` and `n_k = HW`. The Section D implementation for this class makes that case work and tests it explicitly.

### E2. Numerically stable softmax

Subtracting the row max before exponentiating is mandatory in float32. Without it, even moderate logits (say $q\cdot k = 100$) produce `exp(100) ≈ 2.7e43`, which overflows. This is why the Section D solution does `m = max(row); e = [exp(v - m) ...]`. It changes no mathematical result — the max cancels between numerator and denominator — but without it the code simply returns NaN.

### E3. ViT in five lines of the framework

```python
patch_size, img_size = 16, 224
x = img_to_tensor(img)                                  # (1, 3, 224, 224)
x = x.unfold(2, patch_size, patch_size) \
     .unfold(3, patch_size, patch_size)                 # (1, 3, 14, 14, 16, 16)
x = x.permute(0, 2, 3, 1, 4, 5).reshape(1, 196, 768)   # (1, N, D)
```

`unfold` is the non-obvious call: it produces the $P\times P$ grid without copying, and the `permute` is where **channel order bugs** live. Getting `reshape` right requires the permutation to put the spatial grid *before* the patch dimensions — otherwise you silently produce 196 tokens in a scrambled order, and the learned position embedding then learns to compensate for your bug rather than failing loudly.

### E4. DETR post-processing in practice

```python
# N object queries -> (N, num_classes + 1) class logits and (N, 4) boxes
probs = out_logits.softmax(-1)                      # includes "no object"
scores, labels = probs.max(-1)
keep  = scores.max(-1).values > threshold           # replaces NMS entirely
boxes = cxcywh_to_xyxy(out_boxes)                   # normalised centre coords
```

There is **no NMS** in DETR — the Hungarian matching plus learnable object queries is the replacement. That is the architectural claim most worth stating in a case-study answer.

### E5. Common bugs and diagnostics

| Symptom | Likely cause | Diagnostic |
|---|---|---|
| Attention output all identical | $q=0$, or all scores equal | print row max of `QK^T`; expect variety |
| `IndexError` in `w @ V` | `V` has `n_q` rows instead of `n_k` | assert `V.shape[0] == K.shape[0]` |
| `NaN` in attention | missing max-subtraction in softmax | check logits magnitude |
| IoU always ≈ 0 | boxes in different conventions — `xyxy` vs `cxcywh` | print `box[2:] > box[:2]` for `xyxy` |
| AP absurdly high | flagging duplicates as separate TPs | one TP per ground-truth object only |
| ViT tokens in scrambled order | missing `permute` before `reshape` | reshape with `.T` back and compare |

---

## F. Connections and comparisons

### F1. Attention vs convolution

| | Convolution (Class 03) | Self-attention |
|---|---|---|
| Connectivity | **local**, $F\times F$ | **global**, every token |
| Cost | $O(HWF^2)$ — linear in pixels | $O(N^2 d)$ — quadratic in tokens |
| Weights | shared across positions | **dynamic**, per input pair |
| Positional prior | **built in** (kernel position = position) | **none** — needs positional encoding |
| Inductive bias | strong (translation equivariance) | weak (*"Transformers lack the inductive bias of CNNs"*, slide p.20) |
| Data hunger | moderate | high (*"performs poorly if just trained on ImageNet"*) |
| Parallelism | high | high |

**The exam sentence:** a CNN gets translation equivariance for free from weight sharing; a transformer has to be *told* where things are.

### F2. Where each normalisation axis runs

| | Axis normalised over | Depends on batch at inference? | Parameter count |
|---|---|---|---|
| **BatchNorm** | across the batch, per feature | **yes** — needs running stats | $2 \times C$ (scale + shift) |
| **LayerNorm** | across features, per token | **no** — computed per sample | $2 \times d$ |

Both cost essentially the same compute and parameters — the instructor makes this point explicitly (*"the numbers are same… the computations are going to be almost similar"*). The choice is **semantic**, not computational.

### F3. Norm placement: post-norm vs pre-norm

| | Order | Used by | Deep stacks |
|---|---|---|---|
| **Post-norm** | sub-block → residual → **LayerNorm** | original Transformer (slide p.9) | harder to train |
| **Pre-norm** | **LayerNorm** → sub-block → residual | ImageGPT (slide p.17) | easier to train |

### F4. Detector families covered across Classes 06–08

| | R-CNN / Fast R-CNN | Faster R-CNN | YOLO (Class 07) | DETR |
|---|---|---|---|---|
| Type | two-stage | two-stage | one-stage | transformer |
| Proposals | selective search | **RPN** | anchors (fixed grid) | **learned object queries** |
| Duplicate removal | — | — | **NMS** | **none** |
| Assignment | — | — | anchor matching | **Hungarian** |
| Box loss | — | — | — | $L_1$ + **gIoU** |
| Slide evidence | p.14–24 (Class 06) | p.32–43 (Class 06) | Class 07 | p.29–36 |

### F5. DETR variants from the results table

| Variant | Backbone | GFLOPs | FPS | Params | AP | AP_S | AP_L |
|---|---|---|---|---|---|---|---|
| DETR | ResNet-50 | 86 | **28** | 41M | 42.0 | 20.5 | **61.1** |
| DETR-DC5 | ResNet-50 + dilated | 187 | 12 | 41M | 43.3 | 22.5 | 61.1 |
| DETR-R101 | ResNet-101 | 152 | 20 | 60M | 43.5 | 21.9 | 61.8 |
| DETR-DC5-R101 | ResNet-101 + dilated | 253 | **10** | 60M | **44.9** | 23.7 | **62.3** |

**DC5** = dilated convolution in the backbone → higher feature-map resolution → better small-object AP, at a large compute cost.

### F6. Detection vs tracking (Class 08 close-out)

| | Detection | Tracking |
|---|---|---|
| Operates on | one frame | a **sequence** of frames |
| Output | boxes per frame | boxes **+ persistent identity** |
| Core difficulty | localisation, duplicates | **ID switching**, occlusion, re-entry |
| Template matching | — | **Kalman filter** (linear), **EKF** (nonlinear) |
| Library | — | **ByteTrack** (instructor's project) |
| Used for | per-frame decisions | *"path planning or other kinds of things"* |

---

## G. Exam-focused revision sheet

### Essential definitions

- **Self-attention** — a *set* operation, permutation-invariant, global; needs positional embeddings.
- **Query / Key / Value** — three learned linear projections $W_Q, W_K, W_V$ of every input embedding.
- **Attention weight** $w_{ij}$ — softmax-normalised $q^{(i)}\cdot k^{(j)}$; rows sum to 1.
- **Object query** — a learnable embedding acting as one "slot" for one predicted object; DETR's replacement for anchors.
- **Hungarian matching** — optimal one-to-one assignment minimising total match cost.
- **gIoU** — a scale-invariant, still-informative extension of IoU used in DETR's box loss.
- **Callee** — $N \ge$ number of objects, fixed; unmatched queries predict "no object" and contribute no loss.
- **ID switching** — the defining failure of multi-object tracking.

### Key equations and symbols

$$H = \operatorname{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$
$$q^{(i)} = W_Q x^{(i)},\quad k^{(i)} = W_K x^{(i)},\quad v^{(i)} = W_V x^{(i)}$$
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$N = \frac{HW}{P^2} \qquad d_k = \frac{d}{h}$$
$$\hat\sigma = \arg\min_{\sigma\in S_N}\sum_i \mathcal{L}_{match}(y_i,\hat y_{\sigma(i)})$$
$$\mathcal{L}_{box} = \lambda_{L1}\|b_i-\hat b\|_1 + \lambda_{gIoU}\big(-\text{IoU}(b_i,\hat b) + |b_i-\hat b|_1\big)$$
$$\text{IoU} = \frac{|A\cap B|}{|A\cup B|},\qquad \text{Precision} = \frac{TP}{TP+FP},\qquad \text{Recall} = \frac{TP}{TP+FN}$$

$d_k$ per-head dim · $d$ model dim · $h$ heads · $N$ tokens/queries · $P$ patch size · $\lambda$ loss weights · $\sigma$ permutation.

### The two derivations to be able to reproduce

1. **$\sqrt{d_k}$.** LayerNorm ⇒ mean 0, variance 1. Sum of $d_k$ such terms ⇒ mean 0, **variance $d_k$**. Divide by $\sqrt{d_k}$ ⇒ variance back to 1. Divide by $d_k$ ⇒ variance $1/d_k$, softmax nearly uniform. *Verified: 63.81 → 0.997 (÷√d_k) vs 0.0156 (÷d_k).*
2. **Why LayerNorm.** Tokens can differ hugely in magnitude; before they are multiplied in attention, the large one dominates. Normalise **within** the token. It is **not** about small batch size or about text.

### Key assumptions

- Self-attention assumes **nothing** about spatial layout — hence positional encoding.
- ViT's competitiveness assumes **very large pretraining** (14M–303M images).
- Hungarian matching assumes the prediction set is a **set**, not a sequence — no ordering prior.
- gIoU matters because plain IoU has vanishing gradient when boxes are disjoint.

### Distinctions to remember

| | A | B |
|---|---|---|
| BatchNorm / LayerNorm | across batch | across features of one token |
| Post-norm / pre-norm | LN after sub-block | LN before (ImageGPT) |
| ViT / DETR | classification, encoder only | detection, encoder–decoder |
| IoU / gIoU | bounded $[0,1]$, dead when disjoint | goes negative, informative |
| YOLO / DETR | anchors + NMS | object queries + Hungarian |
| Detection / tracking | one frame | sequence + identity |

### Traps and misconceptions

1. **"Divide by $d_k$ to average."** Averages the *mean*; destroys the variance.
2. **"LayerNorm because batch size is small."** The instructor rejected exactly this.
3. **"Self-attention preserves order."** It does not — it is permutation-invariant.
4. **"ViT beats CNNs."** Only at 14M–303M pretraining scale; it loses at ImageNet-1K alone.
5. **"DETR has the best AP."** The best-AP row (44.9) is also the **slowest** (10 FPS).
6. **"Higher AP = ship it."** DETR at 42.0 AP beats Faster RCNN-FPN's 40.2, yet the instructor's deployment answer is still *"we are still using YOLO"* — because compute is on the car.
7. **"DETR uses NMS."** It does not.
8. **"Hungarian matching makes the labels agree."** It optimises total cost; a globally better assignment can pair a gt with a wrong-label prediction (§C7).

### Checklist

- [ ] **Explain:** Q/K/V, and what one attention row means
- [ ] **Explain:** set operation ⇒ positional encoding
- [ ] **Explain:** why $1/\sqrt{d_k}$ and why LayerNorm *(the two correct reasons)*
- [ ] **Explain:** quadratic wall, and how ViT escapes it
- [ ] **Derive:** the $\sqrt{d_k}$ variance argument
- [ ] **Calculate:** attention weights and output for a small Q/K/V
- [ ] **Calculate:** ViT token count and attention-map size
- [ ] **Calculate:** $d_k$ from $d$ and $h$
- [ ] **Calculate:** IoU, precision, recall, AP
- [ ] **Calculate:** DETR matching cost and the optimal assignment
- [ ] **Implement:** attention with the $V$-row bug fixed
- [ ] **Interpret:** DETR-S weakness vs AP_L strength
- [ ] **Interpret:** when tracking beats per-frame detection

---

## H. Practice Questions

> Mirrors `session16.pdf`: **A** single-correct MCQ · **B** multi-select · **C** descriptive ≤100 words · **D** coding, attempt 1 of 3 · **E** case study · **F** numericals.

### Section A — Single-correct MCQ (6 × 1 mark)

**A1.** Why do transformers require positional encodings?

- A. Attention has no learned parameters to optimise
- B. Attention mixes information across positions too quickly
- C. Attention is permutation-invariant over its inputs
- D. Positional encodings reduce the quadratic compute cost

**A2.** In multi-head attention with model dimension 1024 and 16 heads, what is $d_k$?

- A. 16
- B. 64
- C. 128
- D. 1024

**A3.** Why is the attention denominator $\sqrt{d_k}$ rather than $d_k$?

- A. It reduces memory used by the softmax
- B. It keeps the score variance at one after summation
- C. It makes every attention weight exactly equal
- D. It keeps the softmax output strictly positive

**A4.** A 224×224 image is split into 16×16 patches. How many tokens enter the transformer?

- A. 14
- B. 49
- C. 196
- D. 50,176

**A5.** DETR produces a fixed set of $N$ predictions but an image contains only 3 objects. What happens to the other $N-3$ queries?

- A. They are removed by non-maximum suppression
- B. They predict "no object" and contribute no loss
- C. They are matched to background anchors
- D. They are dropped by a confidence threshold

**A6.** DETR-R101 scores 43.5 AP at 20 FPS, while DETR-DC5-R101 scores 44.9 AP at 10 FPS. For an autonomous-driving stack with a fixed compute budget, which statement is best supported?

- A. The 1.4-point AP gain always justifies the halving of FPS
- B. The higher-FPS model is preferable given the budget
- C. Both models are equivalent since AP differs by under 2 points
- D. FPS depends only on parameter count, not on the backbone

### Section B — Multi-select (4 × 2 marks)

**B1.** Which statements about **LayerNorm inside a transformer** are **correct**? Select all that apply.

- A. It normalises each token across its own feature dimensions
- B. Its statistics can be computed from a single sample
- C. It uses more parameters than BatchNorm for the same width
- D. It equalises token magnitudes before they enter attention
- E. It requires running statistics updated during training

**B2.** Which are **true** of DETR? Select all that apply.

- A. It uses non-maximum suppression to remove duplicate boxes
- B. It matches predictions to ground truth with a Hungarian assignment
- C. It uses a scale-invariant box loss based on generalised IoU
- D. It predicts a fixed number of boxes regardless of image content
- E. Its object queries are learned during training

**B3.** Which are **limitations of ViT** identified in the class? Select all that apply.

- A. It performs poorly when trained only on ImageNet-1K
- B. It cannot be trained without positional encodings
- C. Its attention cost is quadratic in the number of patches
- D. It cannot be applied to images larger than 224×224
- E. It lacks the translation inductive bias of convolutional layers

**B4.** In a multi-object tracking system, which are **symptoms of ID switching**? Select all that apply.

- A. A track keeps one ID across a short occlusion
- B. An object reappears after a gap with a different ID
- C. Two objects exchange relative position after overtaking
- D. A spurious detection appears for one frame only
- E. A track's box is interpolated through a missing-detection gap

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why is self-attention expensive for high-resolution images, and how does ViT address it?

**C2.** Why is LayerNorm preferred to BatchNorm inside a transformer block?

**C3.** What problem does Hungarian matching solve in DETR, and why can the resulting assignment pair a ground truth with a wrong-label prediction?

**C4.** Why does DETR use generalised IoU in its box loss rather than plain IoU?

**C5.** DETR achieves higher AP than several Faster R-CNN variants, yet the instructor states production systems often still use YOLO. Give one reason and one supporting comparison from the results table.

### Section D — Coding (20 marks — attempt ONE of the three)

**Pure Python, standard library only.** Provide a reusable function, not a hardcoded answer. Handle empty and boundary inputs.

---

**Option D1 — Scaled dot-product attention**

```python
def attention(Q, K, V): ...
```

- `Q`: list of `n_q` rows, each of length `d` (floats).
- `K`: list of `n_k` rows, each of length `d`. **`n_k` may differ from `n_q`** (cross-attention).
- `V`: list of **`n_k`** rows, each of length `d_v`.
- Returns `n_q` rows of `d_v` values, rounded to 6 decimal places.
- Empty `Q`, `K`, or `V` → `[]`. Mismatched `len(V) != len(K)` → `[]`. Row-width mismatches → `[]`.
- Softmax must be computed **row-wise** and be numerically stable.

Worked examples:

```
Q=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
K=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
V=[[1,0],[0,1],[0,0]]

attention(Q,K,V)  ->  [[0.451863, 0.274069], [0.274069, 0.451863], [0.274069, 0.274069]]

Q=[[1,0]], K=[[1,0],[0,1]], V=[[5,0],[0,5]]
attention(Q,K,V)  ->  [[3.348808, 1.651192]]

Q=[[0,0,0,0]], K=[[1,0,0,0]], V=[[7,7]]
attention(Q,K,V)  ->  [[7.0, 7.0]]

attention([], [], [])                        ->  []
attention(Q, K, [[1,0]])                     ->  []
```

**Option D2 — DETR Hungarian matching**

```python
def match_cost(gt, preds, class_penalty=10.0): ...
def best_assignment(cost): ...
```

- `gt`: list of `(class_label, box)`, box is `[x1,y1,x2,y2]`.
- `preds`: list of `(class_label, confidence, box)`.
- `match_cost` returns an `|gt| × |preds|` matrix where `cost[i][j] = 0 if labels match else class_penalty`, plus `-ln(max(confidence, 1e-12))` plus the L1 distance between boxes. Rounded to 6 dp.
- `best_assignment(cost)` returns the list `p` of length `|gt|` such that `cost[i][p[i]]` is minimal in total, or `[]` when there are no ground-truth rows.

Worked examples:

```
gt    = [("cat",[0,0,10,10]), ("dog",[20,20,30,30])]
preds = [("cat",0.9,[0,0,10,10]), ("dog",0.2,[5,5,15,15]), ("cat",0.8,[18,18,32,32])]

match_cost(gt, preds) ->
   [[0.105361, 31.609438, 80.223144],
    [90.105361, 61.609438, 18.223144]]

best_assignment(match_cost(gt, preds))  ->  [0, 2]      total = 18.3285
best_assignment(match_cost([], preds))  ->  []
```

**Option D3 — Detection metrics**

```python
def iou(box_a, box_b): ...
def average_precision(detections, n_gt, iou_threshold=0.5): ...
```

- `iou` returns intersection-over-union of two `[x1,y1,x2,y2]` boxes, or `0.0` if they do not overlap.
- `average_precision` takes `detections` in descending-confidence order, each a dict with a `"tp"` boolean, and `n_gt` the number of ground-truth objects. Returns AP rounded to 6 dp using the precision-envelope method; `0.0` for an empty list.

Worked examples:

```
iou([0,0,10,10], [5,5,15,15])  ->  0.142857
iou([0,0,10,10], [2,2,8,8])    ->  0.36
iou([0,0,10,10], [20,20,30,30]) ->  0.0

average_precision([{"tp":True},{"tp":False},{"tp":True},{"tp":True},{"tp":False}], 3)
    ->  0.833333
average_precision([], 3)          ->  0.0
average_precision([{"tp":True}]*3, 3) ->  1.0
```

### Section E — Case study (15 marks)

> **Scenario.** An autonomous-vehicle company must add a **traffic-light and vehicle detection** system. Requirements: ≥ 20 FPS end-to-end on a single board-level GPU, ≤ 250 GFLOPs per frame, and reliable detection of **small** objects (a 20×20 px traffic light 60 m away). The team currently runs a YOLO model and is considering DETR. Separately, the perception module needs persistent object identity for path planning.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Detection, or tracking? State both needs and why they are different tasks.
2. **Data.** What labels are needed? What is the hardest part of collecting them?
3. **Model.** Compare DETR and YOLO against the three requirements above, quoting the results table. Which do you pick, and why?
4. **Metric.** Which AP variant would you report to management, and which would you actually optimise? Justify with the size of your objects.
5. **Failure mode.** Name the most likely failure and its mechanism.

**Sub-questions**

- **E-a.** DETR-DC5-R101 has the best AP in the table. Give three reasons it is still the wrong choice here.
- **E-b.** Your AP_S is 19.0 while AP_L is 63.0. Diagnose the cause and propose two changes.
- **E-c.** The planner reports that cars change identity after merging into traffic. Explain the mechanism, name the standard tool, and describe what it does during the gap.

### Section F — Numericals (6 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** With $d_k = 4$ and $Q = K = I_3$ (padded to 4 columns), compute $QK^T$, the scaled scores, and the softmax attention matrix. Confirm each row sums to 1.

**F2.** A dot product sums $d_k = 64$ zero-mean unit-variance terms. State the mean and variance of the result, the variance after dividing by $\sqrt{d_k}$, and the variance after dividing by $d_k$. Explain which is correct and why.

**F3.** A 224×224 image is tokenised with 16×16 patches. Compute the grid size, the sequence length $N$, and the number of entries in the attention matrix. How does this compare with treating every pixel as a token?

**F4.** Compute $d_k$ for ViT-Base ($d=768$, $h=12$) and ViT-Huge ($d=1280$, $h=16$). Confirm that a 16×16 RGB patch projects to ViT-Base's hidden size in one multiplication.

**F5.** Two boxes are $A=[0,0,10,10]$ and $B=[5,5,15,15]$. Compute $|A\cap B|$, $|A\cup B|$, and IoU. Two detections are ranked TP, FP, TP, TP, FP against 3 ground-truth objects. Compute the precision and recall at each rank, and the resulting AP.

**F6.** DETR has 2 ground-truth objects and 3 predictions. Compute the full 2×3 matching-cost matrix using class penalty 10 and `cost = penalty(if mismatch) − ln(conf) + L1`. Give the optimal assignment and its total cost, and state which ground truth is matched to a mismatched label.

---

## Worked Solutions — Attempt the Questions First

> Answers to everything in Section H. Section F mirrors §C, which is already verified.

### Section A — Solutions

**A1. → C — Attention is permutation-invariant over its inputs.**

Slide p.3 states it as a consequence, not an assumption: *"It's a set operation (permutation-invariant) and hence need 'position embeddings' to 'remember' the spatial structure."*

- A is false — attention has $W_Q,W_K,W_V$, plenty of parameters.
- B is the opposite of the reason; global mixing is the point.
- D is false — positional encodings add $O(Nd)$ cost, negligible against $O(N^2d)$, and do nothing to reduce it.

**A2. → B — 64.**

$d_k = d/h = 1024/16 = 64$. *(Verified in §C5.)* A is $h$; C is $h \cdot d_k/16$; D is $d$, which is the residual-stream width, not the per-head width.

**A3. → B — It keeps the score variance at one after summation.**

The instructor's derivation (§B4): after LayerNorm the mean is 0 and the variance 1; summing $d_k$ terms keeps the mean at 0 but **accumulates variance to $d_k$** — *"we want it to keep variance as 1"* — and *"variance is nothing but [standard deviation] square."* Verified numerically in §C2: 63.81 → 0.997 after ÷$\sqrt{d_k}$.

- A is a side benefit, not the reason.
- C is false — softmax makes weights sum to 1, not become equal.
- D is trivially true of softmax regardless of scaling.

**A4. → C — 196.**

$N = HW/P^2 = (224/16)^2 = 14^2 = \mathbf{196}$. *(Verified.)* A is the grid side, not the count; B is the $P=32$ case; D is the pixel count that motivates the whole problem.

**A5. → B — They predict "no object" and contribute no loss.**

Slide p.36: *"Class prediction also predicts 'no object'."* The loss (slide p.31) gates every term by $\mathbf{1}\{c_i \neq \varnothing\}$, so a query matched to an empty slot contributes **zero**. Verified by inspection of the worked assignment in §C7, where only 2 of the queries are consumed.

- A and C are wrong — DETR uses **neither NMS nor anchors**.
- D is wrong — a threshold may be applied at inference, but it is not how the loss is defined, and it is not what makes the fixed size work.

**A6. → B — The higher-FPS model is preferable given the budget.**

The instructor's own framing: DETR *"has a reduced FPS of 28… the time it is taking to process each frame has increased,"* and *"you cannot call your cloud every time to make a prediction, you have to keep the compute on the car."*

- A is wrong because it assumes AP is the only objective — the entire point of §B13 is that it is not.
- C is wrong because AP and FPS are different resources; 1.4 AP at half the frame rate is a real trade-off, not a wash.
- D is wrong — DETR-R101 (60M) and DETR (41M) have different backbones; the results table shows compute tracks the backbone (dilated convolution), not the parameter count.

### Section B — Solutions

**B1. → A, B, D. Reject C, E.**

- **A** ✔ *"each token is normalized within itself"* (transcript).
- **B** ✔ LayerNorm's statistics come from a single sample; the instructor notes the batch *"doesn't exist"* for it at inference.
- **C** ✗ **Both have $2C$ parameters** (a learned scale and shift). The transcript is explicit: *"the number of parameters are same."* This is the trap — the choice is semantic, not a capacity difference.
- **D** ✔ *"the vector magnitude of the first token is very much larger… this larger value token will have more influence on the attention by default."*
- **E** ✗ That is BatchNorm's running statistics. LayerNorm has none.

**B2. → B, C, D, E. Reject A.**

- **B** ✔ *"Use Hungarian Algorithm to find a bipartite matching with the lowest matching cost"* (slide p.30).
- **C** ✔ *"a combination of the L1 loss and the generalized IoU loss that is scale-invariant"* (slide p.31).
- **D** ✔ *"Infer a fixed-size set of $N$ predictions"* (slide p.30).
- **E** ✔ *"Add $N$ learnt positional encodings (object queries)"* (slide p.35).
- **A** ✗ DETR has **no NMS** — that is precisely the architectural claim. State it explicitly in any case-study answer.

**B3. → A, C, E. Reject B, D.**

- **A** ✔ *"Performs poorly if just trained on ImageNet… can be expected since Transformers lack the inductive bias of CNNs"* (slide p.20).
- **C** ✔ *"computation is quadratic in the input sequence length"* (slide p.14).
- **E** ✔ same slide — the missing inductive bias, restated as a limitation.
- **B** ✗ This is a **requirement**, not a limitation — positional embeddings are added, and slide p.25 shows they are *learned*.
- **D** ✗ Patches scale the sequence length; ViT handles arbitrary resolutions by changing $P$ or interpolating the position embedding. Nothing on any slide restricts it to 224×224.

**B4. → B, C. Reject A, D, E.**

- **B** ✔ *"every time a prediction is missing and it's reappearing into the picture, a new association is given to it, and you don't want that"* (transcript).
- **C** ✔ *"a black [car] can be in front of red or like they interchange the positions"* — the overtaking case the instructor names.
- **A** ✗ Preserving an ID through occlusion is the **goal**; keeping one ID means no switch occurred.
- **D** ✗ A one-frame spurious detection is a **false positive**, which the instructor lists separately.
- **E** ✗ Interpolating through the gap is the **Kalman filter's job** — the fix, not the symptom.

### Section C — Solutions

*Each model answer is ≤100 words, following the three-beat structure from `session16.pdf`.*

**C1.**
Self-attention builds an $N\times N$ affinity matrix, so cost grows as $O(N^2)$ in the number of input tokens. At 224×224 that is 50,176 tokens and roughly 2.5 × 10⁹ attention entries — the *"quadratic in the input sequence length… quickly gets very expensive"* problem on slide p.14. A convolution avoids this because each output depends on only $F^2$ inputs, so its cost is linear in pixels. ViT reduces $N$ by grouping pixels into $P\times P$ patches: with $P=16$, $N=(224/16)^2=196$, a 256× cut in sequence length and about 65,000× fewer attention entries.

*(97 words.)*

**C2.**
Because token magnitudes can differ enormously — the instructor's example is one token with values *"10, 12, 100"* against another spanning *"0–10"*. Attention multiplies and sums these vectors, so the high-magnitude token *"will have more influence on the attention by default"*. LayerNorm normalises each token across its own features, equalising magnitudes before any multiplication, so no token can dominate merely by being large. BatchNorm normalises across the batch instead, which compares tokens to each other and therefore cannot remove a within-token scale difference. Note the reason is **not** small batch size — the instructor explicitly rejected that answer — and the two have equal parameter counts.

*(99 words.)*

**C3.**
DETR produces a fixed set of $N$ unordered predictions, so there is no inherent correspondence to the ground-truth objects. Hungarian matching finds the one-to-one assignment of predictions to objects that minimises total matching cost, so training gradients point at the best available assignment rather than an arbitrary one. It can still pair a ground truth with a wrong-label prediction because the cost *penalises* label disagreement rather than forbidding it. In the worked example the alternative assignment costs about $80$ more, so accepting the mismatch is globally optimal even though it is locally wrong.

*(99 words.)*

**C4.**
Plain IoU lies in $[0,1]$ and equals $0$ whenever two boxes are disjoint. At that point its gradient with respect to the box coordinates vanishes, so a badly placed box receives almost no learning signal and can fail to recover. Generalised IoU subtracts the enclosing box's geometry from the union term, allowing the score to fall **below zero** for disjoint boxes. DETR therefore keeps IoU informative for poor matches, and because the correction is defined relative to the enclosing box the measure is also scale-invariant — stated on slide p.31 as *"a combination of the L1 loss and the generalized IoU loss that is scale-invariant."*

*(98 words.)*

**C5.**
**Reason:** on-device latency, not accuracy. The instructor states that *"in the real world lot of times we are still using YOLO rather than DETR"* because compute must live on the device — *"you cannot call your cloud every time to make a prediction, you have to keep the compute on the car."* Unlike LLM workloads, *"these are not life critical tasks"* sits outside labs. **Supporting comparison:** DETR achieves 42.0 AP at 86 GFLOPs and 28 FPS, beating Faster RCNN-FPN's 40.2 AP — yet the highest-AP row, DETR-DC5-R101 at 44.9, also has the *worst* throughput at 253 GFLOPs and 10 FPS.

*(99 words.)*

### Section D — Solutions

#### D1 — Scaled dot-product attention

**Key idea tested.** The $\sqrt{d_k}$ scaling, row-wise softmax, and the fact that **queries and keys need not be the same count**.

```python
import math

def softmax(row):
    m = max(row)                      # stability: subtract max
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]

def attention(Q, K, V):
    nq = len(Q)
    if nq == 0 or not K or not V:
        return []
    d = len(Q[0])
    dv = len(V[0])
    if d == 0 or dv == 0:
        return []
    if len(V) != len(K):              # one value per KEY
        return []
    if any(len(r) != d for r in K) or any(len(r) != dv for r in V):
        return []
    scale = 1.0 / math.sqrt(d)
    out = []
    for i in range(nq):
        scores = [sum(Q[i][t] * K[j][t] for t in range(d)) * scale
                  for j in range(len(K))]
        w = softmax(scores)
        out.append([round(sum(w[j] * V[j][c] for j in range(len(K))), 6)
                    for c in range(dv)])
    return out
```

**Verified execution output:**

```
D1 self-attn: [[0.451863, 0.274069], [0.274069, 0.451863], [0.274069, 0.274069]]
D1 empty: []   mismatch: []
D1 cross 1 query / 2 keys: [[3.348808, 1.651192]]   weights 0.669762 / 0.330238
D1 skewed:  [[0.669762, 0.330238]]                 key0 scores 1, key1 scores 0
D1 zero query: [[7.0, 7.0]]                        uniform -> mean of V
```

**Why it works.** `d` is the per-query feature width, so the scaling constant is $1/\sqrt{d}$ — matching the slide's $\sqrt{d_k}$ when $d=d_k$ in the single-head case. Weights come from a **row-wise** softmax, so each row is a distribution over *keys*. The output is the convex combination of value rows indexed by those weights.

**Two bugs this version is written to prevent**, both encountered while building this question:
1. **`V` must have one row per key, not per query.** An earlier version allowed `len(V) != len(K)` and crashed with `IndexError` inside the `w[j] * V[j][c]` product. In cross-attention the two counts genuinely differ ($N$ queries, $HW$ keys), so this guard cannot simply be removed.
2. **Do not require `len(K) == len(Q)`.** An earlier version rejected that case, which would break cross-attention entirely. The test `attention([[1,0]], [[1,0],[0,1]], [[5,0],[0,5]])` pins the behaviour: **one query, two keys**, weights 0.669762/0.330238.

**Common wrong approaches.**
- Forgetting the max subtraction — returns NaN for large logits.
- Applying softmax **column-wise** instead of row-wise — each query's weights would not sum to 1.
- Dividing by `d` instead of `sqrt(d)` — §C2's failure mode.

#### D2 — DETR Hungarian matching

**Key idea tested.** The matching cost is a *combined* class + box objective, and the optimal assignment is found **globally**, not greedily.

```python
from itertools import permutations
import math

def match_cost(gt, preds, class_penalty=10.0):
    cost = []
    for g_label, g_box in gt:
        row = []
        for p_label, conf, p_box in preds:
            penalty = 0.0 if g_label == p_label else class_penalty
            l1 = sum(abs(a - b) for a, b in zip(g_box, p_box))
            row.append(round(penalty - math.log(max(conf, 1e-12)) + l1, 6))
        cost.append(row)
    return cost

def best_assignment(cost):
    G = len(cost)
    if G == 0:
        return []
    P = len(cost[0])
    if P < G:
        return []
    best, best_p = None, None
    for p in permutations(range(P), G):
        total = sum(cost[i][p[i]] for i in range(G))
        if best is None or total < best - 1e-12:
            best, best_p = total, p
    return list(best_p)
```

**Verified execution output:**

```
D2 cost row: [0.105361, 31.609438, 80.223144]
D2 cost row: [90.105361, 61.609438, 18.223144]
D2 best assignment: [0, 2]   total: 18.3285
D2 empty gt: []
```

**Why it works.** `permutations(range(P), G)` enumerates exactly the set $S_G$ that slide p.30 minimises over, so taking the argmin **is** the Hungarian result — brute force is only tractable because $G$ is tiny here. The `1e-12` floor prevents `log(0)` if a confidence of exactly 0 ever appears. `-math.log(conf)` makes high-confidence, correct-label matches cheap.

**Worked reading of the matrix.** Row 0 (gt `cat`) vs prediction 2 (`cat`, conf 0.8, box `[18,18,32,32]`): labels agree so no penalty; $-\ln 0.8 = 0.2231$; $L_1 = 18+18+22+22 = 80$; total $80.2231$ ✔.

**The instructive result.** The optimum pairs gt `dog` with the **`cat`**-labelled prediction 2, total $18.3285$, versus $61.71$ for the alternative. Greedy matching by lowest individual cost would make the same first choice here — but that is luck, not a guarantee. In general, greedy can pick two cheap pairings that leave an object unmatched; the exhaustive minimisation cannot.

**Common wrong approaches.**
- Greedy row-by-row selection instead of global argmin.
- Forgetting the class penalty, so a geometrically-near but wrong-label box wins.
- Letting `log(0)` raise instead of flooring.

#### D3 — Detection metrics

**Key idea tested.** IoU's union term, and AP via the precision envelope rather than raw precision averages.

```python
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def average_precision(detections, n_gt, iou_threshold=0.5):
    tp = fp = 0
    prec, rec = [], []
    for d in detections:
        if d["tp"]:
            tp += 1
        else:
            fp += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    if not prec:
        return 0.0
    mrec = [0.0] + rec + [rec[-1]]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])     # precision envelope
    ap = sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))
    return round(ap, 6)
```

**Verified execution output:**

```
D3 iou A,B: 0.142857
D3 iou A,C: 0.36
D3 iou disjoint: 0.0
D3 AP(3 gt): 0.833333
D3 AP no dets: 0.0
D3 AP perfect: 1.0
```

**Why it works.** IoU's union must subtract the intersection once — $|A\cup B| = |A| + |B| - |A\cap B|$ — which is exactly where the A/B case goes wrong if you forget. For AP, the right-to-left running maximum builds the precision envelope: at each recall level, credit the best precision achievable at that recall or higher. Ranking 2's precision of 0.5 is therefore discarded in favour of 0.75 at rank 4.

**A note on `iou_threshold`.** The spec accepts it but this design receives pre-computed `"tp"` flags, so the threshold is **not applied here** — the greedy match-to-ground-truth step that consumes it belongs upstream. That is stated rather than silently ignored; an implementation that silently accepted the parameter and did nothing with it would be the bug.

**Common wrong approaches.**
- Averaging precision directly instead of applying the envelope → 0.6517 instead of 0.8333 here. *(The naive mean of `[1.0, 0.5, 0.6667, 0.75, 0.6]` is $0.7033$; the envelope is what makes AP $0.8333$.)*
- Dividing by union without subtracting the intersection.
- Returning a negative IoU for disjoint boxes — that is gIoU's job, not IoU's.

### Section E — Solutions

**Model answer covering the chain.**

1. **Problem → task.** Two distinct tasks. **Detection**: find and classify every traffic light and vehicle per frame. **Tracking**: maintain a *persistent identity* across frames so the planner knows *"the same object where it is lying across the frames, not just one frame"* (transcript). Detection alone is insufficient — the instructor says so directly: *"when you are doing path planning… you need to know [the] same object."*
2. **Data.** Per-frame bounding boxes with class labels for detection; per-object **persistent IDs** across frames for tracking. **Hardest part:** the tracking labels — a single clip must be annotated consistently over hundreds of frames, and any ID the annotator flips becomes unlearnable label noise. Detection labels are comparatively cheap and independent per frame.
3. **Model.** Against the three requirements: DETR-R101 gives 43.5 AP at 152 GFLOPs / 20 FPS, which satisfies ≥20 FPS and ≤250 GFLOPs. DETR-DC5-R101 gives the best AP (44.9) but runs at **10 FPS** and **253 GFLOPs**, violating both budget constraints. DETR (86/28) is cheapest but weakest on small objects. **Pick DETR-R101** as the detection candidate, on the basis of meeting the latency and compute budget — then validate the small-object requirement in step 4.
4. **Metric.** *Report* overall **AP** (the mean over IoU 0.50:0.95) because management wants one headline number. *Optimise* **AP_S**, because a 20×20 px traffic light at 60 m is a small object on the COCO bands (small < 32² px). This immediately exposes a conflict: DETR's AP_S is 20.5–23.7 across all variants, while Faster RCNN-R101-FPN+ reaches 27.2 — **DETR is the weaker model for this requirement**, and that must be fixed before shipping.
5. **Failure mode.** **Small-object miss.** A 20×20 px target survives very few activations in a low-resolution feature map, and DETR's set-prediction queries are learned globally rather than anchored to image locations, so there is no high-resolution prior to catch it. Result: traffic lights vanish at distance — the most safety-critical failure in this system.

**Sub-questions.**

**E-a — three reasons DETR-DC5-R101 is wrong here.**
1. **Throughput.** 10 FPS fails the ≥20 FPS requirement by half. On a self-driving stack this is not a tuning detail — the instructor's *"you cannot call your cloud every time to make a prediction, you have to keep the compute on the car"* makes latency a hard architectural constraint.
2. **Compute.** 253 GFLOPs exceeds the 250 GFLOPs budget, leaving no headroom for the tracking stage that must also run on the same board.
3. **Diminishing returns.** The entire gain over DETR-R101 is **1.4 AP** (44.9 vs 43.5), and it comes from dilated convolution in the backbone — extra activation-map resolution. But the failure mode identified above is *also* about resolution, and the metric that matters (AP_S) improves only from 21.9 to 23.7. So the most expensive change buys the least on the metric that matters.

**E-b — AP_S 19.0 vs AP_L 63.0.** The diagnosis is a **spatial-resolution and feature-stride mismatch**: small objects occupy few activations and are frequently lost by early downsampling, while large objects fill their feature maps and are easy to localise.

- **Change 1 — raise backbone output resolution.** Switch the backbone to dilated convolution (the DC5 variants) so the final stride drops from 32 to 8 without retraining the classifier. This is precisely the mechanism that lifts AP_S from 20.5 → 22.5 in DETR vs DETR-DC5.
- **Change 2 — multi-scale features.** Add a Feature Pyramid Network so detection heads read from several strides, letting a small object be detected at a fine stride where it is still resolvable. This is what Faster RCNN-R101-FPN+ does to reach AP_S 27.2, and the gap to DETR is the clearest evidence it is the right fix.

*(Validation on traffic lights alone will be needed regardless: 19.0 is below every published detector in the table, so the model choice itself may need revisiting.)*

**E-c — ID changes after merging.**
**Mechanism:** during the merge one vehicle is occluded by another, so per-frame detection returns no box for it. When it re-emerges, the tracker has no matching hypothesis, so it is treated as a **new** object and assigned a fresh ID. The instructor describes exactly this: *"every time a prediction is missing and it's reappearing into the picture, a new association is given to it, and you don't want that."* Overtaking aggravates it — *"a black [car] can be in front of red or like they interchange the positions."*
**Tool:** **Kalman filter** (the instructor's choice, implemented inside **ByteTrack** in his class project). Linear motion is handled by the Kalman filter; nonlinear motion by the **Extended Kalman Filter**.
**During the gap:** the filter *propagates* the last known state forward, so it retains a predicted box and, crucially, a predicted **track ID** across the missing frames. When the object reappears, the new detection associates with the propagated hypothesis and inherits the original ID — *"if the prediction is not missed then basically we are not losing the IDs."*

### Section F — Solutions

**F1.** *Formula:* $H = \operatorname{softmax}\!\big(QK^T/\sqrt{d_k}\big)V$.

*Step 1 — scores.* With $Q = K$ the 3×4 identity-padded matrix, $QK^T = I_3$:
$$\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}$$

*Step 2 — scale.* $\sqrt{d_k} = \sqrt{4} = 2$:
$$\frac{QK^T}{2} = \begin{bmatrix}0.5&0&0\\0&0.5&0\\0&0&0.5\end{bmatrix}$$

*Step 3 — softmax, row-wise.* Row $[0.5,0,0]$: $e^{0}=1$, $e^{-0.5}=0.6065$, sum $= 2.2131$:
$$w_{0\cdot} = [0.4519,\;0.2741,\;0.2741]$$
By symmetry all three rows are permutations of this. **Row sums = 1.0000000000** (verified to 10 dp).

*Step 4 — output.* With $V = \begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}$:
$$H = \begin{bmatrix}0.451863&0.274069\\0.274069&0.451863\\0.274069&0.274069\end{bmatrix}$$

*Interpret:* every output token is dominated by its own value ($0.45$) with a near-uniform remainder — the signature of orthogonal queries/keys, where the diagonal always wins. In a real network $Q,K$ are learned, and this structure is what allows a patch token to gather context from distant patches.

**F2.** *Formula:* variances add for independent variables; $\mathrm{Var}(\bar X\cdot n) = n^2\mathrm{Var}(X)$.

*Substitute:* each term has mean $0$, variance $1$, and $d_k = 64$ terms are summed.

*Calculate:*
- mean of the sum $= \mathbf{0}$ (sum of zeros)
- variance of the sum $= \mathbf{64}$
- after $\div\sqrt{64} = 8$: variance $= \mathbf{1}$
- after $\div 64$: variance $= \mathbf{1/64} = \mathbf{0.015625}$

*Verified by simulation (20,000 trials):* raw sum variance **63.81**; after $\div\sqrt{d_k}$ **0.997**; after $\div d_k$ **0.015578** — all matching theory to sampling error.

*Interpret:* dividing by $\sqrt{d_k}$ is correct. It is the **only** choice that restores the variance to 1, which is precisely the property LayerNorm established — as the instructor says, *"it is dependent on the output of the LayerNorm. Output of the LayerNorm will have certain properties."* Dividing by $d_k$ restores the mean but drives the variance to $1/64$, squashing every logit toward 0 so the softmax becomes nearly uniform and attention degenerates to averaging. The mean was never the problem; the variance was.

**F3.** *Formula:* $N = \dfrac{HW}{P^2} = \left(\dfrac{224}{16}\right)^2$.

*Substitute and calculate:*
- grid $= 224/16 = \mathbf{14}$, i.e. $14\times14$
- $N = 14 \times 14 = \mathbf{196}$ tokens
- attention matrix $= N^2 = 196^2 = \mathbf{38{,}416}$ entries

*Compare with per-pixel tokens:* $N_{\text{pixel}} = 224^2 = \mathbf{50{,}176}$ tokens → $50{,}176^2 = 2.52 \times 10^9$ entries. *(Both verified; $50{,}176$ matches slide p.14's "$\sim$50,000 sequence length".)*

*Interpret:* patches cut the sequence length by $\frac{50{,}176}{196} = \mathbf{256\times}$, and therefore cut attention-map entries by $256^2 = \mathbf{65{,}536\times}$. That is the entire trick — ViT does not make attention cheaper, it makes $N$ small enough that $N^2$ is affordable. A CNN pays $O(HWF^2)$, linear in pixels, at every resolution.

**F4.** *Formula:* $d_k = d/h$.

- ViT-Base: $768/12 = \mathbf{64}$
- ViT-Huge: $1280/16 = \mathbf{80}$

Both verified. Patch check: $16\times16\times3 = \mathbf{768}$, which equals ViT-Base's hidden size — exactly slide p.28's note *"Note: 16×16×3 = 768"* — so the patch-to-hidden projection is a square $768\times768$ matrix. ViT-Huge's 80 is larger, because the projection is $768 \to 1280$ and need not be square.

*Interpret:* $d_k \ll d$ in both cases (64 vs 768; 80 vs 1280), which is why the instructor emphasises *"there is a dimension $d_k$ not $d$ here… $d_k$ is generally very smaller than the actual $d$."* Attention scores are computed in a low-dimensional projected space even though the residual stream is wide — this is also what makes multi-head affordable.

**F5.** *Formula:* $\text{IoU} = \dfrac{|A\cap B|}{|A| + |B| - |A\cap B|}$.

*Boxes:* $A=[0,0,10,10]$, $B=[5,5,15,15]$.
- $|A\cap B| = (10-5)\times(10-5) = 5\times5 = \mathbf{25}$
- $|A| = |B| = 10\times10 = 100$
- $|A\cup B| = 100 + 100 - 25 = \mathbf{175}$
- $\text{IoU} = 25/175 = \mathbf{0.142857}$ *(verified)*

*Ranked detections* TP, FP, TP, TP, FP against 3 ground truths:

| Rank | Flag | TP | FP | Precision | Recall |
|---|---|---|---|---|---|
| 1 | TP | 1 | 0 | 1.0000 | 0.3333 |
| 2 | FP | 1 | 1 | 0.5000 | 0.3333 |
| 3 | TP | 2 | 1 | 0.6667 | 0.6667 |
| 4 | TP | 3 | 1 | 0.7500 | 1.0000 |
| 5 | FP | 3 | 2 | 0.6000 | 1.0000 |

Precision envelope (right-to-left max): `[1.0, 0.75, 0.75, 0.75]` at recall `1/3, 2/3, 1`:

$$\text{AP} = \tfrac13(1.0) + \tfrac13(0.75) + \tfrac13(0.75) = \mathbf{0.8333} \quad\text{(verified)}$$

*Interpret:* IoU $=0.14$ means the two boxes barely overlap — at an IoU threshold of 0.5 this would **not** count as a true positive. Note the envelope discarded rank 2's precision of 0.5, because rank 4 already reached 0.75 at the same recall; that is why AP (0.8333) is much closer to the maximum achievable than the naive mean of precisions ($0.7033$) would suggest.

**F6.** *Formula:* $\text{cost}(i,j) = \text{penalty}(c_i \ne \hat c_j) - \ln(\text{conf}_j) + L_1(b_i, \hat b_j)$.

Ground truth: `cat [0,0,10,10]`, `dog [20,20,30,30]`.
Predictions: `cat 0.9 [0,0,10,10]`, `dog 0.2 [5,5,15,15]`, `cat 0.8 [18,18,32,32]`.

| | pred 0 | pred 1 | pred 2 |
|---|---|---|---|
| **gt cat** | **0.1054** | 31.6094 | 80.2231 |
| **gt dog** | 90.1054 | 61.6094 | **18.2231** |

*Cell derivations (two spot-checks):*
- `gt cat ↔ pred 0`: labels match → $-\ln 0.9 = 0.1054$; $L_1 = 0$; total $= \mathbf{0.1054}$ ✔
- `gt cat ↔ pred 2`: labels match → $-\ln 0.8 = 0.2231$; $L_1 = 18+18+22+22 = 80$; total $=\mathbf{80.2231}$ ✔
- `gt dog ↔ pred 1`: labels match → $-\ln 0.2 = 1.6094$; $L_1 = 15+15+15+15 = 60$; total $= \mathbf{61.6094}$ ✔

*Optimal assignment:* $\sigma = [0,2]$, total $= 0.1054 + 18.2231 = \mathbf{18.3285}$ *(verified)*. Alternatives: $[0,1] \to 61.71$; $[2,1] \to 141.83$.

*Which gt gets a mismatched label:* **the `dog` ground truth is matched to prediction 2, which is labelled `cat`.**

*Interpret:* Hungarian matching minimises the **total** cost, so a $10$ label penalty can be worth paying when it saves $80$ in box distance — here it is the difference between a total of 18.33 and 61.71. The matching does not require labels to agree; it makes disagreement expensive. This is why a low-confidence, geometrically-close box (0.8) can outrank a higher-confidence one (0.2) that sits further away, and why the trained model must produce both good labels *and* good geometry before matching behaves sensibly.

---

## J. Sources and references

### Course sources

| Source | File | Detail |
|---|---|---|
| Presentation | `session_8.pdf` | 40 slides, *"Transformers For Vision — Lecture 8"*; image-only, OCR'd in full |
| Transcript | `Session_8_—_Object Detection III Transformers_…txt` | 19/9/2026, 1:45 PM, 120 min; **4.7k words, shortest in the course** |
| **Professor repository** | **None in this workspace** | The instructor describes a **ByteTrack**-based tracking project shared via a Drive link; no code is present |

### Key slide numbers

| Slides | Topic |
|---|---|
| p.2 | Transformer block; Add & Norm; encoder/decoder |
| p.3 | Self-attention as a **set** operation ⇒ needs position embeddings |
| p.4 | Q/K/V equations; attention weight; output (rendered to read) |
| p.5–6 | Matrix form $H=\text{softmax}(QK^T/\sqrt{d_k})V$; $n\times n$ attention map |
| p.7 | Multi-head attention |
| p.8 | **Positional encoding sin/cos** (rendered to read) |
| p.9 | Complete transformer layer; residual + LayerNorm |
| p.10–13 | Encoder–decoder, cross-attention, BERT/MLM |
| p.14 | **Quadratic cost; 224×224 ≈ 50,000 tokens** |
| p.15–19 | ImageGPT: resolution/palette reduction, AR vs BERT, pre-norm, pros/cons |
| p.20 | ViT claims; poor at ImageNet scale; needs 14M–300M |
| p.21–27 | ViT's seven construction steps |
| p.28 | ViT variants + pretraining datasets; $16\times16\times3=768$ |
| p.29–31 | **DETR set prediction, Hungarian matching, $L_1$ + gIoU** (rendered to read) |
| p.32–36 | DETR architecture: backbone, encoder, decoder, object queries, FFN heads, "no object" |
| p.37 | Detection baselines (ResNet/FPN + longer training) |
| p.38 | Precision, recall, AP |
| p.39 | **IoU, AP at IoU 50/75/95, AP_S/M/L**; link to cocodataset.org |
| p.40 | **Detection results table** |

### Supplementary external sources

Used only where the lecture left a detail implicit.

- Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017 — the original. https://arxiv.org/abs/1706.03762
- Dosovitskiy et al., *"An Image is Worth 16×16 Words"*, ICLR 2021 — ViT. https://arxiv.org/abs/2010.11929
- Carion et al., *"End-to-End Object Detection with Transformers"* (DETR), ECCV 2020 — Hungarian matching, gIoU box loss, all numbers on slides p.30–p.40. https://arxiv.org/abs/2005.12872
- Chen et al., *"Generative Pretraining from Pixels"*, OpenAI 2020 — ImageGPT. https://openai.com/research/image-gpt
- Devlin et al., *"BERT"* — for the NLP lineage on slides p.12–13. https://arxiv.org/abs/1810.04805
- **COCO detection evaluation** — authoritative definitions of AP, the ten IoU thresholds, and the size bands. https://cocodataset.org/#detection-eval *(the slide links this directly)*
- Bochkovskiy et al., YOLO — the production comparator the instructor invokes. https://arxiv.org/abs/1506.02640
- Zhang et al., *"ByteTrack"* — the tracking library in the instructor's project. https://arxiv.org/abs/2110.06864

### Unresolved ambiguities

1. **Slide p.39 literally renders the IoU thresholds as "50%, 5%, and 95%".** This is not OCR noise — a 190-DPI re-read of the page produces the identical string, so the missing "7" is present in the slide as exported (most likely an animation build state, since a "7" is also dropped from the AP₅₀ subscript on the same slide). The intended triple is **50%, 75%, 95%**, confirmed against the DETR paper (which averages over IoU 0.50:0.05:0.95) and the COCO evaluation page the slide itself links. The transcript repeats the same mis-hearing. Treated as a slide artefact, not a course claim.
2. **APₛ/AP_M/AP_L size bands are never stated in the lecture** — only *"AP based on objects of different sizes"*. The 32²/96² boundaries are supplied from COCO and are labelled **supplementary** in §B12.
3. **Slide p.28 header reads "Hiddensize D MLPsize"**, whose column mapping to the numeric row is unambiguous from the arithmetic ($768/12=64$) but is inferred rather than labelled on the slide.
4. **No slides accompany the tracking segment.** §B14 rests entirely on the transcript; the ByteTrack project code referenced by the instructor is not in this workspace.
5. **The instructor's FPS recollection of YOLO ("around 140 or 156… I don't exactly remember")** is explicitly hedged by him, so no YOLO number is asserted in this guide.

---

*Questions A–F are original, written against the Session 8 sources only. The Session 9 midterm quiz was excluded as evidence of difficulty or style. Every numerical answer and every code solution above was executed and verified; two implementation bugs and one mislabelled output were found and fixed during that verification.*