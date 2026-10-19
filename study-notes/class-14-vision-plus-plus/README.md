# Class 14 — Vision++ with Foundational Models

> **Exam weight: Tier 1 everywhere.** This class is a *tour of paper walkthroughs* — DINO, Grounding DINO, SAM, Depth Anything — all built on one idea: **a large pre-trained backbone plus a light, task-specific head** (and, in DINO's case, no labels at all). It is unusually rich in quotable numbers (ViT scale ladder, DINO kNN, Grounding-DINO's zero-shot table, SAM's 1.1B masks and 20:1 loss ratio, Depth Anything's 1.5M + 62M images), so it feeds A, B, C **and** F.
>
> **Sources:** `session_14__2_.pdf` (51 slides, image-only, upright, *"Vision++, with Foundational Models"*) · `Session_14_—_Vision++,_with_Foundational_Models_….txt` (transcript).
> **Professor repository: none in this workspace.**

**Every §B concept carries an intuitive explanation *and* a concrete "▸ Walk through it".** The examiner's favourite question in this class is *"what does the frozen backbone do, and what is actually trained?"* — every subsection ends by answering it.

---

## A. Lecture overview

**The one-sentence version.** Self-supervised ViTs (DINO) learn a **general visual representation without labels**; that single frozen representation is then reused by three different heads — **language grounding** (Grounding DINO), **promptable segmentation** (SAM), and **monocular depth** (Depth Anything) — which is what "Vision++" means.

**The whole class in eight lines:**

```
  ViT            image → 16×16 patches → tokens → transformer
                 needs LOTS of supervised data (JFT-300M)

  SiT            ViT + denoising autoencoder + rotation + contrastive
                 ⇒ self-supervised ViT (beats SimCLR)

  DINO           student/teacher self-distillation, NO labels
                 ⇒ features so good a kNN gets 74.5%

  Grounding DINO DINO detector + TEXT  ⇒ open-set detection
  SAM            ViT (run once) + PROMPT encoder + mask decoder
                 ⇒ segment anything you can point at
  Depth Anything frozen DINOv2 + 62M unlabeled images
                 ⇒ monocular depth from one photo

  The pattern    big frozen backbone  +  small trained head
                 + a data engine to make labels cheap
```

**The arc of the lecture.** (1) ViT recap and its appetite for data. (2) SiT: make ViT self-supervised. (3) DINO: self-distillation, the representation that everything else borrows. (4) Closed-set → open-set detection; Grounding DINO and language. (5) SAM: promptable segmentation, its architecture, its losses, its 1.1B-mask data engine. (6) Depth Anything: the same teacher–student + frozen-backbone recipe applied to depth.

---

## B. The ideas

### B1. ViT in one slide — patches, tokens, CLS, positions

**The intuition.** A transformer eats a **sequence of vectors**, not an image grid. So chop the image into fixed patches, flatten each patch, project it linearly to `D` dimensions — each patch is now a "word". Add a learnable **CLS token** whose final state is the image representation, and add **position embeddings** so the transformer knows where each patch came from.

**▸ Walk through it.** A $224\times224$ RGB image with $16\times16$ patches gives $(224/16)^2 = 14^2 = 196$ patches. Each $16\times16\times3$ patch is flattened to 768 numbers (for ViT-B) and linearly projected; prepend one CLS token → **197 tokens**. The transformer does self-attention over all 197 (order does not matter, hence the position embeddings), and the CLS output goes to a softmax classifier. For a $384\times384$ input the same arithmetic gives $24^2=576$ patches → **577 tokens**.

**The takeaway.** *ViT = patchify → linear embed → prepend CLS → add positions → transformer. The number of tokens grows with the square of resolution, which is why attention cost explodes.*

---

### B2. ViT's dirty secret — it needs enormous supervised data

**The intuition.** Convolutions have **built-in priors** (locality, translation equivariance). A transformer has almost none, so it must *learn* those priors from data. Give it too little and it loses to a CNN.

**▸ Walk through it.** The slide's three-rung ladder: on **ImageNet-1k (1M images)** ViT is merely "acceptable"; on **ImageNet-21k (14M)** it is "great"; on **JFT-300M (300M)** it is state of the art — ImageNet **88.55%** vs **88.5%** for the previous best, ImageNet-Real **90.72%**, CIFAR-10 **99.50%**, CIFAR-100 **94.55%**, Pets **97.56%**, Flowers **99.68%**, plus VTAB's 19 tasks. The compute axis on the same slide rises from ~1 to ~80 exaFLOPs.

**The takeaway.** *ViT trades inductive bias for scale. That is exactly why the rest of the lecture is about getting a good representation **without** 300M labels.*

---

### B3. SiT — a self-supervised ViT

**The intuition.** BERT and GPT learn from **corrupting** text and predicting what was removed. Do the same to patches: mask / replace / colour-distort / blur parts of the image and make the ViT **reconstruct** the corrupted patches. That is a denoising autoencoder with a transformer inside.

**▸ Walk through it.** Take an image, corrupt some patches, feed the visible patches through the ViT encoder, and ask a decoder to reconstruct the missing ones **autoregressively**. Because reconstructing pixels alone can produce a representation that ignores semantics, SiT adds two **auxiliary** objectives: a **rotation loss** — predict the orientation the image was rotated by (e.g. 30°) — and a **contrastive loss** — force high-level representations of *different* images apart. On CIFAR-10 linear evaluation SiT reaches **83.50%** vs **78.75%** for SimCLR with a ResNet-56; on CIFAR-100 **57.75%** vs **44.33%**; on Tiny-ImageNet **43.06%** vs **25.79%**.

**The takeaway.** *SiT = masked/denoising reconstruction + rotation prediction + contrastive separation. Reconstruction learns pixels; the two auxiliary losses force the features to be semantic.*

---

### B4. DINO — self-distillation with no labels at all

**The intuition.** Drop reconstruction and drop negatives. Use **two copies of the same network**: a **teacher** and a **student**. Show the teacher only *global* views (whole object), the student both *global and local* views (crops/zooms). Train the student to reproduce the teacher's **softmax distribution** over prototypes. No labels, no contrastive negatives — just "agree with the teacher, through a harder view".

**▸ Walk through it.** For each image, make 2 global crops and several local crops. Teacher sees the 2 global crops; student sees all of them. Both end in an MLP head producing a softmax over $K$ prototypes. The loss is cross-entropy (equivalently KL) between teacher and student distributions. The teacher is **not** trained by gradients — it is an **exponential moving average** of the student:

$$\theta_{\text{teacher}} \leftarrow \beta\,\theta_{\text{teacher}} + (1-\beta)\,\theta_{\text{student}}.$$

The "classes" are meaningless during pre-training; what matters is the **pre-softmax representation**. That representation is so well clustered that a plain **kNN** classifier on frozen DINO features reaches **74.5%** on ImageNet, versus **79.3%** for a fully supervised ResNet-50.

**The takeaway.** *DINO's product is not a classifier, it is a **feature extractor**. kNN 74.5 vs 79.3 says the gap to full supervision is under 5 points — with zero labels.*

---

### B5. Closed-set vs open-set detection

**The intuition.** A normal detector is trained on a fixed label set (COCO's 80 classes). Ask it for "zebra crossing" or "the red mug on the left" and it cannot answer — not because it is bad, but because the class was never a row in its output layer.

**▸ Walk through it.** Slide 10 draws the two boxes side by side: **closed-set** — *"COCO pre-defined categories"*, output is one of 80; **open-set** — *"human-input novel categories"*, output is whatever the prompt names. The lecture's framing of the fix is a three-part equation: `transformer detector (DINO) + grounded pre-training = Grounding DINO`.

**The takeaway.** *Closed-set = classes fixed at training time; open-set = classes supplied at inference time, usually as text. The bridge is language.*

---

### B6. Grounding DINO — putting language into the detector

**The intuition.** Keep DINO's detector, add a **text encoder**, and let image features and text features **talk to each other** before queries are chosen. Then a query is not "object 37" — it is "object that matches this phrase".

**▸ Walk through it.** Input: an image and a prompt such as `"the cat"` or `"a red car . a dog ."`. The prompt is encoded by a text backbone (BERT in the paper), the image by a Swin transformer. A **feature enhancer** performs image self-attention, **text-to-image cross-attention**, and **image-to-text cross-attention**, so each modality is updated using the other. Then **language-guided query selection** picks the image tokens most relevant to the text, the **cross-modality decoder** refines boxes, and two heads fire: a **contrastive** score (does this box match this phrase?) and a **localization** loss (**L1 + GIoU**).

**The takeaway.** *Grounding DINO = DINO + a text tower + cross-modal fusion + language-guided queries. Detection becomes "find the thing this sentence describes".*

---

### B7. What zero-shot grounding actually buys (the numbers)

**The intuition.** If language really carries the class, then a model trained on *grounding* data should detect COCO objects it never saw labelled — and fine-tuning should add a smaller, separate gain.

**▸ Walk through it.** The slide's COCO table, Swin-T rows: DINO(Swin-T)-F gets **46.2** zero-shot AP; Grounding-DINO-T gets **46.7** with the same O365 pre-training. Adding **GoldG** grounding data lifts it to **48.1**; adding **Cap4M** captions lifts it to **48.4**. Fine-tuning on COCO then gives **56.9 / 57.1 / 57.2**. The large Swin-L model goes from **52.5** zero-shot (O365 + O1 + GoldG) to **60.7** when COCO RefC is included, and **63.0** fine-tuned; GLIP-L, the prior open-set reference, is **49.8**.

**The takeaway.** *Language grounding is worth ~+2 AP zero-shot over a detector of the same size; fine-tuning is worth another ~+9. Both matter, and they are not the same thing.*

---

### B8. SAM's three innovations

**The intuition.** SAM is not "a better segmentation net". It is a **task**, a **model**, and a **dataset** designed together so that "segment" becomes a *promptable* operation, like next-token prediction for LLMs.

**▸ Walk through it.** (1) **Promptable segmentation task** — given *any* prompt indicating what to segment (foreground/background points, a rough box, a mask, free-form text, or a combination), return a valid mask. (2) **Segment Anything Model** — a fast encoder–decoder that returns a mask in about **50 ms in a browser**, and is **ambiguity-aware**: one point may mean the part, the subpart, or the whole object, so it returns the **three most likely** masks. (3) **Segment Anything Dataset** — **1.1 billion** masks collected with the Segment Anything Engine, **all generated automatically**, with no human mask labelling.

**The takeaway.** *SAM = promptable task + ambiguity-aware encoder–decoder + a 1.1B-mask automatic data engine. Its novelty is the **task formulation**, not a new convolution.*

---

### B9. SAM's architecture — encoder once, decoder per prompt

**The intuition.** Image encoding is expensive; prompting is interactive. So split them: run the heavy image encoder **once per image**, then answer each prompt with a cheap decoder. That is what makes 50 ms interaction possible.

**▸ Walk through it.** **Image encoder:** an MAE-pre-trained ViT, minimally adapted for high resolution; it runs once and can be cached before any prompt arrives. **Prompt encoder:** two families. *Sparse* prompts (points, boxes, text) are each mapped to a **256-dimensional** vector — a point is a positional encoding of its coordinate **plus** one of two learned embeddings (foreground/background); a box is a pair of positional encodings, one tagged "top-left", one "bottom-right"; text uses **CLIP's** text encoder. *Dense* prompts (masks) go through convolutions at **4× lower** resolution, then two stride-2 $2\times2$ convolutions (channels 4 then 16), then a $1\times1$ conv to 256, with GELU and layer norm between them. **Mask decoder:** self-attention over image features, cross-attention from prompt tokens, a transposed convolution to upsample, an MLP, and **three** mask outputs plus an IoU score per mask. If there is no mask prompt, a learned "no mask" embedding is added everywhere.

**The takeaway.** *SAM's cost model is "one encoder pass per image, one cheap decoder pass per prompt". Everything a prompt touches is 256-dimensional, which is why prompts of wildly different types can be mixed.*

---

### B10. SAM's losses — focal, dice, and the IoU head

**The intuition.** Segmentation is **extremely imbalanced**: almost every pixel is background. Plain cross-entropy is dominated by easy background pixels, so the model learns nothing about the small object. Two fixes are combined: **focal loss** down-weights easy examples, and **dice loss** measures overlap directly.

**▸ Walk through it.** **Focal loss** adds a modulating factor to cross-entropy:

$$\mathrm{FL}(p_t) = -(1-p_t)^{\gamma}\log p_t .$$

With $\gamma=2$ and $p_t=0.9$ (easy, correct) the loss is $0.01\times0.105=0.00105$; with $p_t=0.6$ (harder) it is $0.16\times0.511=0.0817$ — **77× larger**. **Dice** is the F1 of two sets:

$$\mathrm{Dice}=\frac{2|A\cap B|}{|A|+|B|},\qquad \text{Dice loss}=1-\mathrm{Dice}.$$

SAM trains masks with **focal : dice = 20 : 1**, and trains a separate **IoU-prediction head** with **MSE** against the true IoU of the predicted mask, added with weight 1.0.

**The takeaway.** *Focal fixes class imbalance; dice fixes overlap; the IoU head lets the model *rank* its own three masks at inference. The 20:1 ratio is the fact to remember.*

---

### B11. Why dice is always ≥ IoU

**The intuition.** Both measure overlap, but they penalise differently when the masks are small.

**▸ Walk through it.** Write $|A\cap B|=i$, $|A|=a$, $|B|=b$. Then

$$\mathrm{IoU}=\frac{i}{a+b-i},\qquad \mathrm{Dice}=\frac{2i}{a+b}.$$

Compare $\frac{2i}{a+b}$ with $\frac{i}{a+b-i}$: cross-multiplying, Dice ≥ IoU iff $2(a+b-i)\ge a+b$, i.e. $a+b\ge 2i$, which is true because $i\le a$ and $i\le b$. Equality only when $a=b=i$. With $i=300,a=400,b=500$: IoU $=300/600=0.5$, Dice $=600/900=0.667$.

**The takeaway.** *Dice is the gentler, more forgiving overlap score; on tiny objects it still gives a usable gradient where IoU is near zero. That is why segmentation losses use dice.*

---

### B12. Monocular depth — the problem, and three old answers

**The intuition.** A single image has lost the third dimension. A big car far away and a toy car close up can project to the same pixels. Concave and convex look identical under the wrong lighting. Depth from one camera is **ill-posed** — but the world is regular enough that a strong prior can recover a lot.

**▸ Walk through it.** Sensors (LiDAR, stereo, time-of-flight) are accurate but expensive, bulky, and complex. A single camera is cheap, light, and already on every device. Before foundation models there were three families: **regression** (predict a depth value per pixel), **classification** (discretise depth into bins, predict the bin), and **generative** (GANs/VAEs/diffusion generate the depth map). All three are typically **application-specific** — indoor vs outdoor, relative vs metric, sparse vs dense — so a model trained for one setting does not transfer.

**The takeaway.** *Depth is ill-posed from one view; classical methods solve it per setting. The foundation-model opportunity is **zero-shot generalisation**, limited only by the shortage of depth labels.*

---

### B13. Depth Anything — the teacher–student data engine

**The intuition.** Web images are abundant; depth labels are not. So train a **teacher** on the small labelled set, let it **pseudo-label** 62M unlabelled images, and train a **student** on both. Exactly the DINO playbook, applied to depth.

**▸ Walk through it.** Stage 1: train a teacher on **1.5M** labelled images (labels from LiDAR, stereo matching, or structure-from-motion) with a pre-trained **DINOv2** encoder. Stage 2: the teacher labels **62M** unlabelled indoor and outdoor images; the student — same architecture, **randomly initialised decoder**, DINOv2 encoder — trains on labelled data plus teacher pseudo-labels. Stage 3 (optional): fine-tune for metric depth or segmentation.

**The takeaway.** *Depth Anything = DINOv2 encoder + teacher pseudo-labelling + a student trained on 1.5M labelled and 62M unlabelled images. The data engine, not the architecture, is the contribution.*

---

### B14. The two tricks that make the student work

**The intuition.** Adding unlabelled data *alone did not beat the labelled-only baseline*. Two ingredients were needed: make the student's inputs harder, and stop the student's features from drifting away from DINOv2's semantics.

**▸ Walk through it.** **Trick 1 — strong perturbations.** The teacher sees the clean image; the student sees a corrupted version: **colour jitter** (brightness/contrast/hue), **noise**, **spatial distortion** (warping), and **CutMix** (paste two images together with a mask, 50%). Same logic as DINO/BYOL: a car is still a car when blurred or mirrored. **Trick 2 — feature alignment.** Keep the student encoder's features close to **frozen DINOv2**, measured by **cosine similarity**. An auxiliary segmentation head was tried and **failed**. The caveat is a **margin $\sigma=0.15$**: DINOv2 considers a car's front and rear "alike", but depth must distinguish them, so full alignment would destroy the depth signal.

**The takeaway.** *Perturb the student, align it to a frozen semantic teacher — but only up to a margin. The margin is the numerical hook: $\sigma=0.15$.*

---

### B15. Three losses, one sentence each

**The intuition.** Each loss answers a different question: *is the depth right where we have labels?*, *are the features still semantic?*, *is the depth right where we only have pseudo-labels?*

**▸ Walk through it.** **$L_l$ — labelled:** mean absolute error between student prediction and ground-truth depth, averaged over pixels. **$L_{feat}$ — alignment:** $1-\text{mean cosine}$ between student and frozen DINOv2 features, with margin $\sigma$ (so it is clamped once similarity is high enough). **$L_u$ — unlabelled:** MAE between the student's prediction on the perturbed/CutMix image and the teacher's pseudo-label for that image, computed **separately per image region of the CutMix mask $M$** so each region is compared to the right target.

**The takeaway.** *$L_l$ teaches depth, $L_{feat}$ preserves semantics, $L_u$ exploits 62M unlabelled images. The mask in $L_u$ is what makes CutMix supervision correct.*

---

### B16. The through-line — frozen backbone, small head, cheap labels

**The intuition.** All four systems are the same recipe with a different head:

| system | frozen/reused part | trained part | label source |
|---|---|---|---|
| SiT | — | whole ViT | self-supervision |
| DINO | teacher (EMA) | student | none |
| Grounding DINO | — | detector + text fusion | grounding data + captions |
| SAM | image encoder (run once per image) | prompt encoder + mask decoder | 1.1B auto masks |
| Depth Anything | DINOv2 encoder (alignment target) | decoder | 1.5M labels + 62M pseudo |

**▸ Walk through it.** Ask of any of these systems: *what is expensive and fixed, what is cheap and learned, and where do the labels come from?* For SAM the encoder runs once per image and the prompt decoder is tiny; for Depth Anything the DINOv2 encoder is frozen for alignment while the decoder learns depth; for Grounding DINO nothing is frozen but the text tower supplies the open vocabulary.

**The takeaway.** *"Vision++" = reuse a general representation and train a small head against a cheap label source. If asked "what is trained?", answer with the middle column.*

---

## C. Numbers worth knowing

All re-derived by execution.

### C1. ViT scale ladder (slide 3)

| pre-training set | size | verdict |
|---|---|---|
| ImageNet-1k | **1 M** | "acceptable" |
| ImageNet-21k | **14 M** | "great" |
| JFT-300M | **300 M** | state of the art |

| benchmark | ViT (JFT-300M) | previous SOTA |
|---|---|---|
| ImageNet | **88.55** | **88.5** |
| ImageNet-Real | **90.72** | — |
| CIFAR-10 | **99.50** | **99.37** |
| CIFAR-100 | **94.55** | **93.51** |
| Pets | **97.56** | **96.62** |
| Flowers | **99.68** | **99.63** |

### C2. SiT vs SimCLR, linear evaluation (slide 6)

| method | backbone | CIFAR-10 | CIFAR-100 | Tiny-IN | C100→C10 | C10→C100 |
|---|---|---|---|---|---|---|
| SimCLR | ResNet-56 | 78.75 | 44.33 | n/a | 66.19 | 36.79 |
| **SiT** | Transformer | **83.50** | **57.75** | **43.06** | **75.52** | **57.89** |

### C3. DINO representation quality (slide 9)

| classifier on frozen features | accuracy |
|---|---|
| kNN on **DINO** | **74.5%** |
| supervised **ResNet-50** | **79.3%** |

### C4. Grounding DINO zero-shot COCO (slide 24)

| model | backbone | pre-training | zero-shot AP | fine-tuned |
|---|---|---|---|---|
| Faster R-CNN | RN50-FPN | — | — | 40.2 |
| DINO(Swin-T)-F | Swin-T | O365 | **46.2** | 56.9 |
| **Grounding-DINO-T** | Swin-T | O365 | **46.7** | 56.9 |
| **Grounding-DINO-T** | Swin-T | + GoldG | **48.1** | 57.1 |
| **Grounding-DINO-T** | Swin-T | + Cap4M | **48.4** | 57.2 |
| GLIP-L | Swin-L | FourODs, GoldG, Cap24M | **49.8** | 60.8 |
| **Grounding-DINO-L** | Swin-L | O365, O1, GoldG | **52.5** | 63.0 |

### C5. SAM facts (slides 25–41)

| quantity | value |
|---|---|
| masks in SA-1B | **1.1 B** (all automatic) |
| mask given a prompt | **~50 ms** in a browser |
| masks returned per ambiguous prompt | **3** (part, subpart, whole) |
| prompt embedding width | **256** |
| focal : dice loss ratio | **20 : 1** |
| IoU-head loss | **MSE**, weight 1.0 |

### C6. Depth Anything (slides 45–51)

| quantity | value |
|---|---|
| labelled images | **1.5 M** |
| unlabelled (pseudo-labelled) images | **62 M** |
| feature-alignment margin $\sigma$ | **0.15** |
| perturbations | colour jitter, noise, spatial distortion, CutMix 50% |
| losses | $L_l$ (MAE), $L_{feat}$ (1 − cos), $L_u$ (MAE, masked) |

---

## D. Pictures

### D1. ViT

```
   224×224×3
      │  split into 16×16 patches
      ▼
   [p1][p2] ... [p196]        196 patches
      │  flatten + linear
      ▼
   [CLS][t1][t2] ... [t196]   197 tokens, each 768-d
      │  + position embeddings
      ▼
   ┌──────── transformer encoder ────────┐
   │  self-attention over all 197 tokens │
   └─────────────────────────────────────┘
      │
      ▼
   CLS ──► linear ──► class
```

### D2. DINO self-distillation

```
                 2 global crops
   image ──┬────────────────────► [ TEACHER  ViT + MLP ] ──► p_t
           │                              ▲
           │                              │ EMA: θ_t ← βθ_t + (1−β)θ_s
           │                              │ (no gradient)
           └── 2 global + n local ──► [ STUDENT  ViT + MLP ] ──► p_s
                                          │
                                          ▼
                              loss = CE(p_t, p_s)   (or KL)
                              no labels, no negatives
```

### D3. Grounding DINO

```
   text "a red car . a dog ." ──► [text backbone] ──┐
                                                     │ text features
   image ──► [image backbone Swin] ──► image feats ──┤
                                                     ▼
                        ┌────── FEATURE ENHANCER ──────┐
                        │ image self-attn              │
                        │ text→image cross-attn        │
                        │ image→text cross-attn        │
                        └──────────────┬───────────────┘
                                       ▼
                        language-guided query selection
                                       ▼
                        ┌──── CROSS-MODALITY DECODER ────┐
                        │ boxes + phrases refined         │
                        └───────┬─────────────┬───────────┘
                                ▼             ▼
                     contrastive score    L1 + GIoU
```

### D4. SAM

```
                        ┌──────────────┐
   image ──────────────►│ image encoder│──► embedding (run ONCE)
                        │  (MAE ViT)   │
                        └──────────────┘
                                     │
   prompts                           ▼
   ├─ points ─┐              ┌───────────────┐
   ├─ box ────┼─ 256-d ─────►│  mask decoder │──► 3 masks
   ├─ text ───┘  (CLIP)      │ self + cross  │     + 3 IoU scores
   └─ mask ──── conv ───────►│  attn, upconv │
                             └───────────────┘
        sparse: positional enc + learned type embedding
        dense : conv 4×↓ → 2×2/stride2 (4,16 ch) → 1×1 → 256
```

### D5. Depth Anything

```
   labelled 1.5M ──► [ TEACHER: DINOv2 enc + decoder ] ──► depth
                                   │
                                   │ pseudo-labels
                                   ▼
   unlabelled 62M ─────────────► [ STUDENT ]  ◄── perturb:
                                   │   colour jitter, noise,
                                   │   spatial warp, CutMix
                                   ▼
                              depth prediction
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        L_l : MAE vs GT     L_feat : 1−cos vs      L_u : MAE vs
        (labelled only)      frozen DINOv2 (σ)      teacher (masked)
```

---

## E. Code

Section D of the exam asks you to *implement* a concept. For this class the natural targets are exactly the quantities the papers define: **IoU / Dice**, **focal loss + SAM's 20:1 combination**, and **EMA teacher update + margin-clamped feature alignment**. Full specifications and verified outputs are in **Section D**; reference implementations are in the worked solutions.

**Four one-liners worth memorising.**

```
IoU   = |A∩B| / |A∪B|                  Dice = 2|A∩B| / (|A|+|B|)
FL    = −(1−p_t)^γ · log p_t           L_feat = max(0, 1 − cos − σ)
θ_t  ← β θ_t + (1−β) θ_s               (β ≈ 0.996)
```

---

## F. Comparisons

| | **ViT** | **SiT** | **DINO** | **Grounding DINO** | **SAM** | **Depth Anything** |
|---|---|---|---|---|---|---|
| task | classify | representation | representation | open-set detect | promptable segment | monocular depth |
| labels | supervised | **none** | **none** | grounding + captions | **1.1B auto masks** | 1.5M + 62M pseudo |
| backbone | ViT | ViT | ViT (×2: student/teacher) | Swin + BERT | MAE ViT | DINOv2 |
| extra machinery | — | rotation + contrastive loss | EMA teacher, global/local crops | feature enhancer, language-guided queries | prompt encoder, 3-mask decoder | teacher, perturbations, feature alignment |
| headline number | 88.55% IN | 83.5% CIFAR-10 | **74.5% kNN** | **52.5 AP** zero-shot | **50 ms**, 1.1B masks | 62M unlabelled |
| key loss | CE | recon + aux | CE/KL teacher→student | contrastive + L1 + GIoU | **focal 20 : dice 1** + IoU MSE | MAE + cosine + masked MAE |

**Decision line.** Use **DINO** when you need a general frozen feature extractor with no labels; **Grounding DINO** when the object classes are named at inference time; **SAM** when a human (or another model) can point at the object; **Depth Anything** when you need dense depth from one cheap camera.

---

## G. Exam-focused revision sheet

- **ViT:** $224/16=14$ → **196 patches**, +CLS = **197 tokens**; $384$ → **577**. Needs scale: 1M → 14M → 300M.
- **ViT results:** ImageNet **88.55**, CIFAR-100 **94.55**, Pets **97.56**, Flowers **99.68**.
- **SiT:** denoising autoencoder over corrupted patches + **rotation loss** + **contrastive loss**; CIFAR-10 **83.50** vs SimCLR **78.75**.
- **DINO:** **self-distillation**, teacher = EMA of student, teacher sees **global** crops, student sees **global + local**; no labels, no negatives; **kNN 74.5** vs supervised **79.3**.
- **Closed-set vs open-set:** COCO's 80 vs "human-input novel categories".
- **Grounding DINO:** DINO detector + **text**; **dual-encoder, single-decoder**; components = image backbone, text backbone, **feature enhancer**, **language-guided query selection**, **cross-modality decoder**; losses = **contrastive + L1 + GIoU**.
- **Grounding numbers:** T zero-shot **46.7 → 48.1 (+GoldG) → 48.4 (+Cap4M)**; fine-tuned **56.9/57.1/57.2**; L **52.5** zero-shot, **63.0** fine-tuned; GLIP-L **49.8**.
- **SAM's 3 innovations:** promptable task · 50 ms model · **1.1B** automatic masks.
- **SAM prompts:** points (positional + fg/bg embedding), box (top-left + bottom-right), text (**CLIP**), mask (conv); all **256-d**.
- **SAM decoder:** self-attn + cross-attn, transposed conv, MLP, **3 masks** + IoU head.
- **SAM losses:** **focal : dice = 20 : 1**; IoU head **MSE**; Dice **≥** IoU always.
- **Focal:** $-(1-p_t)^\gamma\log p_t$; down-weights easy examples; fixes background-heavy segmentation.
- **Depth is ill-posed:** size vs distance, concave vs convex.
- **Depth Anything:** **1.5M** labelled + **62M** unlabelled; DINOv2 encoder; perturbations + **margin σ = 0.15**; losses $L_l$, $L_{feat}$, $L_u$.
- **Failed idea:** auxiliary segmentation head for Depth Anything — did **not** work.

---

## H. Practice Questions

> **Format mirrors the exam:** A single-correct (1 mark), B multi-select (2), C descriptive ≤100 words (2), D coding (20, attempt one), E case study (15), F numericals (2.5). Attempt before reading the solutions.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** A $224\times224$ image is split into $16\times16$ patches for a ViT. Including the CLS token, the sequence length is:
A. 196  
B. 197  
C. 224  
D. 768  

**A2.** The main reason ViT underperforms CNNs when trained only on ImageNet-1k is that:
A. Transformers cannot represent images  
B. It lacks convolutional inductive biases and must learn them from data  
C. It has too many parameters  
D. Patches destroy all spatial information  

**A3.** SiT makes ViT self-supervised by:
A. Adding a segmentation head  
B. Reconstructing corrupted patches, with rotation and contrastive auxiliary losses  
C. Training on ImageNet labels  
D. Distilling a ResNet teacher  

**A4.** In DINO, the teacher network is updated by:
A. Backpropagation from the student's loss  
B. An exponential moving average of the student's weights  
C. A separate supervised loss  
D. Random re-initialisation each epoch  

**A5.** Grounding DINO turns closed-set detection into open-set detection by:
A. Adding more output classes  
B. Adding a text encoder and cross-modal fusion so classes can be named at inference  
C. Training on COCO twice  
D. Replacing the detector with SAM  

**A6.** SAM's image encoder is run:
A. Once per prompt  
B. Once per image, before prompting  
C. Once per mask  
D. Once per training step  

**A7.** SAM returns three masks for an ambiguous point prompt because:
A. It always predicts part, subpart and whole  
B. It is an ensemble of three models  
C. It predicts three colour channels  
D. Three masks are required by the dice loss  

**A8.** Depth Anything's feature-alignment loss uses a margin $\sigma=0.15$ because:
A. Cosine similarity is unbounded  
B. Full alignment to DINOv2 would erase depth cues that DINOv2 considers irrelevant  
C. It speeds up training by 15%  
D. DINOv2 features are 15-dimensional  

### Section B — Multi-select (5 × 2 marks)

**B1.** Which are true of DINO? Select all that apply.
A. It uses a student and a teacher network  
B. The teacher sees only global crops  
C. The student sees global and local crops  
D. It uses labelled ImageNet data during pre-training  
E. A kNN on its frozen features reaches 74.5%  

**B2.** Which are components of Grounding DINO? Select all that apply.
A. Image backbone  
B. Text backbone  
C. Feature enhancer with cross-modal attention  
D. Language-guided query selection  
E. A diffusion decoder  

**B3.** Which are true of SAM? Select all that apply.
A. Prompts can be points, boxes, text or masks  
B. The dense mask prompt is embedded with convolutions  
C. Sparse prompts are mapped to 256-dimensional embeddings  
D. It always outputs exactly one mask  
E. It was trained on 1.1 billion automatically generated masks  

**B4.** Which are correct about SAM's losses? Select all that apply.
A. Focal loss handles class imbalance  
B. Dice loss is $1-\text{Dice}$  
C. Focal and dice are combined in a 20 : 1 ratio  
D. The IoU head is trained with cross-entropy  
E. Dice is always ≥ IoU for the same pair of masks  

**B5.** Which are true of Depth Anything? Select all that apply.
A. It uses a DINOv2 encoder  
B. The teacher pseudo-labels 62M unlabelled images  
C. The student is trained on perturbed inputs  
D. Feature alignment is enforced without any margin  
E. An auxiliary segmentation head was tried and failed  

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Explain why ViT needs far more training data than a ResNet to reach the same accuracy.

**C2.** Describe DINO's student–teacher setup in three sentences: views, objective, teacher update.

**C3.** List the five components of Grounding DINO and state what each contributes.

**C4.** SAM makes three innovations. Name all three and say why the *task* formulation matters more than the architecture.

**C5.** Depth Anything has three losses. Name each, state what it compares, and say why the third is computed per CutMix region.

### Section D — Coding (20 marks — attempt ONE of the three)

Implement in **pure Python (standard library only)**. Return `None` on malformed input.

**D1. Segmentation overlap metrics.**
- `iou(pred, truth)` → intersection over union of two equal-length sequences of truthy/falsy mask values; `None` if lengths differ, either is `None`, both are empty, or the union is zero.
- `dice(pred, truth)` → the Sørensen–Dice score $2|A\cap B|/(|A|+|B|)$; `None` on the same malformed cases or when the denominator is zero.

```
iou([1,1,0,0],[1,0,1,0])        ->  0.3333333333333333
iou([1,1,1],[1,1,1])            ->  1.0
iou([0,0],[0,0])                ->  None
dice([1,1,0,0],[1,0,1,0])       ->  0.5
dice([1,1,1],[1,1,1])           ->  1.0
```

**D2. Focal loss and SAM's mask loss.**
- `focal_loss(p, y, gamma=2.0)` → binary focal loss for predicted probability `p` of the true class coded by `y ∈ {0,1}`: $-(1-p_t)^\gamma\log p_t$, where $p_t=p$ if $y=1$ else $1-p$. `None` if `p` is not strictly in $(0,1)$, `y` is not 0/1, or `gamma < 0`.
- `sam_mask_loss(focal, dice_score, ratio=20.0)` → `(ratio*focal + dice_score)/(ratio+1)`; `None` if any input is `None` or `ratio <= 0`.

```
focal_loss(0.9, 1)              ->  0.0010536051565782623
focal_loss(0.6, 1)              ->  0.08173209980255854
focal_loss(0.1, 0)              ->  0.0010536051565782623
focal_loss(1.0, 1)              ->  None
sam_mask_loss(0.4, 0.3)         ->  0.3952380952380953
```

**D3. DINO's EMA and Depth Anything's alignment.**
- `ema_update(teacher, student, beta)` → element-wise `beta*t + (1-beta)*s`; `None` if lengths differ, empty, or `beta ∉ [0,1]`.
- `feature_alignment_loss(student_feats, teacher_feats, sigma=0.15)` → given two equal-length lists of per-pixel feature vectors, compute the mean cosine similarity and return `max(0, 1 - mean_cos - sigma)`; `None` if lists differ in length or contain a zero vector.

```
ema_update([0.5,0.5],[1.0,0.0],0.996)                          ->  [0.502, 0.498]
feature_alignment_loss([[1,0],[0,1]],[[1,0],[1,0]],0.15)       ->  0.35
feature_alignment_loss([[1,0],[0,1]],[[1,0],[0,1]],0.15)       ->  0.0
```

### Section E — Case study (15 marks)

> **Scenario.** A warehouse robot must pick **arbitrary products** off shelves. The product list changes weekly, and new SKUs have **no labelled images**. The robot has one RGB camera, a manipulator that needs metric distance, and a 200 ms perception budget per frame. The team proposes: **Grounding DINO** to find the requested product from its name, **SAM** to produce the exact mask, and **Depth Anything** to estimate distance for grasping.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Problem.** Why does a closed-set detector fail here, and what changes when classes can be named as text?
2. **Task.** Write the three sub-tasks in order and say what each consumes and produces.
3. **Data.** What labels does each model need at training time, and what does it need for a new SKU at inference time?
4. **Model.** Explain why SAM's image encoder is run once per frame while its decoder is run per prompt.
5. **Metric.** Name one metric per sub-task and say what it measures.
6. **Failure mode.** Give two concrete failures of this pipeline and one mitigation each.

**Sub-questions**

- **E-a.** The requested prompt is `"the blue box on the left"`. Which model uses the words, and at what point in the architecture?
- **E-b.** SAM returns three masks for one point. How should the robot choose, and what does SAM provide to help?
- **E-c.** Depth Anything is trained on relative, mixed-source depth. What must be added before the arm can use its output in metres?

### Section F — Numericals (8 × 2.5 marks)

**F1.** A ViT uses $16\times16$ patches. For inputs of (a) $224\times224$ and (b) $384\times384$, compute the number of patches, the token count including CLS, and the number of unordered token pairs in self-attention.

**F2.** ViT pre-trained on JFT-300M scores 88.55 on ImageNet vs 88.5 for the previous SOTA, and 94.55 on CIFAR-100 vs 93.51. Compute the absolute and relative improvement on each.

**F3.** SiT scores 57.75 on CIFAR-100 and 43.06 on Tiny-ImageNet; SimCLR (ResNet-56) scores 44.33 and 25.79. Compute SiT's absolute and relative gain on each.

**F4.** DINO's frozen features reach 74.5% with kNN; a supervised ResNet-50 reaches 79.3%. Compute the absolute gap and DINO's percentage of the supervised score.

**F5.** A DINO teacher starts at weight 0.5 and the student is fixed at 1.0. With $\beta=0.996$, compute the teacher after one update, and the number of updates needed for the teacher to reach 0.6.

**F6.** Grounding-DINO-T scores 46.7 zero-shot on COCO; adding GoldG gives 48.1, adding Cap4M gives 48.4, and fine-tuning gives 57.2. Compute each zero-shot increment, the total zero-shot gain, and the fine-tuning gap. Compare with DINO(Swin-T)-F at 46.2.

**F7.** A predicted mask has 400 pixels, the ground truth has 500, and 300 overlap. Compute IoU and Dice. Then, given focal loss 0.4 and dice loss 0.3, compute SAM's combined mask loss at a 20 : 1 ratio.

**F8.** Depth Anything uses 1.5M labelled and 62M unlabelled images. Compute the total and the unlabelled fraction. Then compute the margin-clamped feature loss for cosine similarity 0.80 and 0.95 with $\sigma=0.15$.

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — 197.** $14^2=196$ patches plus one CLS token.

**A2. → B — It lacks convolutional inductive biases.** Locality and translation equivariance must be learned, which needs data.

**A3. → B — Reconstruct corrupted patches, with rotation and contrastive auxiliary losses.** That is the SiT recipe.

**A4. → B — An exponential moving average of the student's weights.** No gradient flows into the teacher.

**A5. → B — A text encoder and cross-modal fusion.** Classes become prompt strings, not output rows.

**A6. → B — Once per image, before prompting.** This is exactly what makes interactive latency possible.

**A7. → A — It always predicts part, subpart and whole.** Ambiguity is treated as multi-modality, not error.

**A8. → B — Full alignment would erase depth cues DINOv2 ignores.** The margin preserves front-vs-rear distinctions.

### Section B — Solutions

**B1. → A, B, C, E. Reject D.** DINO is *self*-supervised; no labels are used.

**B2. → A, B, C, D. Reject E.** Grounding DINO is a detector; diffusion belongs to Class 15.

**B3. → A, B, C, E. Reject D.** SAM deliberately returns **three** masks for ambiguity.

**B4. → A, B, C, E. Reject D.** The IoU head uses **MSE** against the true IoU, not cross-entropy.

**B5. → A, B, C, E. Reject D.** Alignment is **margin-clamped** at $\sigma=0.15$.

### Section C — Solutions

**C1.** A CNN bakes in locality and translation equivariance through its convolution kernels, so it can learn from a small dataset. A ViT treats the image as a bag of patches and must *learn* those spatial regularities from data; with too few examples it overfits the patch positions instead of the concepts. Hence the ladder: acceptable at 1M images, great at 14M, state of the art at 300M.

**C2.** For each image DINO makes two global crops and several local crops; the teacher sees only the global crops while the student sees all of them. The student minimises the cross-entropy between its softmax distribution and the teacher's, with no labels and no negatives. The teacher is not trained by gradients — it is an exponential moving average of the student's weights.

**C3.** (1) **Image backbone** (Swin) extracts visual features. (2) **Text backbone** (BERT) encodes the prompt. (3) **Feature enhancer** performs image self-attention plus text→image and image→text cross-attention, so each modality is conditioned on the other. (4) **Language-guided query selection** chooses image tokens relevant to the text to initialise queries. (5) **Cross-modality decoder** refines boxes and phrase alignment, scored by contrastive loss and localised by L1 + GIoU.

**C4.** The three innovations are the **promptable segmentation task** (any point/box/text/mask prompt yields a valid mask), the **model** (a fast, ambiguity-aware encoder–decoder returning three masks in ~50 ms), and the **dataset** (1.1B automatically generated masks). The task matters most because it defines *what counts as success* and makes pre-training possible at all: once segmentation is phrased as prompt→mask, the SA-Engine can generate training data without humans, and zero-shot transfer becomes "prompt it differently". The architecture is a means to that end.

**C5.** $L_l$ is mean absolute error between the student's depth and ground-truth depth on labelled images. $L_{feat}$ is $1-$ mean cosine similarity between student and frozen DINOv2 features, clamped by the margin $\sigma=0.15$. $L_u$ is mean absolute error between the student's prediction on a perturbed or CutMix image and the teacher's pseudo-label. It is computed **per CutMix region** because a pasted image contains two different scenes with two different depth ranges; comparing the whole image against one pseudo-label would supervise each region with the wrong target.

### Section D — Solutions

**D1.**

```python
def iou(pred, truth):
    if pred is None or truth is None or len(pred) != len(truth) or len(pred) == 0:
        return None
    inter = sum(1 for p, t in zip(pred, truth) if p and t)
    union = sum(1 for p, t in zip(pred, truth) if p or t)
    if union == 0:
        return None
    return inter / union

def dice(pred, truth):
    if pred is None or truth is None or len(pred) != len(truth) or len(pred) == 0:
        return None
    inter = sum(1 for p, t in zip(pred, truth) if p and t)
    denom = sum(1 for p in pred if p) + sum(1 for t in truth if t)
    if denom == 0:
        return None
    return 2 * inter / denom
```

Verified outputs: `iou([1,1,0,0],[1,0,1,0]) → 0.3333333333333333`; `iou([1,1,1],[1,1,1]) → 1.0`; `iou([0,0],[0,0]) → None`; `dice([1,1,0,0],[1,0,1,0]) → 0.5`; `dice([1,1,1],[1,1,1]) → 1.0`.

**D2.**

```python
import math

def focal_loss(p, y, gamma=2.0):
    if p is None or y is None or gamma is None:
        return None
    if not (0.0 < p < 1.0) or y not in (0, 1) or gamma < 0:
        return None
    pt = p if y == 1 else 1.0 - p
    return -((1.0 - pt) ** gamma) * math.log(pt)

def sam_mask_loss(focal, dice_score, ratio=20.0):
    if focal is None or dice_score is None or ratio is None or ratio <= 0:
        return None
    return (ratio * focal + dice_score) / (ratio + 1.0)
```

Verified outputs: `focal_loss(0.9,1) → 0.0010536051565782623`; `focal_loss(0.6,1) → 0.08173209980255854`; `focal_loss(0.1,0) → 0.0010536051565782623`; `focal_loss(1.0,1) → None`; `sam_mask_loss(0.4,0.3) → 0.3952380952380953`.

**D3.**

```python
import math

def ema_update(teacher, student, beta):
    if teacher is None or student is None or beta is None:
        return None
    if len(teacher) != len(student) or len(teacher) == 0:
        return None
    if not (0.0 <= beta <= 1.0):
        return None
    return [beta * t + (1.0 - beta) * s for t, s in zip(teacher, student)]

def _cos(a, b):
    if len(a) != len(b) or len(a) == 0:
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def feature_alignment_loss(student_feats, teacher_feats, sigma=0.15):
    if student_feats is None or teacher_feats is None or sigma is None:
        return None
    if len(student_feats) != len(teacher_feats) or len(student_feats) == 0:
        return None
    cs = []
    for s, t in zip(student_feats, teacher_feats):
        c = _cos(s, t)
        if c is None:
            return None
        cs.append(c)
    mean_cos = sum(cs) / len(cs)
    return max(0.0, 1.0 - mean_cos - sigma)
```

Verified outputs: `ema_update([0.5,0.5],[1.0,0.0],0.996) → [0.502, 0.498]`; `feature_alignment_loss([[1,0],[0,1]],[[1,0],[1,0]],0.15) → 0.35`; identical features → `0.0`.

### Section E — Solutions

**Model answer covering the chain.**

1. **Problem.** A closed-set detector's output layer only contains the classes it was trained on; a new SKU is not a row, so its score is meaningless. Naming the class as text moves the "class definition" out of the weights and into the prompt, so new products work with zero training.
2. **Task.** (i) **Open-set detection** — text prompt + image → boxes and phrase-box scores (Grounding DINO). (ii) **Promptable segmentation** — box/point prompt + image → mask (SAM). (iii) **Monocular depth** — RGB → dense relative depth (Depth Anything). They run in that order because each consumes the previous output.
3. **Data.** Grounding DINO: grounding/caption data (O365, GoldG, Cap4M) plus COCO for fine-tuning. SAM: 1.1B automatically generated masks. Depth Anything: 1.5M labelled depth images + 62M pseudo-labelled. **For a new SKU** the inference-time input is just its **name** (and optionally a click); no images or labels are needed.
4. **Model.** Image encoding is the expensive, prompt-independent part (a high-resolution ViT), so it is computed once per frame and cached. The mask decoder is cheap and prompt-dependent, so it runs once per prompt; this is what keeps interaction inside the latency budget.
5. **Metric.** Grounding DINO: **AP** (or AP50) — detection accuracy at a given IoU. SAM: **IoU / Dice** between predicted and true mask. Depth Anything: **absolute relative error** (or $\delta_1$ accuracy) — how close predicted depth is to truth.
6. **Failure mode.** (i) Grounding DINO misses an uncommon product whose name does not match its visual appearance → mitigate with prompt templates/ensembling or a few labelled examples. (ii) SAM segments the shelf edge instead of the product because the box is loose → mitigate with box+point prompts and use the IoU head to pick the best of three. (iii) Depth is relative, not metric → calibrate with one known distance or fine-tune to metric depth.

**Sub-questions.**

**E-a.** **Grounding DINO** uses the words. The prompt is encoded by the text backbone, fused with image features in the **feature enhancer** (text→image and image→text cross-attention), and used again in **language-guided query selection** and the **cross-modality decoder**. "On the left" is a relational phrase, so this is the hardest part of the pipeline.

**E-b.** SAM returns three masks plus an **IoU score** for each. The robot should take the mask with the highest predicted IoU, and may additionally reject it if the score is below a threshold; if the task is grasping, the "whole object" mask is preferred over the "part".

**E-c.** Depth Anything's output is **relative** (and trained on mixed sources), so it has no absolute scale. Add **metric calibration** — e.g. a known camera height or one measured distance — or **fine-tune** the model on metric depth for the robot's camera and workspace (stage 3 of its training recipe).

### Section F — Solutions

**F1.** *Formulas:* patches $=(H/16)(W/16)$; tokens $=$ patches $+1$; pairs $=n(n-1)/2$.

- $224$: $14\times14=\mathbf{196}$ patches; $\mathbf{197}$ tokens; $197\times196/2=\mathbf{19{,}306}$ pairs.
- $384$: $24\times24=\mathbf{576}$ patches; $\mathbf{577}$ tokens; $577\times576/2=\mathbf{166{,}176}$ pairs.

*Interpret:* moving from 224 to 384 raises the side by **1.71×**, the patch count by **2.94×**, and the attention pairs by **8.6×** — the quadratic cost that makes high-resolution ViT expensive.

**F2.** *Formula:* absolute $=a-b$; relative $=(a-b)/b\times100$.

- ImageNet: $88.55-88.5=\mathbf{0.05}$ pts; $0.05/88.5=\mathbf{0.057\%}$.
- CIFAR-100: $94.55-93.51=\mathbf{1.04}$ pts; $1.04/93.51=\mathbf{1.11\%}$.

*Interpret:* on saturated ImageNet the gain is tiny; on the harder 100-class problem it is ~1 point — the pattern of a model that wins by **scale**, not by a new inductive bias.

**F3.** *Formula:* absolute $=a_{\text{SiT}}-a_{\text{SimCLR}}$; relative $=\text{absolute}/a_{\text{SimCLR}}$.

- CIFAR-100: $57.75-44.33=\mathbf{13.42}$ pts; $13.42/44.33=\mathbf{30.3\%}$.
- Tiny-ImageNet: $43.06-25.79=\mathbf{17.27}$ pts; $17.27/25.79=\mathbf{67.0\%}$.

*Interpret:* the transformer backbone helps most where the task is hardest; on Tiny-ImageNet SiT is **two-thirds better** than SimCLR at the same linear-evaluation protocol.

**F4.** *Formulas:* gap $=79.3-74.5$; share $=74.5/79.3$.

- Gap: $\mathbf{4.8}$ points.
- Share: $74.5/79.3=\mathbf{93.9\%}$ (it closes **6.1%** of the gap).

*Interpret:* with **no labels at all**, DINO is within 5 points of full supervision on a kNN probe — the strongest single argument that self-supervision has "solved" representation learning.

**F5.** *Formula:* $t_n=1-(1-t_0)\beta^n$.

- One update: $0.996(0.5)+0.004(1.0)=\mathbf{0.502}$.
- Reach 0.6: $0.5(0.996)^n=0.4\Rightarrow n=\ln(0.8)/\ln(0.996)=\mathbf{55.7}\Rightarrow\mathbf{56}$ updates.

*Interpret:* $\beta=0.996$ means the teacher moves ~0.4% per step — slow enough to be a stable target, fast enough to track the student over thousands of steps.

**F6.** *Formula:* increments are successive differences; total $=48.4-46.7$; gap $=57.2-48.4$.

- GoldG: $48.1-46.7=\mathbf{+1.4}$; Cap4M: $48.4-48.1=\mathbf{+0.3}$.
- Total zero-shot: $48.4-46.7=\mathbf{+1.7}$ points ($3.6\%$ relative).
- Fine-tuning gap: $57.2-48.4=\mathbf{+8.8}$ points.
- Versus DINO(Swin-T)-F: $46.7-46.2=\mathbf{+0.5}$ zero-shot.

*Interpret:* language grounding is worth ~2 points zero-shot; **fine-tuning is worth five times more**. The two are complementary, not substitutes.

**F7.** *Formulas:* IoU $=i/(a+b-i)$; Dice $=2i/(a+b)$; loss $=(r\cdot FL+DL)/(r+1)$.

- IoU: $300/(400+500-300)=300/600=\mathbf{0.5}$.
- Dice: $2(300)/(400+500)=600/900=\mathbf{0.667}$.
- Combined: $(20\times0.4+0.3)/21=8.3/21=\mathbf{0.395238}$.

*Interpret:* Dice exceeds IoU on the same masks (0.667 vs 0.5) — the inequality from §B11. In the 20:1 combination, the focal term supplies **96.4%** of the loss (8.0/8.3), so the dice term acts as a gentle correction that fixes overlap when the object is small.

**F8.** *Formulas:* total $=1.5+62$; fraction $=62/63.5$; $L_{feat}=\max(0,1-\cos-\sigma)$.

- Total: $\mathbf{63.5}$ M images; unlabelled fraction $62/63.5=\mathbf{97.6\%}$.
- $\cos=0.80$: $\max(0,1-0.80-0.15)=\mathbf{0.05}$.
- $\cos=0.95$: $\max(0,1-0.95-0.15)=\mathbf{0}$ (clamped).

*Interpret:* **97.6%** of the training images carry no human depth label — the data engine is the model. The margin means alignment stops pushing once cosine reaches 0.85, leaving room for depth-specific distinctions such as a car's front versus rear.

---

## J. Sources

### Course sources

| source | what it provided |
|---|---|
| `Session_14_—_Vision++,_with_Foundational_Models_….txt` | the lecture narrative: why ViT needs data, DINO's teacher/student logic, SAM prompts and decoder, Depth Anything's three losses |
| `session_14__2_.pdf` (51 slides) | all figures, tables and numbers (ViT ladder, SiT table, DINO kNN, Grounding-DINO COCO table, SAM architecture and losses, Depth Anything pipeline) |

### Key slides

- **1–3** ViT: patches, tokens, scale ladder and results.
- **4–6** SiT: denoising reconstruction, rotation and contrastive losses, the comparison table.
- **7–9** DINO: global/local crops, EMA teacher, kNN 74.5 vs 79.3.
- **10–11** closed-set vs open-set detection.
- **12–24** Grounding DINO: motivation, architecture, five components, zero-shot COCO table.
- **25–27** SAM's three innovations and the task definition.
- **28–37** SAM architecture: image encoder, prompt encoder, mask convolution, decoder.
- **38–41** IoU, focal loss, dice loss, and the 20:1 combination.
- **42–51** Depth Anything: ill-posedness, prior methods, teacher/student pipeline, perturbations, feature alignment, three losses.

### Supplementary

- Dosovitskiy et al., *"An Image is Worth 16x16 Words"* (ViT), 2021.
- Atito et al., *"SiT: Self-supervised Vision Transformer"*, 2021.
- Caron et al., *"Emerging Properties in Self-Supervised Vision Transformers"* (DINO), 2021.
- Liu et al., *"Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"*, 2023.
- Kirillov et al., *"Segment Anything"*, 2023.
- Yang et al., *"Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"*, CVPR 2024.
- Lin et al., *"Focal Loss for Dense Object Detection"*, 2017 — the source of the focal-loss formula.
- Milletari et al., *"V-Net"*, 2016 — the source of the dice loss.

### Unresolved

- The SAM slide's IoU illustration prints **0.4034 / 0.7330 / 0.9264** as "poor / good / excellent"; these are pedagogical examples, not benchmark results, and are not quoted as model performance here.
- SAM's slide gives **"about 50 ms in a browser"** without stating the hardware or the image resolution; treat it as an order-of-magnitude claim.
- The Grounding-DINO Swin-L row in the OCR is partly garbled (`525`, `62.6) 62.7`, `163.0/ 63.0)`); the values used here (**52.5** zero-shot, **63.0** fine-tuned) are the unambiguous readings and match the visible column structure.
- Depth Anything's exact weighting between $L_l$, $L_{feat}$ and $L_u$ is not printed on the slides used here; only the loss *forms* and the margin $\sigma=0.15$ are quoted.
