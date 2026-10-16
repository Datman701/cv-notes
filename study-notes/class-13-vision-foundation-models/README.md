# Class 13 — Vision Foundation Models

> **Exam weight: Tier 1 for A/B/C, light for F.** This is a *conceptual* class: almost no new maths, but a dense list of named models and one-sentence distinctions that map perfectly onto the MCQ, multi-select and descriptive sections. The only quan­titative hooks are **scale** (400M vs 1.28M pairs), **generalization** (ImageNet vs ObjectNet), and **prompt gains** (+1.3%, +5%).
>
> **Sources:** `session_13_cropped.pdf` (73 image-only slides, rotated 90°, title *"CS231N: Multimodal Foundation Models"* by Shubham Jain) · `Session_13_—_Vision_Foundation_Models_….txt` (transcript, ~7k words of usable content).
> **Professor repository: none in this workspace.**

**Every §B concept below carries an intuitive explanation *and* a concrete "▸ Walk through it".** Foundation-model questions reward the *distinction*, so each subsection ends with the one sentence an examiner wants.

---

## A. Lecture overview

**The one-sentence version.** Until now every model was **specialised** (one dataset, one task); a **foundation model** is **pre-trained once on large, diverse, mostly unlabelled data** so it becomes *generally* useful, and is then adapted to many tasks by fine-tuning, zero-shot, or few-shot.

**The whole class in seven lines:**

```
  Task-specific        one dataset → one model → one task
                       (ResNet on ImageNet)

  Foundation model     large diverse data → ONE pre-trained model
                       → many tasks (zero-shot / few-shot / fine-tune)

  Classification FM    CLIP   (image-text contrastive)
                       CoCa   (CLIP + a caption GENERATION objective)

  LM + Vision FM       LLaVA     (frozen CLIP encoder + linear layer + LLM)
                       Flamingo  (frozen encoder + Perceiver Resampler
                                  + gated cross-attention → in-context learning)

  Why it works         SCALE: 400M image-text pairs, ~6× the parameters
```

**The arc of the lecture.** (1) What is a foundation model and how do you recognise one. (2) The vision-language family: **CLIP → CoCa → LLaVA → Flamingo**. (3) The engineering that makes CLIP work as a *zero-shot classifier*: the text encoder as the classifier, prompt engineering, prompt ensembling. (4) The limits: batch size, weak captions, "you can't know everything in your data". (5) A forward glance at **world models**.

---

## B. The ideas

### B1. Task-specific vs foundation model

**The intuition.** The whole course so far built one model per task. A foundation model flips this: train **one** big model on a **broad** distribution and *reuse* it. The slide's own framing: *"Train a specialized model for each task"* vs *"Pre-train one model that acts as the foundation for many different tasks."*

**▸ Walk through it.** You have four tasks: classify dogs, classify food, detect cars, segment roads. The **old** way (slide 1) trains four pipelines — `Data 1 → Model 1 → Task 1`, `Data 2 → Model 2 → Task 2`, and so on. The **foundation** way (slide 2) trains a *single* model on a large diverse dataset, then fans it out: `Foundation → Task 1 / Task 2 / Task 3 / Task 4`, each reached by only **fine-tuning, zero-shot or few-shot** adaptation.

**The takeaway.** *A task-specific model is closed (one job); a foundation model is open (many jobs). The distinguishing test is generality, not size.*

---

### B2. How do you *recognise* a foundation model?

**The intuition.** Not every large model is a foundation model. The slide separates a **defining** property from **common** properties.

**▸ Walk through it.** Ask two questions. **Always** true of a foundation model: it is **general and robust to many different tasks** — *you don't have to retrain it for each new task*. **Often** true (supporting signs, not the definition): **very many parameters**, **very large training data**, and a **self-supervised pre-training objective**.

Why self-supervision matters: at internet scale you *don't have labels* for all that data, so you cannot use classical supervised learning. Instead you design a **pretext task** the data answers about itself. The slide's recap example: take an `L*a*b*` image, hide the colour, and train the model to generate `a` and `b` from `L` alone — the labels are free, from the image.

**The takeaway.** *Generality is the definition; scale and self-supervision are the usual companions. Self-supervision is what lets the objective scale without humans.*

---

### B3. The families of foundation models

**The intuition.** "Foundation model" is an umbrella. This class only *uses* the classification and LM+vision families; the rest are context.

**▸ Walk through it.** The taxonomy slide groups them: **Language** — ELMo, BERT, GPT, T5. **Classification** — CLIP, CoCa. **LM + Vision** — LLaVA, Flamingo, GPT-4V, Gemini, Molmo. **And more** — Segment Anything, Whisper, DALL·E, Stable Diffusion, Imagen. **Chaining** — LMs + CLIP, "visual programming" (one model's output feeding another, e.g. CuPL). The lecture then declares language models **out of scope** and starts with **classification**.

**The takeaway.** *Language (ELMo/BERT/GPT/T5) · Classification (CLIP/CoCa) · LM+Vision (LLaVA/Flamingo/GPT-4V/Gemini/Molmo) · generation (SAM/Whisper/DALL·E/SD/Imagen) · chaining.*

---

### B4. Self-supervised pre-training, recapped (SimCLR)

**The intuition.** The bridge from the old classes to CLIP is **contrastive learning without labels**. Two augmentations of the same image are a *positive* pair; different images are *negatives*. Pull positives together, push negatives apart, in embedding space.

**▸ Walk through it.** Take a cat photo (slide 8). Make two augmentations; an encoder turns each into a 1-D vector. In a batch of $N$ images you get $2N$ vectors. The **similarity matrix** (slide 9) scores every pair: augmentations of the *same* cat score high; a cat vs a dog score low. The loss *pulls* the matched pair together and *pushes* the others apart. The *hope* (slide 10): the learned representation **generalises to new instances** — give it a *drawing* of a cat and it still lands near the cat cluster.

**The takeaway.** *SimCLR learns concepts without labels by a pull-together / push-apart objective. The test of success is generalisation to unseen (out-of-distribution) inputs.*

---

### B5. CLIP — contrastive *image-text* learning

**The intuition.** SimCLR pairs an image with an *image*. CLIP (slide 11–16) pairs an image with its **text caption**. Now the representation space holds both modalities: *"a cute fluffy cat"* and a photo of a cat map to nearby vectors. The full form of CLIP: **C**ontrastive **L**anguage–**I**mage **P**retraining.

**▸ Walk through it.** A batch of $N$ (image, caption) pairs. The **image encoder** and **text encoder** each produce $N$ unit vectors. Form the $N\times N$ matrix of dot products $u_i\!\cdot\!v_j$ (slide 15/16). The **diagonal** $u_i\!\cdot\!v_i$ are the true pairs — maximise them; the **off-diagonal** $u_i\!\cdot\!v_j\ (i\neq j)$ are mismatches — minimise them. Slide 17 shows where the data comes from: *"Mount Rainier's northwestern slope viewed aerially just before sunset"* scraped together with its Wikipedia image — captions are free, so this scales to **400M pairs**.

**The takeaway.** *CLIP is SimCLR with the second view replaced by a caption. Matching image-text pairs are pulled together; mismatched ones pushed apart.*

---

### B6. The contrastive loss in one line

**The intuition.** The objective (slide 16) is exactly a **softmax**: for each image, "how much probability does the *correct* text get among all texts in the batch?" — i.e. **InfoNCE**.

**▸ Walk through it.** For image $i$ the probability the correct text $v_i$ is picked is

$$p_{ii}=\frac{\exp(u_i\!\cdot\!v_i/\tau)}{\sum_{j=1}^{N}\exp(u_i\!\cdot\!v_j/\tau)},$$

and the loss is $-\log p_{ii}$. The slide writes the symmetric form:

$$\mathcal{L}=-\frac1N\sum_{i=1}^{N}\log\frac{e^{u_i\cdot v_i}}{\sum_{j=1}^{N}e^{u_i\cdot v_j}}\;-\;\frac1N\sum_{i=1}^{N}\log\frac{e^{u_i\cdot v_i}}{\sum_{j=1}^{N}e^{u_j\cdot v_i}}$$

The **first sum** is *image → text* ("given this image, find its caption"); the **second** is the reverse *text → image*. $\tau$ is a **temperature** that sharpens the softmax. Reverse the roles of $u$ and $v$ and you get the second term — that is the transcript's *"combining them together brings text closer to image and image closer to text."*

**The takeaway.** *CLIP's loss is a two-way softmax over the batch similarity matrix: correct pair in the numerator, whole batch in the denominator. Bigger batch ⇒ bigger denominator ⇒ finer discrimination.*

---

### B7. Using CLIP out of the box — the clever trick

**The intuition.** An LLM does zero-shot text by predicting next tokens. Vision has **no grammar** — objects do not follow in sequence — so image zero-shot had to be solved differently. The trick (slides 23–26): **the text encoder *is* the classifier.**

**▸ Walk through it.** You want to classify into {plane, dog, bird} (slide 24). Write each label as text and push it through the **text encoder**, getting a "plane vector", "dog vector", "bird vector". A new image goes through the **image encoder** to an image vector. Compute the **matching score** against each class vector (slide 25: 0.13, 0.27, 0.09) and **take the largest** — here *dog*. Slide 26 calls this exactly **1-NN**: the class vectors are the "training set", and you return the nearest one.

**▸ Walk through it (numbers).** Image vector $(1,2)$; class vectors dog $(2,3)$, cat $(3,1)$, bird $(1,0)$. Cosine similarity = dot / (product of norms):

- dog: $8/(\sqrt5\sqrt{13})=0.992$
- cat: $5/(\sqrt5\sqrt{10})=0.707$
- bird: $1/\sqrt5=0.447$

Arg-max is **dog** at 0.992. No training on labelled images happened — the "classifier weights" were *written in English*.

**The takeaway.** *Zero-shot classification with CLIP = encode each class name as text, encode the image, and pick the closest text vector. The classifier is generated from words, not trained on data.*

---

### B8. Prompt engineering and prompt ensembling

**The intuition.** Because CLIP was trained on *captions*, a bare word is a poor query; a **phrase** matches the training distribution better — and one phrase can be biased, so **average several**.

**▸ Walk through it.** Replace `"dog"` with `"A photo of a dog"` (slide 27) → **+1.3%** on ImageNet, because the phrase embedding is closer to the caption embeddings CLIP saw. But a single phrase is biased (slide 28), so use several templates — *"A photo of a dog"*, *"A drawing of a dog"*, … — encode each, and take the **mean vector** as the class representation (slide 29) → **+5%** on ImageNet. Concretely, the mean of $(1,0)$ and $(0,1)$ is $(0.5,0.5)$; the class vector becomes the *centre* of its prompt cluster.

**The takeaway.** *Prompts beat bare labels (+1.3%); averaging many prompt templates beats a single prompt (+5%). You are nudging the text vector onto the caption manifold and removing phrase-specific bias.*

---

### B9. Why CLIP generalises — and ResNet doesn't

**The intuition.** A task-specific model **memorises** its dataset; a foundation model learns the **concept**. The slide's proof is a distribution shift: train on ImageNet, test on **ObjectNet** (same classes, weird viewpoints).

**▸ Walk through it.** On ImageNet, ResNet-101 scores **76.2%** and zero-shot CLIP ViT-L *matches* it at **76.2%** (slide 31) — with *no human labels*. Now move to ObjectNet (slide 33/34): ResNet-101 **collapses to 32.6%**, while CLIP holds at **72.3%**. The reason (transcript): ResNet *"learns and memorizes the things present in that dataset"*, so ObjectNet is out-of-distribution for it; CLIP *"understood what banana is"* in general, so it survives sketches, graphics and adversarial images too (slide 35: ImageNet-Sketch 25.2% → **60.2%**).

**The takeaway.** *Task-specific accuracy is a property of the dataset; foundation accuracy is a property of the concept. The gap shows up only out of distribution — which is exactly where deployment happens.*

---

### B10. Why CLIP *works*: scale, not labels

**The intuition.** Slide 36 asks *"How can no labels beat labels?"* and answers **Scale**. Both axes grew enormously.

**▸ Walk through it.** Two independent scalings (slides 37/38). **Parameters:** the ImageNet ResNet has **44.5M**; CLIP scaled the transformer up to roughly **6×** that (≈267M). **Data:** ImageNet had **1.28M** labelled training pairs; CLIP scraped **400M** image-text pairs — about **312×** more. The transcript's framing: with *large, high-variation* data you also need *many parameters*; a small model would **underfit** that diversity.

**The takeaway.** *No labels beat labels because of scale: ~6× the parameters on ~312× the data, scraped for free from the internet.*

---

### B11. CoCa — add a generation objective

**The intuition.** CLIP (slides 39/40) is an **early-fusion, dual-encoder** model whose image and text embeddings are *forced to be similar*. CoCa (**C**ontrastive **Ca**ptioner) keeps the contrastive loss **and adds a captioning (generation) loss** with a multimodal text decoder.

**▸ Walk through it.** In CoCa the image encoder feeds **two** heads. One head does the **contrastive loss** — the same "dog vector ≈ dog image" alignment as CLIP. The other feeds a **multimodal text decoder** through **cross-attention** on the image features and is trained to **generate the caption** *"two dogs running in a field"* token by token (a **captioning loss**). So the model is trained to both *match* and *describe*. Slide 41 shows the payoff: CoCa zero-shot ImageNet **86.3%** vs CLIP **76.2%**, and a higher average (**85.7%** vs **74.3%**); fine-tuned, CoCa reaches **91.0%** (slide 42).

**The takeaway.** *CoCa = CLIP's contrastive alignment + a generative caption decoder (early fusion, cross-attention). Adding generation buys a large accuracy jump over CLIP.*

---

### B12. The weaknesses of CLIP-style models

**The intuition.** Three limits (slides 44–51), each a likely exam MCQ.

**▸ Walk through it.**
1. **Batch-size dependence.** Contrastive learning only compares what is *in the batch*. The slide (45) shows the concept you learn scales with batch size: batch **4 → "animal"**, **100 → "dog"**, **32000 → "Welsh Corgi"**. But there is a limit (46): even at 32K it is unlikely both *"a mug in some grass"* and *"some grass in a mug"* appear, so their difference is never learned. Reusing the words in a different order (**compositionality**) is exactly what benchmarks like **Winoground (CVPR 2022)** test — and CLIP is weak there (44). The fix, **hard negatives**, has its own trap (48): *"a black cat and a brown dog"* vs *"a brown cat and a black dog"* are near-duplicate strings, so mining them reintroduces the supervision you were trying to avoid.
2. **Image-level captions are insufficient.** A caption *"living room"* (49) omits *house plant*, *couch* — the fine-grained objects. Region captions with boxes (50) help but need boxes.
3. **You can't know everything in your dataset.** With 5B scraped pairs (51) some content is harmful; the slide's own point is to be *"intentional about data collection and filtering"* (CSAM, etc.).

**The takeaway.** *CLIP is limited by batch composition (word order / relational concepts), by coarse captions (no fine-grained objects), and by the fact that you cannot audit all of a 5B-scale dataset.*

---

### B13. LLaVA — a frozen CLIP encoder bolted onto an LLM

**The intuition.** Slides 52–61. A language model does next-token prediction; can we feed it **image tokens** too, so it answers questions *about an image*? Historical context (53/54): vision-language models start as far back as **2019 (ViLBERT)** — but those were **task-specific**, needing bespoke machinery (e.g. Mask-RCNN box re-ranking for RefCOCO). LLaVA makes it general.

**▸ Walk through it.** Three moves (slide 61). **(i)** Take a **pre-trained, frozen CLIP image encoder** and a **pre-trained LLM** (e.g. LLaMA). **(ii)** Insert a **new trainable linear layer** that projects CLIP features into the LLM's token space — *"a new linear layer to bridge CLIP features to LLM input space."* **(iii)** **Fine-tune the LLM + linear layer together** (the rest frozen). A subtlety (slide 59/60): use the **penultimate** layer's patch tokens, *not* the final CLS token — *"these tokens preserve spatial and linguistic information best for LLMs; can drop CLS for slight gains."* Data need is modest (slide 61): reasonable performance with **>100,000** instruction samples (image + instruction + answer).

**The takeaway.** *LLaVA = frozen CLIP encoder → trainable linear projection → frozen-then-fine-tuned LLM. It turns vision into "LLM tokens", so the language model can describe and answer questions about images.*

---

### B14. Flamingo — frozen vision encoder + gated cross-attention

**The intuition.** Slides 62–71. LLaVA prepends image tokens; **Flamingo** instead **injects** vision into the LLM's layers via **cross-attention**, and adds a **Perceiver Resampler** so a *variable* number of image patches becomes a *fixed* number of tokens. This is what unlocks **in-context (few-shot) learning** — give it a few image-text examples in the prompt and it infers the task.

**▸ Walk through it.** Input is **interleaved** image and text (slide 62): *"\<image\> This is a very cute dog. \<image\> This is …"*. Two components are **learned from scratch** (slide 65): the **Perceiver Resampler** (slide 64: *"a learned method of down-sampling image/video representations"* to a fixed token count) and the **gated cross-attention dense** layers (slide 68). Everything else — the vision encoder and the LM — is **pretrained and frozen**. The gate (slide 68) is a **tanh gating** around the cross-attention and the feed-forward block, so the text stream stays intact until training decides how much vision to let through. Result (slides 69/70): the same model answers *"what are they doing?", "what object are they using?", "why is this surprising?"* about a teddy-bears-on-the-moon image, and it does **in-context arithmetic** (\(2+1=3,\ 5+6=11,\ 3\times6=18\)) purely from prompt examples.

**The takeaway.** *Flamingo = frozen encoders + Perceiver Resampler (fixed-size visual tokens) + tanh-gated cross-attention fused at many layers. It specialises in few-shot/in-context learning, unlike CLIP's single-shot zero-shot.*

---

### B15. Where this is going — world models

**The intuition.** Slide/transcript closing thought: the next step is a foundation model that **understands the world** — physics, what is possible in a scene — so generated video becomes indistinguishable from real.

**▸ Walk through it.** Today, generated videos expose themselves by mistakes (a flip that is "not nice enough"), because the model has not internalised the **laws of physics**. A **world model** would know *what can and cannot happen*, so generation and prediction would be consistent with reality. The transcript is explicit that this is a near-future direction, not a solved result.

**The takeaway.** *A world model is the ambition beyond vision-language foundation models: a model that has internalised physical common sense, so it can generate and reason about the world consistently.*

---

## C. Numbers worth knowing

All re-derived by execution.

### C1. Scale of CLIP vs a task-specific model

| quantity | task-specific (ImageNet ResNet) | CLIP | ratio |
|---|---|---|---|
| training pairs | **1.28 M** (labelled) | **400 M** (image-text, scraped) | **312.5×** |
| parameters | **44.5 M** | ≈ **267 M** (slide: "≈6×") | **6×** |

### C2. Accuracy — in-distribution vs shifted (slides 31–35)

| dataset | ResNet-101 | CLIP ViT-L (zero-shot) |
|---|---|---|
| ImageNet | **76.2%** | **76.2%** |
| ObjectNet | **32.6%** | **72.3%** |
| ImageNet-Sketch | **25.2%** | **60.2%** |
| ImageNet-Adversarial | — | **77.1%** |

CLIP loses **3.9 pts** shifting to ObjectNet; ResNet loses **43.6 pts**.

### C3. Prompt gains on ImageNet (slides 27–29)

| method | gain | result on 76.2% baseline |
|---|---|---|
| bare label `"dog"` | — | 76.2% |
| single phrase `"a photo of a dog"` | **+1.3%** | **77.5%** |
| mean over several phrases | **+5%** | **81.2%** |

### C4. CoCa vs CLIP — six robustness datasets (slide 41)

| model | ImageNet | IN-A | IN-R | IN-V2 | IN-Sketch | ObjectNet | avg |
|---|---|---|---|---|---|---|---|
| CLIP | 76.2 | 77.2 | 88.9 | 70.1 | 60.2 | 72.3 | **74.3** |
| **CoCa** | **86.3** | **90.2** | **96.5** | **80.7** | **77.6** | **82.7** | **85.7** |

CoCa fine-tuned ImageNet: **91.0%**; frozen: **90.6%** (slide 42).

### C5. Learnable pieces (slides 61, 65)

| model | frozen | trained |
|---|---|---|
| **LLaVA** | CLIP vision encoder + LLM (initial) | linear projection (+ LLM fine-tune); **>100,000** samples |
| **Flamingo** | vision encoder + language model | **Perceiver Resampler** + **gated cross-attention** |

### C6. Batch-size concept ladder (slide 45)

| batch size | concept learned |
|---|---|
| 4 | "animal" |
| 100 | "dog" |
| 32 000 | "Welsh Corgi" |

---

## D. Pictures

### D1. Task-specific vs foundation

```
   TASK-SPECIFIC                         FOUNDATION

   Data1 → Model1 → Task1                ┌──────────────┐
   Data2 → Model2 → Task2                │  FOUNDATION  │→ Task1
   Data3 → Model3 → Task3     vs.        │   MODEL      │→ Task2
   Data4 → Model4 → Task4                │ (one model)  │→ Task3
                                         └──────────────┘→ Task4
   four piles of data                    one diverse pile,
   four models                           fine-tune / zero-shot / few-shot
```

### D2. CLIP's contrastive matrix

```
                 TEXT encoder
              v1     v2     v3         (i = image, j = text)
          ┌  u1·v1  u1·v2  u1·v3 ┐     ● = match  (maximise, the diagonal)
   IMAGE  │  u2·v1  u2·v2  u2·v3 │     ○ = mismatch (minimise)
  encoder │  u3·v1  u3·v2  u3·v3 │
          └   ●      ○      ○    ┘     loss = two-way softmax over rows + cols
              ○      ●      ○
              ○      ○      ●
```

### D3. Zero-shot classifier = 1-NN over text vectors

```
   image ──► image encoder ──► v                     scores
                                              "plane"  0.13
   "plane" ─► text encoder ─► plane vec       "dog"    0.27  ◄── argmax
   "dog"   ─► text encoder ─► dog vec    vs   "bird"   0.09
   "bird"  ─► text encoder ─► bird vec
                                              class = "dog"
   the classifier's weights were WRITTEN, not trained
```

### D4. CoCa — contrastive + captioning

```
                      ┌──► contrastive head ──► align with text   (like CLIP)
   image ─► encoder ──┤
                      └──► multimodal text decoder ──► "two dogs running in a field"
                           (cross-attention)          (captioning loss)
```

### D5. LLaVA

```
   image ─► [frozen CLIP encoder] ─► patch tokens ─► [LINEAR proj, trained]
                                                        │  image tokens
   text ────────────────────────────────────────────► [  LLM  ] ─► answer
   (penultimate layer, drop CLS)                         ▲ fine-tune LLM + linear
```

### D6. Flamingo

```
   interleaved:  <image> caption <image> caption ...
        │                 │
        ▼                 ▼
   [frozen vision enc]  [text tokens]
        │                 │
   Perceiver Resampler ──► (fixed # visual tokens)
        │                 │
        └──► gated cross-attention (×N layers, tanh gate) ──► [frozen LLM] ─► text
             ▲ trained from scratch
   ⇒ few-shot / in-context learning
```

---

## E. Code

Section D of the exam asks you to *implement* a concept. For a foundation-model class the natural targets are the **cosine similarity**, the **zero-shot classifier**, the **contrastive (InfoNCE) loss**, the **mean-prompt vector**, and **retrieval**. Full specifications and verified outputs are in **Section D** below; the reference implementations live in the worked-solutions section.

**Two one-liners worth memorising.**

```
cosine(a,b) = dot(a,b) / (|a| |b|)          # scale-invariant similarity
loss_i      = -log( exp(u_i·v_i/τ) / Σ_j exp(u_i·v_j/τ) )   # InfoNCE, per row
```

---

## F. Comparisons

| | **CLIP** | **CoCa** | **LLaVA** | **Flamingo** |
|---|---|---|---|---|
| family | classification | classification | LM + vision | LM + vision |
| objective | contrastive | contrastive **+ captioning** | language modelling (next token) | language modelling (next token) |
| vision↔text | dual encoders, **early fusion** | dual + **generative decoder** | image tokens prepended | **cross-attention** in LLM layers |
| vision encoder | trained jointly | trained jointly | **frozen CLIP** | **frozen** |
| key extra part | — | text decoder | **linear projection** | **Perceiver Resampler** + gates |
| headline skill | **zero-shot** classification | stronger zero-shot | visual **Q&A / description** | **few-shot / in-context** |
| images needed | 400M pairs | 400M+ | **>100K** instructions | interleaved web data |

**Decision line.** Use **CLIP** for cheap open-vocabulary classification; **CoCa** when you want better zero-shot or captioning; **LLaVA** for instruction-following visual Q&A; **Flamingo** when you need to teach a task from a handful of in-prompt examples.

---

## G. Exam-focused revision sheet

- **Definition:** foundation = *general/robust to many tasks*; scale + self-supervision are usual, not defining.
- **Recognition:** always general; often big + large data + self-supervised.
- **CLIP** = Contrastive Language–Image Pretraining; dual encoder; InfoNCE over the batch matrix; **400M** pairs.
- **The trick:** **text encoder as classifier** → zero-shot via arg-max cosine; think **1-NN**.
- **Prompts:** `"a photo of a X"` **+1.3%**; mean of many templates **+5%**.
- **Generalization:** ImageNet 76.2 = 76.2; ObjectNet **32.6 → 72.3**; Sketch **25.2 → 60.2**.
- **Scale:** 44.5M params → ~**6×**; 1.28M → 400M pairs = **312.5×**.
- **CoCa:** + generation/captioning loss; ImageNet 86.3, avg 85.7; fine-tuned 91.0.
- **CLIP weaknesses:** batch size / compositionality (Winoground); weak image-level captions; can't audit a 5B dataset; hard-negative mining reintroduces supervision.
- **LLaVA:** frozen CLIP (penultimate layer, drop CLS) + trainable **linear** + LLM; >100K samples.
- **Flamingo:** frozen encoders + **Perceiver Resampler** + **gated (tanh) cross-attention**; **in-context learning**.
- **History:** VLMs since **2019 (ViLBERT)**, but task-specific until the foundation era.

---

## H. Practice Questions

> **Format mirrors the exam:** A single-correct (1 mark), B multi-select (2), C descriptive ≤100 words (2), D coding (20, attempt one), E case study (15), F numericals (2.5). Attempt before reading the solutions.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** The *defining* property of a foundation model is that it is:
A. Trained on more than one billion images  
B. General and robust across many tasks without retraining  
C. Always self-supervised  
D. Always larger than 1B parameters  

**A2.** In CLIP, the *text* encoder is used at inference time as:
A. A feature extractor for captions only  
B. A fixed classifier whose weights are given by the class-name embeddings  
C. A generator of images from prompts  
D. A tokeniser for the image encoder  

**A3.** CLIP's training objective is best described as:
A. A reconstruction loss on masked patches  
B. A binary cross-entropy on image labels  
C. A symmetric contrastive (InfoNCE) loss over the batch's image-text similarity matrix  
D. A captioning loss with a multimodal decoder  

**A4.** The key reason CLIP can classify *new* categories without training on them is that:
A. It memorises ImageNet's 1000 classes  
B. Class vectors are created from text, so any category with a name can be a class  
C. It fine-tunes a new head per class  
D. It uses k-means on the image embeddings  

**A5.** Compared with ImageNet-trained ResNet-101, CLIP's advantage appears mainly:
A. On ImageNet itself  
B. On out-of-distribution data such as ObjectNet  
C. On very small images  
D. On grayscale images only  

**A6.** The change CoCa makes relative to CLIP is adding:
A. A larger batch size  
B. A captioning (generation) objective with a multimodal text decoder  
C. A segmentation head  
D. A reinforcement-learning stage  

**A7.** In LLaVA the component trained to connect the frozen CLIP encoder to the LLM is:
A. The CLIP encoder  
B. A linear projection layer (with the LLM fine-tuned)  
C. The tokeniser  
D. The Perceiver Resampler  

**A8.** The two components Flamingo learns from scratch are:
A. The image encoder and the LLM  
B. The tokeniser and the embedding table  
C. The Perceiver Resampler and gated cross-attention layers  
D. The classifier head and the optimizer  

### Section B — Multi-select (5 × 2 marks)

**B1.** Which are properties *always* or *often* seen with foundation models? Select all that apply.
A. General / robust to many tasks  
B. Large number of parameters  
C. Large amount of training data  
D. Self-supervised pre-training objective  
E. Guaranteed to beat every task-specific model on its home dataset  

**B2.** Which are correct about CLIP? Select all that apply.
A. Trained on ~400M image-text pairs  
B. Contrastive objective with image and text encoders  
C. Zero-shot classification by matching an image vector to class-name text vectors  
D. Prompt engineering `"a photo of a X"` improves ImageNet accuracy  
E. It requires pixel-level labels for training  

**B3.** Which are limitations of CLIP-style models? Select all that apply.
A. Reliance on batch size to learn fine-grained / compositional concepts  
B. Relational phrases such as "a mug in some grass" vs "some grass in a mug" are confused  
C. Image-level captions omit fine-grained objects  
D. You cannot inspect every item in a 5-billion-pair dataset  
E. They cannot be used for any image classification task  

**B4.** Which are true of LLaVA? Select all that apply.
A. It uses a frozen pre-trained CLIP vision encoder  
B. A trainable linear layer bridges CLIP features into the LLM's token space  
C. It fine-tunes the LLM together with the linear layer  
D. The penultimate-layer tokens are used, often dropping the CLS token  
E. It uses the Perceiver Resampler to downsample images  

**B5.** Which are true of Flamingo? Select all that apply.
A. It handles interleaved image and text input  
B. The vision encoder and language model are frozen  
C. Gated cross-attention uses a tanh gate  
D. A Perceiver Resampler converts variable-size image features to a fixed number of tokens  
E. Its headline capability is in-context (few-shot) learning  

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Define a foundation model and contrast it with a task-specific model in the vision setting.

**C2.** Explain CLIP's zero-shot classification as a "1-nearest-neighbour with text vectors".

**C3.** Why does CLIP hold up on ObjectNet while an ImageNet-trained ResNet collapses?

**C4.** Distinguish *early fusion* (CLIP/CoCa) from the cross-attention design of Flamingo.

**C5.** Describe LLaVA's architecture and training recipe in three steps.

### Section D — Coding (20 marks — attempt ONE of the three)

Implement in **pure Python (standard library only)**. Return `None` on malformed input.

**D1. Similarity and zero-shot classifier.**
- `cosine_similarity(a, b)` → the cosine similarity of two equal-length numeric vectors; `None` if lengths differ, either is `None`, or either is the zero vector.
- `zero_shot_classify(image_vec, class_vecs, labels)` → the `(label, score)` of the class whose vector is most similar to `image_vec`; `None` if `class_vecs` and `labels` differ in length or are malformed.

```
cosine_similarity([1,1,1],[1,1,1])                 ->  1.0
cosine_similarity([1,0],[0,1])                     ->  0.0
cosine_similarity([0,0],[1,2])                     ->  None
zero_shot_classify([1,2], [[2,3],[3,1],[1,0]], ["dog","cat","bird"])
                                                   ->  ("dog", 0.9922778767...)
zero_shot_classify([1,2], [[2,3]], ["dog","cat"])  ->  None
```

**D2. Contrastive (InfoNCE) loss.**
- `clip_contrastive_loss(image_embeds, text_embeds, temperature)` → the symmetric contrastive loss of the two equal-length lists of embedding vectors. Compute the similarity matrix, divide by `temperature`, and average the image→text and text→image softmax cross-entropies. `None` if lists differ in length, are empty, or `temperature <= 0`.

```
clip_contrastive_loss([[1,0],[0,1]], [[1,0],[0,1]], 0.1)         ->  4.54e-05
clip_contrastive_loss([[0.8,0.6],[0.6,0.8]], [[1,0],[0,1]], 0.1) ->  0.126928
clip_contrastive_loss([[0.8,0.6],[0.6,0.8]], [[1,0],[0,1]], 1.0) ->  0.598139
clip_contrastive_loss([[1,0],[0,1]], [[0,1],[1,0]], 0.1)         ->  10.000045
clip_contrastive_loss([[1,0]], [[1,0]], 0)                       ->  None
```

**D3. Prompt ensembling and retrieval.**
- `mean_vector(vectors)` → the element-wise mean of a non-empty list of equal-length vectors; `None` if empty or ragged.
- `retrieve(query, gallery, k)` → the top-`k` `(score, index)` pairs by cosine similarity, sorted by descending score then index; `None` if `k <= 0` or any input is malformed.

```
mean_vector([[1,0],[0,1]])                                 ->  [0.5, 0.5]
mean_vector([[1,0],[0,1],[1,0]])                           ->  [0.6666667, 0.3333333]
mean_vector([[1,0],[1,0,0]])                               ->  None
retrieve([1,2], [[2,3],[3,1],[1,0],[2,4]], 2)              ->  [(0.9999999999999998, 3), (0.9922778767136677, 0)]
retrieve([1,2], [[2,3]], 0)                                ->  None
```

### Section E — Case study (15 marks)

> **Scenario.** A marketplace wants a **zero-shot product classifier** for a catalogue that changes weekly (new categories appear often, and there are no labelled photos for them). The team proposes using **CLIP** with category names as class text. Categories are fine-grained ("wireless earbuds", "over-ear headphones") and some differ only by **word order** ("case for phone" vs "phone case"). Daily volume is ~5M images; latency must stay within ~50 ms/query.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Why is a zero-shot foundation model the right tool here, rather than a supervised classifier?
2. **Data.** What data does CLIP need at *training* time, and what data does it need at *inference* time for a new category?
3. **Model.** Describe how a prediction is produced, from category name to label.
4. **Metric.** State the two prompt-related accuracy gains on ImageNet and what each corresponds to.
5. **Failure mode.** Give two ways CLIP is likely to fail on this catalogue.
6. **Engineering.** Given a 5M-image daily load and 50 ms latency, what does the dot-product design buy you?

**Sub-questions**

- **E-a.** New categories must be added with **no retraining**. Explain the mechanism that makes this possible and its one-sentence cost.
- **E-b.** The catalogue contains "case for phone" and "phone case". Explain, using compositionality and hard negatives, why this is hard and what a fix would cost.
- **E-c.** When would you abandon zero-shot CLIP and fine-tune (or switch to a captioning model like CoCa)?

### Section F — Numericals (8 × 2.5 marks)

**F1.** On ImageNet a ResNet-101 scores 76.2% and zero-shot CLIP ViT-L also scores 76.2%. On ObjectNet ResNet drops to 32.6% while CLIP holds at 72.3%. Compute each model's accuracy *drop* in points and *relative* percent, and the ratio of the two drops.

**F2.** The ImageNet ResNet has 44.5M parameters; CLIP scaled parameters to about 6× that. ImageNet used 1.28M pairs; CLIP used 400M. Compute CLIP's parameter count and the data ratio.

**F3.** A CLIP baseline scores 76.2% on ImageNet. Using a single template `"a photo of a [category]"` adds 1.3%; averaging several templates adds 5% (both measured against the same baseline). Compute the two resulting accuracies.

**F4.** Classify an image vector $(1,2)$ against class vectors dog $(2,3)$, cat $(3,1)$, bird $(1,0)$ by cosine similarity. State the predicted class.

**F5.** For a 2-pair batch with image embeddings $(0.8,0.6),(0.6,0.8)$ and text embeddings $(1,0),(0,1)$, compute the symmetric contrastive loss at temperature $\tau=0.1$. (State the per-row log-loss.)

**F6.** A retrieval index holds 5B image embeddings of dimension 512 in 32-bit floats. Compute (a) the storage in bytes, (b) the floating-point operations for one query, and (c) the time at 1 PFLOP/s ($10^{15}$ FLOP/s).

**F7.** A batch of 32,768 pairs is drawn uniformly from 400M pairs. Compute (a) the probability a *specific* pair is in the batch, and (b) the probability that *two specific* pairs are both in it. Relate this to why fine-grained concepts need hard negatives.

**F8.** Across six robustness datasets CLIP averages 74.3% and CoCa averages 85.7%. Compute the absolute and relative improvement of CoCa.

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — General and robust across many tasks without retraining.** The slide's defining property; scale and self-supervision are "often", not "always".

**A2. → B — A fixed classifier whose weights are the class-name embeddings.** The text encoder *generates* the classifier from words.

**A3. → C — Symmetric contrastive (InfoNCE) over the batch similarity matrix.** Reconstruction is BERT/SAM; image labels are ResNet; captioning is CoCa.

**A4. → B — Class vectors come from text, so any named category becomes a class.** This is why it is "open-vocabulary"; no head is trained (rules out C).

**A5. → B — On out-of-distribution data such as ObjectNet.** In-distribution they tie at 76.2%; the gap opens only under shift.

**A6. → B — A captioning (generation) objective with a multimodal text decoder.** CLIP's alignment plus a decoder.

**A7. → B — A trainable linear projection (with the LLM fine-tuned).** The CLIP encoder is frozen; the Perceiver Resampler is Flamingo.

**A8. → C — The Perceiver Resampler and the gated cross-attention layers.** Encoders and the LLM are frozen.

### Section B — Solutions

**B1. → A, B, C, D. Reject E.** No foundation model is guaranteed to beat every specialist on the specialist's own dataset; the advantage is generality and out-of-distribution behaviour.

**B2. → A, B, C, D. Reject E.** CLIP uses *natural-language supervision* (image-text pairs), never pixel labels.

**B3. → A, B, C, D. Reject E.** CLIP is an image *classifier* — of course it can classify; the listed items are the real limitations.

**B4. → A, B, C, D. Reject E.** The Perceiver Resampler belongs to Flamingo, not LLaVA.

**B5. → A, B, C, D, E. All five.** Each is a defining feature of Flamingo; E is its headline skill.

### Section C — Solutions

**C1.** A **task-specific** model is trained on one dataset for one task and applied to that task. A **foundation model** is pre-trained once on large, diverse (usually unlabelled, self-supervised) data so it becomes generally useful, then adapted to many tasks by fine-tuning, zero-shot or few-shot. The defining property is *generality*; large size and self-supervision are typical but not essential. Vision examples: ResNet (task-specific) vs CLIP (foundation).

**C2.** Write each class name as text and encode it with the text encoder, giving one vector per class. Encode the image and compute its similarity to each class vector; return the class with the highest score. The class vectors act as the "training set" and the image as the query, so it is exactly **1-nearest-neighbour** — except the references were *written*, not learned from images.

**C3.** ResNet **memorises** ImageNet's specific pixels, so a shift to ObjectNet's unusual viewpoints is out-of-distribution and accuracy halves (76→33). CLIP learned the **concept** behind each word from internet captions, so "banana" survives sketches, graphics and odd angles; on ObjectNet it falls only 4 points (76→72). The gap reflects learned concept versus memorised dataset.

**C4.** **Early fusion** (CLIP, CoCa) computes an image vector and a text vector in the *same* space and compares them by dot product — the modalities meet only at the similarity score. **Flamingo** instead keeps a language model and injects vision into its hidden layers through **gated cross-attention** (query from text, keys/values from the Resampler), so fusion happens *inside* the transformer at every layer.

**C5.** (1) Take a **frozen** pre-trained CLIP image encoder and a pre-trained LLM. (2) Add a **trainable linear layer** that projects CLIP's penultimate-layer patch tokens into the LLM's input space as image tokens. (3) **Fine-tune the LLM together with the linear layer** while encoders stay frozen; ~100K image-instruction-answer samples suffice.

### Section D — Solutions

**D1.**

```python
import math

def cosine_similarity(a, b):
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def zero_shot_classify(image_vec, class_vecs, labels):
    if image_vec is None or class_vecs is None or labels is None:
        return None
    if len(class_vecs) != len(labels) or len(class_vecs) == 0:
        return None
    best_i, best_s = None, None
    for i, (cv, lab) in enumerate(zip(class_vecs, labels)):
        s = cosine_similarity(image_vec, cv)
        if s is None:
            return None
        if best_s is None or s > best_s:
            best_s, best_i = s, i
    return labels[best_i], best_s
```

Verified outputs: `cos([1,1,1],[1,1,1]) → 1.0`; `cos([1,0],[0,1]) → 0.0`; `cos([0,0],[1,2]) → None`; `zero_shot_classify([1,2],[[2,3],[3,1],[1,0]],["dog","cat","bird"]) → ("dog", 0.9922778767136677)`; `zero_shot_classify([1,2],[[2,3]],["dog","cat"]) → None`.

**D2.**

```python
import math

def clip_contrastive_loss(image_embeds, text_embeds, temperature):
    if image_embeds is None or text_embeds is None or temperature is None:
        return None
    if len(image_embeds) != len(text_embeds) or len(image_embeds) == 0:
        return None
    if temperature <= 0:
        return None
    n = len(image_embeds)
    logits = [[sum(a * b for a, b in zip(image_embeds[i], text_embeds[j])) / temperature
               for j in range(n)] for i in range(n)]

    def row_logloss(row, k):
        m = max(row)
        denom = sum(math.exp(x - m) for x in row)
        return (m + math.log(denom)) - row[k]

    total = 0.0
    for i in range(n):
        total += row_logloss(logits[i], i)
        col = [logits[j][i] for j in range(n)]
        total += row_logloss(col, i)
    return total / (2 * n)
```

Verified outputs: perfect match at τ=0.1 → `4.54e-05`; matched at τ=0.1 → `0.126928` (= $\ln(1+e^{-2})$); at τ=1.0 → `0.598139`; mismatched → `10.000045`; τ=0 → `None`.

**D3.**

```python
import math

def cosine_similarity(a, b):
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def mean_vector(vectors):
    if vectors is None or len(vectors) == 0:
        return None
    d = len(vectors[0])
    if d == 0 or any(len(v) != d for v in vectors):
        return None
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(d)]

def retrieve(query, gallery, k):
    if query is None or gallery is None or k is None or k <= 0:
        return None
    scored = []
    for i, g in enumerate(gallery):
        s = cosine_similarity(query, g)
        if s is None:
            return None
        scored.append((s, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[:k]
```

Verified outputs: `mean_vector([[1,0],[0,1]]) → [0.5, 0.5]`; `mean_vector([[1,0],[0,1],[1,0]]) → [0.666…, 0.333…]`; ragged → `None`; `retrieve([1,2],[[2,3],[3,1],[1,0],[2,4]],2) → [(0.9999999999999998, 3), (0.9922778767136677, 0)]`; `k=0 → None`.

### Section E — Solutions

**Model answer covering the chain.**

1. **Task.** Categories change weekly and have **no labelled images**, so a supervised classifier would need fresh labels every week. A **zero-shot** foundation model classifies a new category the moment you can *name* it — the "classifier" is text, so adding a category costs one string, not a training run.
2. **Data.** **Training:** ~400M (image, caption) pairs scraped from the web — natural-language supervision, no human labels. **Inference:** just the **category names** (plus optional prompt templates); no images of the new category are required.
3. **Model.** Encode each category name (or the mean of several templates) with the text encoder → class vectors. Encode the product image → image vector. Score by cosine/dot product and take the arg-max (a 1-NN over text vectors). Done.
4. **Metric.** `"a photo of a [category]"` gives **+1.3%** on ImageNet (single phrase matches the caption distribution); a **mean over several templates** gives **+5%** (removes single-phrase bias).
5. **Failure mode.** (i) **Compositional / relational** categories — CLIP confuses "case for phone" with "phone case" (word order), the Winoground failure. (ii) **Fine-grained sub-categories** with little visual-versus-textual separation, worsened by **weak image-level captions** that never named the fine detail. (A third: no calibration/uncertainty for confident-but-wrong matches.)
6. **Engineering.** Scoring is a **dot product** — trivially parallel and cache-friendly — so 5M images/day at 50 ms is feasible if image vectors are precomputed and the class-vector dot products are done in a single matrix multiply.

**Sub-questions.**

**E-a — adding a category with no retraining.** Because the classifier **is the text encoder**: a new class is a new *string*, encoded on the fly to a class vector and appended to the comparison set. The cost is the flip side: **performance is bounded by how well the category name (and prompts) covers the visual concept**, and fine-grained names can be under-specified.

**E-b — word order / compositionality.** CLIP is trained by **contrastive matching over a batch**, so concepts that differ only by **word order** or relation ("case for phone" vs "phone case", "mug in grass" vs "grass in mug") are rarely both present with their distinct images, and their similarity scores stay close — the **Winoground** weakness. **Hard negatives** would help, but mining them means *choosing* confusable pairs by hand — reintroducing the **supervision** that made scaling cheap in the first place.

**E-c — when to leave zero-shot.** Fine-tune (or switch to a captioning model such as **CoCa**) when (i) accuracy on in-catalogue categories matters more than adding new ones, (ii) the deployment distribution is **out-of-distribution** relative to web data (medical, industrial, proprietary styles), or (iii) you need fine-grained/relational distinctions that category *names* cannot express. CoCa in particular gives better zero-shot accuracy and captioning for richer supervision.

### Section F — Solutions

**F1.** *Formulas:* drop (pts) $=a_{\text{in}}-a_{\text{shift}}$; relative drop $=\text{drop}/a_{\text{in}}\times100$.

- ResNet: $76.2-32.6=\mathbf{43.6}$ pts; $43.6/76.2=\mathbf{57.2\%}$.
- CLIP: $76.2-72.3=\mathbf{3.9}$ pts; $3.9/76.2=\mathbf{5.1\%}$.
- Ratio of drops: $43.6/3.9=\mathbf{11.2\times}$.

*Interpret:* identical in-distribution accuracy, but ResNet loses 11× more under the same shift — the quantitative signature of memorisation versus concept.

**F2.** *Formula:* params $=44.5\text{M}\times6$; ratio $=400\text{M}/1.28\text{M}$.

$$44.5\times6=\mathbf{267\text{M}}\text{ params}\qquad \frac{400\times10^{6}}{1.28\times10^{6}}=\mathbf{312.5\times}\text{ more data.}$$

*Interpret:* the transcript's *"≈6× parameters on a ~312× larger dataset"* — both axes scaled, which is why labels were unnecessary.

**F3.** *Formula:* `accuracy = baseline + gain`.

- Single phrase: $76.2+1.3=\mathbf{77.5\%}$.
- Mean of templates: $76.2+5.0=\mathbf{81.2\%}$.

*Interpret:* prompt engineering recovers ~1 point; ensembling several templates recovers ~5 — cheap, training-free accuracy from the *text* side.

**F4.** *Formula:* $\cos(a,b)=\dfrac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$.

Image $(1,2)$, $\lVert\cdot\rVert=\sqrt5$:

| class | vector | dot | cosine |
|---|---|---|---|
| dog | $(2,3)$ | $2+6=8$ | $8/(\sqrt5\sqrt{13})=\mathbf{0.992}$ |
| cat | $(3,1)$ | $3+2=5$ | $5/(\sqrt5\sqrt{10})=\mathbf{0.707}$ |
| bird | $(1,0)$ | $1$ | $1/\sqrt5=\mathbf{0.447}$ |

*Interpret:* arg-max is **dog** (0.992). The classifier was never trained — the three class vectors *are* the entire "model".

**F5.** *Formula:* $\ell_i=-\log\dfrac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$, with $s_{ij}=u_i\!\cdot\!v_j$.

Similarity matrix $S=\begin{bmatrix}0.8&0.6\\0.6&0.8\end{bmatrix}$. Divide by $\tau=0.1$: rows $(8,6)$ and $(6,8)$. For row 0:

$$\ell_0=-\log\frac{e^{8}}{e^{8}+e^{6}}=-\log\frac{1}{1+e^{-2}}=\ln(1+e^{-2})=\mathbf{0.1269}.$$

By symmetry every row/column gives the same value, so the symmetric loss is $\mathbf{0.1269}$.

*Interpret:* when the diagonal is only slightly above the off-diagonal, the loss is small but non-zero; at $\tau=1$ the same matrix gives $0.598$ (the temperature controls how harshly near-misses are penalised).

**F6.** *Formulas:* storage $=N\cdot d\cdot4$ bytes; FLOPs $=2Nd$; time $=\text{FLOPs}/\text{rate}$.

- Storage: $5\times10^{9}\times512\times4=\mathbf{1.024\times10^{13}\ bytes}\approx\mathbf{10.2\ TB}$.
- FLOPs: $2\times5\times10^{9}\times512=\mathbf{5.12\times10^{12}}$.
- Time at $10^{15}$ FLOP/s: $5.12\times10^{12}/10^{15}=\mathbf{5.12\ ms}$.

*Interpret:* a full brute-force scan is ~10 TB of memory traffic but only ~5 ms of pure math — the dot product is so cheap that retrieval over *billions* of images is practical (the slide's *"retrieval over 5B images"*).

**F7.** *Formulas:* $P(\text{one in batch})=B/N$; $P(\text{both})\approx(B/N)^2$.

- $P_1=32768/4\times10^{8}=\mathbf{8.19\times10^{-5}}$.
- $P_2\approx(8.19\times10^{-5})^2=\mathbf{6.71\times10^{-9}}$.

*Interpret:* the chance of *two particular* confusable pairs landing in the same batch is about **one in 150 million** — so fine-grained contrasts like "mug in grass" vs "grass in mug" are almost never learned from random batches. That is exactly why the slide turns to **hard-negative mining**.

**F8.** *Formulas:* absolute $=a_{\text{CoCa}}-a_{\text{CLIP}}$; relative $=\text{absolute}/a_{\text{CLIP}}$.

- Absolute: $85.7-74.3=\mathbf{11.4}$ points.
- Relative: $11.4/74.3=\mathbf{15.3\%}$.

*Interpret:* adding a **generation objective** to the same contrastive recipe buys ~11 points on the robustness suite — the captioning signal sharpens the visual features. (Recomputing the six dataset means gives CLIP 74.15% and CoCa 85.67%, which the slide rounds to 74.3/85.7.)

---

## J. Sources

### Course sources

| source | what it provided |
|---|---|
| `Session_13_—_Vision_Foundation_Models_….txt` | the lecture narrative (CLIP → CoCa → LLaVA → Flamingo), the ObjectNet discussion, the batch-size and hard-negative limitations |
| `session_13_cropped.pdf` (73 slides) | all figures, tables and numbers; note the deck is **rotated 90°** and image-only |

### Key slides

- **1–2** task-specific vs foundation. **3–4** foundation taxonomy. **5** how to recognise a foundation model.
- **8–12** SimCLR recap and the move to language/generalisation.
- **13–18** CLIP objective, the similarity matrix, and web image-text data.
- **19–22** pretext → transfer; zero-shot framing. **23–26** text-encoder-as-classifier (1-NN).
- **27–29** prompt engineering (+1.3%) and ensembling (+5%). **30–35** ImageNet/ObjectNet/Sketch results. **36–38** scale (params + data).
- **39–42** CoCa and its results. **43–51** advantages and limitations.
- **52–61** LLaVA. **62–71** Flamingo. **72–73** Flamingo few-shot results.

### Supplementary

- Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"* (CLIP), 2021 — the source the slides cite throughout.
- Yu et al., *"CoCa: Contrastive Captioners are Image-Text Foundation Models"*, 2022.
- Liu et al., *"Visual Instruction Tuning"* (LLaVA), 2023.
- Alayrac et al., *"Flamingo: a Visual Language Model for Few-Shot Learning"*, 2022.
- Thrush et al., *"Winoground"*, CVPR 2022 — the compositionality benchmark.
- The CS231n multimodal-foundation-models lecture (this deck is adapted from it).

### Unresolved

- The slide's CLIP six-dataset **average is printed as 74.3%**; recomputing the six listed values gives **74.15%** (a rounding artefact, not an error of substance) — the CoCa row recomputes to 85.67% → 85.7✔.
- The **~6× parameter** factor is quoted in the transcript, not shown as an exact count on any slide; the slide only states ImageNet ResNet = **44.5M**.
- Flamingo's 72/73 results tables OCR'd too poorly to quote individual benchmark scores; only the qualitative claim (few-shot beats fine-tuned SOTA on several benchmarks) is used here.
