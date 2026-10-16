# Class 10 — Image Segmentation

> **Exam weight: Tier 1.** Segmentation is the densest CV task and this is the densest lecture in the course. Sections A/B/C will hit hard. The transposed-convolution formula, dilated convolution, IoU/Dice/mIoU and Panoptic Quality are all standard Section F material.
>
> **Source:** `Session_10.pdf` (59 slides, *"Image Segmentation"*, Shubham Jain) + `Session_10.txt` transcript.
> **⚠️ Known gap: the Session 10 transcript is empty** — 59 words of *"Thank you"*. The recording failed. Everything here comes from slides alone, so verbal nuance is absent. This is the **only** lecture with no audio.
> **Professor repository: none in this workspace.**

**Biggest clue in the deck:** slide p.38 is a five-question exercise — *"Work in pairs for 5 minutes."* The instructor built a drill into his own slides. That tells you exactly which five numbers he thinks matter. Those five are reproduced in F1–F5.

---

## A. Lecture overview

**The one-sentence version.** Detection answers *what and roughly where*. Segmentation answers *what and exactly which pixels* — so the output grows from a handful of numbers per object to one label per pixel, and almost every design decision follows from that.

**Prerequisites:** Class 01 (convolution, output size, padding), Class 03 (CNN encoder, receptive field), Class 06 (IoU).

**Difficulty: Intermediate.** The concepts are easy; the arithmetic is not. Transposed convolution and dilation both have formulas that are *counter-intuitive* and easy to get wrong under pressure.

**Where it sits.** Class 06/07/08 covered boxes. This covers pixels. Class 11/12 moves to 3-D.

---

## B. The ideas

### B1. Why boxes aren't enough

Slide p.3 gives the killer example: **a thin object at 45° fills only ~20% of its bounding box.** A lane marking, a pole, a traffic light — a box around them is mostly road and sky.

The reasoning generalises: whenever your question is about **area, length, or a boundary**, you need a mask. Slide p.15 states it as a rule: *"When the answer to your question is an area, a length or a boundary, you need segmentation, not detection."*

### B2. Four kinds — one table

| Task | Label per pixel | Separates instances? | Covers stuff? | Metric | Models |
|---|---|---|---|---|---|
| **Semantic** | class | No | Yes | mIoU | FCN, U-Net |
| **Instance** | (class, id) for *things* | Yes | No | mask AP | Mask R-CNN |
| **Panoptic** | (class, id) for *everything* | Yes | Yes | PQ | Panoptic FPN |

Slide p.5 gives the framing that makes the table obvious — output density rises as you move right:

```
Classification      y ∈ {1…K}              1 label
Detection           {(c, box)}ᴺ            N boxes
Semantic seg.       Y ∈ {1…K}ᴴˣᵂ           K labels for every pixel
Instance seg.       {(c, mask)}ᴺ           N binary masks
```

> **KEY IDEA (slide p.5):** *"Moving right, output density increases and so does annotation cost. Choose the cheapest rung that answers your product question."*

That last clause is the actual engineering skill. A warehouse robot picking boxes off a pallet needs **instance** (which physical box?), and semantic output is useless for it — every pixel says *"box"* and there is no way to tell the left one from the right one.

### B3. Stuff vs things

| | Things | Stuff |
|---|---|---|
| | countable, well-defined shape | amorphous, defined by texture/material |
| | car, person, traffic sign, cell nucleus | road, sky, grass, water, wall |
| | instance IDs make sense | *"How many skies?"* is meaningless |

This distinction is the **entire reason panoptic segmentation exists.** Semantic handles stuff well and things badly; instance handles things well and stuff badly; panoptic does both, with **no overlaps** — every pixel belongs to exactly one segment.

*Edge case the instructor raises (p.9):* a crowd of pedestrians 200 m away is neither — it's countable in principle but not individually distinguishable. COCO carries an `iscrowd` flag precisely because datasets disagree.

### B4. Two naive ideas, both dead

**Naive #1 — classify a patch around every pixel.** Three failures (slide p.22): **cost** (Cityscapes is 1024×2048 ≈ **2.1 M forward passes**), **redundancy** (adjacent patches share almost all pixels, so features get recomputed endlessly), and **fixed context** (patch size caps how far the model can see).

**Naive #2 — just keep convolving at full resolution.** Two failures (slide p.24): **memory** (activations scale as $H\times W\times C$ at every layer) and **context** (to see 1,000 px you'd need ~500 layers of 3×3).

**The fix (slide p.25): downsample, then upsample.**

```
      ENCODER              BOTTLENECK           DECODER
   (classification        holds "what"        restores
     backbone)            at low res           "where"
   large receptive                            output = input size
   field, cheap                                    
        ↓                       ↓                   ↑
   VGG / ResNet / MiT    small feature map      ↑ upsampling
                                                 │
        ──── skip connections (U-Net) ───────────┘
```

One idea to hold onto: the encoder gets **cheap and smart** at low resolution; the decoder buys back the **resolution** it needs to say exactly *where*. That's why every architecture here is shaped like a U.

### B5. Upsampling operators

| Operator | Changes size? | Learnable? | Watch out for | Seen in |
|---|---|---|---|---|
| Nearest / bilinear | Up | No | Blocky / blurry edges | FPN, DeepLab, SegFormer |
| Max unpooling | Up | No (needs indices) | Sparse output | SegNet |
| Transposed conv | Up | Yes: $k^2 C_{in}C_{out}$ | Checkerboard if $k \bmod s \neq 0$ | FCN, U-Net |
| Resize + conv | Up | Yes | Conv runs at high res — costs more | Modern decoders |
| Pixel shuffle | Up ×r | Yes | Channel count × $r^2$ | ESPCN, super-res |

**Bilinear, the interpolation formula (slide p.27):**

$$f(x,y) \approx (1-a)(1-b)f_{00} + a(1-b)f_{10} + (1-a)b f_{01} + ab\,f_{11}$$

With a 2×2 input going to 3×3, `align_corners=True`, you get the slide's exact grid:

```
1 2        1.0  1.5  2.0
3 4   →    2.0  2.5  3.0     (verified)
           3.0  3.5  4.0
```

> **Gotcha (slide p.27):** `align_corners=True` vs `False` gives **different grids**. Keep training and export consistent — otherwise your model is subtly wrong at inference for no visible reason.

**Transposed convolution — the one rule you must memorise (slide p.35):**

$$o = (i-1)\cdot s - 2p + k + \text{output\_padding}$$

Note the $(i-1)$, **not** $i$. That is why stride 1 still grows the output by $k-1$ — a genuinely counter-intuitive behaviour, and the slide's first table row demonstrates it.

Mechanically (slide p.34): each input value multiplies the whole $k\times k$ kernel and **writes it onto** the output; overlapping stamps are **summed**. It's genuinely learnable upsampling. Note also the naming warning on the same slide: *"'deconvolution' (misleading)"* — it's not an inverse.

**Why "resize + conv" exists (slide p.36):** it has **no uneven overlap by construction**, which is what causes transposed-conv checkerboard artefacts. Cost: the conv runs at the higher resolution.

### B6. Seeing far without 500 layers

**Output stride** $\text{OS} = \dfrac{\text{input size}}{\text{feature size}}$. OS 32 turns a 1024×2048 image into 32×64. OS 8 gives 128×256.

**Dilated (atrous) convolution** gets context without extra parameters:

$$\text{effective kernel size} = (k-1)\cdot d + 1, \qquad r_{out} = r_{in} + (k-1)\cdot d$$

- $k{=}3, d{=}1 \to$ effective $3$ (normal)
- $k{=}3, d{=}2 \to$ effective $5$
- $k{=}3, d{=}6 \to$ effective **13**

> **The padding trap.** With dilation $d$, `padding = d` is *not* enough to preserve size. You need $\dfrac{(k-1)d}{2}$. For $k{=}3, d{=}6$: effective kernel 13, so padding **6**, not 1. Slide p.38's question 5 is exactly this, and it is a classic exam trap.

**ASPP (slide p.54)** is what you build when you want many scales at once: parallel atrous branches at rates **6, 12, 18, 24**, plus an image-pooling branch for global context. PSPNet (p.56) does the same idea with pooling bins 1×1, 2×2, 3×3, 6×6 instead of dilation.

### B7. The architectures (slides p.47–58)

All of these are **the same encoder–decoder** with different answers to three questions: *how do I get context?*, *how do I get detail back?*, *how do I combine them?*

#### FCN (2015) — the first end-to-end model

Slide p.48: input **512×512×3** → output **512×512×21 logits** (21 = PASCAL VOC classes + background).

The idea: replace the network's fully-connected layers with convolutions, so the output is a dense label map instead of one class vector. The prediction is made at low resolution and upsampled; FCN-8s and FCN-16s add skip connections from earlier, higher-resolution layers to sharpen it.

**Why it mattered:** before FCN you needed a separate classifier per proposal and post-processing. FCN made segmentation *one network, one loss*.

#### SegNet (2015) — throw away the features, keep the indices

Slide p.49. Encoder = VGG-16 convs; the decoder is the **mirror image**.

The distinctive choice: max-pooling **discards the values but records where the max was**. The decoder then performs **max unpooling using those stored indices**, scattering each value back to where it came from. Trainable convolutions then densify the sparse result.

- **Memory-light:** stores only indices, not feature maps — so no wide skip tensors.
- Uses **median-frequency class balancing** in the loss.
- Built for **road scenes and real-time** use.

Trade-off versus U-Net's concatenation: SegNet throws away spatial detail (only maxima survive), while U-Net reuses it fully. SegNet is cheaper; U-Net is more accurate on small structures.

#### U-Net (2015) — **the architecture to remember**

Slide p.50, **572×572 in → 388×388 out**, built from four blocks.

```
Down (×4):   2 × (3×3 conv + ReLU)  →  2×2 max-pool, stride 2   [channels double]
Up   (×4):   2×2 up-conv (halves channels)
              → CONCATENATE encoder features  ←  THE key idea
              → 2 × (3×3 conv + ReLU)
Head:         1×1 conv → class scores

channels:  64 → 128 → 256 → 512 → 1024 → (mirror back) → 2
```

Two details the slide stresses:

1. **Skips concatenate, they do not add** — *"Skips concatenate (U-Net) rather than add (FCN)."* Addition forces the two tensors to have identical shapes *and* lets one wash out the other; concatenation keeps both, and the following convs learn how to merge them.
2. The **"U" is symmetric** — *"Symmetric 'U' gives the name."*

**Why it worked with ~30 training images (slide p.51, ISBI 2012 EM):** four things, and they are the standard recipe for small-data segmentation —
- skip concatenation gives the decoder **full-resolution detail at every level**;
- **elastic deformation** augmentation (realistic for tissue);
- a **weighted loss up-weighting the thin gaps between touching cells**, which separates instances;
- **overlap-tile with mirrored borders**, so any image size can be segmented;
- and it ran **<1 s for a 512×512 image on a 2015 GPU**.

**The family (slide p.52):** 3D U-Net (volumetric CT/MRI) · nnU-Net (2021, *self-configuring*, now the medical baseline) · Attention U-Net (attention gates filter skip features) · U-Net++ (nested dense skips) · TransUNet (transformer in the bottleneck) · Diffusion U-Net (the denoiser in Stable Diffusion).

> **Takeaway (slide p.52):** *"U-Net is a pattern (multi-scale encoder, symmetric decoder, concat skips), not a fixed network."*

#### DeepLab v1 / v2 (2015) — atrous backbone + CRF

Slide p.53:

```
Backbone (atrous, OS 8) → Coarse scores → Bilinear ×8 → Dense CRF → mask
```

- **Atrous backbone keeps 1/8 resolution** while retaining pretrained context — instead of downsampling further, it *dilates*.
- **Fully-connected CRF** post-processing **snaps labels to image edges**.
- v2 adds **ASPP**: parallel atrous branches at rates **6, 12, 18, 24**.
- **No learned decoder** — upsampling is plain bilinear.

> **The slide's own THINK prompt (p.53) is worth memorising as the answer's shape:** *"Why might a CRF that uses pixel colour help at boundaries but hurt on a black car on dark asphalt?"* The CRF's pairwise terms reward colour agreement between neighbours. At a car/asphalt boundary the colours genuinely differ, so it sharpens correctly. But on a **black car against dark asphalt** the colours are nearly identical, so the CRF has no evidence to separate them and the boundary is left to the coarse scores alone — the sharpening that helps everywhere else does nothing exactly where you needed it.

v3 **removed the CRF** — slide p.53: *"better features made it unnecessary."*

#### ASPP (DeepLab v3) — multi-scale in one layer

Slide p.54, on an OS-16 feature map:

```
                 ┌── 1×1 conv ──────────────────┐
Backbone (OS 16) ┼── 3×3 conv, rate 6 ──────────┤
                 ├── 3×3 conv, rate 12 ─────────┤→ Concat → 1×1 conv → Scores
                 ├── 3×3 conv, rate 18 ─────────┤
                 └── image pooling + 1×1 + up ──┘
```

- *"Multiple receptive fields in parallel on the same features."*
- **Image pooling adds global context** — the whole-scene branch.
- BatchNorm in every branch (v3).

**The arithmetic that makes this concrete.** A 3×3 conv at rate $d$ has effective kernel $2d+1$ in **feature pixels**. At output stride 16, each feature pixel is 16 image pixels, so:

| rate | effective kernel (feature px) | image reach |
|---|---|---|
| 6 | 13 | **208 px** |
| 12 | 25 | 400 px |
| 18 | 37 | 592 px |
| 24 | 49 | **784 px** |

All verified. So ASPP sees a **~3.8× range of object sizes in one layer**, from cars to buildings, at a parameter cost almost identical to a single 3×3 conv (dilation adds no parameters).

#### DeepLab v3+ (2018) — ASPP + a real decoder

Slide p.55. Encoder = atrous backbone + ASPP. Decoder = fuse **stride-4 low-level features (48 channels)** → concat → ×4 up → mask. Uses **depthwise-separable** atrous convs (Xception / MobileNet) for efficiency.

*This is the fix for exactly the weakness SEG introduces below:* the decoder exists to put **sharp boundaries** back, because ASPP itself is low-resolution. Slide p.55 calls it *"a strong, widely deployed CNN baseline to this day."*

#### PSPNet (2017) — the same idea with pooling

Slide p.56: dilated ResNet at OS 8 → pooling branches with **1×1, 2×2, 3×3, 6×6** bins → 1×1 conv each → upsample → concat with the original → conv → mask.

| | ASPP | PSPNet |
|---|---|---|
| Multi-scale by | **dilation** | **pooling** |
| Global context | one image-pooling branch | the **1×1 bin** (whole map) |

Slide p.56's justification for the global branch: *"Global prior fixes context mistakes: boat vs car, pillow vs bed-sheet."* It won the ILSVRC 2016 scene-parsing challenge.

#### Mask R-CNN (2017) — masks as a detection *branch*

Slide p.57. This is the **instance** segmentation answer, and the design point is decoupling:

- A **parallel mask branch**: a small FCN per RoI producing **K class-specific 28×28 masks**.
- **Sigmoid + BCE per pixel** — *not* softmax over K masks. A pixel is in or out of this class.
- **Only the ground-truth class's mask is penalised** → *"mask and class decoupled."* Predicting the right class but the wrong mask costs you the box/class term but not a nonsense mask on every other class.
- **RoIAlign instead of RoIPool** — *"crucial for pixel accuracy."* RoIPool quantises, introducing a half-pixel misalignment that blurs the mask.
- $L = L_{cls} + L_{box} + L_{mask}$; ~5 fps in the original paper.

#### SETR (2021) — a pure transformer encoder

Slide p.58: 16×16 patches + positional embedding → **ViT-Large, 24 layers** → reshape to H/16 × W/16 → decoder (naive / PUP / MLA).

- *"First strong result with a pure-transformer encoder."*
- **Features are single-scale at 1/16** — *"no fine detail from early stages."*
- Heavy: **~318 M parameters** for **50.2 mIoU on ADE20K** (as reported in the SegFormer paper).
- **Positional embeddings tie it to the training resolution.**

**The trade SETR makes.** Class 08 established that attention costs $O(N^2)$. At 224×224 with 16×16 patches that's 196 tokens — affordable. But the encoder sees **only** that one scale, so all the fine detail must be reconstructed by the decoder from a 14×14 map. Compare U-Net, which concatenates stride-1 features: **U-Net wins on boundaries precisely because it never threw that information away.**

#### The timeline (slide p.47)

| Year | CNN era | Year | Transformer / foundation era |
|---|---|---|---|
| 2015 | FCN · SegNet · U-Net · DeepLab v1 | 2019 | Panoptic FPN |
| 2016 | DeepLab v2 | 2021 | **SETR** · SegFormer · MaskFormer |
| 2017 | **PSPNet** · **Mask R-CNN** | 2022 | Mask2Former |
| 2018 | DeepLab v3+ | 2023–25 | SAM · SAM 2 · SAM 3 |

### B8. Losses & metrics

**Pixel-wise cross-entropy** is the default: average over all labelled pixels, skip unlabelled with `ignore_index = 255`, compute on logits upsampled to label size. Simple and stable.

**The imbalance problem (slide p.41):** road, sky and building dominate pixel counts; poles, lights and bikes are tiny.

- **Weighted CE:** $w_k \propto 1/\text{freq}_k$ (or median-frequency balancing, as in SegNet).
- **Focal loss:** down-weights easy pixels so hard ones drive the gradient.

**Dice loss optimises overlap directly (slide p.42).** The killer stat:

> *"Lesion < 1% of pixels: predicting 'all background' gets > 99% accuracy but **Dice = 0**."*

That's the whole argument against pixel accuracy.

**Metrics (slide p.43):**

| Pixel accuracy | IoU (per class) | mIoU | Dice / F1 |
|---|---|---|---|
| correct / total | $TP/(TP{+}FP{+}FN)$ | mean of per-class IoU | $2TP/(2TP{+}FP{+}FN) = \dfrac{2\cdot IoU}{1+IoU}$ |
| **misleading under imbalance** | the standard | rare classes count equally | — |

Two rules the slide insists on:
1. **Accumulate TP/FP/FN over the whole dataset per class, then divide** — *not* a mean of per-image IoUs.
2. **Report per-class IoU too** — a good mIoU can hide a class that is always wrong.

**Panoptic Quality (slide p.45):** match ground truth to predictions at IoU > 0.5, then

$$PQ = SQ \times RQ, \qquad SQ = \frac{1}{|TP|}\sum IoU, \qquad RQ = \frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}$$

SQ = how well matched segments are drawn. RQ = the F1 of detection — did we find the right segments?

---

## C. Numbers worth knowing

All verified against the slides.

### C1. Transposed convolution (slide p.35)

$o = (i-1)s - 2p + k + \text{out\_pad}$

| $i$ | $k$ | $s$ | $p$ | out_pad | $o$ | Comment on slide |
|---|---|---|---|---|---|---|
| 4 | 3 | 1 | 0 | 0 | **6** | stride 1 still grows by $k-1$ |
| 4 | 2 | 2 | 0 | 0 | **8** | U-Net up-conv: no overlap |
| 4 | 4 | 2 | 1 | 0 | **8** | clean ×2, $k$ divisible by $s$ |
| 4 | 3 | 2 | 1 | 1 | **8** | ×2 but uneven overlap |
| 16 | 4 | 2 | 1 | 0 | **32** | a typical decoder step |

All five reproduced. Row 1 is the one that surprises people.

### C2. Slide p.38's five exercises

| # | Question | Answer | How |
|---|---|---|---|
| 1 | `ConvTranspose2d(k=4, s=2, p=1)` on 16×16 | **32×32** | $(16-1)\cdot2 - 2 + 4 = 32$ |
| 2 | 3×(3×3 conv) dilation 1,2,4 → receptive field | **15** | $1 \to 3 \to 7 \to 15$ |
| 3 | OS 32 on 512×1024 | **16×32** | $512/32,\ 1024/32$ |
| 4 | Why did FCN initialise transposed convs to bilinear? | stability | removes random-init checkerboard artefacts |
| 5 | 3×3 conv, $d{=}6$, $p{=}6$, stride 1, on 64×64 | eff. kernel **13**, output **64×64** | $(k{-}1)d{+}1 = 13$; $\lfloor(64+12-12-1)/1\rfloor+1 = 64$ |

### C3. Output stride

$$\text{feature size} = \left\lceil \frac{H}{\text{OS}} \right\rceil$$

512×1024: OS 32 → **16×32** · OS 16 → 32×64 · OS 8 → **64×128**. All verified.

### C4. The slide p.44 worked example (4×4, two classes)

| class | \|GT\| | \|Pred\| | ∩ | ∪ | IoU |
|---|---|---|---|---|---|
| 0 | 9 | 9 | 8 | 10 | **0.80** |
| 1 | 7 | 7 | 6 | 8 | **0.75** |

$$\text{mIoU} = \tfrac{0.80+0.75}{2} = \mathbf{0.775}, \qquad \text{pixel acc} = \tfrac{14}{16} = \mathbf{0.875}$$
$$\text{Dice}_1 = \frac{2(6)}{2(6)+1+1} = \mathbf{0.857} = \frac{2(0.75)}{1.75}$$

All three verified — and note the slide's own identity $\text{Dice} = \frac{2\cdot IoU}{1+IoU}$ checks out.

**A trap hiding in this example:** pixel accuracy is $14/16$, **not** $14/18$. Summing $TP+FP+FN$ gives 18 because every misclassified pixel is counted twice — once as a false positive for the wrong class, once as a false negative for the right one. The denominator is the **actual pixel count**, $\sum(TP+FN) = 16$. *(I wrote $14/18$ first; it gave 0.778 and disagreed with the slide.)*

### C5. Panoptic Quality (slide p.45)

TP with IoU 0.8 and 0.6; FP = 1; FN = 1:

$$SQ = \tfrac{0.8+0.6}{2} = 0.70, \qquad RQ = \frac{2}{2 + 0.5 + 0.5} = 0.667, \qquad PQ = \mathbf{0.467}$$

Verified.

### C6. Annotation cost (slide p.18)

| | Cost |
|---|---|
| One Cityscapes image, finely annotated | **~1.5 hours** |
| One COCO object mask | **~79 seconds** |
| SA-1B masks, produced with a model in the loop | **1.1 billion** |

The slide's remedy list: weak labels, semi-supervised training on unlabelled frames, synthetic data, and **model-in-the-loop annotation** — *"a promptable model proposes, a human corrects."*

---

## D. Pictures

### D1. Why a box fails on a thin diagonal

```
   ground truth          a box around it        the box says "road" here
   ╱  lane marking      ┌───────────────┐      but this is the lane!
  ╱                     │               │
 ╱   ~20% filled        │  lane + road  │
                        └───────────────┘

   a mask knows the difference; a box does not
```

### D2. The four tasks on one picture

```
  semantic            instance             panoptic
  ┌──────────┐        ┌──────────┐          ┌──────────┐
  │ sky      │        │  sky     │          │ sky      │  stuff:
  │          │        │          │          │  (id 1)  │  one id
  │ ▓▓  ▓▓   │        │  (no sky)│          ├────┬─────┤
  │ road     │        │ ▓1▓ ▓2   │          │car1│car2 │  things:
  └──────────┘        └──────────┘          │(id2)(id3)│  own id each
   car1 = car2         car1 ≠ car2          └──────────┘
```

### D3. Encoder–decoder, with the trap

```
  resolution   1/1  1/2   1/4   1/8   1/16   1/32
                │     │     │     │      │      │
   what the    ██████████████████░                 ← encoder: "what"
   net knows         grows ▓▓▓▓▓                   ← bottleneck: semantics
                                          ▒▒▒▒▒            ← decoder: "where"

   deep + cheap          shallow + expensive
   large RF, few px      big maps, costly convs

   ◄──────── cheap, sees far ────────►◄── expensive, precise ──►
```

### D4. Dilation: same 3×3 weight count, much bigger reach

```
  d=1          d=2          d=6
  ● ● ●        ● · ● · ●    ● · · · · · · ● · · · · · ●
  ● ● ●        ● · ● · ●    · · · · · · · · · · · · ·
  ● ● ●        ● · ● · ●    · · · · ● · · · · · · · ·
   9 taps        9 taps       · · · · · · · · · · · · ·      9 taps
  effective 3  effective 5   effective 13
                padding must be d, and only (k-1)d/2 preserves size
```

### D5. Checkerboard from transposed conv

```
  k=4, s=2, p=1  → clean 2× upsample
  k=3, s=2, p=0  → uneven overlap between stamps
                   ░▓░▓░
                   ▓░▓░▓      "checkerboard"
                   ░▓░▓░
  why: k % s != 0, so stamps do not tile uniformly
  fix : resize (bilinear) + conv  →  no overlap problem by construction
```

### D6. Pixel accuracy lies

```
  lesion covers 1% of pixels
  ─────────────────────────────────────────────
  predict all background:
     pixel accuracy  99.4%     ← looks perfect
     Dice                0     ← actually useless
  ─────────────────────────────────────────────
  this is why mIoU / Dice, not accuracy, is reported
```

---

## E. Code

> No instructor code exists in this workspace (the transcript is empty). The blocks below are **illustrative PyTorch** and were **not executed** — PyTorch isn't installed here. The *Section D* solutions in §H are pure-Python and **were** executed, with real output pasted in.

### E1. The two lines that decide your loss

```python
# logits: (B, K, H, W)   target: (B, H, W) with values 0..K-1
loss = F.cross_entropy(logits, target, ignore_index=255)

# what the models actually do: predict at low res, upsample logits to label size
logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
loss   = F.cross_entropy(logits, target, ignore_index=255)
```

**Shape + dtype notes that bite:**
- Channel-first `(B, K, H, W)` for the loss, but **(B, H, W)** class indices for the target. Mismatching these is the #1 beginner error.
- `ignore_index=255` marks void/unlabelled pixels — they must be excluded, or the model is trained to predict a class that doesn't exist.
- Keep `align_corners` **identical** between training and export (slide p.27).

### E2. Transposed conv

```python
up = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                       kernel_size=4, stride=2, padding=1)   # 16×16 → 32×32
```

`kernel_size=4, stride=2, padding=1` is the clean ×2 recipe (slide p.35 row 4). Avoid `k % s != 0` unless you want checkerboards.

### E3. Effective kernel — get padding right

```python
EFF = lambda k, d: (k - 1) * d + 1          # 3, d=6  ->  13
KEEP = lambda k, d: ((k - 1) * d) // 2      # 3, d=6  ->   6
```

`padding=6` for `k=3, d=6` is what preserves a 64×64 input. `padding=1` would give 74.

### E4. Metrics

```python
confusion = torch.zeros(K, K, dtype=torch.long, device='pred.device')
for t, p in zip(target.flatten(), pred.flatten()):
    if t != 255:
        confusion[t, p] += 1

inter = confusion.diag()
union = confusion.sum(0) + confusion.sum(1) - inter
iou    = inter.float() / union.clamp(min=1)
miou   = iou.mean()
```

Accumulate `confusion` across the **whole dataset**, then compute — never average per-image IoU (slide p.43).

### E5. Bugs worth memorising

| Symptom | Cause |
|---|---|
| Cross-entropy rejects your target | target must be `(B,H,W)` class indices, not `(B,K,H,W)` logits |
| Model output vanishes toward one class | imbalance — see Dice/focal, not CE |
| Loss ignores the whole test set | forgetting `ignore_index`; a Void region set to 255 |
| Inference differs from training, no visible bug | `align_corners` mismatch |
| Checkerboard in the mask | `k % s != 0` in transposed conv |
| Output larger than input | dilation without `padding=(k-1)d/2` |

---

## F. Comparisons

**Semantic vs instance vs panoptic — what each actually gives you**

| | Semantic | Instance | Panoptic |
|---|---|---|---|
| Answers "what is here?" | ✅ | ✅ | ✅ |
| Answers "how many?" | ❌ | ✅ | ✅ (things only) |
| Answers "exactly which pixels?" | ✅ | ✅ (things) | ✅ |
| Handles road/sky | ✅ | ❌ | ✅ |
| Metric | mIoU | mask AP | PQ |

**The eight architectures, side by side**

| | Year | Upsample by | Context via | Skips | OS | Standout fact |
|---|---|---|---|---|---|---|
| **FCN** | 2015 | transposed conv | FCN-8s/16s skip | add | coarse→upsample | first fully end-to-end; 512×512×3 → 512×512×21 |
| **SegNet** | 2015 | **max unpool (stored indices)** | none | **none** — indices only | 16 | memory-light, real-time; median-freq loss |
| **U-Net** | 2015 | 2×2 transposed conv | none | **concatenate** | 4→32 | 572→388; works on **~30 images** |
| **DeepLab v1/v2** | 2015–16 | **plain bilinear ×8** | atrous backbone (OS 8) | none | 8 | **+ Dense CRF** to snap to edges |
| **PSPNet** | 2017 | bilinear | **pooling 1/2/3/6 bins** | none | 8 | 1×1 bin = global prior (boat vs car) |
| **Mask R-CNN** | 2017 | RoIAlign + resample | detection backbone | none | per-RoI | **K × 28×28 masks**, only GT class penalised |
| **DeepLab v3+** | 2018 | ×4 bilinear after a real decoder | **ASPP** 6/12/18/24 | **stride-4, 48 ch** | 16 | depthwise-separable; deployed baseline |
| **SETR** | 2021 | PUP / MLA decoder | self-attention | none | **16, single scale** | 318 M params, 50.2 mIoU ADE20K |

**ASPP vs PSPNet** — same goal, different mechanism:

| | ASPP (DeepLab v3) | PSPNet |
|---|---|---|
| Multi-scale by | **dilation** (rates 6/12/18/24) | **pooling** (bins 1×1/2×2/3×3/6×6) |
| Global context | one image-pooling branch | the **1×1 bin** (whole map) |
| Extra params | none (dilation is free) | one 1×1 conv per branch |

**Which fix solves which problem**

| Problem | Fix | Model |
|---|---|---|
| Need context, cheaply | dilate instead of downsampling | DeepLab v1/v2 |
| Need several object sizes at once | parallel dilation rates | **ASPP** |
| Need global scene context | pool the whole map | PSPNet / ASPP branch |
| Need boundaries back | concat high-res features | **U-Net**, DeepLab v3+ |
| Need instances, not just classes | per-RoI mask branch | **Mask R-CNN** |

**U-Net variants** (slide p.52): 3D U-Net · nnU-Net (self-configuring, 2021) · Attention U-Net (gates on skips) · U-Net++ (nested dense skips) · TransUNet (transformer bottleneck) · Diffusion U-Net (the denoiser in Stable Diffusion).

> Takeaway from slide p.52: *"U-Net is a pattern (multi-scale encoder, symmetric decoder, concat skips), not a fixed network."*

---

## G. Exam-focused revision sheet

**Definitions**
- Semantic / instance / panoptic — differ in *whether instances are separated* and *whether stuff is covered*.
- Stuff vs things — countable-with-a-shape vs amorphous.
- Output stride (OS) — input ÷ feature size.
- Effective kernel — $(k{-}1)d + 1$.
- SQ / RQ / PQ — segmentation quality / recognition quality / their product.

**Formulas**

$$o = (i{-}1)s - 2p + k + \text{out\_pad} \qquad \text{(transposed conv)}$$
$$\text{eff. kernel} = (k{-}1)d + 1 \qquad \text{keep-size padding} = \tfrac{(k{-}1)d}{2}$$
$$r_{out} = r_{in} + (k{-}1)d \qquad \text{feature size} = \lceil H/\text{OS} \rceil$$
$$\text{IoU} = \tfrac{TP}{TP{+}FP{+}FN} \qquad \text{Dice} = \tfrac{2TP}{2TP{+}FP{+}FN} = \tfrac{2\,IoU}{1+IoU}$$
$$PQ = SQ \times RQ, \qquad SQ = \tfrac{1}{|TP|}\textstyle\sum IoU, \qquad RQ = \tfrac{|TP|}{|TP|+\frac12|FP|+\frac12|FN|}$$

**Traps — these lose marks**
1. Transposed conv uses **$(i-1)$**, not $i$. Stride 1 with $k{=}3$ still grows 4→6.
2. Dilation needs **padding 6** for $k{=}3,d{=}6$ — not padding 1.
3. Pixel accuracy denominator is **actual pixels** ($TP+FN$), not $TP+FP+FN$ (which double-counts).
4. Accumulates TP/FP/FN over the **dataset** before dividing — not a mean of per-image IoUs.
5. mIoU can hide a class that is always wrong — **report per-class IoU too**.
6. Plain IoU is dead for disjoint boxes → that's why gIoU exists (Class 08); Dice overlaps on the same issue.

**Checklist**
- [ ] Pick the right task for a product requirement, and justify it
- [ ] Explain why boxes fail on thin/thin-diagonal objects
- [ ] Compute transposed-conv output size
- [ ] Compute effective kernel and padding for dilation
- [ ] Compute output stride and feature map size
- [ ] Compute IoU, Dice, mIoU, PQ from counts
- [ ] Explain why pixel accuracy is misleading
- [ ] Name what each of FCN / SegNet / U-Net / DeepLab / PSPNet contributes

---

## H. Practice Questions

> Mirrors `session16.pdf`: **A** single-correct MCQ · **B** multi-select · **C** descriptive ≤100 words · **D** coding, attempt 1 of 3 · **E** case study · **F** numericals.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** A warehouse robot must pick a specific box off a pallet where several identical boxes sit side by side. Which output do you need?

- A. Semantic segmentation, since every box is one class
- B. Instance segmentation, to tell the boxes apart
- C. Panoptic segmentation, since boxes are countable things
- D. A bounding box per box, since that is enough to grasp

**A2.** `ConvTranspose2d(in=16, k=4, s=2, p=1)` applied to a 16×16 map produces what output size?

- A. 30×30
- B. 31×31
- C. 32×32
- D. 34×34

**A3.** A 3×3 convolution with dilation 6 and stride 1 has what effective kernel size?

- A. 3
- B. 7
- C. 13
- D. 19

**A4.** A tumour occupies 1.5% of an MRI slice. A model predicts all background and achieves 98.5% pixel accuracy. Its Dice score is approximately:

- A. 0.985
- B. 0.49
- C. 0.015
- D. 0.000

**A5.** Panoptic Quality with two correctly matched segments (IoU 0.8 and 0.6), one false positive and one false negative is approximately:

- A. 0.47
- B. 0.70
- C. 0.83
- D. 1.40

**A6.** Why do most modern segmentation decoders prefer resize-then-convolve over a transposed convolution?

- A. It is more accurate at every resolution
- B. It avoids uneven overlap and checkerboard artefacts
- C. It requires no learnable parameters
- D. It makes the encoder receptive field larger

**A7.** Mask R-CNN predicts K class-specific masks for each RoI, but penalises only the ground-truth class's mask. What is the main benefit?

- A. It cuts the mask loss to a single term per RoI
- B. It prevents class and mask errors from corrupting each other
- C. It lets the same mask be reused for every class
- D. It removes the need for RoIAlign

**A8.** SETR reports strong results using a pure transformer encoder. What is its main architectural weakness for dense prediction?

- A. Attention cost grows quadratically with token count
- B. Encoder features exist at a single coarse scale
- C. It cannot accept positional embeddings
- D. It produces fewer than 100 tokens per image

### Section B — Multi-select (5 × 2 marks)

**B1.** Which statements about **panoptic segmentation** are **correct**? Select all that apply.

- A. Every pixel receives a class label
- B. Every pixel receives an instance identifier
- C. Segments may overlap
- D. It is evaluated with Panoptic Quality
- E. It is naturally suited to full scene-understanding stacks

**B2.** A transposed convolution with `kernel_size=3` and `stride=2` produces visible checkerboard artefacts. Which changes would help? Select all that apply.

- A. Set kernel_size to 4
- B. Add a bilinear resize before a convolution
- C. Reduce the stride to 1
- D. Increase the padding to 2
- E. Initialise the weights to bilinear kernels

**B3.** Which are **correct** statements about evaluation for segmentation? Select all that apply.

- A. Pixel accuracy is unreliable under class imbalance
- B. mIoU should be computed from dataset-level counts
- C. Per-class IoU should always accompany mIoU
- D. Dice and IoU are unrelated measures
- E. Panoptic Quality multiplies a shape term by a detection term

**B4.** Which are **valid ways** to raise the effective receptive field cheaply? Select all that apply.

- A. Increase dilation rate
- B. Increase output stride in the encoder
- C. Add more fully connected layers at full resolution
- D. Use an encoder backbone instead of stacking convolutions
- E. Raise the kernel size in the decoder only

**B5.** Which statements about the classic segmentation architectures are **correct**? Select all that apply.

- A. U-Net concatenates encoder features rather than adding them
- B. SegNet's decoder recovers detail from stored max-pool indices
- C. PSPNet obtains multi-scale context using dilation rates
- D. DeepLab v3 removed the CRF because better features made it unnecessary
- E. Mask R-CNN applies softmax over K masks at each pixel

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why can a bounding box be actively misleading for a thin diagonal object?

**C2.** Why does segmentation use an encoder–decoder rather than classifying every pixel?

**C3.** Why does dilating a convolution change the padding required, and what is the consequence of getting it wrong?

**C4.** Why is pixel accuracy a poor metric for lesion segmentation?

**C5.** What do the two factors of Panoptic Quality each measure, and why is separating them useful?

### Section D — Coding (20 marks — attempt ONE of the three)

**Pure Python, standard library only.** Provide a reusable function. Handle empty and boundary inputs.

---

**Option D1 — Bilinear upsampling**

```python
def upsample_bilinear(img, out_h, out_w, align_corners=True): ...
def upsample_nearest(img, out_h, out_w): ...
```

- `img`: non-empty list of rows of numbers.
- Both return `out_h` rows of `out_w` numbers, rounded to 6 dp for the bilinear version.
- `align_corners=True` samples at `(i-1)/(out-1)`; `False` samples at `i/out`. Edge samples are clamped.
- Empty input or non-positive output → `[]`.

Worked examples:

```
img = [[1,2],[3,4]]

upsample_bilinear(img, 3, 3, True)
    -> [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]]
upsample_nearest(img, 4, 4)
    -> [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]
upsample_bilinear(img, 3, 3, False)
    -> [[1.0, 1.666667, 2.0], [2.333333, 3.0, 3.333333], [3.0, 3.666667, 4.0]]
upsample_bilinear(img, 2, 2)
    -> [[1,2],[3,4]]
upsample_bilinear([], 3, 3)   ->  []
upsample_bilinear(img, 0, 3)  ->  []
```

**Option D2 — Segmentation metrics**

```python
def seg_metrics(conf): ...
```

- `conf`: dict `{class_index: (TP, FP, FN)}`.
- Returns a dict with `IoU` (per class), `Dice` (per class), `mIoU`, `pixel_acc`, `global_IoU`.
- `pixel_acc` denominator is **actual labelled pixels**, $\sum(TP + FN)$ — *not* $\sum(TP+FP+FN)$.
- Empty dict → `{}`. All values rounded to 6 dp.

Worked examples:

```
seg_metrics({0: (8,1,1), 1: (6,1,1)})
  IoU       ->  {0: 0.8,  1: 0.75}
  Dice      ->  {0: 0.888889, 1: 0.857143}
  mIoU      ->  0.775
  pixel_acc ->  0.875
  global_IoU->  0.777778

seg_metrics({0: (90,0,10), 1: (0,0,5)})
  IoU       ->  {0: 0.9, 1: 0.0}
  mIoU      ->  0.45
  pixel_acc ->  0.857143

seg_metrics({})  ->  {}
```

**Option D3 — Panoptic Quality**

```python
def panoptic_quality(ious, n_fp, n_fn, match_threshold=0.5): ...
```

- `ious`: list of IoU values, one per proposed GT↔prediction match.
- A match counts as TP only if `IoU > match_threshold`.
- Returns `{"SQ":…, "RQ":…, "PQ":…}`, each rounded to 6 dp. If no TP, all three are `0.0`.

Worked examples:

```
panoptic_quality([0.8, 0.6], 1, 1)   ->  {"SQ":0.7,"RQ":0.666667,"PQ":0.466667}
panoptic_quality([], 0, 0)           ->  {"SQ":0.0,"RQ":0.0,"PQ":0.0}
panoptic_quality([0.2], 0, 1)        ->  {"SQ":0.0,"RQ":0.0,"PQ":0.0}
panoptic_quality([0.9, 0.9], 2, 0)   ->  {"SQ":0.9,"RQ":0.666667,"PQ":0.6}
```

### Section E — Case study (15 marks)

> **Scenario.** A mining company uses drones to survey open-pit mine walls. They need to measure **the area of unstable rock** each week and **the depth of each crack**, to decide which sections to cordon off. Images are 4000×3000, captured at varying zoom. Annotating pixel masks is far too slow: the current team manages 2 images per day. They already run a detector that finds "large rocks" reliably.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Semantic, instance, or panoptic? Justify in terms of the two questions being asked (area, depth).
2. **Data.** Why is a 4000×3000 image a specific problem? State what slide p.18's remedies you would adopt and why.
3. **Model.** Sketch an encoder–decoder. State the output stride you'd choose and why not OS 32.
4. **Metric.** You have 1,200 images of crack masks and the "large rocks" detector. What do you report, and what do you *not* report?
5. **Failure mode.** Zoom level varies between flights. Name the failure and the mechanism.

**Sub-questions**

- **E-a.** Your mask model reports 97% pixel accuracy and mIoU 0.61. Explain why these two numbers are not in conflict, and what that implies about which class is failing.
- **E-b.** The decoder upsamples with `ConvTranspose2d(k=3, s=2, p=0)` and the masks show visible grid artefacts. Diagnose and give two fixes.
- **E-c.** Ground truth masks were drawn as 4000×3000 but training ran at 512×512. Compute the output stride implied, and describe one consequence for very thin cracks.

### Section F — Numericals (8 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** Compute the output size of `ConvTranspose2d` for $(i,k,s,p,\text{out\_pad}) = (16,4,2,1,0)$, $(8,3,1,0,0)$ and $(4,3,2,1,1)$. Explain why the first of these is the "clean ×2" case.

**F2.** Three 3×3 convolutions are stacked with dilation 1, 2 and 4. Compute the receptive field, showing the running value at each step. Then state the effective kernel size of a single 3×3 convolution with dilation 6.

**F3.** An encoder has output stride 32 and receives a 512×1024 image. Compute the feature map size, then repeat for output stride 8. State the ratio of pixel counts.

**F4.** For a two-class 4×4 case with ground-truth sizes 9 and 7, prediction sizes 9 and 7, and per-class intersections 8 and 6: compute each class's union, both IoUs, mIoU, pixel accuracy and Dice for class 1. State which of these is misleading and why.

**F5.** Two ground-truth segments match predictions at IoU 0.8 and 0.6, with one false positive and one false negative. Compute SQ, RQ and PQ. Then recompute RQ if the threshold were raised so that only the 0.8 match counts.

**F6.** A 3×3 convolution with dilation 6 and stride 1 is applied to a 64×64 map with padding 6. Compute the effective kernel size and the output size. Then state what the output size would be if padding were 1, and what that implies about the model.

**F7.** DeepLab v3's ASPP operates on an output-stride-16 feature map and uses 3×3 branches with rates 6, 12, 18 and 24. Compute the effective kernel size of each branch *in feature-map pixels* and convert each to *image pixels*. What ratio of object sizes does one ASPP layer therefore cover, and how does that compare with the parameter count of a plain 3×3 branch?

**F8.** SegNet and U-Net both undo encoder downsampling in the decoder, but they keep different things. State what SegNet stores and what U-Net keeps, give the parameter-free consequence of SegNet's choice, and explain which one you would choose for 3-pixel-wide vessels in a retinal scan.

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — Instance segmentation, to tell the boxes apart.**

Semantic output labels every box pixel "box" — identical instances. *"You can count, track, grasp"* requires separating them (slide p.7).

- **A** is exactly the failure: one class for all boxes.
- **C** is overkill — panoptic adds stuff segmentation (floor, walls) the robot doesn't need, and the slide's own framing is *"choose the cheapest rung that answers your product question."*
- **D** is the trick option — a box *is* enough for a coarse grasp, so arguably true; but the question says "a **specific** box… side by side", and the slide's own THINK prompt (p.11) uses exactly this scenario with the answer being the instance row.

**A2. → C — 32×32.**

$o = (i{-}1)s - 2p + k + \text{out\_pad} = (16{-}1)\cdot2 - 2 + 4 + 0 = 30 - 2 + 4 = 32$. **Verified.**

- **A** (30) forgets the $+k$.
- **B** (31) uses $i$ instead of $(i-1)$: $16\cdot2 - 2 + 4 = 34$, no — $31$ comes from $(16{-}1)\cdot2 - 2 + 3$, i.e. dropping $k$ to 3.
- **D** (34) is the classic error of using $i$ instead of $i{-}1$: $16\cdot2 - 2 + 4 = 34$.

**A3. → C — 13.**

$(k{-}1)d + 1 = (3{-}1)\cdot6 + 1 = 13$. **Verified.** A is the undilated size; B would be $d=4$; D would be $d=9$.

**A4. → D — 0.000.**

Dice $=\frac{2TP}{2TP+FP+FN}$. Predicting all background gives $TP = 0$, so Dice $= 0$ exactly — slide p.42's example: *"predicting 'all background' gets > 99% accuracy but Dice = 0."*

- A (0.985) is the pixel accuracy, which is exactly the number the question tells you to distrust.
- B (0.49) and C (0.015) look like plausible near-misses but neither follows from the formula.

**A5. → A — 0.47.**

$SQ = (0.8+0.6)/2 = 0.70$; $RQ = 2/(2 + 0.5 + 0.5) = 0.667$; $PQ = 0.70 \times 0.667 = \mathbf{0.467}$. **Verified** against slide p.45.

- **B** (0.70) is SQ alone.
- **C** (0.83) is closer to RQ but wrong — the ½ weights make RQ 0.667.
- **D** (1.40) is $2\times SQ$, a plausible sign error from multiplying instead of dividing.

**A6. → B — It avoids uneven overlap and checkerboard artefacts.**

Slide p.36: resize + conv has *"No uneven overlap by construction"*, and slide p.37's cheat-sheet lists *"Checkerboard if $k \bmod s \neq 0$"* as the transposed-conv warning.

- **A** is false — the whole point of slide p.36 is that it's a trade: *"Cost: conv runs at the higher resolution."*
- **C** is false — it has exactly the same number of learnable weights $k^2C_{in}C_{out}$.
- **D** is nonsense — the decoder doesn't change the encoder's receptive field.

**A7. → B — It prevents class and mask errors from corrupting each other.**

Slide p.57: *"only the GT class's mask is penalised → mask and class decoupled."*

- **A** is the *mechanism*, not the benefit — and the mask term is still summed over RoIs, not zeroed.
- **C** is backwards: masks are **class-specific**, so they cannot be shared.
- **D** is false — RoIAlign is still required, and slide p.57 calls it *"crucial for pixel accuracy."*

**A8. → B — Encoder features exist at a single coarse scale.**

Slide p.58: *"Features are single-scale at 1/16 — no fine detail from early stages."* All detail must be reconstructed by the decoder from a 14×14 map — which is why **U-Net's concatenation of stride-1 features wins on boundaries**: it never discarded that information.

- **A** is true of transformers generally but is **not** what the slide flags, and at 196 tokens it is affordable. This is the Class 08 problem, and SETR is not a demonstration of it.
- **C** is false — the slide explicitly shows *"16×16 patches + pos. embed."*
- **D** is false — 224/16 = 14, so 196 tokens.

### Section B — Solutions

**B1. → A, B, D, E. Reject C.**

- **A** ✔ *"Every pixel gets (class, instance-id)"* (p.8).
- **B** ✔ same line — stuff pixels simply share one id per class.
- **C** ✗ The defining constraint is **non-overlapping**: *"each pixel belongs to exactly one segment."* Overlap is the failure mode, not a feature.
- **D** ✔ *"with the Panoptic Quality (PQ) metric"* (p.8).
- **E** ✔ *"The natural output for a full scene understanding stack (e.g. AV perception)."*

**B2. → A, B, E. Reject C, D.**

- **A** ✔ $k{=}4$ with $s{=}2$ gives $k \bmod s = 0$ → uniform tiling (slide p.35 row 3).
- **B** ✔ slide p.36's fix exactly.
- **E** ✔ slide p.34: *"Weights are learned like any conv; often initialised to bilinear (FCN did this)"*; slide p.38 Q4 asks why.
- **C** ✗ stride 1 doesn't *remove* overlap — it makes things worse, since the output then grows by $k{-}1$ and stamps still don't tile.
- **D** ✗ padding shifts where the overlap happens; it does not make $k$ divisible by $s$. Note $k{=}3,s{=}2$ needs `output_padding` to hit ×2, which slide p.35 row 4 flags as *"×2, but uneven overlap"* — still artefact-prone.

**B3. → A, B, C, E. Reject D.**

- **A** ✔ slide p.43: *"Misleading under imbalance."*
- **B** ✔ *"Accumulate TP/FP/FN over the whole dataset per class, then divide (not a mean of per-image IoUs)."*
- **C** ✔ *"a good mIoU can hide a class that is always wrong."*
- **D** ✗ They're the same thing: slide p.43 prints Dice as $=\frac{2\,IoU}{1+IoU}$. **Verified** numerically in C4 (0.75 → 0.857).
- **E** ✔ $PQ = SQ \times RQ$.

**B4. → A, B, D. Reject C, E.**

- **A** ✔ effective kernel $=(k{-}1)d+1$ — more reach at identical parameter count.
- **B** ✔ larger OS = smaller feature map = larger receptive field per unit.
- **D** ✔ slide p.25: the encoder *"is a classification backbone (VGG, ResNet, MiT): cheap, large receptive field."*
- **C** ✗ slide p.24 rules this out as naive idea #2: *"to see 1,000 px you would need ~500 layers of 3×3"*, plus the memory blow-up.
- **E** ✗ the decoder runs at high resolution — a bigger decoder kernel increases cost without addressing the context problem, which lives in the encoder.

**B5. → A, B, D. Reject C, E.**

- **A** ✔ slide p.50: *"Skips concatenate (U-Net) rather than add (FCN)."*
- **B** ✔ slide p.49: *"Max unpooling with indices from the matching encoder layer."* SegNet stores **indices, not feature maps** — that is what makes it memory-light.
- **C** ✗ **PSPNet uses pooling**, not dilation — slide p.56: *"Pooling-based multi-scale context (ASPP is dilation-based)."* Dilation rates 6/12/18/24 belong to **ASPP**, which slide p.53 attributes to DeepLab **v2**.
- **D** ✔ slide p.53: *"v3 removed the CRF: better features made it unnecessary."*
- **E** ✗ slide p.57 specifies *"Sigmoid + BCE per pixel"* — a binary in/out decision for one class. Softmax would force exactly one class per pixel, which is precisely the thing instance segmentation must avoid.

### Section C — Solutions

*Each ≤100 words, three beats: define → reason → stop.*

**C1.**
A bounding box is axis-aligned, so a thin object at an angle leaves large empty regions inside its own box. Slide p.3's figure states a 45° thin object fills only **~20%** of its bounding box. A box therefore cannot distinguish "thin object here" from "empty background here" — every pixel inside is equally claimed. For lane markings, poles, cracks or cables this destroys exactly the information needed. Where the question concerns **area, a length, or a boundary** — slide p.15's rule — only a per-pixel mask answers it. Counting or rough localisation can tolerate boxes; area measurement cannot.

*(97 words.)*

**C2.**
Classifying every pixel independently means one forward pass per pixel. On a Cityscapes image ($1024\times2048$) that is **~2.1 million passes** (slide p.22), and adjacent patches share almost all their pixels so features are recomputed endlessly. Keeping full resolution instead fails differently (slide p.24): activations scale with $H\times W\times C$, and to see 1,000 px you would need ~500 layers. The encoder–decoder splits the problem: the encoder downsamples cheaply to gain a **large receptive field**, the bottleneck holds semantics at low resolution, and the decoder upsamples to recover **where**. One forward pass, and every pixel still gets a label.

*(96 words.)*

**C3.**
Dilation spaces the kernel taps $d$ apart, so the $k$ weights now span $(k{-}1)d + 1$ pixels instead of $k$. Output size depends on the **effective** kernel, so keeping the input size requires padding $\frac{(k{-}1)d}{2}$, not $d$. For $k{=}3, d{=}6$ the effective kernel is 13 and the padding must be 6. Using padding 1 instead gives $\lfloor(64+2-13)/1\rfloor+1 = 64$… using the raw kernel gives $\lfloor(64+2-3)/1\rfloor+1 = 64$ as well, which **expands** the map to 74 and misaligns every downstream stage. The layer's real output size no longer matches its input, so skip connections no longer line up and the network can no longer be built as designed.

*(99 words.)*

**C4.**
Segmentation images are dominated by a few classes. Slide p.42's example: a lesion occupying under 1% of pixels. A model that predicts all background scores **>99% pixel accuracy** because the vast majority of pixels are genuinely background — while contributing **Dice = 0**, since $TP = 0$. The metric rewards the majority class and ignores exactly the pixels you care about. Slide p.43 therefore marks pixel accuracy *"misleading under imbalance"* and pairs it with IoU / mIoU / Dice, which weight every class equally and score the predicted *region* rather than the pixels.

*(94 words.)*

**C5.**
$RQ$ (recognition quality) answers *"did we find the right segments?"* — it is the F1 of detection, $RQ = |TP|/(|TP| + \frac12|FP| + \frac12|FN|)$, penalising both spurious and missed segments. $SQ$ (segmentation quality) answers *"are the matched segments drawn accurately?"* — it is the mean IoU over matched pairs. Separating them is useful because they fail independently: a model can draw the right segments beautifully but miss some ($SQ$ high, $RQ$ low), or detect the right number of segments but draw them poorly ($RQ$ high, $SQ$ low). Slide p.45 reports PQ per class, averaged over classes, separately for things and stuff.

*(97 words.)*

### Section D — Solutions

#### D1 — Bilinear upsampling

**Key idea tested.** Computing the source coordinate correctly under both `align_corners` conventions, and clamping at edges.

```python
def upsample_bilinear(img, out_h, out_w, align_corners=True):
    if not img or not img[0] or out_h <= 0 or out_w <= 0:
        return []
    ih, iw = len(img), len(img[0])
    if out_h == ih and out_w == iw:
        return [r[:] for r in img]
    sy = (ih - 1) / (out_h - 1) if align_corners and out_h > 1 else ih / out_h
    sx = (iw - 1) / (out_w - 1) if align_corners and out_w > 1 else iw / out_w

    def sample(a, scale, n):
        v = a * scale
        i = int(v)
        if i < 0:
            i, frac = 0, 0.0
        elif i > n - 1:
            i, frac = n - 1, 0.0
        else:
            frac = v - i
        return i, frac

    out = []
    for r in range(out_h):
        y0, fy = sample(r, sy, ih)
        y1 = min(y0 + 1, ih - 1)
        row = []
        for c in range(out_w):
            x0, fx = sample(c, sx, iw)
            x1 = min(x0 + 1, iw - 1)
            top = img[y0][x0] * (1 - fx) + img[y0][x1] * fx
            bot = img[y1][x0] * (1 - fx) + img[y1][x1] * fx
            row.append(round(top * (1 - fy) + bot * fy, 6))
        out.append(row)
    return out

def upsample_nearest(img, out_h, out_w):
    if not img or not img[0] or out_h <= 0 or out_w <= 0:
        return []
    ih, iw = len(img), len(img[0])
    return [[img[r * ih // out_h][c * iw // out_w] for c in range(out_w)]
            for r in range(out_h)]
```

**Verified execution output:**

```
bilinear 2x2->3x3 ac=True : [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]]
nearest   2x2->4x4       : [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]
bilinear 2x2->3x3 ac=False: [[1.0, 1.666667, 2.0], [2.333333, 3.0, 3.333333], [3.0, 3.666667, 4.0]]
identity                 : [[1, 2], [3, 4]]
empty -> []    zero -> []
```

**Why it works.** The output pixel $(r,c)$ samples the input at a real-valued position and blends its four neighbours with weights $(1-f_x)(1-f_y)$ etc. — slide p.27's formula. With `align_corners=True` and 2→3, the scale is $(2{-}1)/(3{-}1) = 0.5$, so the three samples land on $0.0,\ 0.5,\ 1.0$ → values $1.0,\ 1.5,\ 2.0$ across the top row, **reproducing the slide's grid exactly**.

**The two examples side by side are the point.** With `align_corners=False` the scale is $2/3$, so the middle sample lands at $0.667$ and gives **1.667, not 1.5**. Same input, same output size, different answer — which is exactly slide p.27's *"Gotcha"* about keeping training and export consistent.

**Common wrong approaches.**
- Using `int()` without clamping, so the last output row raises `IndexError`.
- Forgetting the `out_h == 1` guard, which divides by zero in the `align_corners` branch.
- Nearest sampling with `round()` instead of `//` — shifts the whole grid by half a pixel.

#### D2 — Segmentation metrics

**Key idea tested.** Per-class counting, and the distinction between *actual* pixels and *summed* counts.

```python
def seg_metrics(conf):
    if not conf:
        return {}
    ious, dices = {}, {}
    tp_t = fn_t = correct = total = 0
    for k, (tp, fp, fn) in conf.items():
        u = tp + fp + fn
        ious[k] = round(tp / u, 6) if u else 0.0
        d = 2 * tp + fp + fn
        dices[k] = round(2 * tp / d, 6) if d else 0.0
        tp_t += tp
        fn_t += fn
        correct += tp
        total  += tp + fn          # actual pixels, NOT tp+fp+fn
    return {
        "IoU": ious,
        "Dice": dices,
        "mIoU": round(sum(ious.values()) / len(ious), 6),
        "pixel_acc": round(correct / total, 6) if total else 0.0,
        "global_IoU": round(tp_t / (tp_t + fn_t + (sum(
            f for _, (t, f, n) in conf.items()) )), 6) if total else 0.0,
    }
```

**Verified execution output:**

```
slide-44 example: IoU {0: 0.8, 1: 0.75}  mIoU 0.775  pixel_acc 0.875  Dice1 0.857143
rare class never predicted: IoU {0: 0.9, 1: 0.0}  mIoU 0.45  pixel_acc 0.857143
```

Both reproduce slide p.44's three headline numbers exactly.

**The bug this caught — and it's the one slide p.44 is designed to catch.** My first version used `total += tp + fp + fn`, giving $14+2+2 = 18$ and a pixel accuracy of **0.778**, contradicting the slide's **0.875**. The reason: every misclassified pixel is counted **twice** — once as an FP for the class it was wrongly assigned, once as an FN for the class it truly belongs to. The correct denominator is the **actual labelled pixel count**, $\sum(TP+FN) = 16$. If your framework hands you a confusion matrix rather than `(TP,FP,FN)`, the safe formulation is `pixel_acc = confusion.diag().sum() / confusion.sum()`.

**The second example is the slide's own THINK prompt (p.44).** A class present once in ground truth and never predicted gets IoU $= 0$, so mIoU falls from 0.775 to **0.45** — while pixel accuracy barely moves (0.857). That is the entire argument for mIoU, and it is why slide p.43 insists on reporting per-class IoU too.

**Common wrong approaches.**
- Averaging per-image IoUs instead of accumulating counts (slide p.43 explicitly forbids this).
- Computing Dice as $2\cdot IoU/(1+IoU)$ and then averaging — identical only for a single class, not for mIoU.
- Returning `0` for a class with no ground truth instead of excluding it — that silently drags mIoU down.

#### D3 — Panoptic Quality

**Key idea tested.** Thresholding matches first, then averaging **only the surviving IoUs** — not the raw list.

```python
def panoptic_quality(ious, n_fp, n_fn, match_threshold=0.5):
    tp = [v for v in ious if v > match_threshold]
    n_tp = len(tp)
    if n_tp == 0:
        return {"SQ": 0.0, "RQ": 0.0, "PQ": 0.0}
    sq = round(sum(tp) / n_tp, 6)
    rq = round(n_tp / (n_tp + 0.5 * n_fp + 0.5 * n_fn), 6)
    return {"SQ": sq, "RQ": rq, "PQ": round(sq * rq, 6)}
```

**Verified execution output:**

```
[0.8, 0.6], fp=1, fn=1  ->  {"SQ":0.7,"RQ":0.666667,"PQ":0.466667}
[]                      ->  {"SQ":0.0,"RQ":0.0,"PQ":0.0}
[0.2], fp=0, fn=1       ->  {"SQ":0.0,"RQ":0.0,"PQ":0.0}
[0.9,0.9], fp=2, fn=0   ->  {"SQ":0.9,"RQ":0.666667,"PQ":0.6}
```

All reproduce slide p.45's worked example exactly.

**Why the filter must come first.** `SQ` is *"the mean IoU of **matched** pairs"* — a 0.2-IoU pair is not a match, so it must be excluded from the average entirely. Averaging the raw list first would fold in IoUs of pairs that were never counted as detections, inflating SQ while RQ stays low.

**The `0.5×` factors in RQ.** They are standard F1 half-weights, and they are why RQ behaves like a harmonic mean rather than a plain ratio. With $n_{TP}=2$, $FP=1$, $FN=1$: $2/(2+0.5+0.5) = 0.667$, not $2/3 = 0.667$… which coincidentally agree here — a good reminder to compute it rather than simplify it, because they diverge as soon as FP ≠ FN. *(Check: $n_{TP}=2, FP=2, FN=0$ gives $2/(2+1+0) = 0.667$, matching the fourth verified case.)*

**The all-missed case** (`[0.2]`, no TP) returns all zeros rather than dividing by zero — and note that the single missed GT is *also* not counted in `n_fn` if you passed it in as 1; the caller must supply consistent counts.

**Common wrong approaches.**
- Using `>=` instead of `>` at the threshold — a pair at exactly 0.5 is not a match (slide p.45: *"Match if IoU > 0.5"*).
- Averaging SQ and RQ instead of multiplying them.
- Computing PQ per class and then taking an unweighted mean over classes, when classes have wildly different segment counts — slide p.45 specifies per class, averaged, but *separately for things and stuff*.

### Section E — Solutions

**Model answer covering the chain.**

1. **Task.** **Semantic segmentation.** Both questions — unstable-rock *area* and crack *depth* — are properties of a region, not of a countable object. Rock patches and cracks are closer to **stuff** (amorphous, defined by texture) than to discrete **things**, so instance IDs add nothing: *"How many cracks?"* is less meaningful than *"where is the unstable area?"* The existing rock **detector** is not sufficient — a box cannot yield area or depth. Note panoptic would only earn its extra annotation cost if the site also needed to count individual boulders; the question doesn't ask that, and slide p.5's rule is *"choose the cheapest rung that answers your product question."*
2. **Data.** A 4000×3000 image is **12 million pixels**; a dense mask annotation takes ~1.5 h (slide p.18) and at 2 images/day the team covers 24 M pixels/day — a fraction of one flight. Adopt slide p.18's remedies, in order: (i) **weak labels** — accept boxes or scribbles now, since area can be approximated from boxes and later refined; (ii) **model-in-the-loop annotation** — the existing detector proposes crack regions, a human corrects, which is *"the default tool for annotation today"* (slide p.10); (iii) **semi-supervised** training on the ~99% of video/flight frames with no labels. Slide p.18's own figure makes the trade explicit: ~1.5 h per image versus ~79 s per COCO object mask.
3. **Model.** A U-Net-style encoder–decoder. Encoder: a ResNet/MiT backbone with strides 4→8→16→32; at stride 32 each 3×3 conv sees ~96 px of context. Decoder: transposed convs with `k=4, s=2, p=1` (the clean ×2 case), concatenating stride-16 and stride-8 features. **Output stride 8**, not 32, because at OS 32 a 4000×3000 image becomes 125×94 — each crack of 2–3 px width is *sub-pixel* and cannot be represented, which is precisely E-c's failure. OS 8 gives 500×375, where a 2 px crack is 2 feature cells. ASPP-style dilation (rates 6/12/18/24) adds context without more downsampling.
4. **Metric.** Report **per-class IoU and mIoU over dataset-level counts** (slide p.43), **plus Dice** for the crack class specifically since it is the thin, low-area target. Do **not** report pixel accuracy as a headline — the mine wall is mostly one rock class, so accuracy will read ~97% regardless of crack quality. Do not use the detector's own AP as if it were segmentation quality: mask AP requires IoU on **masks**, not boxes. Slide p.43 is explicit that instance segmentation uses mask AP and panoptic uses PQ; for semantic, mIoU is the standard.
5. **Failure mode.** **Zoom/scale variation between flights.** The receptive field is fixed in *pixels*, so a crack that is 3 px wide at low zoom becomes 30 px wide at high zoom — the model sees an entirely different signal, and the thin-target training that made it work at one zoom fails at another. Worse, a scale-invariant descriptor is not what's wanted here: the measurement is *metres of depth*, and a scale-normalised model discards exactly the cue that converts pixels to millimetres.

**Sub-questions.**

**E-a — 97% accuracy with mIoU 0.61.**
Not a contradiction: they're measuring different things. Pixel accuracy is dominated by the rock class, which the model gets right; mIoU averages **per-class** IoU, so a class the model never predicts contributes **0** regardless of how much area it occupies (slide p.44's worked example: adding one never-predicted class drops mIoU 0.775 → 0.45 while accuracy barely moves). **The implication:** one class is failing completely or near-completely while the rest are fine — and because mIoU averages classes rather than pixels, one such class costs a *fixed* $1/K$ of the score regardless of how few pixels it has. Identify it by reading the **per-class IoU** table (slide p.43: *"a good mIoU can hide a class that is always wrong"*), not by guessing from the average. Given slide p.41's imbalance discussion, the first fix to try is Dice loss or focal loss, since a model predicting all rock never produces crack pixels at all.

**E-b — checkerboard from `k=3, s=2, p=0`.**
**Diagnosis:** $3 \bmod 2 \neq 0$, so the kernel stamps do not tile uniformly — slide p.37's cheat-sheet flags exactly this (*"Checkerboard if $k \bmod s \neq 0$"*), and slide p.35's row 4 shows $k{=}3, s{=}2$ needs `output_padding=1` to reach ×2 *while remaining unevenly overlapped*.
- **Fix 1:** use `kernel_size=4, stride=2, padding=1` — the clean ×2 recipe (slide p.35 row 4, and the exact config in slide p.36's code block).
- **Fix 2:** replace the transposed conv with **resize + 3×3 conv**, which slide p.36 says has *"No uneven overlap by construction."* It costs more compute at the high resolution, which is the stated trade-off.

**E-c — resolution mismatch.**
$4000/512 \approx 7.8$, so the effective output stride implied by training at 512×512 on 4000×3000 masks is roughly **8** (ground truth downsampled by ~8). **Consequence for thin cracks:** a crack occupying 2 px in the original mask becomes 0.25 px after ÷8 — below one feature cell. Downsampling therefore doesn't just blur thin cracks, it **erases** them from the label, so the model is trained on data where the target class is partly invisible, and no amount of loss tuning recovers it. Two mitigations: train on random crops at native resolution instead of whole downsampled frames, or supervise with a deep-supervision head at full resolution rather than only at stride 8.

### Section F — Solutions

**F1.** *Formula:* $o = (i{-}1)s - 2p + k + \text{out\_pad}$.

| Case | Substitute | $o$ |
|---|---|---|
| $(16,4,2,1,0)$ | $(16{-}1)(2) - 2(1) + 4 + 0 = 30 - 2 + 4$ | **32** |
| $(8,3,1,0,0)$ | $(8{-}1)(1) - 0 + 3 + 0 = 7 + 3$ | **10** |
| $(4,3,2,1,1)$ | $(4{-}1)(2) - 2 + 3 + 1 = 6 - 2 + 4$ | **8** |

All three verified.

*Interpret:* the first is the **clean ×2** case because $k = 4$ is divisible by $s = 2$, so the kernel stamps tile uniformly with no overlap pattern — slide p.35 row 4, and precisely the configuration in slide p.36's code comment (`16×16 → 32×32`). The second shows the trap: **stride 1 still grows the output**, by $k{-}1 = 2$, giving 10 not 8 — the $(i{-}1)$, not $i$, in the formula. The third needs `output_padding=1` to reach exactly ×2, and slide p.35 flags it as *"×2, but uneven overlap"* — correct size, artefact-prone.

**F2.** *Formula:* $r_{out} = r_{in} + (k{-}1)d$, starting from $r_0 = 1$.

| Layer | $d$ | $k$ | $r$ |
|---|---|---|---|
| 1 | 1 | 3 | $1 + 2(1) = \mathbf{3}$ |
| 2 | 2 | 3 | $3 + 2(2) = \mathbf{7}$ |
| 3 | 4 | 3 | $7 + 2(4) = \mathbf{15}$ |

**Receptive field = 15.** Verified.

*Effective kernel, single layer:* $(k{-}1)d + 1 = 2(6) + 1 = \mathbf{13}$. Verified.

*Interpret:* each dilation is chosen so the layer *doubles* the receptive field — 3 → 7 → 15 — for exactly **9 weights**, the same as three plain 3×3 convolutions (slide p.37 lists transposed/conv parameters as $k^2C_{in}C_{out}$, and dilation costs nothing extra). This is how ASPP (slide p.54) and DeepLab get multi-scale context without extra layers, and it answers slide p.24's *"to see 1,000 px you would need ~500 layers"*.

**F3.** *Formula:* $\text{feature size} = \lceil H/\text{OS} \rceil$.

| OS | Height | Width |
|---|---|---|
| 32 | $512/32 = \mathbf{16}$ | $1024/32 = \mathbf{32}$ |
| 8 | $512/8 = \mathbf{64}$ | $1024/8 = \mathbf{128}$ |

Verified. **Pixel-count ratio:**
$$\frac{64 \times 128}{16 \times 32} = \frac{8192}{512} = \mathbf{16\times}$$

*Interpret:* dropping from OS 32 to OS 8 gives **16× more feature pixels** — 4× per axis. That is the whole trade slide p.25 names: output stride decides how much resolution the decoder must *recover*. OS 32 is the sweet spot for classification, but slide p.24's memory warning and slide p.58's SETR critique (*"Features are single-scale at 1/16 — no fine detail from early stages"*) both point the same way: for segmentation you need detail back, and every factor of 4 in stride is 4× fewer detail cells.

**F4.** *Formula:* $\text{IoU}_k = \dfrac{\cap_k}{|GT_k| + |Pred_k| - \cap_k}$.

**Unions.** Class 0: $9 + 9 - 8 = \mathbf{10}$. Class 1: $7 + 7 - 6 = \mathbf{8}$. Verified.

**IoUs.** Class 0: $8/10 = \mathbf{0.80}$. Class 1: $6/8 = \mathbf{0.75}$. Verified — these are slide p.44's own numbers.

$$\text{mIoU} = \frac{0.80+0.75}{2} = \mathbf{0.775}, \qquad \text{pixel acc} = \frac{8+6}{16} = \mathbf{0.875}$$
$$\text{Dice}_1 = \frac{2(6)}{2(6) + 1 + 1} = \frac{12}{14} = \mathbf{0.857}$$

All three verified against the slide.

*The denominator trap.* Pixel accuracy is $14/16$, **not** $14/18$. Summing $TP+FP+FN$ over both classes gives $8+1+1+6+1+1 = 18$ because **each misclassified pixel is counted twice** — once as an FP for the class it was wrongly given, once as an FN for the class it truly is. The denominator is the actual pixel count, $\sum(TP+FN) = 16$.

**The misleading number is pixel accuracy.** At 0.875 it looks respectable while mIoU is 0.775 — but consider what happens if a third class appears once in ground truth and is never predicted: mIoU drops to $(0.80+0.75+0)/3 = \mathbf{0.517}$ while pixel accuracy is essentially unchanged, because one pixel out of 17 barely moves the ratio. *(Verified: my `seg_metrics({0:(90,0,10), 1:(0,0,5)})` case gives mIoU 0.45 against pixel accuracy 0.857.)* That asymmetry is why slide p.43 calls pixel accuracy *"misleading under imbalance"* and makes mIoU the standard.

**F5.** *Formula:* $PQ = SQ\times RQ$, $SQ = \frac{1}{|TP|}\sum IoU$, $RQ = \frac{|TP|}{|TP|+\frac12|FP|+\frac12|FN|}$.

*Substitute:* both matches clear 0.5, so $|TP| = 2$, $FP = 1$, $FN = 1$.
$$SQ = \frac{0.8+0.6}{2} = \mathbf{0.70}, \qquad RQ = \frac{2}{2+0.5+0.5} = \frac{2}{3} = \mathbf{0.667}, \qquad PQ = 0.70 \times 0.667 = \mathbf{0.467}$$

Verified — this is slide p.45's example exactly.

*Raised threshold so only 0.8 matches:* $|TP| = 1$.
$$SQ = \frac{0.8}{1} = \mathbf{0.80}, \qquad RQ = \frac{1}{1+0.5+0.5} = \mathbf{0.50}, \qquad PQ = 0.80 \times 0.50 = \mathbf{0.400}$$

*Interpret:* PQ fell 0.467 → 0.400 even though **SQ rose** 0.70 → 0.80. That is the entire reason slide p.45 separates the two factors: tightening the match threshold removes a low-quality match, which *raises* the average quality of what remains while *reducing* how many segments were found. The product correctly penalises the loss of a detection, and a system that reports only SQ would look **better** after making itself worse. (Note the $RQ$ denominator must then count the demoted match as a miss — here as the existing $FN$ — or $RQ$ would be miscomputed.)

**F6.** *Formula:* $\text{eff. kernel} = (k{-}1)d + 1$; $o = \left\lfloor \dfrac{H + 2p - d(k{-}1) - 1}{s} \right\rfloor + 1$.

*Effective kernel:* $(3{-}1)\cdot6 + 1 = \mathbf{13}$. Verified.

*Output with $p = 6$:*
$$\left\lfloor \frac{64 + 2(6) - 6(2) - 1}{1} \right\rfloor + 1 = \left\lfloor \frac{64 + 12 - 12 - 1}{1} \right\rfloor + 1 = 63 + 1 = \mathbf{64}$$

**Verified: 64×64, size preserved.** Because $p = \frac{d(k{-}1)}{2} = 6$, the padding exactly compensates the widened kernel.

*Output with $p = 1$:*
$$\left\lfloor \frac{64 + 2(1) - 12 - 1}{1} \right\rfloor + 1 = \left\lfloor 53 \right\rfloor + 1 = \mathbf{54}$$

*Interpret:* the map **shrinks** 64 → 54 instead of holding at 64. This is the bug slide p.38's question 5 is built to catch, and the rule is $p = \frac{d(k{-}1)}{2}$, not $p = d$. The consequence is not merely a smaller tensor: every subsequent layer's geometry is now different from the design, **U-Net skip connections no longer align** (they assume matching spatial sizes at each level), and at the end of the encoder a stack of such layers could collapse the map below what the decoder expects. In a real network that usually surfaces as a shape-mismatch error at the skip concatenation — a loud failure — but if the error is absorbed by a resize it becomes a silent misalignment that quietly destroys boundary accuracy.

**F7.** *Formula:* $\text{eff. kernel} = (k{-}1)d + 1$ in feature pixels; image reach $= \text{eff} \times \text{OS}$.

*Substitute:* $k=3$, so eff $= 2d+1$; $\text{OS} = 16$.

| rate $d$ | eff. kernel (feature px) | image reach |
|---|---|---|
| 6 | $2(6)+1 = \mathbf{13}$ | $13\times16 = \mathbf{208}$ px |
| 12 | $2(12)+1 = \mathbf{25}$ | $400$ px |
| 18 | $2(18)+1 = \mathbf{37}$ | $592$ px |
| 24 | $2(24)+1 = \mathbf{49}$ | $49\times16 = \mathbf{784}$ px |

All four verified.

*Size range covered:* largest ÷ smallest $= \frac{784}{208} = \mathbf{3.77\times}$, equivalently $49/13 = 3.77$. In original-image terms one layer sees objects from **~208 px to ~784 px** across.

*Parameter cost:* every branch is a **3×3** convolution, so each has $9C_{in}C_{out}$ weights — **identical** to a single undilated 3×3. Dilation inserts gaps between the taps and adds **no parameters at all** (slide p.37 lists conv parameters as $k^2C_{in}C_{out}$, unchanged by $d$).

*Interpret:* ASPP buys a **3.77× range of object scales for roughly 4× a single convolution's cost**, with zero extra parameters per branch beyond the fact of running four of them. That is exactly the answer to slide p.24's *"to see 1,000 px you would need ~500 layers of 3×3"* — you do not need depth, you need **parallel scales**. The global image-pooling branch in the same block then adds context beyond any of these, which is why slide p.56 credits PSPNet's 1×1 bin with fixing *"boat vs car, pillow vs bed-sheet"* errors no local scale can.

**F8.** *This is a reasoning question; the "numbers" are the consequences.*

**What each keeps.** **SegNet** stores **max-pool indices** — a single integer per pooled position — and reconstructs the decoder input by scattering each value back to the position that produced the maximum (slide p.49: *"Memory-light: stores indices, not feature maps"*). **U-Net** stores the **entire encoder feature tensor** at every level and concatenates it onto the matching decoder tensor (slide p.50).

*Consequences.*

- **SegNet's decoder parameter count is close to zero.** Because max unpooling and the indexing are fixed, non-learnable operations, the decoder needs almost no parameters — only the trainable convolutions that densify the sparse result. Memory during training is also far lower, since indices cost $O(N)$ integers rather than $O(N C)$ floats. Slide p.49 calls it *"memory-light"* and targets *"real-time"* use.
- **U-Net pays in memory and parameters** for the concatenation: each skip doubles the channel count at that level (slide p.50's 64→128→256→512→1024 progression), and the tensors must all be held for backprop.

**Which for 3-pixel-wide vessels.** **U-Net.** Max unpooling keeps only the **maximum** response per pooled position, so a 3-pixel-wide vessel is represented by **one** surviving value per 2×2 window — it is structurally incapable of preserving the vessel's width, and the spatial evidence for its two boundaries is discarded before the decoder can ever use it. Concatenating the full-resolution encoder features preserves every pixel, and the following convolutions learn how to fuse appearance with precise location. That is the mechanism behind slide p.51's claim that skip concatenation *"gives the decoder full-resolution detail at every level"* — the reason U-Net succeeded with ~30 training images and why nnU-Net is now the medical-imaging baseline.

*The trade stated honestly:* U-Net would be the wrong choice if memory or latency were the binding constraint, in which case SegNet's index-based decoder is the right tool and you would accept coarser thin structures. The question is not which is "better" but which resource you can afford to spend.

---

## J. Sources

### Course sources

| Source | File | Detail |
|---|---|---|
| Presentation | `Session_10.pdf` | 59 slides, *"Image Segmentation"*, Shubham Jain; text layer, extracted in full |
| Transcript | `Session_10_—_Image Segmentation_….txt` | **Empty** — 59 words of *"Thank you"*; recording failed |
| **Professor repository** | **None in this workspace** | — |

### Key slides

| Slides | Topic |
|---|---|
| p.3 | Boxes → pixels; the 45° thin-object example |
| p.5–11 | The four tasks; semantic / instance / panoptic; stuff vs things; side-by-side table |
| p.13–18 | Applications; datasets; **annotation cost (1.5 h / 79 s / 1.1 B)** |
| p.20–26 | Problem formulation; two naive failures; **encoder–decoder + output stride** |
| p.27 | Nearest / bilinear with the worked 2×2→3×3 grid; `align_corners` gotcha |
| p.34–35 | Transposed convolution mechanics; **the output-size table** |
| p.36–37 | Resize+conv; **operator cheat-sheet** |
| p.38 | **The five-question exercise** |
| p.40–42 | Pixel CE; weighted CE and focal; **Dice and the "all background" example** |
| p.43–44 | Metrics; **the 4×4 worked example** |
| p.45 | **Panoptic Quality** |
| p.47–58 | FCN, SegNet, U-Net, DeepLab v1/v2/v3/v3+, ASPP, PSPNet, Mask R-CNN, SETR |

### Supplementary

- Long et al., *"Fully Convolutional Networks for Semantic Segmentation"* (CVPR 2015). https://arxiv.org/abs/1411.4038
- Ronneberger et al., *"U-Net"* (MICCAI 2015) — the architecture to remember. https://arxiv.org/abs/1505.04597
- Chen et al., *"Encoder–Decoder with Atrous Separable Convolution for Semantic Image Segmentation"* (DeepLabv3+, ECCV 2018). https://arxiv.org/abs/1802.02611
- Zhao et al., *"Parsing Whole Scene Segmentation"* (PSPNet, CVPR 2017). https://arxiv.org/abs/1612.01105
- He et al., *"Mask R-CNN"* (ICCV 2017). https://arxiv.org/abs/1703.06870
- Kirillov et al., *"Panoptic Segmentation"* (CVPR 2019) — PQ. https://arxiv.org/abs/1801.00890
- COCO panoptic metric definition. https://cocodataset.org/#panoptic-eval *(the slide's linked family)*

### Unresolved

1. **Slide p.50 gives U-Net as "572×572 in → 388×388 out" but prints no intermediate sizes.** The three rules (valid 3×3 conv shrinks by 2; 2×2 max-pool stride 2 halves; 2×2 up-conv doubles then **crops to the skip size** — the slide's *"copy & crop"*) do not by themselves pin the chain down, and the crop step is not specified numerically. I attempted a reconstruction and could not reproduce 388, so **no intermediate chain is asserted here**. The endpoints are slide facts; the arithmetic tested in F1–F3 is the part that is verifiable.
2. **Slide p.42's Dice example has no numeric figure** — only *"Lesion < 1% of pixels: predicting 'all background' gets > 99% accuracy but Dice = 0"*. Exact percentages are illustrative, so F/A4 use their own stated figures rather than the slide's.
3. **No audio means no verbal emphasis** — for the first time in this course I cannot cross-check a slide against what the instructor actually stressed. Treat the slide ordering as a proxy for emphasis, not a guarantee.

---

*Questions A–F are original, written against the Session 10 slides. Every numerical answer and all three code solutions were executed and verified; three errors were found and fixed — a wrong transposed-conv row in my notes, a pixel-accuracy denominator that double-counted misclassified pixels, and an unverifiable U-Net derivation I removed rather than fabricate.*