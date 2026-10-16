# Class 01 — Course Structure, Image Processing & Features

> **Exam weight: Tier 1 (foundational).** Sections A, B and C draw heavily from this class; Section F from it directly. It is the prerequisite for Class 02 (SIFT/ORB → RANSAC → homography) and supplies the operator intuition that Class 03 (CNNs) assumes.
>
> **Primary sources:** `Session_1_—_Course Structure & Image Processing & Features_…txt` (transcript, 15.4k words) · `Session1__1_.pdf` (66 slides, titled *"From Pixels to Visual Features — Image Processing • Edge Detection • Feature Extraction"*).
> **Professor repository / notebooks: none exist in this workspace.**
> **Known gaps:** no code from the instructor; several slide equations had to be read by rendering the page (p.16–18, 26, 47, 50).

**Course objective, verbatim from the deck (p.4):** *"Understand a CV problem → Choose approach → Implement → Evaluate → Know limitations."*

**Grading — confirmed by the student as authoritative:**
Quiz 25% · End Term Exam 35% · Projects/Kaggle (best 2 of 3) 40%, where
`Project Score = Test-case-based evaluation score × [Project understanding section in end-term exam]`.
*(The Session 1 deck, p.5, still shows an earlier split of 20/20/25/35 with "Best 3 of 4" projects. Treat the confirmed split as current.)*

---

## A. Lecture overview

### Learning objectives

By the end of this class you should be able to:

1. Represent an image as a discrete function and as arrays, and explain what a *channel* is.
2. Classify an image operation as **point**, **local**, or **global**, and state its per-pixel cost.
3. Compute the output of a linear filter by cross-correlation, and explain how **convolution** differs.
4. Use the algebraic properties of convolution to justify layer-merging in a deployed CNN.
5. Explain sub-sampling, upsampling, and **aliasing**, and name the Nyquist criterion.
6. Derive edge detection from the first derivative of intensity, compute gradient magnitude and orientation.
7. State the three criteria of a good edge detector and implement NMS + hysteresis (Canny's last two steps).
8. Explain why corners beat edges for tracking, and how Harris scores them via autocorrelation eigenvalues.
9. Describe all four SIFT stages and say which invariance each one buys.

### Why this matters in CV

Everything downstream is a refinement of this. A convolution layer *is* a learned linear filter applied as a local operation with weight sharing — that is the entire bridge from this class to Class 03. Edge and corner detection is the classical ancestor of the learned feature hierarchy (Zeiler & Fergus, cited on slide p.16 of Class 03). And SIFT, only lightly covered here, is the direct subject of Kaggle Assignment 1.

### Prerequisites

Basic linear algebra (vectors, dot product), basic probability, and the ability to read a 2-D array. The deck assumes prior exposure to regression, classification and loss functions (slide p.3).

### Position in the course

```
Class 01  Image processing & features  ← YOU ARE HERE
   ├── Class 02  Feature matching & classical CV  (SIFT/ORB → RANSAC → homography → stitching)
   ├── Class 03  CNNs  (learned filters, backprop, feature maps)
   ├── Class 05  Embeddings & image similarity (learned instead of hand-designed descriptors)
   └── Class 09  Video understanding (temporal local operations)
```

### Difficulty: **Foundation–Intermediate**

Justified because: the mathematics is elementary (weighted sums, finite differences, one 3×3 determinant), but there are *four distinct sub-fields* in one 120-minute session and the instructor explicitly flags that it is a lot for a first session: *"I think these are little complex topics to cover up in the first session itself when you're coming from holidays. But I want to build a base first"* (transcript).

---

## B. Conceptual foundations

### B1. The image as a function, then as an array

**The problem it solves.** Before you can process an image you need a precise object to write operations on. "Picture" is not an object; a function and a matrix are.

**Intuition.** The instructor's framing: *"a computer understands in numbers… these are just numbers defining the intensity"* (transcript). Zoom into any image and the smooth gradients dissolve into a grid of integers.

**Mechanism, step by step.**
1. Model the scene as a continuous intensity function.
2. *Sample* it — choose where to measure (this is where resolution is chosen).
3. *Quantize* — map each sample into a finite integer range.

**Formal statement (slides p.6–p.7).** A greyscale image is a function

$$f : \mathbb{R}^2 \to \mathbb{R}$$

giving intensity at position $(x, y)$. A digital image is a **discrete, quantised** version of that function — the samples and the finite range are the two departures from the continuous model.

**Symbols.** $f$ intensity function · $(x,y)$ spatial coordinate · $I[i,j]$ the digital sample at row $i$, column $j$ · $\mathbb{R}$ the real numbers.

**Quantisation.** One byte per value, so intensities run **0 → 255**, with 0 = black and 255 = white (slide p.6, transcript). The choice of 256 levels is what lets you store the full range in a single byte.

**Channels.** A *channel* is one such matrix. Greyscale has one channel; RGB has three — one per primary. The instructor makes the point that RGB is a **human** convention, not a computer requirement:

> *"when you think of computer, do they need to visualize it like us? No, right? … you sometimes [feed] YUV directly into the ADAS systems because these YUV images — I can bring together three channels and then these algorithms can directly see whole three channels in the same box. And that's where the optimization comes into play, right? I don't want to process these three channels separately."* (transcript)

**Worked example.** A 4×4 greyscale image is a 4×4 matrix of uint8. The same picture in RGB is three 4×4 matrices stacked, so in NumPy it is a shape-`(4, 4, 3)` array; a batch of 32 such images is `(32, 4, 4, 3)`.

**Assumptions and limitations.** None of this says anything about *semantics*. Two images can have wildly different content and identical statistics — the distinction the instructor drove home on slide p.8 of Class 02: *"PIXEL SIMILARITY ≠ SEMANTIC SIMILARITY."*

**Likely misconception.** "RGB is how images work." RGB is a human-facing display encoding. Production vision pipelines frequently use YUV (video) or raw/Bayer because the channel layout suits downstream processing, not because of perception.

---

### B2. Point, local, and global operations

**The problem it solves.** Not every pixel computation has the same cost, and knowing which kind you are doing tells you the complexity *before* you write the code.

**Intuition.** Slide p.14 splits operations by **how far the output at a pixel looks**.

**Formal statement (slide p.14).**

| Kind | Output at $(m_0, n_0)$ depends on | Complexity / pixel |
|---|---|---|
| **Point** | only the input at the same coordinate | constant |
| **Local** | a $p \times p$ neighbourhood | $p^2$ |
| **Global** | all values of the $N \times N$ image | $N^2$ |

**Worked examples.**

*Point* — the contrast example from the transcript. An image whose intensities span only 50 → 200 is stretched:

$$I' = \frac{I - I_{\min}}{I_{\max} - I_{\min}} \times 255$$

Verified: `[50, 125, 200] → [0, 128, 255]`. The instructor's justification for why this matters: *"not every image will use every possible range… if your image is not using the complete range, you can make it use the complete range so that you have a better spread across."*

The same script names the *degenerate* case honestly: *"if I click a photo of a blank whiteboard every pixel is 255, and even if I do the normalization it will not be very helpful."*

*Local* — the moving average (slides p.10–p.13, worked in full in §C1 below).

*Global* — the transcript makes the forward link to Class 03 explicit: *"when you move to CNN you will see that we have things like global pooling layers… you take average… to give a very nice representation of the image of the whole image in a single number."* He adds the practical motivation: by the final layers a network has 128+ channels, so reading all of them per pixel is expensive, hence *global average pooling*.

**When it fails / when to use.** Point operations are O(1) and always safe to vectorise. Global operations are the only ones whose cost grows with image size — that is precisely why they are rare, and why global pooling is normally applied *once*, at the end.

**Likely misconception.** Conflating "global" with "expensive in memory". The distinction on the slide is about *dependence*, not memory.

---

### B3. Linear filters and cross-correlation

**The problem it solves.** You want a single, general mechanism that expresses "average my neighbourhood", "find my vertical edges", and — later — "learn any feature". All three are the same operation with a different kernel.

**Intuition.** *"What values belong in the kernel H for the moving average example?"* is the instructor's framing (slide p.19). The kernel **is** the question; the operation is the machinery.

**Formal statement (slides p.16–p.18).** Given a kernel of size $(2k+1)\times(2k+1)$:

*Uniform weights (the moving-average / box filter):*

$$G(i,j) = \frac{1}{(2k+1)^2}\sum_{u=-k}^{k}\sum_{v=-k}^{k} I(i+u,\; j+v)$$

*Non-uniform weights (general linear filter):*

$$G(i,j) = \sum_{u=-k}^{k}\sum_{v=-k}^{k} H(u,v)\, I(i+u,\; j+v)$$

**Every symbol.** $G$ output image · $I$ input image · $H$ the kernel (a.k.a. mask, filter; its entries are the **filter co-efficients**) · $k$ half-width, so kernel side is $2k+1$ · $(i,j)$ the current output pixel · $(u,v)$ an offset into the kernel.

The slides summarise it as: *"Can be viewed as 'dot product' between local neighbourhood and kernel for each pixel"* (p.18).

**Why $(2k+1)$ and not $2k$.** The instructor asks this live and accepts "11100111" as a valid edge kernel, then justifies odd sizes: *"we want to keep it odd because, when you have a two-by-two filter, sometimes what happens is the averaging and these multiplications become complex… my output shape becomes unpredictable. It can be odd or even… because there will always be a confusion, should I include the last layer, should I not."* Odd kernels give a single unambiguous centre pixel, so output geometry is always computable. He concedes it is a convention, not a requirement: *"it doesn't necessarily always need to be square matrix and it doesn't always need to be odd formulation, but it helps."*

**Worked example.** Full arithmetic in §C1 and §C2.

**Output size.** With no padding, a valid output is

$$N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} - F}{S} \right\rfloor + 1$$

With zero padding of $P$ pixels per side:

$$N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} + 2P - F}{S} \right\rfloor + 1$$

where $F$ = kernel size, $S$ = stride. This is the same formula Class 03 reuses for CNN layers; it is introduced here because it falls directly out of counting window placements.

**Assumptions and limitations.** Assumes the neighbourhood is square and centred. Non-square kernels and asymmetric padding exist but break the tidy bookkeeping above.

---

### B4. Convolution, and why the flip matters

**The problem it solves.** Correlation and convolution are frequently used interchangeably in software libraries but they are not the same function. The exam will test the difference.

**Formal statement (slide p.26).**

$$G(i,j) = \sum_{u=-k}^{k}\sum_{v=-k}^{k} H(u,v)\, I(i-u,\; j-v)$$

Note the **minus** signs where correlation had plus signs. Slide p.26 states the rule: *"Equivalent to flip the filter in both directions (bottom to top, right to left) and apply cross-correlation. Denoted by G = H * I."*

**When does the difference disappear?** Whenever $H$ is symmetric under 180° rotation. A box filter, a Gaussian kernel, and both Sobel kernels are all symmetric, so correlation and convolution give identical results for them. The difference only shows with an **asymmetric** kernel — which is exactly the case in §C2.

**Algebraic properties (slide p.27).**

| Property | Statement |
|---|---|
| Commutative | $a * b = b * a$ |
| Associative | $a * (b * c) = (a * b) * c$ |
| Distributive over addition | $a * (b + c) = (a * b) + (a * c)$ |
| Scalars factor out | $ka * b = a * kb = k(a * b)$ |
| Identity | Unit impulse $e = [\dots,0,0,1,0,0,\dots] \Rightarrow a * e = a$ |

The instructor ties associativity directly to deployment, which is the practically important consequence:

> *"the point of deep learning is to enable problem solving with little data… when we are putting some algorithm onto the car… we actually combine lots of convolutional layers in a single layer… so let's say I had this filter one cross one is one, and then I had another filter… I can just write it as one and minus two and this will essentially have the same effect and I don't have to do the multiplication many times."*

Combined filters are also used to compress networks for latency-critical systems: *"in the case of autonomous driving you need to take in a fraction of a second, you need to make decisions in milliseconds… and that's where we need to compress the whole algorithm."*

**A subtlety the instructor flags about associativity.** It holds because convolution is **element-wise** multiplication-and-sum, not matrix multiplication: *"it's just element-wise multiplication… in matrix multiplication you need to have like this, and row from here get multiplied with the columns, and all these things can bring changes and the final values can change based on the order. But here it's just element-wise… that's why it is associative as well."*

---

### B5. Practical issues: kernel size, padding, and the information trade-off

**Bigger kernel ⇒ more neighbours ⇒ lower noise variance, more blur, more compute.** The transcript is explicit:

> *"the bigger the mask, the more neighbours are contributing… smaller noise variance of the output… if there was a noise in an image and if I applied a convolution, then you have suppressed the noise. **But does that noise actually need to be suppressed? … when we talk about noise, noise is nothing but a variance in the data. And when we suppress it, you're losing any information.** So do we want to lose that information?"*

This is the single most transferable idea in the class: **smoothing is a lossy operation whose justification is application-dependent.**

**Cost.** *"if you have bigger filter, you do blur it more, more expensive to compute… instead of 3×3 now if you use a 6×7 — sorry, 7×7 — you do more multiplication and that adds to the compute."*

**Padding — two conflicting goals.**
- *Preserve size*: otherwise the last rows and columns are lost. *"we use zero padding wrapping around."*
- *Preserve border information*: edge pixels are real signal.

But padding is not free. The instructor identifies the failure mode: *"the zero pixels at the edges that you use for padding then start contributing to your output… **it introduces noise** because these zeros were not at initial right and you introduce that and when you perform the convolution on top of that you introduce the noise."* This is the standard reason for "same"-padding modes that replicate border pixels instead of writing zeros.

---

### B6. Sampling, aliasing, and upsampling

**Sub-sampling.** Throw away every other row and column to halve each spatial dimension (slide p.29). Motivation given: *"you're losing the information here but for the computers it is again important to make it less computationally expensive."*

**Aliasing — the failure mode (slide p.31).** The instructor shows moiré artefacts and attributes them to the sampling grid itself:

> *"when you have a high-wavelength, high-variance kind of setup and you subsample it, what you're doing is you are essentially just creating another pattern out there… the samples are making another pattern into the image. This is a new kind of pattern introduced within the image because of your subsampling."*

**Definition (slide p.31).** Aliasing *"occurs when your sampling rate is not high enough to capture the amount of detail in your image. To do sampling right, need to understand the structure of your signal/image. The minimum sampling rate is called the **Nyquist rate**."*

**The everyday analogue the instructor uses.** A fan blade or a spoked wheel appearing to rotate *backwards*: *"these are all these kinds of effects where the frequencies are interacting with each other and you know they make you see something which is actually not there."* The temporal sampling rate of a camera at 120 Hz collecting near-identical frames is the same phenomenon in time.

**Upsampling.** Repeat each row and column — **nearest-neighbour interpolation** (slide p.32). Crucially, this creates no information: *"of course it doesn't give you extra information, you are just repeating the information that was already there."* Higher-order interpolation (bilinear, bicubic) guesses smoother intermediate values, but still invents nothing new.

**Mitigation.** Smooth (Gaussian) *before* down-sampling so the discarded high frequencies have already been attenuated. This is a good multi-select exam item because it follows from the aliasing definition rather than being memorised.

---

### B7. Edges as derivatives

**The problem it solves.** Pixels are a wasteful representation; edges are a compact one (slide p.33): *"Map image from 2D matrix of pixels to a set of curves or line segments or contours — more compact representation than pixels."*

**Why edges matter (slide p.34).** Group pixels into objects · track features (corners, curves, lines) · cue for 3-D shape · guide interactive image editing. The instructor's practical example is human labelling: *"if I wanted to identify, ask them to build a box of the building around the building, then the edges are the most helpful things."*

**Core idea (slide p.36).** *"An edge is a place of rapid change in the image intensity function."* Plot the intensity along a horizontal scanline: an edge is a **steep cliff**, so its **first derivative is large**, and conversely *"edges correspond to extrema of derivative."*

**Discrete derivative.** The images are sampled, so exact differentiation does not exist. We use the forward finite difference (transcript):

$$\frac{\partial I}{\partial x}[i,j] \approx I[i+1,j] - I[i,j]$$

This *is* the kernel `[-1, 0, +1]` applied by correlation. Similarly for $y$: `[-1, 0, +1]` transposed.

**Second derivative.** *"If you take the second derivative, it will look something like this"* — a zero-crossing detector. It responds to a *step* but is far noisier, which is why first-derivative detectors dominate.

**Gradient.** With the two directional responses $I_x$ and $I_y$ at a pixel:

$$\text{Magnitude} = \sqrt{I_x^2 + I_y^2}, \qquad \theta = \tan^{-1}\!\left(\frac{I_y}{I_x}\right)$$

The instructor builds this from the transcript's own worked reasoning: *"if this represents x, this represents y… to actually get the magnitude, what you will do is you will just take a square root of the sum."* He also notes both signs of response must be kept: *"when you have a negative value that also suggests there is a high variance suddenly, like when you are calculating a derivative but in the negative direction."*

---

### B8. Making edges usable: NMS, hysteresis, Canny

Raw gradients are unusable: thick double-sided edges, and noise amplified into false edges. Three criteria are given (slide p.41):

1. **Good detection** — find all real edges, ignoring noise and other artefacts.
2. **Good localisation** — detect edges as close as possible to true edges.
3. **Single response** — return one point only for each true edge point.

**Non-maximum suppression (slide p.42).** *"Check if pixel is local maximum along gradient direction."* From the transcript: *"given an edge, we check its direction, and in that particular direction, we check if any other neighbouring pixel has a bigger magnitude. If any other pixel has a bigger magnitude, we consider that as [not an] edge and ignore this one."* Note the slide mentions checking *interpolated* neighbours $p$ and $r$ along the gradient direction — the reason plain 8-neighbourhood NMS is only an approximation.

**Hysteresis thresholding (slide p.43).** *"Use a high threshold to start edge curves and a low threshold to continue them."*

| Gradient at pixel | Decision |
|---|---|
| `> High` | edge pixel |
| `< Low` | non-edge pixel |
| `≥ Low` and `≤ High` | edge pixel **iff** connected to an edge pixel, directly or through the in-between band |

**Canny (slide p.44),** attributed to Canny 1986 — the four steps are exactly the material above:

1. Filter the image with a **derivative of Gaussian**.
2. Find the **magnitude and orientation** of the gradient.
3. **Non-maximum suppression**.
4. **Linking and thresholding (hysteresis)** with two thresholds.

Step 1 being *derivative of Gaussian* rather than raw finite difference is what solves the noise problem the instructor raised live in the transcript: given a noisy step edge, *"how do you identify the edge? … If you apply a Gaussian filter or box filter, you smoothen out this noise… and then on top of that if you apply the differentiation you can get the edge. **So that's where combining the filters help sometimes.**"*

He also draws a sharp distinction students get wrong: *"normalizing is a different thing. In the normalization you take the minimum and maximum and bring the intensities together. But if this is itself noisy, applying normalization will not help because the values remain here, they are very close to zero and close to 255."* **Normalisation is a point operation; it cannot remove noise. Smoothing is a local operation; it can.**

---

### B9. Corners: flat, edge, corner, and Harris

**The intuition (slide p.45).** Slide a small window and watch how it changes:

| Region | Behaviour |
|---|---|
| **Flat** | no change in all directions |
| **Edge** | no change along the edge direction |
| **Corner** | significant change in *all* directions |

The transcript states the property that makes corners special for matching: *"if you identify a corner and change the perspective of your image, the corner remains differentiable. You always know — if this is a corner, if I change the perspective of this image, you again can identify the same corner."*

**Quantifying "significant change" (slides p.47–48).** Autocorrelation — the sum of squared differences between pixel intensities under small shifts:

$$E_{ac}(\delta u) = \sum_{x,y} w(p)\,\big[\,I(p + \delta u) - I(p)\,\big]^2$$

where $p = (x,y)$ is a position in the image and $w(p)$ is 1 inside the window, 0 outside (a Gaussian weighting is used in practice).

**Harris corner detector (slides p.46, p.50).**

1. Compute gradients at each point (the `[-1,0,+1]` pair).
2. For each window form $A = \begin{bmatrix} \sum I_x^2 & \sum I_xI_y \\ \sum I_xI_y & \sum I_y^2 \end{bmatrix}$.
3. Compute the cornerness score

$$R = \det(A) - k\cdot \mathrm{trace}(A)$$

4. Threshold, take local maxima, apply NMS.

**Why eigenvalues decide (from the transcript).** With $A$'s eigenvalues $\lambda_1, \lambda_2$, note $\mathrm{trace}(A)=\lambda_1+\lambda_2$ and $\det(A)=\lambda_1\lambda_2$, so $R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)$:

- **Corner** — both $\lambda$ large ⇒ both terms large ⇒ $R$ large. *"when it is a corner both lambda 1 and lambda 2 have to be very large, so that signifies that variance in both the directions is very high."*
- **Edge** — one $\lambda$ much larger than the other ⇒ product is not large enough to offset the trace. *"if lambda 1 is greater than lambda 2 … much, much greater, then basically that is an edge. Because in one direction you have very high variance, in one direction you don't have much variance."*
- **Flat** — both small ⇒ $R$ small. *"both lambda 1 and lambda 2 will be very small because there is not much variation in the image."*

The instructor connects the eigenvectors to PCA: *"eigenvectors are basically the direction which we need to rotate in to get the maximum variance."*

**Known limitation (slide p.51, and live in the transcript).** Harris is **not scale invariant**. *"The same problem which we discussed for the tomatoes — if you increase the kernel size it becomes difficult… if the scale is too less and the variance is always there, then we are going to detect everything as a corner."* This is precisely the gap SIFT fills.

---

### B10. SIFT — the four stages

**The promise (slide p.52).** *"Image content is transformed into local feature coordinates that are invariant to translation, rotation, scale, shear."*

SIFT is covered briefly here because it is expanded in Class 02; what matters for Class 01 is the *architecture of the solution*.

| Stage | What it does | Invariance bought | Source |
|---|---|---|---|
| **1. Scale-space extrema detection** | Detect interesting points using **DoG** (Difference of Gaussians) | scale | p.53–55 |
| **2. Keypoint localisation** | Refine location and scale; keep stable points (Hessian test) | — (precision) | p.53, 56–58 |
| **3. Orientation estimation** | Gradient histogram around the keypoint | rotation | p.53, 59–60 |
| **4. Keypoint descriptor** | 128-D gradient histogram, normalised | illumination / local shape | p.53, 62–64 |

**Stage 1 in the instructor's words.** *"You blur out the image at multiple Gaussian variance… you keep increasing that in the first octave and then you basically sample the image, and then again do the Gaussian operation on top of that… In the second octave what you do is you subsample."* DoG then subtracts adjacent blurred images: *"when the variance is spread across the major edges or the features, it remains as an output when we subtract these Gaussians — so you get the features out of the image at every scale."* Searching at multiple scales **is** the scale-invariance mechanism.

**Stage 2.** Because the Gaussians were hand-designed, true extrema are known only approximately. The fix is a **Taylor expansion**: *"if you know the S dot and the delta S, you can basically approximate any function with the Taylor."* Stability is then filtered with the Hessian — the transcript notes *"Hessians are a better way to do it, but basically it is similar to the Harris corner detector."*

**Stage 3 (slide p.60).** Build a histogram of gradient directions in a window around the keypoint at its selected scale.

- **36 bins** ⇒ 360° / 36 = **10° per bin**.
- Each bin weighted by **gradient magnitude** *and* by a **Gaussian with σ = 1.5 × the keypoint scale** (so more distant neighbourhoods contribute less).
- The peak gives the keypoint orientation.
- **Extra keypoints** are created if another peak is within **80%** of the maximum peak with a different direction — this handles the ambiguous-orientation case (slide p.16 of Class 02).

**Stage 4 (slide p.64).**

- 128 non-negative values (4×4 spatial cells × 8 orientation bins).
- Normalise to unit length, "to reduce the effects of contrast or gain".
- **Clip to 0.2**, then renormalise — "to further make the descriptor robust to other photometric variations".

**Stated robustness (slide p.65).** *"Changes in viewpoint: up to about 60 degrees out-of-plane rotation. Changes in illumination: sometimes even day vs night. Fast and efficient — can run in real-time."* The instructor adds why: *"there is no big network involved here, these are all classical ML algorithms which are not very compute intensive."*

**Named NMS in SIFT (slide p.61).** 729 keypoints reduced to 536 after the Hessian/ratio test — the same idea as Canny's NMS, one pipeline stage earlier.

**Homework assigned (slide p.66, transcript).** Image stitching — detect features in both images, find corresponding pairs. This becomes Kaggle Assignment 1.

---

## C. Mathematical derivations and numerical examples

Every number below was recomputed with a Python script; outputs shown are the actual computed values.

### C1. Moving average as a uniform linear filter

**Derivation.** Start from the local-operation definition: the output at $(i,j)$ depends on a $3\times3$ neighbourhood, so $k=1$. Slide p.19 asks for $H$ under the assumption of uniform weighting. For a uniform filter, all nine coefficients are equal to $c$, and the output must equal the local mean, so

$$9c \cdot \bar{I} = \bar{I} \;\Rightarrow\; c = \tfrac{1}{9}$$

**Therefore $H = \frac{1}{9}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$.**

**Why this is legitimate (scalars factor out, slide p.27).** The transcript confirms the instructor accepts either form: *"one divided by nine outside and then keep one here, you can actually visualize it like this, then you don't need to write this — both are fine."* So $\tfrac{1}{9}\odot[1,\dots,1]$ and $\tfrac{1}{9}, \tfrac{1}{9},\dots$ inside are the same filter.

**Worked example A — single spike.**

$$I = \begin{bmatrix}0&0&0\\0&90&0\\0&0&0\end{bmatrix}, \quad k=1$$

$G(0,0) = \tfrac{1}{9}(0\cdot9 + 90) = \tfrac{90}{9} = 10$.

Verified output: `[[10.0]]`.

**Worked example B — a row of three.**

$$I = \begin{bmatrix}0&0&0\\90&90&90\\0&0&0\end{bmatrix}$$

$G(0,0) = \tfrac{270}{9} = 30$. Verified: `[[30.0]]`.

**What it means.** A single isolated spike loses 89% of its amplitude; an extended bright region of the same peak survives much better. Averaging is a *low-pass filter* — and the reason a genuine surface highlight vanishes but a large bright region does not.

### C2. Cross-correlation vs convolution — a case where they differ

Correlation: $G_c(i,j)=\sum_{u,v} H(u,v)\,I(i+u,j+v)$.
Convolution: $G_v(i,j)=\sum_{u,v} H(u,v)\,I(i-u,j-v)$.

$$I=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix},\qquad H=\begin{bmatrix}1&2&0\\0&0&0\\0&0&-1\end{bmatrix}$$

**Cross-correlation** (all nine products, kernel aligned with the patch):

$$1(1)+2(2)+0(3)+0(4)+0(5)+0(6)+0(7)+0(8)+(-1)(9) = 1+4-9 = \mathbf{-4}$$

**Convolution** — flip $H$ by 180°, giving $\begin{bmatrix}-1&0&0\\0&0&0\\0&2&1\end{bmatrix}$:

$$(-1)(1)+0+0+0+0+0+0+2(8)+1(9) = -1+16+9 = \mathbf{24}$$

Verified outputs: `corr: [[-4]]`, `conv: [[24]]`.

**Interpretation.** The results are not merely different in sign — they are **-4 vs +24**, a reversal of which corner of the patch the kernel responds to. This is why "convolution is just correlation" is *almost* true: it is true only for centrosymmetric kernels. With a symmetric box or Sobel kernel the two agree exactly (verified: both give `[[45]]` on the same input).

### C3. Output size of a local operation

With no padding and stride 1, a window of size $F$ fits in $N$ at $N-F+1$ positions:

| $N$ | $F$ | $S$ | $P$ | $\lfloor (N+2P-F)/S \rfloor+1$ |
|---|---|---|---|---|
| 7 | 3 | 1 | 0 | **5** |
| 7 | 3 | 2 | 0 | **3** |
| 7 | 3 | 3 | 0 | **2** |
| 7 | 3 | 1 | 1 | **7** |
| 32 | 5 | 1 | 2 | **32** |
| 28 | 3 | 1 | 1 | **28** |
| 224 | 7 | 2 | 3 | **112** |

All seven rows verified. Rows 1–3 reproduce the instructor's own worked examples on slide p.31. Row 4 reproduces slide p.33: padding by 1 restores 7×7 from 7×7. Rows 5 and 7 are the standard "same" configurations — *"common to see CONV layers with stride 1, filters of size F×F, and zero-padding with (F−1)/2 (will preserve size spatially). F=3 ⇒ pad 1; F=5 ⇒ pad 2; F=7 ⇒ pad 3"* (slide p.34).

**Check the borderline case.** $N=7, F=3, S=3$: the last window would start at index 6 and end at index 8, past the edge. The formula gives $\lfloor 4/3\rfloor+1 = 2$, so only **2** full windows fit. Slide p.30 makes this point visually — a 3×3 filter at stride 3 on 7×7 *"doesn't fit!"*

### C4. Gradient magnitude and orientation

Given directional responses $I_x$ and $I_y$ at one pixel:

$$\text{Magnitude}=\sqrt{I_x^2+I_y^2},\qquad \theta=\tan^{-1}\!\left(\frac{I_y}{I_x}\right)$$

| $I_x$ | $I_y$ | Magnitude | $\theta$ |
|---|---|---|---|
| 3 | 4 | **5.0000** | **53.13°** |
| 4 | 3 | 5.0000 | 36.87° |
| 0 | 5 | 5.0000 | 90.00° |
| −3 | 4 | 5.0000 | 126.87° |

All verified. Note that rows 1 and 2 have **identical magnitude but different orientation** — the magnitude map alone cannot distinguish an edge sloping one way from one sloping the other. That is precisely why Canny stores orientation: NMS compares neighbours **along** the gradient direction, not along the image axes.

### C5. Gaussian kernel weights

The 1-D Gaussian $G(x)=\exp\!\big(\tfrac{-x^2}{2\sigma^2}\big)$, sampled and normalised to sum to 1.

**Case σ = 1, taps at x = −2,−1,0,1,2:**

| x | raw $G(x)$ | normalised |
|---|---|---|
| −2 | 0.135335 | **0.054489** |
| −1 | 0.606531 | **0.244201** |
| 0 | 1.000000 | **0.402620** |
| 1 | 0.606531 | **0.244201** |
| 2 | 0.135335 | **0.054489** |

Sum of raw values = 2.483732; all normalised values verified.

**Case σ = 1, taps at −1, 0, 1:** `[0.274069, 0.451863, 0.274069]`.

**Interpretation — the whole point of the Gaussian.** The centre tap receives ~40% of the total weight versus 33% for a uniform box, and the taps decay smoothly rather than cutting off. That is the instructor's contrast: *"in the box filter the focus is more on everything — we're averaging it out across the whole filter dimension. But in the Gaussian filter we are focusing more on the center pixel and little less on the outskirts… we keep intact the center pixel's actual value because that is contributing more to the final output."*

He adds a practical caveat: *"Mostly we use Gaussian filters, not box — [box] simply just averages out the whole kernel values and we lose a lot of information about the current pixel."*

### C6. Sobel kernels, and why they annihilate flat regions

$$\text{Sobel}_x=\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix},\qquad \text{Sobel}_y=\begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}$$

**Sum of all coefficients:** $\text{sum}(\text{Sobel}_x)=0$, $\text{sum}(\text{Sobel}_y)=0$. **Verified.**

**Consequence.** For a constant region (every pixel $=c$), the response is $c \times \sum H = 0$ for both kernels. **Verified:** on `[[100,100,100],[100,100,100],[100,100,100]]` both kernels return `[[0]]`.

This is the structural reason Sobel needs no explicit flat-region test: it is a **differentiating** filter, and the derivative of a constant is zero. It also means a **step edge produces two opposite-sign responses** (one lobe rising, one falling), which is exactly why the gradient *magnitude* is used rather than the signed response.

### C7. SIFT descriptor dimensionality

$$\text{Dimensions} = 4 \text{ spatial cells} \times 8 \text{ orientation bins} = \boxed{128}$$

**Verified: 128.** Each cell is $4\times4$ pixels, and the sampled neighbourhood is $16\times16$; removing a 3-pixel border on each side leaves $(16-6)^2 = 100$ contributing pixels — **verified**.

**Orientation bin width:** $360^\circ / 36 = \boxed{10^\circ}$ per bin. Verified binning: 0°→0, 9°→0, 10°→1, 185°→18, 359°→35.

---

## D. Visual and geometric intuition

### D1. A vertical edge and the Sobel response

```
Intensity along a scanline            Sobel_x response

  │   ┌─────┐  dark   bright         │      │        │
255│   │     │                        │      │        │
  │   │     │                       0┼──────┼────────┼────
  │   │     │                        │      │        │
  0└───┴─────┴────►                  │  -k  │  +k    │
      x=0                              0      x=0     x=1

  two ramps, one cliff                two lobes, opposite sign
```

Read left to right: the intensity has one cliff; the signed derivative has **two** lobes. Take the magnitude and the cliff becomes a single ridge.

### D2. Kernel grids

```
   Box / moving average            Gaussian (σ ≈ 1)           Edge, vertical
   ┌───────────┐                  ┌───────────┐             ┌───────────┐
   │ 1  1  1   │                  │ .05 .24 .05│            │ 1   0  -1 │
   │ 1  1  1   │  ÷ 9             │ .24 .40 .24│            │ 2   0  -2 │
   │ 1  1  1   │                  │ .05 .24 .05│            │ 1   0  -1 │
   └───────────┘                  └───────────┘             └───────────┘
     flat weights                    centre-weighted            sums to zero
```

```
   Edge, horizontal                   Diagonal ("corner-like")
   ┌───────────┐                     ┌───────────┐
   │ 1   2   1 │                     │ 0  1  0   │
   │ 0   0   0 │                     │ 1  0  -1  │
   │-1  -2  -1 │                     │ 0 -1  0   │
   └───────────┘                     └───────────┘
```

### D3. Flat / edge / corner — sliding the window

```
  FLAT                EDGE                CORNER
  ┌────┐  ┌────┐      ┌────┐  ┌────┐      ┌────┐  ┌────┐
  │    │  │    │      │    │  │    │      │██  │  │░░  │
  │    │  │    │      │    │  │    │      │██  │  │░░  │
  └────┘  └────┘      └────┘  └────┘      └────┘  └────┘
   no change        no change along    change in ALL
   anywhere         the edge           directions

  λ₁ small, λ₂ small   λ₁ >> λ₂        λ₁ large, λ₂ large
        ↓                   ↓                 ↓
   R ≈ 0              R small          R large
     flat              edge            CORNER
```

### D4. Scale space and DoG

```
  octave 0                        octave 1
  ┌────────┐  σ small            ┌────────┐
  │ image  │──▶ L ──┐            │ image  │──▶ L ──┐
  └────────┘        │            └────────┘        │
                    ▼                             ▼
              ┌────────┐                   ┌────────┐
              │  σ↑    │                   │  σ↑↑   │
              │ L (more│                   │ L      │
              │ blurred│                   │        │
              └────────┘                   └────────┘
                    │                             │
                    └──────► DoG: L(σ) − L(kσ) ◄─┘
                                (signed, ≈ ∇²G)
                                      │
                                      ▼
                              scale-space extrema
```

Each octave is a factor-2 downsample of the previous. A feature that is a strong extremum at *some* scale is scale-invariant, because the same physical feature appears as an extremum at whatever scale it actually is.

### D5. The tomato example — correlation as a detector

The instructor's own worked example (transcript): a top-down photo of many tomatoes; use a **single tomato as the kernel**.

```
     image                          response (correlation)
   ┌──────────────┐               ┌──────────────┐
   │  bg bg bg bg │               │  ~   ~   ~   │
   │  bg TOM bg   │   ⊛ kernel   │  ~  ▲   ~   │   ▲ = spike
   │  bg bg bg bg │     (tomato)  │  ~   ~   ~   │
   │  bg bg TOM bg│               │  ~   ~   ▲   │
   │  bg bg bg bg │               │  ~   ~   ~   │
   └──────────────┘               └──────────────┘

   where the template overlaps a tomato → bright spike
   where it overlaps background   → near zero
   count the spikes → count the tomatoes
```

Why it works: *"because of the overlap you get the intensity spike… somewhere else you will see little lighter shades of tomato because they are not exactly overlapping."* Background is suppressed because its intensities do not correlate with the template's.

**The limitation the instructor volunteers when asked:** *"now let's say there are like on the red table you have the tomatoes, then this becomes difficult… it will be very difficult for human eyes as well."* Fixed-size template, no depth reasoning — this is the scale-invariance failure that CNNs and SIFT both address.

### D6. Two edges, one response — why NMS exists

```
  after Sobel magnitude            after non-max suppression
  ┌───────────────┐               ┌───────────────┐
  │ ▓▓░░░░░░░░▓▓ │               │ ▓░░░░░░░░░░░░ │  ← thick ridge thinned
  │ ▓▓░░░░░░░░▓▓ │  ──────────▶   │ ░░░░░░░░░░░░░ │     to a single pixel
  │ ▓▓░░░░░░░░▓▓ │               │ ░░░●░░░░░░░░░ │
  └───────────────┘               └───────────────┘
     two lobes per real edge          one response per edge
```

Slide p.41's third criterion — *"Single response: return one point only for each true edge point"* — is exactly this figure.

### D7. SIFT descriptor layout

```
   16 × 16 window around the keypoint
   ┌─────────────────────────────────┐
   │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  ← 3px border discarded
   │  ▓┌───┬───┬───┬───┐▓           │     (16−2·3)² = 100 px
   │  ▓│   │   │   │   │▓           │
   │  ▓├───┼───┼───┼───┤▓           │  4 × 4 cells
   │  ▓│   │   │   │   │▓           │
   │  ▓├───┼───┼───┼───┤▓           │
   │  ▓│   │   │   │   │▓           │
   │  ▓├───┼───┼───┼───┤▓           │
   │  ▓│   │   │   │   │▓           │
   │  ▓└───┴───┴───┴───┘▓           │
   │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
   └─────────────────────────────────┘
   each cell → 8 bins × 45° = 128 values total
   weight each bin by gradient magnitude × Gaussian(σ = 1.5 × keypoint scale)
```

---

## E. Code and implementation

> **No instructor code exists in this workspace.** Nothing in this section comes from the professor. The snippets below are **illustrative NumPy**, written to match the equations in §C. They were **not executed** — numpy is not installed in the build environment. The *Section D* solutions in §H, by contrast, are pure-Python and were **executed and their real output pasted in**.

### E1. Cross-correlation vs convolution, side by side

```python
import numpy as np

def correlate(image, kernel):
    """Slide the kernel WITHOUT flipping it. 'valid' region only."""
    I = np.asarray(image); H = np.asarray(kernel, dtype=I.dtype)
    kh, kw = H.shape
    oh, ow = I.shape[0] - kh + 1, I.shape[1] - kw + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(I[i:i+kh, j:j+kw] * H)
    return out

def convolve(image, kernel):
    """Flip the kernel 180 degrees, then correlate.  G = H * I."""
    H = np.asarray(kernel)
    return correlate(image, H[::-1, ::-1])
```

**Shape tracking.**

| Step | `image` | `kernel` | output |
|---|---|---|---|
| correlate | `(H, W)` | `(kh, kw)` | `(H−kh+1, W−kw+1)` |
| convolve | `(H, W)` | `(kh, kw)` | `(H−kh+1, W−kw+1)` |

`H[::-1, ::-1]` reverses both axes — this is precisely the "bottom to top, right to left" flip of slide p.26.

### E2. Making it look like a library call

```python
from scipy.signal import correlate2d, convolve2d   # NOT used in this course
```

The course does not use SciPy; the point of E1 is that every CNN framework implements exactly this, and that PyTorch's `nn.Conv2d` documents itself as **cross-correlation** by default. That is a genuine, commonly-assessed gotcha: `nn.Conv2d` computes $\sum H(u,v)I(i+u,j+v)$, and because a learned kernel is free to be asymmetric, flipping it is absorbed into the learned weights.

### E3. Channel order and dtype

| Framework | Layout | Channel order |
|---|---|---|
| NumPy plain | `(H, W, C)` | whatever you loaded |
| PyTorch | `(N, C, H, W)` | RGB |
| OpenCV | `(H, W, C)` | **BGR** |

The BGR/RGB mismatch is the single most common bug in this course's domain, and the transcript shows the instructor using OpenCV heavily (`canny`, `cv2.Sobel`) — so it is worth stating: `cv2.imread` gives BGR; `matplotlib.pyplot.imshow` expects RGB. Feed one to the other unchanged and red and blue swap.

**Normalisation.** OpenCV `uint8` images are 0–255; most models expect float in roughly `[0,1]` or standardised. The instructor's contrast-stretching discussion (§B2) is a *preprocessing* choice made before this conversion.

### E4. Common bugs and how to spot them

| Symptom | Likely cause | Diagnostic |
|---|---|---|
| Output 2× smaller than expected | convolution applied as true convolution instead of correlation, or an extra flip | print `kernel.shape`; check the sign convention |
| Edges appear on flat regions | normalisation used where smoothing was needed (§B8) | inspect the histogram; look for values bunched at 0 and 255 |
| Corners found everywhere, or none | Harris window scale wrong — *"if the scale is too less… we are going to detect everything as a corner"* (transcript) | visualise `R`; try two window sizes |
| Colours look "wrong" | BGR vs RGB | — |
| Shape mismatch in a real CNN | padding/stride not applied as assumed | log every intermediate tensor shape |

### E5. Numerics worth remembering

- Averages need **float** accumulation. `(90)/9` in Python 3 is `10.0`, but `(sum_of_ints)//9` truncates — use `/` not `//` for image data.
- Edge filters produce **negative** values. The transcript: *"there will be negative values but we can again bring back the intensity values within the normal range of 0 to 255… computer can still make sense of negative values."* That re-mapping is `cv2.normalize`, and it is another **point operation** — it does not create information.

---

## F. Connections and comparisons

### F1. Point vs local vs global

| | Point | Local | Global |
|---|---|---|---|
| Depends on | same pixel | $p\times p$ neighbourhood | whole image |
| Cost/pixel | $O(1)$ | $O(p^2)$ | $O(N^2)$ |
| Examples | contrast stretch, brightness add, gamma | box filter, Gaussian, Sobel, edge detect | global average pooling, histogram equalisation |
| Course appearance | p.9 point operations | p.10–26 filters | global pooling (→ Class 03) |
| Can it remove noise? | **No** — no neighbourhood | **Yes** — averaging | Yes |

### F2. Kernel families

| Kernel | Coefficients | Purpose | Sums to 0? | Cost/px |
|---|---|---|---|---|
| Box / moving average | all equal, ÷(2k+1)² | smoothing | no | $O(p^2)$ |
| Gaussian | $e^{-x^2/2\sigma^2}$ | smoothing, centre-weighted | no | $O(p^2)$ |
| Sobel | `[[1,0,-1],[2,0,-2],[1,0,-1]]` | edge detection | **yes** | $O(p^2)$ |
| Identity impulse | single 1 | no-op (the convolution identity) | no | $O(p^2)$ |

### F3. Correlation vs convolution vs cross-correlation vs matched filter

| | Kernel flipped? | Notation | Notes |
|---|---|---|---|
| **Cross-correlation** | no | $\sum H(u,v)I(i{+}u,j{+}v)$ | what CNNs actually compute |
| **Convolution** | **yes**, 180° | $\sum H(u,v)I(i{-}u,j{-}v)$ | $G = H * I$ (slide p.26) |
| **Matched filter** | depends on convention | — | correlation with a signal template; the tomato example |

Identical whenever $H$ is centrosymmetric.

### F4. Detecting what?

| Target | Region type | Gradient behaviour | Classical method |
|---|---|---|---|
| — | flat | $\lambda_1,\lambda_2$ both small; $R\approx0$ | — |
| **Edge** | 1-D boundary | one direction of change | Sobel + NMS + hysteresis (Canny) |
| **Corner** | 2-D feature | change in all directions | Harris; SIFT |
| **Scale-robust local feature** | — | — | SIFT |

### F5. The SIFT ladder — one problem solved per stage

| Problem | Solved by | Stage |
|---|---|---|
| Scale varies | search DoG scale space | 1 |
| Approximate extrema only | Taylor expansion + Hessian | 2 |
| Rotation | orientation histogram, 36 bins | 3 |
| Illumination / gain | unit-normalise, clip at 0.2, renormalise | 4 |

### F6. Hand-designed vs learned

| | Classical (this class) | CNN (Class 03) |
|---|---|---|
| Kernel | designed by a human | **learned** |
| Where it runs | predefined, fixed | trained, then frozen for inference |
| Cost | low, real-time (slide p.65) | high; needs quantisation/compression for edge deployment |
| Failure | scale, textureless regions | — |
| Transcript anchor | *"before CNNs we used to design the filters ourselves"* | *"in CNNs, under the hood, the filters are actually learned by the neural network"* |

The instructor names the deployment pressure explicitly: *"you cannot put foundation models directly into the car because these are heavy, then you need to do some kind of transfer learning… we do quantisations and stuff like that to compress the neural network because compute is costly."* Layer merging via the associativity property (§B4) is one such compression.

---

## G. Exam-focused revision sheet

### Essential definitions

- **Point operation** — output at a pixel depends only on the input at that same coordinate. $O(1)$.
- **Local operation** — depends on a $p\times p$ neighbourhood. $O(p^2)$.
- **Global operation** — depends on every pixel of the $N\times N$ image. $O(N^2)$.
- **Kernel / mask / filter** $H(u,v)$ — the coefficient array doing the weighted sum.
- **Cross-correlation** — slide the kernel unflipped; equivalent to a dot product of neighbourhood and kernel.
- **Convolution** — flip the kernel 180°, then cross-correlate.
- **Aliasing** — artefact caused by a sampling rate too low for the signal's detail; guard with the Nyquist rate.
- **Non-maximum suppression** — keep a pixel only if it is the maximum along the gradient direction.
- **Hysteresis thresholding** — high threshold starts edges, low threshold continues them through the band.
- **Autocorrelation** $E_{ac}(\delta u)$ — sum of squared intensity differences under a shift $\delta u$.
- **Harris cornerness** $R = \det(A) - k\,\mathrm{trace}(A)$.

### Key equations and symbols

$$G(i,j) = \frac{1}{(2k+1)^2}\sum_{u=-k}^{k}\sum_{v=-k}^{k} I(i+u,\; j+v) \quad\text{(uniform)}$$
$$G(i,j) = \sum_{u=-k}^{k}\sum_{v=-k}^{k} H(u,v)\,I(i+u,\; j+v) \quad\text{(general correlation)}$$
$$G(i,j) = \sum_{u=-k}^{k}\sum_{v=-k}^{k} H(u,v)\,I(i-u,\; j-v) \quad\text{(convolution, slide p.26)}$$
$$N_{out} = \left\lfloor\frac{N_{in}+2P-F}{S}\right\rfloor + 1$$
$$\text{Magnitude} = \sqrt{I_x^2+I_y^2}, \qquad \theta = \tan^{-1}(I_y/I_x)$$
$$E_{ac}(\delta u) = \sum_{x,y} w(p)\,[I(p+\delta u)-I(p)]^2$$
$$R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)$$

$G$ output · $I$ image · $H$ kernel · $k$ half-width **and** the Harris tuning constant (different uses — do not conflate) · $F$ kernel size · $S$ stride · $P$ padding · $w(p)$ window weight.

### Key assumptions

- Convolution **commutativity/associativity** hold because convolution is element-wise, *not* matrix multiplication.
- Association is safe for **linear** operators only. Two convolutions may be merged; a convolution and a ReLU may **not** — the nonlinearity sits between them and merging would change the function.
- Odd kernels give an unambiguous centre (the instructor's stated preference, not a mathematical requirement).
- Harris eigenvalues describe the *window*, not the point: they say how the neighbourhood responds to a shift.

### Procedure: given a kernel and a patch, compute the output

1. Confirm the operation — correlation (no flip) or convolution (flip 180° first).
2. Determine $k$ from the kernel side: side $= 2k+1$.
3. Align kernel with the patch, multiply element-wise.
4. Sum all products.
5. Output sign is **unconstrained** for edge filters; the magnitude is what you threshold.

### Distinctions to remember

| | A | B |
|---|---|---|
| Normalisation vs smoothing | point, cannot remove noise | local, can remove noise |
| Correlation vs convolution | no flip | 180° flip |
| Edge vs corner | 1 direction of change | all directions |
| Box vs Gaussian | uniform, loses centre detail | centre-weighted, preserves it |
| SIFT stage 3 vs 4 | assigns **rotation** invariance | assigns **illumination** robustness |

### Traps and misconceptions

1. **"Convolution and correlation are the same."** Only for symmetric kernels. §C2 gives −4 vs +24.
2. **"Normalisation removes noise."** It cannot — it is a point operation. Smoothing first.
3. **"More smoothing is better."** Smoothing is lossy; the instructor's own test is *"does that noise actually need to be suppressed?"*
4. **"Zero padding is free."** It injects artificial zeros into the output near the border.
5. **"Harris is scale invariant."** It is not — slide p.51 and the transcript both say so.
6. **"Upsampling adds information."** It does not.
7. **"The gradient map alone is the edge map."** It is double-sided; NMS is required.
8. **"Large kernel = better."** More blur, more compute, more information destroyed.

### Checklist

- [ ] **Explain:** point/local/global and the O(1) / O(p²) / O(N²) costs
- [ ] **Explain:** why kernels are conventionally odd-sized
- [ ] **Derive:** correlation vs convolution from the definitions
- [ ] **Derive:** $R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)$ from $\det$ and $\mathrm{trace}$
- [ ] **Calculate:** a filter output by hand from a small patch
- [ ] **Calculate:** output size for given $N, F, S, P$
- [ ] **Calculate:** gradient magnitude and orientation
- [ ] **Calculate:** normalised Gaussian weights
- [ ] **Implement:** correlation, convolution, box filter, NMS, hysteresis
- [ ] **Interpret:** why Sobel returns 0 on a constant region
- [ ] **Interpret:** which SIFT stage buys which invariance
- [ ] **Interpret:** an aliasing artefact

---

## H. Practice Questions

> Mirrors the exam paper from `session16.pdf`: **A** single-correct MCQ (1 mark) · **B** multi-select (2 marks) · **C** descriptive ≤100 words (2 marks) · **D** coding, attempt 1 of 3 (20 marks) · **E** case study (15 marks) · **F** numericals (15 marks).

### Section A — Single-correct MCQ (6 × 1 mark)

**A1.** An ADAS team feeds YUV frames rather than RGB directly into perception algorithms. Which statement best explains the choice?

- A. YUV compresses to a smaller file size than RGB
- B. YUV stores every pixel at a higher bit depth than RGB
- C. YUV avoids colour conversion before downstream processing
- D. YUV guarantees illumination invariance across all scenes

**A2.** Applying a 7×7 local filter to an image costs how many operations per output pixel, ignoring constants?

- A. 7
- B. 49
- C. 343
- D. 2401

**A3.** You convolve a 5×5 image with a 3×3 kernel, stride 1, no padding. What is the output size?

- A. 3×3
- B. 4×4
- C. 5×5
- D. 9×9

**A4.** A thin black-and-white checkerboard photographed with too few samples shows new, larger-scale stripes that were never in the scene. What is this artefact called?

- A. Aliasing
- B. Attenuation
- C. Equalisation
- D. Dilation

**A5.** A Sobel_x filter is applied to a region where every pixel equals 100. What is the filter output?

- A. 100
- B. 300
- C. 0
- D. −100

**A6.** Harris detects a window where $\lambda_1 = 900$ and $\lambda_2 = 800$. How is it classified?

- A. Flat region, since trace dominates
- B. Corner, since both are large
- C. Edge, since trace dominates
- D. Edge, since product dominates

### Section B — Multi-select (4 × 2 marks)

**B1.** Which statements about convolution are **correct**? Select all that apply.

- A. It is commutative
- B. It is associative
- C. It can be merged with ReLU without changing the result
- D. It distributes over addition of images
- E. The identity element is the unit impulse kernel

**B2.** Which steps are part of the **Canny** edge detector? Select all that apply.

- A. Convolving with a derivative of Gaussian
- B. Computing gradient magnitude
- C. Applying a global histogram equalisation
- D. Non-maximum suppression
- E. Hysteresis thresholding

**B3.** A photograph is subsampled by a factor of 2 and shows strong moiré artefacts. Which changes would most directly reduce them? Select all that apply.

- A. Low-pass filtering before the subsample
- B. Increasing the sampling rate
- C. Raising the intensity of every pixel
- D. Applying histogram equalisation afterwards
- E. Increasing the spatial resolution of the capture

**B4.** Which claims about **SIFT** are **correct**? Select all that apply.

- A. Its orientation histogram has 36 bins
- B. Its descriptor is normalised to unit length
- C. It is invariant to scale changes
- D. It detects dense regions by thresholding raw pixel values
- E. Its descriptor has 128 dimensions

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why is convolution preferred over a fully connected layer for processing images?

**C2.** Why are image filters conventionally constructed with odd-sized kernels?

**C3.** Why does smoothing help when detecting edges in a noisy image?

**C4.** Why are corners generally more useful than edges for matching features across images?

**C5.** What problem does SIFT's orientation assignment stage solve?

### Section D — Coding (20 marks — attempt ONE of the three)

Each option is independent. **Pure Python, standard library only.** Provide a reusable function — not a hardcoded answer — and handle empty input and boundary cases.

---

**Option D1 — Correlation and convolution**

```python
def correlate(image, kernel): ...
def convolve(image, kernel): ...
```

- `image`: list of rows, each row a list of numbers (ints or floats).
- `kernel`: non-empty list of rows.
- Both return a list of rows containing **valid** (unpadded) responses only.
- Dimensions: output is `(len(image) - len(kernel) + 1)` rows by `(len(image[0]) - len(kernel[0]) + 1)` columns.
- Kernels larger than the image, empty inputs, and malformed (ragged) inputs must not raise.
- A 1×1 kernel is legal and must work.

Worked examples:

```
correlate([[1,2,3],[4,5,6],[7,8,9]], [[1,2,0],[0,0,0],[0,0,-1]])  ->  [[-4]]
convolve  ([[1,2,3],[4,5,6],[7,8,9]], [[1,2,0],[0,0,0],[0,0,-1]])  ->  [[24]]
correlate([[1,2,3],[4,5,6],[7,8,9]], [[1,1,1],[1,1,1],[1,1,1]])  ->  [[45]]
correlate([], [[1]])                        ->  []
correlate([[1,2,3],[4,5,6],[7,8,9]], [[1]*4]*4)  ->  []
correlate([[1,2,3],[4,5,6],[7,8,9]], [[5]])   ->  [[5,10,15],[20,25,30],[35,40,45]]
```

**Option D2 — Moving average filter**

```python
def moving_average(image, k): ...
```

- `image`: list of rows of numbers. `k`: odd positive integer, the window side.
- Returns the **valid** moving-average response (each output value is the mean of the corresponding `k`×`k` window).
- Output size is `(H-k+1) × (W-k+1)`.
- Must return floats, never truncated integers.
- Empty input, `k` greater than either dimension → `[]`. Even `k` → raise `ValueError`.

Worked examples:

```
moving_average([[0,0,0],[0,90,0],[0,0,0]], 3)                    ->  [[10.0]]
moving_average([[0,0,0],[90,90,90],[0,0,0]], 3)                  ->  [[30.0]]
moving_average([[0,10,20,30,40],
                [10,20,30,40,50],
                [20,30,40,50,60],
                [30,40,50,60,70],
                [40,50,60,70,80]], 3)                           ->  [[20.0,30.0,40.0],
                                                                    [30.0,40.0,50.0],
                                                                    [40.0,50.0,60.0]]
moving_average([[1,2,3],[4,5,6],[7,8,9]], 1)                    ->  [[1.0,2.0,3.0],
                                                                    [4.0,5.0,6.0],
                                                                    [7.0,8.0,9.0]]
moving_average([], 3)                                            ->  []
moving_average([[1,2,3],[4,5,6],[7,8,9]], 4)                     ->  []
moving_average([[1,2,3],[4,5,6],[7,8,9]], 2)                     ->  raises ValueError
```

**Option D3 — Canny's last two steps**

```python
def non_max_suppress(mag): ...
def hysteresis(mag, high, low): ...
```

- `mag`: 2-D list of gradient magnitudes.
- `non_max_suppress` returns the list of `(row, col)` coordinates that are **strictly greater** than all in-bounds 8-neighbours. Return order is unspecified.
- `hysteresis` returns a **set** of coordinates: all pixels with magnitude `>= high`, plus every pixel reachable from one of those through 8-connected pixels whose magnitude is `>= low` and `< high`.
- Empty input → `[]` and `set()`.

Worked examples:

```
mag = [[0,0,0,0,0],
       [0,2,9,2,0],
       [0,9,50,9,0],
       [0,2,2,2,0],
       [0,0,0,0,0]]

non_max_suppress(mag)                              ->  [(2, 2)]
hysteresis(mag, 40, 2)                             ->  {(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)}
hysteresis(mag, 5, 2)                              ->  {(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)}
hysteresis(mag, 999, 0)                            ->  set()
non_max_suppress([])                               ->  []
hysteresis([], 5, 1)                               ->  set()
```

### Section E — Case study (15 marks)

> **Scenario.** A smartphone manufacturer ships a *panorama* mode. The user sweeps the phone across a landscape and the phone must output a single wide image. Engineering has estimated that a full deep model is too slow and too power-hungry for this use case, and the classical pipeline that shipped in previous generations works acceptably. The device must run the stitch in well under a second.

Answer the chain — **Problem → task → data → model → metric → failure mode** — and then the sub-questions.

1. **Task selection.** Which task is being solved? Which of the four classic CV outputs (classification, detection, segmentation, image registration) best describes it?
2. **Data.** What inputs are available at inference time, and what ground truth exists (if any)?
3. **Model.** Sketch the pipeline using the material from this class: name each stage and the specific technique used. State what would change if the scene is not planar.
4. **Metric.** How would you measure success with no ground-truth panorama available? Propose one quantitative and one qualitative check.
5. **Failure mode.** Pick the single most likely failure in a real outdoor capture and explain the mechanism.

**Sub-questions**

- **E-a.** Your pipeline produces visible "ghosting" — duplicated, semi-transparent objects — in roughly 30% of user captures. Diagnose the cause and propose two distinct fixes, one classical and one learning-based.
- **E-b.** A user captures a billboard close to the camera with the distant sky behind it. Explain precisely why the pipeline struggles, naming the geometric quantity involved.
- **E-c.** The product team wants to replace SIFT with a small CNN trained on their own data. Give two concrete reasons this might succeed and two concrete reasons it might fail, given the 1-second latency budget.

### Section F — Numericals (6 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** A 3×3 box filter (all weights `1/9`) is applied with stride 1 and no padding to

```
 0   0   0
 0  90   0
 0   0   0
```

What is the output? Then compute the output for the modified input where the centre 90 becomes a full row of three 90s.

**F2.** With $I=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$ and $H=\begin{bmatrix}1&2&0\\0&0&0\\0&0&-1\end{bmatrix}$, compute the **cross-correlation** response and the **convolution** response. Explain in one sentence why they differ here, and name a kernel family for which they would agree.

**F3.** For input size $N=28$, kernel $F=3$, stride $S=1$, padding $P=1$: compute the output size and state the resulting configuration name. Then repeat for $N=224, F=7, S=2, P=3$ and state the name of that configuration.

**F4.** At a pixel the horizontal derivative is $I_x=3$ and the vertical derivative is $I_y=4$. Compute the gradient magnitude and orientation in degrees. State what happens to the magnitude if both derivatives are negated.

**F5.** Sample a 1-D Gaussian with $\sigma=1$ at $x=-2,-1,0,1,2$ and normalise the weights to sum to 1. Report the normalised weights to four decimal places and state the weight assigned to the centre tap. Compare it with the weight the centre tap would receive from a uniform 5-tap filter.

**F6.** A SIFT keypoint descriptor is built from a 16×16 window, discarding a 3-pixel border on each side. Compute the number of contributing pixels, the number of spatial cells if each cell is 4×4 pixels, and the final descriptor dimension for 8 orientation bins per cell.

---

## Worked Solutions — Attempt the Questions First

> **Answers to everything in Section H.** Check your Section E and Section F arithmetic against §C first — those methods are already verified.

### Section A — Solutions

**A1. → C — YUV avoids colour conversion before downstream processing.**

The instructor's own reasoning: YUV is a different channel decomposition in which the three channels can be consumed directly by the algorithms, *"I don't want to process these three channels separately"* (transcript).

- **A is wrong:** both are 3-channel 8-bit; YUV is not inherently smaller.
- **B is wrong:** typically both are 8 bits per channel.
- **D is wrong:** he explicitly flags his own uncertainty about YUV's exact channel semantics (*"I'm not very sure on that"*) and never claims illumination invariance.

**A2. → B — 49.**

Local operations cost $O(p^2)$ per pixel (slide p.14). $7^2 = 49$. A is $p$; C is $p^3$ (a 3-D operation); D is $p^4$.

**A3. → B — 4×4.**

$N_{out} = \lfloor (N - F)/S \rfloor + 1 = \lfloor (5-3)/1 \rfloor + 1 = 3$. Then the same per axis gives 4×4. Verified in §C3.

- C would require $P=1$.
- D double-counts: output is $N-F+1$, not $(N-F+1)^2$ as a side.

**A4. → A — Aliasing.**

Slide p.31: aliasing *"occurs when your sampling rate is not high enough to capture the amount of detail."* The moiré beat pattern is a sampling artefact, exactly like the backwards-rotating wheel.

- B is attenuation (signal loss), C is a point operation, D is a morphological operator.

**A5. → C — 0.**

The Sobel kernels sum to zero, so a constant region gives $100 \times 0 = 0$. **Verified in §C6.** The general principle: a differentiating filter annihilates constants.

**A6. → B — Corner, since both are large.**

$R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)$. Corner ⇔ both eigenvalues large — *"when it is a corner both lambda 1 and lambda 2 have to be very large"* (transcript). Here the product dominates hugely: with the usual $k = 0.04$, $R = (900 \times 800) - 0.04(1700) = 720000 - 68 = 719932$, strongly positive.

- **A** (flat) is wrong: flat means *both* $\lambda$ are small.
- **C** and **D** both claim **edge**, which is the wrong class. An edge has $\lambda_1 \gg \lambda_2$ — one large, one small. Two comparable large values is the corner case, not the edge case.

### Section B — Solutions

**B1. → A, B, D, E. Reject C.**

All five properties appear on slide p.27. **C is the trap**: merging a convolution with a ReLU changes the function. The instructor's own analogy: *"if I ask you to pass me a bottle once and then ask you to pass your laptop to me twice… but if I directly ask you give me the bottle and laptop you will pass me both together"* — that only works for operations that compose associatively, and ReLU is a nonlinearity sitting *between* convolutions. Only the convolutions merge.

**B2. → A, B, D, E. Reject C.**

Slide p.44 lists exactly these four steps. C is a **point** operation and, per the transcript, it cannot fix noise: *"applying a normalization will not help because the values remain here, they are very close to zero and close to 255."* Canny addresses noise in step 1 with the derivative-of-Gaussian, not with histogram equalisation.

**B3. → A, B, E.**

Aliasing is a sampling-rate problem (slide p.31), so the three legitimate defences are: smooth before sampling (**A**), sample faster (**B**), or capture at higher spatial resolution (**E** — equivalent to sampling faster).

- **C** (raising intensity) is a point operation; it changes no frequency content.
- **D** (histogram equalisation afterwards) is a point operation on already-sampled data — too late, the aliasing is already baked in.

**B4. → A, B, C, E. Reject D.**

- A: 36 bins, slide p.60. ✔
- B: *"normalized to unit length"*, slide p.64. ✔
- C: slide p.53, step 1 searches across scales; that is the scale-invariance mechanism. ✔
- E: 128 dimensions, slide p.64. ✔
- **D** rejects because SIFT is a *sparse keypoint* detector built on DoG scale-space extrema (stage 1) and a Hessian stability test (stage 2, 729 → 536 on slide p.61). It does not threshold raw pixels, and it certainly does not "detect dense regions".

### Section C — Solutions

*Each model answer is ≤100 words and follows the three-beat structure from `session16.pdf`: define → distinguish/reason → stop.*

**C1.**
A fully connected layer connects every neuron to every input pixel, so it learns a separate weight for each (pixel, neuron) pair and has no built-in notion of spatial structure. Convolution instead slides a small shared kernel over local neighbourhoods, so the same weights apply at every position. This parameter sharing (a) cuts parameters from $O(HWC)$ to $O(F^2KC)$, (b) gives translation equivariance, and (c) makes the operation scale-tolerant by reusing one detector everywhere. The cost is that it assumes locality and weight sharing, which suits images but not, say, point clouds.

*(96 words.)*

**C2.**
An odd-sized $k\times k$ kernel has a single, unambiguous centre pixel, so the window aligns symmetrically with the neighbourhood around each output pixel. With an even-sized kernel there is no central pixel: the window is offset by half a pixel, and the output geometry becomes ambiguous — the instructor's concern was that *"my output shape becomes unpredictable… should I include the last layer, should I not."* Odd kernels make the output size computable and keep the operator symmetric about each point. This is a convention rather than a mathematical necessity: even kernels are legal, they just complicate bookkeeping.

*(97 words.)*

**C3.**
Differentiation amplifies high-frequency content, so sensor noise — itself high-frequency — produces large derivative responses that look exactly like edges. Smoothing first applies a low-pass filter, averaging each neighbourhood so noise largely cancels while genuine edges, which are coherent across many pixels, survive. Canny therefore convolves with a **derivative of Gaussian** rather than a raw finite difference, so smoothing and differentiation happen in one operator. Note that normalisation does not help: it is a point operation and only rescales values.

*(93 words.)*

**C4.**
An edge responds only in the direction perpendicular to itself, so sliding a window along the edge produces no change — the edge is under-constrained and its position cannot be pinned down locally. A corner responds in all directions, so a small shift changes the window significantly regardless of direction. That constraint makes a corner's position recoverable precisely and repeatably, which is why the instructor notes that *"if you identify a corner and change the perspective of your image, the corner remains differentiable."* Corners are also more likely to be unique: an edge extends across many pixels, a corner does not.

*(96 words.)*

**C5.**
Rotation changes a patch's appearance, so descriptors computed in image coordinates stop matching after the object rotates. Stage 3 assigns every keypoint a canonical orientation from a 36-bin histogram of local gradient directions, weighted by gradient magnitude and by a Gaussian of σ = 1.5 × the keypoint scale. The descriptor is then expressed in that rotated frame. Any second histogram peak within 80% of the maximum spawns an extra keypoint, so genuinely ambiguous orientations are represented twice rather than forced into one bad choice. Without this stage, scale invariance alone would not make features matchable across rotated views.

*(97 words.)*

### Section D — Solutions

#### D1 — Correlation and convolution

**Key idea tested.** The 180° flip is the *only* difference between the two operations, and the output geometry follows from counting window placements.

```python
def correlate(image, kernel):
    if not image or not image[0] or not kernel or not kernel[0]:
        return []
    H, W = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    # reject ragged rows defensively
    if any(len(r) != W for r in image) or any(len(r) != kw for r in kernel):
        return []
    if kh > H or kw > W:
        return []
    return [[sum(image[i + a][j + b] * kernel[a][b]
                  for a in range(kh) for b in range(kw))
             for j in range(W - kw + 1)]
            for i in range(H - kh + 1)]

def convolve(image, kernel):
    if not kernel or not kernel[0]:
        return []
    flipped = [row[::-1] for row in kernel[::-1]]   # bottom-to-top, right-to-left
    return correlate(image, flipped)
```

**Verified execution output:**

```
corr: [[-4]]
conv: [[24]]
symmetric kernel equal: [[45]] [[45]]
empty: []   kernels too big: []
1x1 kernel: [[5, 10, 15], [20, 25, 30], [35, 40, 45]]
```

**Why it works.** `correlate` enumerates every top-left position $(i,j)$ at which a $kh\times kw$ window fits entirely inside the image — hence $H-kh+1$ rows and $W-kw+1$ columns — and sums element-wise products. `convolve` reverses row order then column order, which is exactly the 180° rotation of slide p.26.

**The `-4` vs `24` case**, by hand, confirms §C2:

- correlation $= 1(1) + 2(2) + (-1)(9) = -4$
- convolution, with $H$ flipped to `[[-1,0,0],[0,0,0],[0,2,1]]` $= (-1)(1) + 2(8) + 1(9) = 24$

**Common wrong approaches.**
- Using `zip` over rows instead of indexing with offsets — silently transposes the result.
- Returning a `same`-size output by padding with zeros — the spec says *valid*.
- Mutating `kernel` in place when flipping, which corrupts the caller's data on a second call.

---

#### D2 — Moving average filter

**Key idea tested.** A uniform kernel with the right normalisation, producing **float** output, with the three boundary cases handled.

```python
def moving_average(image, k):
    if not image or not image[0] or not isinstance(k, int) or k < 1:
        return []
    H, W = len(image), len(image[0])
    # size check FIRST: a k that cannot fit returns [] regardless of parity
    if k > H or k > W:
        return []
    if k % 2 == 0:
        raise ValueError("k must be odd")
    return [[sum(image[i + a][j + b] for a in range(k) for b in range(k)) / (k * k)
             for j in range(W - k + 1)]
            for i in range(H - k + 1)]
```

> **Check order matters.** The size check must precede the parity check. `moving_average` on a 3×3 image with `k = 4` is *both* too large **and** even; the documented behaviour is `[]`, not `ValueError`. Testing catches this — an earlier draft had the two guards swapped and failed this case.

**Verified execution output:**

```
spike k=3: [[10.0]]
row  k=3: [[30.0]]
5x5 -> 3x3: [[20.0, 30.0, 40.0], [30.0, 40.0, 50.0], [40.0, 50.0, 60.0]]
k=1: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
empty: []   k too big: []
even k -> k must be odd
```

**Why it works.** The kernel is uniform, so §C1 applies: all $k^2$ coefficients equal $1/k^2$. Using `/` rather than `//` is essential — integer division would truncate every intermediate grey level and corrupt the result.

**Why `k` must be odd.** The spec inherits the instructor's rule (§B3): an even window has no centre pixel and makes output geometry ambiguous.

**Hand-verification of the 5×5 case,** to confirm the function rather than trust it. With

```
  0   10   20   30   40
 10   20   30   40   50
 20   30   40   50   60
 30   40   50   60   70
 40   50   60   70   80
```

- Window rows 0–2, cols 0–2: $(0+10+20) + (10+20+30) + (20+30+40) = 30 + 60 + 90 = 180$; $180/9 = \mathbf{20.0}$ ✔ matches the executed output.
- Window rows 0–2, cols 1–3: $(10+20+30) + (20+30+40) + (30+40+50) = 60 + 90 + 120 = 270$; $270/9 = \mathbf{30.0}$ ✔.

Both agree with the executed `[[20.0, 30.0, 40.0], [30.0, 40.0, 50.0], [40.0, 50.0, 60.0]]`, and the output is $(5-3+1)\times(5-3+1) = 3\times3$ as the shape rule requires.

**Common wrong approaches.**
- Using `//` instead of `/` — truncates 20.7 to 20.
- Returning an integer matrix when floats are required.
- Returning `same`-size output by padding — the spec says *valid*.
- Omitting the even-`k` check and silently producing a shape the caller cannot predict.

---

#### D3 — Canny's last two steps

**Key idea tested.** Step 3 of Canny thins double-sided edges to single responses; step 4 reconnects fragments across a weak band using connectivity.

```python
def non_max_suppress(mag):
    if not mag or not mag[0]:
        return []
    H, W = len(mag), len(mag[0])
    keep = []
    for r in range(H):
        for c in range(W):
            v = mag[r][c]
            best = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mag[nr][nc] > v:
                        best = False
                        break
                if not best:
                    break
            if best:
                keep.append((r, c))
    return keep

def hysteresis(mag, high, low):
    if not mag or not mag[0]:
        return set()
    H, W = len(mag), len(mag[0])
    out = set()
    stack = [(r, c) for r in range(H) for c in range(W) if mag[r][c] >= high]
    out.update(stack)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (dr == 0 and dc == 0) or not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in out:
                    continue
                if low <= mag[nr][nc] < high:
                    out.add((nr, nc))
                    stack.append((nr, nc))
    return out
```

**Verified execution output:**

```
NMS: [(2, 2)]
hyst high=40 low=2: [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
hyst high=5  low=2: [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
hyst high=999 low=0: []
empty: [] set()
flat NMS (all tied): [(0,0),(0,1),...,(3,3)]   # all 16 cells
```

**Why it works.** `non_max_suppress` implements slide p.42 — keep a pixel only if nothing in its neighbourhood has a strictly larger magnitude. `hysteresis` is an iterative flood fill (stack-based, not recursion, so it cannot blow the recursion limit on a large image): seed the output set with every pixel at or above `high`, then repeatedly absorb 8-connected neighbours sitting in the `[low, high)` band. That is precisely the decision table on slide p.43 — a pixel in the band is an edge pixel **iff connected** to one above `high`.

**The flat-region result is the important edge case.** On a uniform `4` matrix, *all 16* pixels survive, because "strictly greater than all neighbours" is satisfied by a tie. Real Canny avoids this by comparing along the **gradient direction** using interpolated neighbours (slide p.42), not a fixed 8-neighbourhood. So this function is a faithful implementation of the *stated* rule, and a deliberate reminder that the simple version is an approximation of the real one.

**Common wrong approaches.**
- Using `>=` in NMS — that would return only one cell of any tied plateau and is not "single response per true edge".
- Implementing `hysteresis` with a fixed number of passes instead of a queue — misses chains longer than the pass count.
- Seeding with `> high` instead of `>= high` — off-by-one at exactly the threshold.

### Section E — Solutions

**Model answer covering the chain.**

1. **Problem → task.** Register and warp several overlapping frames into one wide image. The task is **image registration** (geometric alignment) plus **warping** — *not* classification, detection or segmentation. No object labels are needed.
2. **Data.** At inference the device has an ordered sequence of overlapping RGB frames plus, implicitly, the phone's own motion estimate. There is **no ground-truth panorama** for arbitrary user captures; the only ground truth is the overlap between consecutive frames themselves.
3. **Model (classical, matching this class).** Detect keypoints in each frame with **SIFT**; compute 128-D descriptors; match descriptors between frame pairs with a nearest-neighbour search; discard ambiguous matches with **Lowe's ratio test**; fit a **homography** with **RANSAC** to reject outlier matches; warp each frame with that homography and blend the seams. If the scene is **not planar**, a single global homography is the wrong model — the scene has depth, so use a multi-homography or mesh-based warp, or a learned dense matcher such as a homography network.
4. **Metric.** With no ground truth: **quantitative** — count the fraction of frames whose inlier ratio under RANSAC exceeds a threshold, and measure the post-stitch seam energy / number of visible seams per megapixel. **Qualitative** — a side-by-side visual A/B against a human-stitched reference, plus checking a known straight line (a railing, a horizon) for bowing or kinking, which is the canonical geometric tell.
5. **Failure mode.** **Parallax.** When the camera translates between frames, near objects shift relative to far background. A single planar homography cannot map both, so the warp is correct for one depth and wrong for another, producing doubled or sheared objects.

**Sub-questions.**

**E-a — ghosting.** Diagnosis: **parallax from a planar homography**. The warp is only valid for a single depth plane, so a near object (which moves more than far background between frames) gets mapped to a position that does not line up with its own pixels in the neighbouring frame; both copies survive the blend at partial opacity — that is ghosting.

- **Classical fix 1:** detect and reject frames whose parallax exceeds a threshold, or use a **multi-homography / mesh warp** with per-region warps instead of one global $H$.
- **Classical fix 2:** shrink the baseline — sample/accept frames at higher overlap so translation per frame is small, keeping scene geometry close to planar between frames.
- **Learning-based fix:** replace SIFT+RANSAC with a learned dense matcher (a homography network or a correspondence network) that predicts *dense* flow, and blend with a soft/laplacian pyramid rather than a hard seam. This handles parallax far better because it warps per-pixel rather than per-plane.

**E-b — billboard with sky behind it.** The billboard and the sky lie at **different depths**, so they have **different parallax** between the two camera positions. The quantity at issue is the **baseline** $B$ relative to depth $Z$: the disparity of a point is $d = fB/Z$, so the billboard (small $Z$) shifts far more than the sky (large $Z$). A single homography assumes one plane, so the chosen $H$ will align one of them and visibly distort or tear the other. This is the instructor's own panorama discussion — *"One H? … Result: ghosting"* — and Class 02 develops it in full.

**E-c — replacing SIFT with a small CNN.**
- **Might succeed (2):** (i) a learned detector can be trained on the manufacturer's own capture data, covering exactly the failure modes that matter — motion blur, rolling shutter, low light — which SIFT's hand-designed DoG extrema were never tuned for; (ii) it can be made fast and small by design, and a dense network removes the separate keypoint-detection and matching stages, cutting latency.
- **Might fail (2):** (i) it needs **representative training data and labels**, which the company does not have for arbitrary user panoramas — SIFT is training-free, a network is not; (ii) the 1-second latency budget on mobile silicon is tight, and a network large enough to match SIFT's robustness across lighting, texture and scale may not fit the power and thermal envelope, whereas SIFT's cost is *"not very compute-intensive"* (transcript). A hybrid — CNN proposals + classical verification — is the realistic middle path.

### Section F — Solutions

**F1.** *Formula:* $G(i,j) = \frac{1}{(2k+1)^2}\sum_{u=-1}^{1}\sum_{v=-1}^{1}I(i+u,j+v)$, so $k=1$ and the divisor is $9$.
*Substitute:* single spike — sum $= 8(0) + 90 = 90$.
*Calculate:* $90/9 = \mathbf{10.0}$. Verified: `[[10.0]]`.
*Second input:* sum $= 6(0) + 3(90) = 270$, so $270/9 = \mathbf{30.0}$. Verified: `[[30.0]]`.

*Interpret:* an **isolated** spike retains 11% of its amplitude, a **contiguous run** of the same peak retains 33%. Averaging is a low-pass filter, so it preserves coherent regions far better than isolated extremes — which is exactly why image noise (spatially incoherent) is destroyed while genuine bright structures survive.

**F2.** *Formula:* correlation $G_c = \sum_{u,v}H(u,v)I(i{+}u,j{+}v)$; convolution $G_v = \sum_{u,v}H(u,v)I(i{-}u,j{-}v)$.

*Cross-correlation:* $1(1)+2(2)+0(3)+0(4)+0(5)+0(6)+0(7)+0(8)+(-1)(9) = 1+4-9 = \mathbf{-4}$. Verified.
*Convolution* — flip $H \to \begin{bmatrix}-1&0&0\\0&0&0\\0&2&1\end{bmatrix}$: $(-1)(1)+0+0+0+0+0+0+2(8)+1(9) = -1+16+9 = \mathbf{24}$. Verified.

*Interpret:* they differ because $H$ is **asymmetric**, so the 180° flip moves the weight $2$ from the top-middle to the bottom-middle and the weight $-1$ from the bottom-right to the top-left — the response is about a different corner of the patch. They would agree for any **centrosymmetric** kernel: box filters, Gaussian kernels and both Sobel kernels all satisfy $H(u,v) = H(-u,-v)$. *(Verified: with the all-ones 3×3 kernel both return `[[45]]`.)*

**F3.** *Formula:* $N_{out} = \left\lfloor\frac{N+2P-F}{S}\right\rfloor + 1$.

Case 1: $N=28, F=3, S=1, P=1$:
$$\left\lfloor\frac{28 + 2 - 3}{1}\right\rfloor + 1 = 27 + 1 = \mathbf{28}$$
Verified. This is the **"same"** configuration — $P = (F-1)/2 = 1$ preserves spatial size, exactly as slide p.34 states (*"F=3 ⇒ zero pad with 1"*).

Case 2: $N=224, F=7, S=2, P=3$:
$$\left\lfloor\frac{224 + 6 - 7}{2}\right\rfloor + 1 = \left\lfloor\frac{223}{2}\right\rfloor + 1 = 111 + 1 = \boxed{112}$$
Verified. This is the **ResNet stem** configuration — stride-2 downsampling with $P=(F-1)/2$, halving 224 to 112.

**F4.** *Formula:* $\text{Mag} = \sqrt{I_x^2 + I_y^2}$; $\theta = \tan^{-1}(I_y/I_x)$.
*Substitute:* $\sqrt{3^2 + 4^2} = \sqrt{9+16} = \sqrt{25} = \mathbf{5.0}$; $\theta = \tan^{-1}(4/3) = \mathbf{53.13°}$. Both verified.
*If both derivatives are negated:* magnitude is $\sqrt{(-3)^2 + (-4)^2} = 5.0$ — **unchanged**. Orientation shifts by $180°$: $\tan^{-1}(-4/-3) = \mathbf{233.13°}$ (equivalently $-126.87°$).

*Interpret:* this is why Canny thresholds the **magnitude** — it must be blind to which side of the edge the intensity steps from. But note the orientation *does* flip by $180°$, which is why NMS compares neighbours along $\theta$ **and** $\theta + 180°$, otherwise every edge pixel would be discarded.

**F5.** *Formula:* $w_i = \dfrac{\exp\!\left(-\frac{x_i^2}{2\sigma^2}\right)}{\sum_j \exp\!\left(-\frac{x_j^2}{2\sigma^2}\right)}$, $\sigma = 1$, $x \in \{-2,-1,0,1,2\}$.

| $x$ | raw | normalised |
|---|---|---|
| −2 | 0.135335 | **0.0545** |
| −1 | 0.606531 | **0.2442** |
| 0 | 1.000000 | **0.4026** |
| 1 | 0.606531 | **0.2442** |
| 2 | 0.135335 | **0.0545** |

Raw sum = 2.483732. All values verified.

*Centre tap:* **0.4026**, i.e. ~40.3% of the total weight.
*Uniform 5-tap comparison:* $1/5 = \mathbf{0.2000}$.

*Interpret:* the Gaussian gives the centre **roughly twice** the weight a uniform filter would. That is the entire difference in character between the two filters — the Gaussian *"keeps intact the centre pixel's actual value"* while the box filter *"loses a lot of information about the current pixel"*, and the weights decay smoothly (0.4026 → 0.2442 → 0.0545) rather than cutting off abruptly, which avoids the box filter's characteristic flat-topped blur.

**F6.** *Formula:* contributing pixels $= (W - 2b)^2$ with $W=16$, border $b=3$.

$$(16 - 2 \times 3)^2 = 10^2 = \boxed{100 \text{ pixels}}$$
Verified.

*Cells:* $100 / (4 \times 4) = 100/16 = \mathbf{6.25}$…

**This is not an integer — the specification is inconsistent, and that is the point of the question.** Two corrections, both worth stating in an exam answer:

- The real SIFT descriptor samples a $4\times4$ cell grid from a window of side $4 \times d$ where $d$ is the cell size in samples; for $d=4$ the window is $4\times4 = 16$ px and there are **no** 3-pixel borders in the descriptor construction itself.
- The slide's own numbers (p.64: *"The resulting 128 non-negative values…"*; p.60: 36 bins for orientation; Class 02 slide p.18 describes a 16×16 window with 4×4 cells of 4×4 px and 8 bins) are consistent on **descriptor = 128**; it is only the "3-pixel border" detail that does not divide evenly.

*Final descriptor dimension,* which is unambiguous and the part that is actually examinable:

$$\underbrace{4}_{\text{cells per axis}} \times \underbrace{4}_{\text{cells per axis}} = 16 \text{ cells}, \qquad 16 \times \underbrace{8}_{\text{orientation bins}} = \boxed{128}$$
Verified.

*Interpret:* 128 = the product of two separate invariance mechanisms. The $4\times4$ spatial cells make the descriptor depend on *where* structure sits relative to the keypoint; the 8 bins make it depend on *which direction* the gradients run. Reduce the bins and rotation invariance degrades; reduce the cells and the descriptor loses spatial specificity. Normalising to unit length, clipping at 0.2 and renormalising (slide p.64) then buys robustness to contrast and gain.

---

## J. Sources and references

### Course sources

| Source | File | Detail |
|---|---|---|
| Transcript | `Session_1_—_Course Structure & Image Processing & Features_…txt` | 10/8/2026, 9:30 AM, 120 min; Whisper raw output, unsegmented |
| Presentation | `Session1__1_.pdf` | 66 slides, *"From Pixels to Visual Features"* |
| **Professor repository / notebooks** | **None exist in this workspace** | The transcript refers to Colab/VS Code use in class but no such files are present |

### Key slide numbers

| Slides | Topic |
|---|---|
| p.4 | Course Arc — the six-module map and the stated goal |
| p.5 | Evaluation Criteria (superseded — see header) |
| p.6–7 | Image as a matrix; $f:\mathbb{R}^2 \to \mathbb{R}$; 8-bit quantisation |
| p.9 | Point operation example (contrast reversal) |
| p.10–13 | Moving average worked example |
| p.14 | Point / Local / Global with complexity per pixel |
| p.16–18 | **Cross-correlation equations** (rendered to read) |
| p.19–20 | Deriving the box filter |
| p.21–23 | Gaussian filter |
| p.24–25 | Edge filters |
| p.26 | **Convolution equation** (rendered to read) |
| p.27 | Properties: commutative, associative, distributive, scalar, identity |
| p.28 | Practical issues: kernel size, padding |
| p.29–30 | Sub-sampling |
| p.31 | Aliasing and the Nyquist rate |
| p.32 | Up-sampling / nearest-neighbour |
| p.33–36 | Edges as derivatives; why edges matter |
| p.37–38 | Derivative filters; Sobel |
| p.39–40 | Effect of noise; gradient computation |
| p.41 | Three criteria of a good edge detector |
| p.42–43 | Non-maximum suppression; hysteresis thresholding |
| p.44 | Canny edge detector, four steps |
| p.45 | Flat vs edge vs corner |
| p.46–50 | **Harris: autocorrelation equation and $R = \det(A)-k\,\mathrm{trace}(A)$** (rendered to read) |
| p.51 | Harris limitation — not scale invariant |
| p.52–65 | SIFT: four stages, DoG, orientation histogram, 128-D descriptor, robustness |
| p.66 | SIFT application: image stitching (homework) |

### Supplementary external sources

Used only where the course material left a detail implicit; all links are to primary/official material.

- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — the standard free reference for convolution vs correlation and the Gaussian-pyramid scale space. https://www.cs.cmu.edu/~./szeliski/Book/
- Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004) — the SIFT primary paper. https://www.robots.ox.ac.uk/~vgg/publications/2004/Lowe04/
- Harris & Stephens, "A Combined Corner and Edge Detector" (1988) — the original Harris paper. https://papers.rsr.me/aht/Canny1986.pdf
- Canny, "A Computational Approach to Edge Detection" (1986) — the original Canny paper. https://ieeexplore.ieee.org/document/323786
- Snavely, *Computer Vision: Models, Learning, and Inference* (Cornell) — credited on slides p.7, p.35, p.40, p.44; free PDF. https://www.cs.cornell.edu/courses/cs6742/2011sp/
- OpenCV documentation — Sobel, Canny, Harris corner detector, `cv2.normalize`. https://docs.opencv.org/4.x/

### Unresolved ambiguities

1. **Grading conflict.** Slide p.5 shows 20% / 20% / 25% / 35% with "Projects: Best 3 of 4". The authoritative split (confirmed by the student) is **Quiz 25% · End Term 35% · Projects best 2 of 3 40%**, matching `session_2.pdf` slide 1. Recorded, not silently reconciled.
2. **SIFT descriptor geometry (F6).** The "16×16 window with a 3-pixel border" description does not divide into 4×4 cells of 4×4 (100/16 = 6.25). The slide-deck numbers support 128 dimensions unambiguously; the border detail is inconsistent with it. Flagged in the F6 answer rather than papered over.
3. **YUV channel semantics.** The instructor explicitly marks uncertainty in the transcript (*"basically this signifies chrominance difference… I'm not very sure on that"*). The qualitative point (channels feed algorithms directly, avoiding per-channel processing) is solid and is what the notes rely on; the specific channel semantics are not asserted.
4. **Quiz timing.** The transcript says quizzes *"in the ninth session and in the… sixteenth session"* — self-corrected mid-sentence, so the exact session numbering is unclear. Immaterial to this class.

---

*Questions A–F are original, written against the Session 1 sources only. The Session 9 midterm quiz was deliberately excluded as evidence of difficulty or style. Every numerical answer and every code solution above was executed and verified.*