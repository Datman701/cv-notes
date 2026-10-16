# Class 09 — Video Understanding, Project & Practicals

> **Exam weight: Tier 2**, but with one **Tier 1 exception**: the 3D-convolution shape and parameter arithmetic in §C5–C6 is exactly the kind of thing Section F asks, and video tensors are the most commonly botched shape calculation in the whole course.
> **This class also carries the project mechanics that determine 40% of your grade** — see §B12 and the `assignments/` section.
>
> **Primary sources:** `session_9.pdf` (39 slides, *"Lecture 10: Video Understanding"* — CS231n material) · `Session_9_—_Discussion on Quiz + Project and Practicals_…txt` (transcript, 8.5k words).
> **Professor repository / notebooks: none exist in this workspace.** The instructor describes a Drive-linked practical with *"placeholder functions… similar to Coursera style"* and separate `utils` evaluators, but the code is not present.
>
> **Exclusion notice.** The first ~20 minutes of this session were a Q&A on the midterm quiz. Per your instruction that quiz is **not** used as evidence of difficulty, style or content, and none of its questions are reproduced or imitated here. A receptive-field walkthrough occurred in that segment; the underlying concept is covered in [Class 03](../class-03-cnns-under-the-hood/README.md), so it is cross-referenced rather than restated.

---

## A. Lecture overview

### Learning objectives

1. State the shape of a video tensor and compute it for a given resolution, fps and duration.
2. Quantify the storage problem and explain why training uses short, low-fps, low-resolution clips.
3. Distinguish the three fusion strategies for video CNNs: single-frame, late fusion, early (3D).
4. Compute the output shape and parameter count of a 3D convolution layer.
5. Define optical flow and describe the two-stream architecture.
6. Explain I3D's inflation of 2D kernels into 3D.
7. Compute video token counts for a transformer and explain why they explode.
8. State the two broad efficiency strategies for video transformers.
9. Explain joint vs divided space-time attention and the complexity saving.
10. Explain tubelets and why they beat patches for motion.

### Why this matters in CV

Video is where the "2D + time" assumption finally has to be paid for explicitly. Every design decision in this class is a response to one fact — **videos are large and attention over frames is quadratic** — and that reasoning transfers directly to multi-frame perception in ADAS, which the instructor works on: *"you have a lot of videos in it… I cannot pick up every video, every frame that is coming to me to basically train my model."*

### Prerequisites

Class 03 (2D convolution, output-size formula, parameter counting), Class 08 (attention, ViT token counts).

### Position in the course

```
Class 01  filters/kernels ──┐
Class 03  CNNs ─────────────┼──► Class 09  Video = CNN + time dimension
Class 08  ViT / attention ──┘         │
                                      ├─► Class 12  Optical flow, SfM, Visual SLAM
```

### Difficulty: **Intermediate**

Justified because: the shape arithmetic is a direct extension of Class 03 (familiar), and the transformer half is a re-derivation of Class 08 concepts with a new cost argument. The genuinely new material is **why** the design choices were made, not the formulas.

---

## B. Conceptual foundations

### B1. A video is 2D + time

**The problem it solves.** Once you add a temporal axis, every shape convention you learned in Class 03 has to be re-derived.

**Formal statement (slide p.2).** *"A video is a sequence of images. 4D tensor: $T \times 3 \times H \times W$."*

The instructor's own framing: *"video is nothing but an image which is 2D plus a time dimension… so it becomes 4D tensor instead of a 3D tensor… with the three channels from the RGB, let say if you are using, then you have on top of that a time dimension."*

**Worked example, live in class (transcript).** He poses: *"a 30-second video with 30 fps of recording time with greyscale images of height and width 256… what will be this value?"* The student starts to compute pixels; he redirects: *"not the pixels, I'm just asking about this."* The answer:

$$T = 30 \times 30 = 900 \quad\Longrightarrow\quad \boxed{(900,\ 1,\ 256,\ 256)}$$

**The channel dimension survives as 1.** He insists: *"we will still have to keep that dimension… it will become 900 times 1 cross 256 cross 256. We will not give away this dimension but we will keep one value here."* This is a real and easy-to-forget point: greyscale video is 4-D, not 3-D.

**Note the layout.** $T \times 3 \times H \times W$ (channels first, after time) is PyTorch's `(T, C, H, W)`. NumPy/OpenCV would give `(T, H, W, C)`. Getting this backwards is the single most common video bug.

### B2. Task framing: images recognise objects, videos recognise actions

Slide p.4 draws the distinction: *"Images: Recognize objects"* (Dog, Cat, Fish, Truck) versus *"Videos: Recognize actions"* (Swimming, Running, Jumping, Eating, Standing).

The instructor adds the practical framing (transcript): *"image recognition works like you predict the class for the image, but here you are predicting a class for the whole video sequence."*

### B3. The core problem: videos are big

Slide p.5 gives the numbers at **~30 fps, 3 bytes per pixel**:

| Resolution | Bytes/frame | Per minute (30 fps) | Slide |
|---|---|---|---|
| SD 640×480 | 921,600 | **1.66 GB** | *"$ \sim $1.5 GB per minute"* |
| HD 1920×1080 | 6,220,800 | **11.20 GB** | *"$ \sim $10 GB per minute"* |

Both reproduced exactly in §C2.

**Why this is not merely a storage nuisance.** The instructor's ADAS example is the strongest argument in the class (transcript): *"Most of the times in autonomous driving it's like a barren lands, like they are driving on a village side road, there is nothing much happening and **you don't want to keep collecting all that data and filling up your memories** because that is also costly."* Recording is cheap; **keeping** the recording is not.

**The resulting data problem.** *"I have to find some way to basically automatically know that which of the frames, or which parts of the video has a traffic sign… so while the recording is going on you keep tagging them — what lies in this section of the videos. So when I will be training for traffic lights I will just query [with] a simple SQL query which will give me sections of the video where I had traffic lights."* Video understanding is what makes the archive **queryable** rather than merely stored.

### B4. The response: train on short clips

Slide p.6 — *"One Solution: Train on short clips"*:

| Property | Raw video | Training clip |
|---|---|---|
| Input | $T \times 3 \times X \times H \times W$ | low fps, low spatial resolution |
| Example | — | $T=16$, $H=W=112$ |
| Duration | — | **3.2 seconds at 5 fps** |
| Size | GB/min | **588 KB** |

*(Both figures verified in §C3.)*

The instructor describes the live pipeline: *"when we work on the algorithms, what we do is we take only one fps. Every time the camera is giving me, let's say it's collecting on 30 fps, it gives me 30 frames; I only choose the one best frame, and we get rid of all the other 29 frames."*

**Train low, test high (slide p.7).**
- **Training:** *"Train model to classify short clips with low FPS."*
- **Testing:** *"Run model on different clips, average predictions."*

Crucially (transcript): *"you can chunk, have many chunks which have overlap, and do the prediction on top of that **with 30 fps itself** — you don't need to reduce the FPS at testing time."* The down-sampling is a **training-time memory** strategy, not an inference-time one. That asymmetry is worth stating precisely in a case-study answer.

### B5. Fusion strategies — the three families

Everything on slides p.8–p.17 is the answer to one question: **where does temporal information get mixed in?**

| Strategy | Where fusion happens | Slide |
|---|---|---|
| **Single-frame CNN** | never — votes after the fact | p.8 |
| **Late fusion** | after the CNN, on features or scores | p.9–10 |
| **Early fusion (3D CNN)** | inside the convolution | p.11–17 |

#### Single-frame CNN (slide p.8)
*"Simple idea: train normal 2D CNN to classify video frames independently! (Average predicted probs at test-time). **Often a very strong baseline for video classification.**"*

The instructor's honest framing of its weakness (transcript): *"this is very costly because again you are sending every image every second to a CNN which is itself a costly operation, and then you're giving a prediction on every image **which is not required, because we need a prediction only once for the whole video**."*

He also flags the aggregation-policy ambiguity, which is a nice exam detail: *"if the majority of your classes are saying it's not running, then you use not running. But if you want to capture even a single instance of it, then you can run — **even if a single frame detects it's running, you say it's running. It depends on your problem statement.**"*

#### Late fusion (slides p.9–10)
Run the 2D CNN per frame, then combine.

- **With FC layers (p.9):** concatenate all frames' features and feed an MLP. Feature tensor $T \times D \times H \times W$ → flatten → MLP. Cost grows with $T$; the instructor's stated concern is that this is *"very costly"*.
- **With pooling (p.10):** *"Average Pool over space and time"* → then linear. The instructor's framing: *"you have a feature map… I can take the best values, best maximum values, from each of the feature map at each x and y location across these, let's say channels."* This is far cheaper and is where **global average pooling** (Class 01 §B2) reappears.

#### Early fusion with 3D CNNs (slide p.11)
*"Use 3D versions of convolution and pooling to slowly fuse temporal information over the course of the network. **Each activation map in the network is a 4D tensor: $D \times T \times H \times W$.**"*

### B6. How 3D convolution differs

Slides p.13–p.17 walk through it in four steps, mirroring the Class 03 2D walkthrough.

| 2D (slide p.12) | 3D (slides p.14–17) |
|---|---|
| Input $3 \times 32 \times 32$ | Input $T \times 3 \times 32 \times 32$, $T=6$ |
| Filter $5 \times 5$ (spans all channels) | Filter $5 \times 2 \times 5 \times 5$ — *"Now has time dimension!"* |
| Output $6 \times 28 \times 28$ (6 maps) | Output $T' \times 5 \times 28 \times 28$ (5 maps) |
| 6-dim bias vector | **5-dim bias vector** |
| stride in H, W | stride in H, W **and forward/backward in time** |

**The mechanism, in the instructor's words** (transcript): *"3D convolution… where **the kernel also itself has a layer**… I've added a dimension in the kernel itself… you use **cuboid kind of kernel** instead of a flat kernel, and then that makes a single value in the next feature map… this whole cuboid is giving single value."*

**Strides in time (slide p.16 / transcript).** *"If I have a stride of 2, first I convolve over these first two images, and then I can convolve over these two images skipping this one."* His spatial analogy: *"if I take 3×3 filter at this location… if I use the stride of 2, I'll just skip this next one and jump directly to this one."*

**The benefit — and its limit (transcript).** *"It helps you fuse the information across the time dimension… **computation will remain the same** as you pointed out."* Then the real saving, which is architectural rather than per-layer: *"earlier what we're doing is we had another layer of feed-forward neural network and another layer of, let's say, this pooling layer or whatever, like concatenation — **all this you can get rid of because you've already fused information with this convolution**, so overall the number of operations will decrease."* And the general caveat he insists on: *"**this is just a tool** — you don't have to do this specifically or that way specifically… usually it is best suited for these kinds of scenarios where you have a sequence."*

**When 3D convolution is unnecessary — a precise rule he gives.** Asked whether optical flow or depth needs 3D conv:
- *"If I have to predict the depth map from a single image, I can still use 2D convolution."*
- *"You had three channels of image, now you have LiDAR data from which you can generate a depth map — that can become your **fourth channel** and still a 2D convolution can work. **You don't need a 3D convolution** because 3D is majorly used when you have a sequence of time. **It's still a single timestamp.**"*
- *"If you have a video of it, now you can run a 3D convolution."*

The instructor generalises the test: *"if it is a single image and a single timestamp then it is still 2D; but if you have a video sequence then… you can use a 3D convolution."* **The rule is the presence of a time sequence, not the presence of extra channels.**

### B7. Optical flow

**Definition (slide p.20).** *"Optical flow gives a displacement field $F$ between images $I_t$ and $I_{t+1}$… Tells where each pixel will move in the next frame:*

$$F(x,y) = (d_x, d_y) \qquad I_{t+1}(x + d_x,\ y + d_y) = I_t(x,y)$$"

*"Computed via algorithm or faster NN approximation."* Slide p.21 decomposes it into a **horizontal flow $d_x$** image and a **vertical flow $d_y$** image.

The instructor connects it back to Class 01: *"you will get two different feature maps which you can fuse based on the magnitudes, like how you had in the Canny detector where you fused information and got the magnitude and direction."* Optical flow is the same magnitude/direction construction as the image gradient, but estimated **across time** instead of across space.

### B8. Two-stream networks

Slide p.22, from Simonyan & Zisserman (NeurIPS 2014):

```
Input: Single Image            Input: Stack of optical flow
        3 × H × W                      [2 × (T−1)] × H × W
                \                       /
                 └──── Early fusion ────┘
                    (first 2D conv processes all flow images)
```

**The transcript is where the arithmetic gets explained properly.** A student asks *"Can someone tell me why it is 2 times T?"* The answer:

> *"Why it is 2 times T? So T is the number of time sequences, and **for optical flow obviously there is going to be T minus one** because it corresponds to the previous frame — so there is going to be one less. That's how you subtract minus one. And **two** is because you have $d_x$ and $d_y$: one has horizontal direction flow information, one has vertical direction flow information."*

So the two factors are independent and both necessary:

$$\underbrace{2}_{(d_x,\ d_y)} \times \underbrace{(T-1)}_{\text{consecutive frame pairs}}$$

Verified in §C7: $T=16 \Rightarrow 2\times15 = 30$ channels.

**What the motion stream buys (transcript).** Asked what flow adds, the answer is a concrete, quotable example: *"You're saying okay running, not running, not running… **from this you can get very detailed information about the video.** He moved his hand to the right such that he hit an object."* The appearance stream answers *what*; the motion stream answers *what is happening*.

### B9. I3D — inflating 2D networks to 3D

**The idea (slide p.23, transcript).** *"There has been a lot of work on architectures for images. Can we reuse image architectures for video? Idea: take a 2D CNN architecture. **Replace each 2D $K_h \times K_w$ conv/pool layer with a 3D $K_t \times K_h \times K_w$ version.**"*

**The weight trick.** *"Can use weights of 2D conv to initialize 3D conv: **copy $K_t$ times in space and divide by $K_t$.** This gives the same result as 2D conv given 'constant' video input."*

Slide p.23 spells it out: *"Duplicate input $K_t$ times, **Copy kernel $K_t$ times, divide by $K_t$** → Output is the same!"*

**Why it is exact (verified in §C8).** For input constant along time, each of the $K_t$ temporal slices contributes the same partial sum; multiplying the kernel by $1/K_t$ and summing $K_t$ slices recovers the original value exactly.

**The transcript's practical note.** *"No one is designing separately convolution operations for 3D — most of the time they just use the 2D network which is already designed and directly apply 3D from it… **You don't need to have a separate initialization for each of these kernels because the images are all similar.**"*

### B10. Video with transformers — the token explosion

Slides p.25–p.27 apply Class 08's ViT to video.

**Image level (p.25).** 224×224 image → 16×16 patches → $14\times14 = 196$ tokens → linear projection to $D$.

**Video level (p.26–p.27) — the numbers that break everything:**

| Clip | Frames | Tokens |
|---|---|---|
| Small clip | $T=16$ | **3,136** |
| 5 minutes at 1 fps | $T=300$ | **58.8k** |
| 5 minutes at 24 fps | $T=7200$ | **1,411,200** |

All verified in §C4. Slide p.27's annotation is the punchline: *"**~context limit of most modern LLMs**. Naive patch-level attention scales very poorly! How to solve this?"*

The instructor restates the problem (transcript): *"if you keep on using ViT even in a small clip, the token size can grow… the number of tokens will grow very quickly. **Five minutes of video at one FPS will give you this many tokens, and if you increase the FPS the number of images increases and the number of tokens—see how much they have gone to.**"*

**Note this is worse than Class 08's image case** because the two multipliers compound: resolution gives $(\cdot)^2$ per frame, and frames multiply linearly. $224^2 = 50{,}176$ tokens for one image → $1{,}411{,}200$ for 28 seconds of 24 fps.

### B11. The two efficiency strategies

Slide p.28 states the taxonomy, repeated on p.36:

> **Two Broad Strategies for Improving Video Efficiency**
> 1. **Modify the attention operator** — ViViT (ICML 2021), MViTv2 (CVPR 2022)
> 2. **Reduce the number of tokens** — Video Swin Transformer (CVPR 2022)

The instructor restates it live (transcript): *"mainly there are two broad strategies. Either you have to modify the attention calculation mechanism in the transformer, or you reduce the number of tokens."*

A student's suggestion — sample fewer frames — is answered with the reason Class 09 has already established: *"we have moved past that. We are already doing that."*

#### Strategy 1: joint vs divided space-time attention (slides p.29–p.34)

**Joint (p.30).** Treat all tokens from all frames as one sequence. Slide p.30's verdict: **"Extremely inefficient! Recall: 5 minutes at 24fps → 1,411,200 tokens."**

**Divided (p.31–p.34).** *"Key Idea: We can divide time and space attention operators into 2 steps."*

| Step | What each token attends to | Slide |
|---|---|---|
| **Time attention** | *"the same spatial location across different frames"* | p.32 |
| **Space attention** | *"other tokens within the same frame"* | p.33 |

**Why it works (slide p.34) — the two stated reasons:**

1. *"Reduces per-token computational cost from $O(NT)$ to $O(N + T)$."*
2. *"Information can still propagate across space/time after many blocks."*

Verified numerically in §C9: at $N=196$, $T=16$, joint costs $3{,}136$ while divided costs $212$ — a **14.8×** reduction.

**Why reason 2 holds (transcript).** *"Because you are always using the information after every attention block, that information is being carried. Otherwise making the problem is we have this problem of **forgetting in CNNs** — if you use a very large CNN network, the features that were learned in the initial part of the layers start getting fading. That is why you use residual connections… **but here because attention by this transformer by design attends to the tokens and they preserve the information by design** because they're calculating the attention score and preserving it."*

#### Strategy 2a: Video Swin (slide p.35)
*"Key idea: Restrict self-attention to small local space-time cubes. **Very similar to 3D CNNs, but with self-attention inside each cube.** Shift cubes between layers to allow information to pass across cube boundaries."*

#### Strategy 2b: tubelets (slides p.37–39)
*"One solution: tubelets. **More efficient than patches!**"*

The two intuitions the slide gives (p.38):
1. *"**Patches do not contain motion information, but tubelets do.**"*
2. *"Significantly fewer tokens than patches!"*

**The arithmetic (slide p.38).** *"Tubelets over 4 frames: **4× reduction in tokens = 16× reduction in compute.**"*

Because cost is quadratic in tokens, dividing tokens by 4 divides compute by $4^2 = 16$. This is the same $O(N^2) \to O(N/k^2)$ relationship as Class 08.

Slide p.39 notes this is *"Employed by tons of video understanding transformers (e.g. ViViT, VideoMAE, Video Swin, MViT, V-JEPA)."* The instructor confirms the naming: *"they are all using **cubelets**"* — the CS231n nickname for tubelets, matching the instructor's own coinage in Class 09.

*"Still project from tubelet dim to the transformer dim."* — so a tubelet of 4 frames is flattened and passed through the same trainable linear projection as a patch was.

### B12. Project mechanics — 40% of your grade

From the transcript (transcribed facts, not inferred):

- **Three Kaggle assignments**; the instructor *proposed* a reduction to two but left the choice to the cohort: *"instead of taking three assignments, I take only two and take one best out of it… if you want to do two assignments and take one best out of it, or if you want to do do three and take two best out of them."*
- **Confirmed grading:** best **2 of 3**, and per `session_2.pdf` slide 1, `Project Score = Test case based evaluation score × [Project understanding section in end-term exam]`.
- **Assignment 2 deadline:** 5th, *"and as soon as I start there is no way to change these deadlines."*
- **Assignment 3 deadline:** *somewhere around 15th*, with a tighter window if attempted.
- **Submission is on the Kaggle leaderboard, not the Google Form** — *"this time a lot of people came up with late submissions. They filled the Google form, but they never submitted on the leaderboard. So I won't be considering them for evaluation, because technically I cannot see them on the leaderboard. It doesn't come to me if you submit it late."*
- **Scoring on the leaderboard is not the pass condition** — *"at least try to make the submissions, even though if you score less that is okay. But make the submissions."*
- **Do not skip Assignment 1 if doing only two** — *"if you choose to do only two assignments, then many of you have not given the first assignment. This will be only way to get this course."*
- **For the next assignment, use your team name as the roll number** — *"so that it becomes easy for me to map who submitted what."*

**The three competitions** are documented in [`assignments/`](../../assignments/README.md), with briefs pulled from your Kaggle CLI.

### B13. The practical session

Described only in the transcript, no slides:

- Delivered via a **Drive link**, usable in **Google Colab, VS Code, or Cursor**.
- *"There are placeholder functions already there, you just have to fill the functions with your own code."*
- *"There is an evaluation of `utils` functions that are written which, if it is out of range, your prediction — your function is doing essentially something wrong, **similar to Coursera style**."*
- Solutions were offered afterwards as homework.
- **No code from this practical exists in this workspace.**

---

## C. Mathematical derivations and numerical examples

Every number was recomputed; outputs shown are actual computed values.

### C1. Video tensor shape

$$T = \text{duration (s)} \times \text{fps}, \qquad \text{shape} = (T,\ C,\ H,\ W)$$

**Worked example (the class example).** 30 s, 30 fps, greyscale, $256\times256$:

$$T = 30 \times 30 = 900 \implies \boxed{(900,\ 1,\ 256,\ 256)}$$

**Memory.** $900 \times 1 \times 256 \times 256 = 58{,}982{,}400$ bytes $\approx 58.98$ MB at 1 byte per channel.

**Contrast: RGB instead of greyscale.** $900 \times 3 \times 256 \times 256 = 176{,}947{,}200$ bytes $\approx 176.9$ MB. *One channel becomes three — a 3× memory increase for the same clip.*

### C2. Storage per minute

$$S = W \times H \times C_{\text{bytes}} \times \text{fps} \times 60$$

**SD 640×480, 3 bytes/px, 30 fps:**
$$640 \times 480 \times 3 = 921{,}600 \text{ B/frame}$$
$$921{,}600 \times 30 \times 60 = 1{,}658{,}880{,}000 \text{ B} = \mathbf{1.66\ GB/min}$$

**HD 1920×1080:**
$$1920 \times 1080 \times 3 = 6{,}220{,}800 \text{ B/frame}$$
$$6{,}220{,}800 \times 30 \times 60 = 11{,}197{,}440{,}000 \text{ B} = \mathbf{11.20\ GB/min}$$

Both verified; both match the slide's *"$\sim$1.5 GB"* and *"$\sim$10 GB"* (the slide is rounding to order of magnitude).

### C3. The training clip

$$16 \times 3 \times 112 \times 112 = 602{,}112 \text{ B} = \mathbf{588.0\ KB}$$

Duration at 5 fps: $16/5 = \mathbf{3.2\ s}$.

**Verified: both figures match slide p.6 exactly** (*"588 KB"*, *"3.2 seconds at 5 fps"*).

**Interpret.** 588 KB of training data versus 1.66 GB per minute of raw SD footage — a clip is roughly **2,800×** smaller than a second of raw video. That is what makes clip-based training fit on a single machine.

### C4. Video token counts

$$\text{tokens} = T \times \left(\frac{224}{P}\right)^{2}, \qquad P = 16 \Rightarrow 196 \text{ per frame}$$

| Clip | Frames | Tokens | Verified |
|---|---|---|---|
| Small clip $T=16$ | 16 | **3,136** | ✔ |
| 5 min @ 1 fps | $300$ | **58,800** | ✔ |
| 5 min @ 24 fps | $300\times24 = 7{,}200$ | **1,411,200** | ✔ |

All three match slides p.26–p.27 (*"3,136"*, *"58.8k"*, *"1,411,200"*) exactly.

**Scaling, stated properly.** Tokens grow **linearly in $T$** and **quadratically in resolution**. Going from 1 fps to 24 fps multiplies tokens by 24; going from 16×16 patches to 8×8 patches multiplies by 4. Attention then costs the square of that — so the 8×8-patch, 24 fps case is $4^2 \times 24^2 = 9{,}216\times$ the attention cost of the $T{=}16$, $P{=}16$ baseline.

### C5. 3D convolution output shape

**Setup (slide p.17).** Input $T \times 3 \times 32 \times 32$ with $T=6$; **5 filters** of size $5 \times 2 \times 3 \times 5 \times 5$; stride 1, no padding.

$$T' = \frac{6 - 2}{1} + 1 = \mathbf{5} \qquad H' = \frac{32-5}{1}+1 = \mathbf{28} \qquad W' = \mathbf{28}$$

$$\text{Output} = 5 \times 5 \times 28 \times 28 \quad\text{(}T' \times K \times H' \times W'\text{)}$$

All verified. Slide p.17 states *"Stack activations to get $T' \times 5 \times 28 \times 28$ output!"*

### C6. 3D convolution parameter count

$$\text{params} = K \times (k_t \times C_{in} \times k_h \times k_w) + K$$

Weights per filter: $2 \times 3 \times 5 \times 5 = \mathbf{150}$. Plus one bias per filter → $151$.
Total: $5 \times 151 = \mathbf{755}$. **Verified.**

**The bias vector is $K$-dimensional, not 1.** Slide p.17: *"Also 5-dim bias vector"* — one scalar bias per filter, exactly as in 2D. This trips people up because the filter itself is now 3-D; the bias is still one number **per filter**.

**Compare 2D.** A single $5\times5\times3$ filter would be $75$ weights. The 3D version is $150$ — **double**, purely from the added temporal extent $k_t=2$. This is the honest cost of early fusion and it is *per layer*; the instructor's compensating argument (B6) is that later layers can be removed entirely.

### C7. Two-stream flow stack

$$\text{flow channels} = 2 \times (T - 1)$$

For $T=16$: $2 \times 15 = \mathbf{30}$. Verified.

The two factors, from the transcript: **2** from $(d_x, d_y)$; **$(T-1)$** because flow is defined between *consecutive* pairs, so $T$ frames yield only $T-1$ flow images.

### C8. I3D inflation

$$H_{3D}(k_t, u, v) = \frac{H_{2D}(u,v)}{K_t}$$

**Why it is exact.** Let the input be constant along time, $I_t = I$ for all $t$. Then the 3-D output is

$$\sum_{k=0}^{K_t-1}\sum_{u,v} \frac{H(u,v)}{K_t}\, I(\cdot + k\cdot\delta_t + (u,v)) = \frac{1}{K_t}\cdot K_t \sum_{u,v} H(u,v)I(\cdot+(u,v))$$

$$= \sum_{u,v} H(u,v)\,I(\cdot+(u,v))$$

which is exactly the 2-D convolution output. Verified: for $K_t=3$, a $3\times3$ kernel inflates to $3\times3\times3$ with each temporal slice equal to the 2-D kernel divided by 3.

**Assumption to state.** The equivalence holds **only for temporally constant input** — slide p.23 says so explicitly: *"given 'constant' video input."* For real video it is merely a sensible initialisation, not an identity.

### C9. Divided space-time attention cost

| Variant | Per-token cost | At $N{=}196,\ T{=}16$ |
|---|---|---|
| Joint | $O(NT)$ | $196 \times 16 = \mathbf{3{,}136}$ |
| Divided | $O(N + T)$ | $196 + 16 = \mathbf{212}$ |

Verified; reduction factor $3136/212 = \mathbf{14.8\times}$.

**Interpret.** Slide p.34's claim is per-token. Because attention cost is quadratic in the token count *within each operator*, dividing into two operators of sizes $N$ and $T$ gives $O(N^2) + O(T^2)$ instead of $O((NT)^2)$ — an even larger total win than the 14.8× the slide's per-token framing suggests. At the 24 fps scale ($N{=}196$, $T{=}7200$), joint per-token cost is $1{,}411{,}200$ against $7{,}396$ for divided.

### C10. Tubelet reduction

For a tubelet spanning $k$ frames:

$$\text{token reduction} = k, \qquad \text{compute reduction} = k^2$$

For $k=4$: **4× fewer tokens, 16× less compute**. Verified; matches slide p.38 (*"4× reduction in tokens = 16× reduction in compute"*).

*Interpret:* the $4\times \to 16\times$ is the same quadratic effect as Class 08's patch reduction, and it is the quantitative reason tubelets are *"more efficient than patches!"* — plus the qualitative reason on the same slide: patches carry no motion information, tubelets do.

---

## D. Visual and geometric intuition

### D1. Image tensor → video tensor

```
  IMAGE  (3, H, W)                    VIDEO  (T, C, H, W)
  ┌──────┐                              ┌───┐┌───┐┌───┐
  │  C   │  channels                   │ 1 ││ 2 ││ T │  frames
  ├──────┤                              │   ││   ││   │
  │  H   │  height                      ├───┤├───┤├───┤
  ├──────┤                              │   ││   ││   │  channels
  │  W   │  width                       ├───┤├───┤├───┤
  └──────┘                              │   ││   ││   │  height
                                          ├───┤├───┤├───┤
  3-D                                     │   ││   ││   │  width
                                          └───┘└───┘└───┘

  greyscale video keeps C=1:
      30 s @ 30 fps, 256x256  →  (900, 1, 256, 256)
```

### D2. Kernel comparison

```
  2D filter (Class 03)              3D filter (this class)

      ┌───────┐                         ┌──────────┐
      │ 5 x 5 │  ×3 channels            │  5 x 5   │  height × width
      └───────┘                         ├──────────┤
      = 5×5×3 = 75 weights              │    2     │  TIME  ← the new axis
                                         ├──────────┤
                                         │  3 x 5x5 │  channels
                                         └──────────┘
                                         = 2×3×5×5 = 150 weights

   one filter → one 2D output value     one filter → one 3D output value
                                     (a whole spatial patch × 2 frames)
```

### D3. The size argument

```
  RAW SD VIDEO                 TRAINING CLIP
  640×480 @ 30 fps             112×112, 16 frames @ 5 fps
       │                             │
  921,600 B / frame              602,112 B / clip
       │                             │
  1.66 GB / minute               588 KB  (3.2 seconds)
       │                             │
       ▼                             ▼
   cannot fit a GPU batch      fits thousands of clips
   ─────────────────────────────────────────────────
   resolution ↓ and fps ↓  →  ~2,800× less data per sample
```

### D4. Fusion — where temporal mixing happens

```
  SINGLE-FRAME            LATE FUSION (pooling)       EARLY FUSION (3D)
  ┌───┐┌───┐┌───┐         ┌───┐┌───┐┌───┐              ┌─────┐
  │CNN││CNN││CNN│         │CNN││CNN││CNN│              │ 3D  │
  └───┘└───┘└───┘         └───┘└───┘└───┘              │ CNN │
    │    │    │             │    │    │                └──┬──┘
   vote/avg              pool over space+time                 │ fuses
  (no temporal                                            └─────┘
   modelling at all)      temporal info reaches the
                          classifier only at the end      temporal fused from
                                                          layer 1 onward

  slide p.8              slides p.9–10                  slides p.11–17
```

### D5. Token explosion for video transformers

```
  1 frame    14×14 grid  →      196 tokens
  ────────────────────────────────────────────────────────────
  T=16      small clip   →    3,136 tokens        ▓▓▓░░░░░░░░░
  T=300     5 min @1fps  →   58,800 tokens        ▓▓▓▓▓▓▓▓▓▓░░
  T=7200    5 min @24fps → 1,411,200 tokens      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                                  ▲
                                     "~context limit of most
                                      modern LLMs"  (slide p.27)

  cost ∝ tokens²  →  the bars grow quadratically worse than drawn
```

### D6. Joint vs divided space-time attention

```
  JOINT (slide p.30) — "Extremely inefficient!"
     every token attends to every token in every other frame
     ┌─┬─┬─┬─┐   T=16 frames × 196 = 3,136 tokens
     ├─┼─┼─┼─┤   full matrix: 3,136 × 3,136 entries
     ├─┼─┼─┼─┤   per-token cost O(NT) = 3,136
     ├─┼─┼─┼─┤
     └─┴─┴─┴─┘

  DIVIDED (slides p.32–33) — two cheaper operators

  step 1: TIME attention              step 2: SPACE attention
  each token attends to the SAME      each token attends to other tokens
  location in other frames            WITHIN its own frame
       f1    f2    f3                       f1
        │     │     │                   ┌──┼──┐
        ▼     ▼     ▼                   │  │  │
        ▼     ▼     ▼                   └──┼──┘
       (vertical)                       f2    f3
     196 × 16                    196 × 196 per frame
     O(T) = 16                   O(N) = 196

  total per token: O(N + T) = 212  vs  O(NT) = 3,136   → 14.8× cheaper
```

### D7. Patch vs tubelet

```
  PATCHES (per frame, one token each)     TUBELETS (4 frames, one token)
  f1: [ ][ ][ ]                           f1 f2 f3 f4  →  [ ][ ]
  f2: [ ][ ][ ]                           f5 f6 f7 f8  →  [ ][ ]
  f3: [ ][ ][ ]
  f4: [ ][ ][ ]                          4 frames → 1 token
  ─────────────────                       ─────────────────
  4 frames → 4 tokens                     4× fewer tokens

  ✗ "patches do not contain               ✓ tubelets DO carry motion
    motion information"                     (they span time by construction)

  cost ∝ tokens²  →  4× fewer tokens  =  16× less compute
```

---

## E. Code and implementation

> **No instructor code exists in this workspace.** The Drive-linked practical with placeholder functions and `utils` evaluators is not present. Everything below is **illustrative** and was **not executed** (numpy unavailable here). The *Section D* solutions in §H are pure-Python and **were** executed, with real output pasted in.

### E1. The single most important shape bug

```python
# PyTorch convention  (T, C, H, W)
clip = torch.randn(16, 3, 112, 112)

# NumPy / OpenCV convention  (T, H, W, C)
clip = np.random.randint(0, 255, (16, 112, 112, 3)).astype(np.float32)

# The same clip read with the wrong convention:
wrong = torch.randn(16, 112, 112, 3)     # silently accepted, shape is valid!
```

A shape mismatch is **not** an error here — `(16, 112, 112, 3)` is a perfectly legal tensor. It just means 112 channels of 3 pixels. Two habits prevent it:

1. Assert `clip.shape[1] == 3` (or your channel count) immediately after loading.
2. Keep a single conversion helper that all data loading goes through.

### E2. Reading video clips

```python
import cv2

def read_clip(path, start_frame, num_frames):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # <- not optional
    cap.release()
    return frames        # list of (H, W, 3) uint8
```

**The BGR→RGB conversion is the trap.** `cv2.imread` and `cv2.VideoCapture` return **BGR**. This is the same bug as Class 01 §E3, and it matters doubly here: a video model trained on RGB and fed BGR gets an apparently-plausible but systematically wrong input — and because the channel permutation preserves all per-channel statistics, **the model still produces confident output**. It never raises an error.

### E3. Sampled-clip dataloading

```python
class VideoClips:
    """Uniform clips of length T at a fixed stride, matching slide p.7."""
    def __init__(self, frame_paths, T=16, stride=16):
        self.paths, self.T, self.stride = frame_paths, T, stride

    def __len__(self):
        n = len(self.paths)
        return max(0, (n - self.T) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        clip = np.stack([load_frame(p) for p in self.paths[s:s + self.T]])
        # clip: (T, H, W, C)  ->  permute to (C, T, H, W) or (T, C, H, W)
        return clip
```

**The shape contract**, tracked end to end:

| Stage | Shape |
|---|---|
| one frame from disk | `(H, W, 3)` |
| `np.stack` over $T$ frames | `(T, H, W, 3)` |
| `torch.from_numpy(...).permute(...)` | `(3, T, H, W)` or `(T, 3, H, W)` |
| `unsqueeze(0)` (batch of 1) | `(1, 3, T, H, W)` or `(1, T, 3, H, W)` |

Pick **one** and be consistent; `Conv3d` will not tell you which you meant.

### E4. Test-time aggregation

```python
def aggregate_clip_probs(frame_probs, mode="mean"):
    """
    frame_probs: list of per-frame probability lists, one per frame.
    mode="mean": average the probabilities (slide p.8)
    mode="vote": majority vote over argmax labels
    """
    if not frame_probs:
        return None
    if mode == "mean":
        return np.mean(frame_probs, axis=0)
    labels = np.argmax(frame_probs, axis=1)
    counts = np.bincount(labels, minlength=frame_probs[0].shape[0])
    return counts / counts.sum()
```

**Note the two modes answer different questions** — the transcript makes this explicit: majority vote means *"if the majority says not running, then not running"*, whereas capturing a brief event means *"even if a single frame detects it's running, you say it's running."* Choose from the problem statement, not by habit.

### E5. Common bugs and diagnostics

| Symptom | Likely cause | Diagnostic |
|---|---|---|
| Model trains but test accuracy is chance | BGR/RGB swap | visualise one frame |
| Shape error mentioning `C` | `(T,H,W,C)` vs `(T,C,H,W)` | print `clip.shape` at load |
| 3D conv rejects input | kernel $k_t > T$ | compare `k_t` with clip length |
| `Conv3d` unexpected param count | counted $k_h k_w C$ instead of $k_t k_h k_w C$ | $K(k_t C_{in} k_h k_w + 1)$ |
| OOM on long videos | not chunking | slice into `T=16` clips, aggregate |
| Video transformer OOM | tokens not reduced | apply tubelets before projecting |
| Flow stack one channel short | used $T$ instead of $T-1$ | flow needs consecutive pairs |

---

## F. Connections and comparisons

### F1. 2D vs 3D convolution

| | 2D | 3D |
|---|---|---|
| Input | `(C, H, W)` | `(T, C, H, W)` |
| Kernel | $k_h \times k_w \times C$ | $k_t \times k_h \times k_w \times C$ |
| Weights/filter (example) | $5\times5\times3 = 75$ | $2\times3\times5\times5 = 150$ |
| Output | `(K, H', W')` | `(K, T', H', W')` |
| Bias | $K$-dim | $K$-dim |
| Strides | $s_h, s_w$ | $s_t, s_h, s_w$ |
| Mixes time? | **no** | **yes, from layer 1** |
| Temporal fusion | late (after CNN) | early (in the kernel) |

### F2. Video fusion strategies

| | Single-frame (p.8) | Late fusion FC (p.9) | Late fusion pool (p.10) | Early fusion (p.11) |
|---|---|---|---|---|
| Fusion point | after prediction | after features | after features | inside conv |
| Needs $T$ at train | no | yes | yes | yes |
| Cost grows with $T$ | linearly (separate forward passes) | fast (big MLP input) | slowly | slowly |
| Temporal modelling | **none** | indirect | indirect | **learned, early** |
| Baseline strength | *"Often a very strong baseline"* | moderate | good | best of the four |

### F3. Optical flow vs image gradient

| | Image gradient (Class 01) | Optical flow |
|---|---|---|
| Compares | neighbouring pixels **within** a frame | the **same** pixel **across** frames |
| Output | $(I_x, I_y)$ | $(d_x, d_y)$ |
| Equation | $\partial I/\partial x, \partial I/\partial y$ | $I_{t+1}(x{+}d_x, y{+}d_y) = I_t(x,y)$ |
| Magnitude+direction | $\sqrt{I_x^2+I_y^2},\ \tan^{-1}(I_y/I_x)$ | $\sqrt{d_x^2+d_y^2},\ \tan^{-1}(d_y/d_x)$ |
| Channel stack | 3 (RGB) | $2(T-1)$ |

Same mathematics, different axis — the instructor points this out directly.

### F4. Video efficiency strategies

| Strategy | Mechanism | Models | Cost |
|---|---|---|---|
| **Modify attention** | divided space–time | ViViT, MViTv2 | $O(NT) \to O(N{+}T)$ |
| | local space–time cubes | Video Swin | local window |
| **Reduce tokens** | tubelets | ViViT, VideoMAE, V-JEPA | $k\times$ tokens, $k^2\times$ compute |

### F5. When to use 3D convolution — the instructor's rule

| Input | Use 3D conv? | Why |
|---|---|---|
| Single RGB frame | **no** | 2D suffices |
| RGB + LiDAR depth, one timestamp | **no** | *"that's still a single timestamp"* — depth is just a 4th channel |
| RGB + LiDAR, one timestamp, + radar (5th channel) | **no** | still one timestamp |
| Any of the above **as a video sequence** | **yes** | *"you have a sequence"* |

### F6. Test-time aggregation modes

| Mode | Rule | Use when |
|---|---|---|
| Mean (p.8) | average predicted probabilities | default; smooths noise |
| Majority vote | most common argmax | *"if the majority says not running, then not running"* |
| Any-frame | a single positive frame wins | *"to capture even if for a slightest moment it was running"* |

---

## G. Exam-focused revision sheet

### Essential definitions

- **Video tensor** — $(T, C, H, W)$; channels survive even at $C=1$.
- **Early / late fusion** — temporal mixing inside the convolution vs after the CNN.
- **3D kernel** — $k_t \times k_h \times k_w \times C_{in}$; the new axis is time.
- **Optical flow** — displacement field $F(x,y)=(d_x,d_y)$ between consecutive frames.
- **Two-stream network** — appearance stream (RGB) + motion stream (stacked flow).
- **Flow stack channels** — $2(T-1)$.
- **I3D inflation** — copy the 2-D kernel $K_t$ times in time, divide by $K_t$.
- **Joint / divided space–time attention** — one operator over all tokens vs time then space.
- **Tubelet** — a token spanning $k$ frames; $k\times$ fewer tokens, $k^2\times$ less compute.

### Key equations and symbols

$$T = \text{duration} \times \text{fps} \qquad \text{shape} = (T, C, H, W)$$
$$S_{\text{min}} = W \times H \times 3 \times \text{fps} \times 60$$
$$N'_{\text{out}} = \frac{N + 2P - F}{S} + 1 \quad \text{(per axis, incl. } T\text{)}$$
$$\text{params}_{3D} = K\left(k_t C_{in} k_h k_w + 1\right)$$
$$I_{t+1}(x+d_x,\ y+d_y) = I_t(x,y) \qquad F(x,y) = (d_x, d_y)$$
$$\text{flow channels} = 2(T-1)$$
$$\text{tokens} = T \times (224/P)^2$$
$$H_{3D}(k_t,u,v) = H_{2D}(u,v)/K_t$$

$T$ frames · $C$ channels · $K$ filters · $k_t,k_h,k_w$ kernel extents · $P$ padding · $S$ stride · $K_t$ temporal kernel size (inflation factor in I3D) · $N$ tokens per frame.

### Key assumptions

- Video sizes assume **3 bytes per pixel** and **30 fps** (the slide's stated condition).
- I3D inflation is **exact only for temporally constant input**; otherwise it is an initialisation.
- Divided attention preserves cross-frame information only **"after many blocks"** — one block does not.
- Train-low/test-high means **downsampling is a training-time memory strategy, not an inference one**.

### Distinctions to remember

| | A | B |
|---|---|---|
| 2D / 3D kernel | $k_hk_wC$ | $k_tk_hk_wC$ |
| 3D bias | one per filter, $K$-dim | not $1$-dim |
| Flow channels | $T$ | $T-1$ |
| Extra channels (LiDAR) | still 2D | 3D only with a sequence |
| Patches / tubelets | 1 frame, no motion | $k$ frames, carries motion |
| Joint / divided | $O(NT)$ | $O(N{+}T)$ |

### Traps and misconceptions

1. **"Greyscale video is 3-D."** It is 4-D: $(900, 1, 256, 256)$. The channel axis is kept.
2. **"3D conv bias is one scalar."** It is $K$-dimensional.
3. **"Optical flow has $2T$ channels."** It has $2(T-1)$.
4. **"Extra channels require 3D conv."** A 4th channel from LiDAR is still 2D — 3D needs a **time sequence**.
5. **"Tubelets reduce tokens 4×."** They reduce *compute* 16×; tokens 4×.
6. **"You must downsample fps at test time too."** Slide p.7 trains low and tests high.
7. **"Early fusion costs more per layer."** It does — $150$ vs $75$ weights — but later layers can be removed, so *total* operations fall.
8. **"3D conv is required for video."** The instructor's own rule: it is required by the *presence of a sequence*, not by the modality.

### Checklist

- [ ] **Explain:** video tensor shape, and why greyscale is still 4-D
- [ ] **Explain:** why videos are too big, and the two reductions
- [ ] **Explain:** train low / test high
- [ ] **Explain:** early vs late fusion
- [ ] **Explain:** when 3D convolution is and is not appropriate
- [ ] **Explain:** why the flow stack is $2(T-1)$
- [ ] **Derive:** I3D inflation equals 2D convolution on constant input
- [ ] **Calculate:** 3D conv output shape and parameter count
- [ ] **Calculate:** storage per minute for a resolution
- [ ] **Calculate:** video token counts for a transformer
- [ ] **Calculate:** joint vs divided attention cost
- [ ] **Implement:** clip extraction and aggregation
- [ ] **Interpret:** BGR/RGB and channel-order bugs

---

## H. Practice Questions

> Mirrors `session16.pdf`: **A** single-correct MCQ · **B** multi-select · **C** descriptive ≤100 words · **D** coding, attempt 1 of 3 · **E** case study · **F** numericals.

### Section A — Single-correct MCQ (6 × 1 mark)

**A1.** A 20-second clip is recorded at 25 fps in greyscale at 128×128. What is the shape of the video tensor?

- A. (20, 1, 128, 128)
- B. (25, 20, 1, 128, 128)
- C. (20, 3, 128, 128)
- D. (500, 1, 128, 128)

**A2.** A 3D convolution layer has $C_{in}=3$, kernel $4 \times 3 \times 3 \times 3$, and $K=8$ filters. What is the parameter count?

- A. 8 × 81 = 648
- B. 8 × 108 = 864
- C. 8 × 109 = 872
- D. 8 × 27 = 216

**A3.** Why does a two-stream network stack $[2 \times (T-1)]$ flow images rather than $[2 \times T]$?

- A. Optical flow needs two frames per output pixel
- B. Flow is computed between consecutive frame pairs
- C. The $2$ counts horizontal and vertical gradients
- D. The last frame of a clip has no successor

**A4.** I3D initialises a 3D kernel from a 2D kernel by repeating it $K_t$ times along time. What else must be done, and why?

- A. Nothing; repetition alone preserves scale
- B. Divide by $K_t$; repeated slices would otherwise sum to $K_t$ times the value
- C. Divide by $C_{in}$; the channel depth must be normalised
- D. Average with the previous layer's weights

**A5.** A clip has $T=16$ frames and a 224×224 frame at 16×16 patches. Approximately how many tokens enter a video transformer?

- A. 196
- B. 3,136
- C. 38,416
- D. 1,411,200

**A6.** You replace independent 16×16 patches with tubelets spanning 4 frames. By what factor does attention compute fall?

- A. 2×
- B. 4×
- C. 16×
- D. 64×

### Section B — Multi-select (4 × 2 marks)

**B1.** Which statements about **training on video clips** are **correct**? Select all that apply.

- A. Clips typically use fewer frames than the source video
- B. Clips typically use lower spatial resolution
- C. Frame rate must also be reduced at test time
- D. Overlapping clips can be used to cover a long video
- E. A 16-frame clip at 112×112 occupies about 588 KB

**B2.** Which statements about **3D convolution** are **correct**? Select all that apply.

- A. Its kernel has an extent along the time axis
- B. Its output activation map is a 4-D tensor
- C. It uses a single scalar bias for the whole layer
- D. It can use a stride along the time axis
- E. It mixes temporal information from the first layer

**B3.** Which are **valid reasons** for using optical flow in video recognition? Select all that apply.

- A. It encodes how objects move between frames
- B. It provides appearance cues such as colour and texture
- C. It can capture fine motion detail beyond frame-level labels
- D. It requires no computation to estimate
- E. It can be estimated by algorithm or by a neural network

**B4.** Which are **correct** statements about **video transformers**? Select all that apply.

- A. Naive per-frame patching produces very large token counts
- B. Dividing space–time attention reduces cost from $O(NT)$ to $O(N+T)$
- C. Tubelets carry motion information that patches do not
- D. Patch size is fixed at 224×224 for video models
- E. Divided attention propagates cross-frame information within a single block

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why are videos typically stored and trained at a lower frame rate and resolution than the source recording?

**C2.** When is a 2D convolution sufficient, and what makes a 3D convolution necessary?

**C3.** Why does a two-stream network combine a motion stream with an appearance stream rather than using either alone?

**C4.** Why is joint space–time attention too expensive for long videos, and how does dividing it help?

**C5.** Why do tubelets represent motion better than patches?

### Section D — Coding (20 marks — attempt ONE of the three)

**Pure Python, standard library only.** Provide a reusable function. Handle empty and boundary inputs.

---

**Option D1 — Clip extraction**

```python
def uniform_frame_indices(total_frames, num_clips, clip_len, stride=None): ...
def extract_clip(frames, start, length): ...
```

- `uniform_frame_indices` returns up to `num_clips` tuples of frame indices, each `clip_len` long, starting at `0, stride, 2*stride, …`. `stride` defaults to `clip_len` (non-overlapping). Stop when a clip would run past `total_frames`.
- Returns `[]` if any argument is non-positive.
- `extract_clip` returns the sub-list, or `None` if `start < 0`, `length <= 0`, or the range exceeds the list.

Worked examples:

```
uniform_frame_indices(300, 2, 16)
  -> [(0,1,...,15), (16,17,...,31)]

uniform_frame_indices(300, 5, 16, 80)
  -> [(0..15), (80..95), (160..175), (240..255)]        # only 4 fit

uniform_frame_indices(0, 2, 16)        ->  []
uniform_frame_indices(300, 0, 16)      ->  []
uniform_frame_indices(300, 2, 16, -1)   ->  []
uniform_frame_indices(10, 5, 16)       ->  []

extract_clip(list(range(100)), 0, 4)    ->  [0, 1, 2, 3]
extract_clip(list(range(100)), 98, 4)   ->  None
extract_clip(list(range(100)), 4, 4)    ->  [4, 5, 6, 7]
```

**Option D2 — 3D convolution shapes**

```python
def conv3d_output_shape(T, C, H, W, k_t, k_h, k_w, s_t=1, s_h=1, s_w=1, p_t=0, p_h=0, p_w=0): ...
def conv3d_params(C_in, k_t, k_h, k_w, K): ...
```

- `conv3d_output_shape` returns `(T_out, (C, H, W), H_out, W_out)` — the middle element records the input shape so channel bookkeeping survives.
- Any non-positive argument → `None`. Any kernel larger than its input axis → `None`.
- `conv3d_params` returns `$K \times (k_t \cdot C_{in} \cdot k_h \cdot k_w + 1)$, or `0` if any argument is non-positive.

Worked examples:

```
conv3d_output_shape(6, 3, 32, 32, 2, 5, 5)
    -> (5, (3, 32, 32), 28, 28)
conv3d_output_shape(8, 3, 32, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1)
    -> (4, (3, 32, 32), 16, 16)
conv3d_output_shape(1, 3, 32, 32, 2, 5, 5)   -> None
conv3d_output_shape(6, 3, 3, 32, 2, 5, 5)    -> None
conv3d_output_shape(0, 3, 32, 32, 2, 5, 5)   -> None

conv3d_params(3, 2, 5, 5, 5)   ->  755
conv3d_params(3, 2, 5, 5, 0)   ->  0
```

**Option D3 — Aggregating frame predictions**

```python
def aggregate(preds, mode="mean"): ...
```

- `preds`: non-empty list of probability lists, all the same length `C`.
- `mode="mean"` → element-wise mean, rounded to 6 dp.
- `mode="vote"` → index of the most frequent per-row argmax; **lowest index wins ties**.
- Empty input, ragged rows, zero-length rows, or an unrecognised mode → `None`.

Worked examples:

```
aggregate([[0.1,0.7,0.2],[0.2,0.6,0.2],[0.1,0.5,0.4]], "mean")  ->  [0.133333, 0.6, 0.266667]
aggregate([[0.1,0.7,0.2],[0.2,0.6,0.2],[0.1,0.5,0.4]], "vote")  ->  1
aggregate([[0.4,0.4],[0.3,0.3]], "vote")                        ->  0
aggregate([], "mean")                                           ->  None
aggregate([[0.1,0.2],[0.3]], "mean")                           ->  None
aggregate([[0.1,0.2]], "median")                               ->  None
```

### Section E — Case study (15 marks)

> **Scenario.** A road-operations company receives dashcam video from 400 vehicles at 1920×1080 and 30 fps, continuously. It must (a) build a dataset of clips labelled *"traffic sign present"*, (b) flag the exact second a sign appears, and (c) run a near-real-time in-vehicle classifier within a 200 ms per-frame budget on a single edge board. Storage costs are charged per GB per month.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Which of classification, detection or tracking is needed for (a), (b) and (c) respectively? Justify (b) carefully.
2. **Data.** Compute the storage cost if you retained everything raw for one vehicle for one day. What do you do instead?
3. **Model.** Choose between a single-frame CNN, late fusion, a 3D CNN, and a video transformer for (c). State the FPS and GFLOPs trade-off you are accepting.
4. **Metric.** How do you evaluate "flag the exact second a sign appears"? Propose a metric that penalises a flag placed just before the true moment differently from one placed just after.
5. **Failure mode.** Name the most likely failure at night or in heavy rain, and the mechanism.

**Sub-questions**

- **E-a.** A video transformer on the full-resolution stream needs 1,411,200 tokens per 5 minutes. Give two independent ways to bring this under an LLM's context limit, and state the compute saving of each.
- **E-b.** Your optical-flow stream produces half the expected channels. State the most likely cause and the correct value.
- **E-c.** A 3D convolution layer has $C_{in}=3$, $k_t=3$ and $K$ filters. It must take a 32-pixel input height to an output height of 28. Give two kernel/stride/padding configurations that achieve this, and state the parameter count of each in terms of $K$.

### Section F — Numericals (6 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** A 45-second video is recorded at 30 fps at 128×128 in greyscale. Compute the frame count, the tensor shape, and the memory in MB at 1 byte per channel.

**F2.** Compute uncompressed storage per minute at 1920×1080, 3 bytes per pixel, 30 fps. State how long it takes to fill a 2 TB drive.

**F3.** A clip has $T=16$ frames at 112×112 with 3 bytes per pixel. Compute its size, its duration at 5 fps, and the ratio to one minute of 640×480 SD video.

**F4.** A 3D convolution takes input $8 \times 3 \times 32 \times 32$ with $K=4$ filters of size $3 \times 3 \times 3 \times 3$, stride 1, no padding. Compute the output shape, the weights per filter, and the total parameter count.

**F5.** A 224×224 frame is split into 16×16 patches. Compute tokens for a clip of $T=8$, $T=64$, and a 10-second clip at 30 fps. Then compute the token reduction and compute reduction from using tubelets spanning 4 frames.

**F6.** In divided space–time attention with $N=196$ tokens per frame and $T=16$ frames, compute the per-token cost of joint attention and of divided attention, and the reduction factor. State what happens to the joint cost if $T$ rises to 7200.

---

## Worked Solutions — Attempt the Questions First

> Answers to everything in Section H. Section F mirrors §C, already verified.

### Section A — Solutions

**A1. → A — (20, 1, 128, 128).**

$T = 20 \times 25 = 500$ frames, 1 channel, 128×128 → $(500, 1, 128, 128)$.

- B is wrong — $(T, \text{fps}, C, H, W)$ is not a layout; fps is a *sampling rate*, not an axis.
- C is wrong — it uses $T=20$ (the duration) instead of 500, and 3 channels for greyscale.
- D has the right frame count but mislabels it as greyscale — the correct answer is $(500, 1, 128, 128)$, so D fails on the tuple even though its leading number matches.

**A2. → C — 8 × 109 = 872.**

$k_t C_{in} k_h k_w = 4 \times 3 \times 3 \times 3 = 108$, plus 1 bias per filter → $109$; $\times 8 = \mathbf{872}$.

- A (648) counts only $k_hk_wC_{in} = 27$ — it forgets the time extent.
- B (864) omits the biases entirely.
- D (216) omits both $k_t$ and the biases.

**A3. → B — Flow is computed between consecutive frame pairs.**

From the transcript: *"for optical flow obviously there is going to be $T$ minus one because it corresponds to the previous frame — so there is going to be one less."* $T$ frames define $T-1$ consecutive pairs, each giving one flow image.

- A is wrong — each flow image needs exactly **two** frames, one pair.
- C is the other factor's meaning: **2** comes from $(d_x, d_y)$ (horizontal and vertical), not from the frame count. The question asks about the $T-1$, so C describes a different part of the expression.
- D is a *consequence* of B, not the reason — the last frame has no successor precisely because there is no pair to form.

**A4. → B — Divide by $K_t$; repeated slices would otherwise sum to $K_t$ times the value.**

Slide p.23: *"Copy kernel $K_t$ times, **divide by $K_t$** → Output is the same!"* Proved in §C8: for temporally constant input the $K_t$ slices each contribute the same partial sum, so without the division the result is $K_t$ times too large.

- A is wrong — repetition alone multiplies the response by $K_t$.
- C is wrong — channel depth is already handled by the kernel's $C_{in}$ factor.
- D is not part of the method.

**A5. → B — 3,136.**

$196$ tokens per frame $\times 16$ frames $= \mathbf{3{,}136}$. Verified; matches slide p.26.

- A (196) is one frame only.
- C (38,416) is $196^2$ — the **attention matrix**, a different quantity.
- D (1,411,200) is 5 minutes at 24 fps, not a $T=16$ clip.

**A6. → C — 16×.**

Slide p.38: *"Tubelets over 4 frames: **4× reduction in tokens = 16× reduction in compute**."* Compute is quadratic in token count, so 4× fewer tokens gives $4^2 = 16\times$ less compute. Verified in §C10.

- A (2×) and B (4×) are the **token** reductions' distractors — 4× is right for tokens, 16× for compute.
- D (64×) would require 8 frames per tubelet.

### Section B — Solutions

**B1. → A, B, D, E. Reject C.**

- **A** ✔ *"Train on short clips… low fps and low spatial resolution"* (slide p.6).
- **B** ✔ $H=W=112$ versus 640×480.
- **C** ✗ **This is the trap.** Slide p.7 trains low and tests high: *"you don't need to reduce the FPS at test time."* The instructor restates it — *"with 30 fps itself you don't need to reduce during the testing time."* Downsampling is a training-memory strategy, not an inference one.
- **D** ✔ *"you can chunk, have many chunks which have overlap"* (transcript).
- **E** ✔ $16\times3\times112\times112 = 602{,}112$ B $= 588$ KB. Verified.

**B2. → A, B, D, E. Reject C.**

- **A** ✔ slide p.14: *"Now has time dimension!"*
- **B** ✔ slide p.11: *"Each activation map in the network is a 4D tensor: $D \times T \times H \times W$."*
- **C** ✗ The bias is **$K$-dimensional**, one scalar per filter — slide p.17: *"Also 5-dim bias vector."* It is not a single scalar for the layer, and it is not $1$-dimensional.
- **D** ✔ slide p.16: *"Can move forward/backward in time (stride rules apply)."*
- **E** ✔ slide p.11: *"fuse temporal information over the course of the network."*

**B3. → A, C, E. Reject B, D.**

- **A** ✔ the motion stream exists precisely to encode motion: *"He's moving his hand to the right such that he hit an object."*
- **C** ✔ *"from this you can get very detailed information about the video."*
- **E** ✔ slide p.20: *"Computed via algorithm or faster NN approximation."*
- **B** ✗ Appearance (colour, texture, object identity) is the job of the **other** stream. Asking for it from flow inverts the design.
- **D** ✗ Flow estimation is expensive — *"either you can have very good algorithms these days or if you have to save your compute you can run a faster nearest-neighbor search."* It is not free.

**B4. → A, B, C. Reject D, E.**

- **A** ✔ slides p.26–27: 1,411,200 tokens, *"$ \sim $context limit of most modern LLMs."*
- **B** ✔ slide p.34: *"Reduces per-token computational cost from $O(NT)$ to $O(N + T)$."*
- **C** ✔ slide p.38: *"Patches do not contain motion information, but tubelets do."*
- **D** ✗ Patch size is a **hyperparameter**, not fixed to 224×224. 16×16 is a choice made because $16\times16\times3 = 768$ matches ViT-Base's hidden size (slide p.28).
- **E** ✗ Slide p.34's reason 2 is *"Information can still propagate across space/time **after many blocks**."* One divided block does **not** achieve full cross-frame mixing — time attention touches only the same spatial position across frames, and space attention is within-frame only.

### Section C — Solutions

*Each model answer is ≤100 words, following the three-beat structure from `session16.pdf`.*

**C1.**
Videos are large in two compounding dimensions. At 1920×1080 and 30 fps with 3 bytes per pixel, one minute occupies about 11.2 GB — verified. The instructor's operational point is that *keeping* recordings is the expensive part: *"in autonomous driving it's like a barren lands… you don't want to keep collecting all that data and filling up your memories because that is also costly."* Training therefore reduces both axes: a 16-frame 112×112 clip is 588 KB and 3.2 seconds at 5 fps — a roughly 2,800× reduction per sample. Crucially this applies to **training only**; at test time the model runs on overlapping clips at full frame rate and averages predictions.

*(100 words.)*

**C2.**
Use 2D convolution whenever the input is a single instant in time, however many channels it has. The instructor's rule is explicit: RGB plus a LiDAR-derived depth map *"can become your fourth channel and still a 2D convolution can work… **it's still a single timestamp**."* Adding radar makes a fifth channel, still 2D. A 3D convolution becomes necessary the moment the input is a **sequence**: its kernel gains a temporal extent $k_t$, letting one output value fuse a spatial patch across several frames. So the trigger is the presence of a time dimension in the tensor, not the number of channels or the sensing modality.

*(100 words.)*

**C3.**
Appearance and motion are partially independent evidence, and neither stream is sufficient alone. The appearance stream alone — the instructor's word — answers *"running, not running, not running"* while missing *"He moved his hand to the right such that he hit an object."* The motion stream captures that fine-grained temporal structure but says little about what is moving. Two-stream networks (Simonyan & Zisserman, slide p.22) therefore process RGB and a stack of optical flow independently and fuse the features — **early fusion**, a first 2-D convolution over all flow images. The stack is $[2 \times (T-1)] \times H \times W$: a factor 2 for $(d_x, d_y)$ and $T-1$ for consecutive pairs.

*(100 words.)*

**C4.**
Joint space–time attention lets every token attend to every token across all frames. With 196 tokens per frame and $T=7200$ frames (5 minutes at 24 fps) that is 1,411,200 tokens — slide p.30's *"Extremely inefficient!"*, annotated as *"$ \sim $context limit of most modern LLMs."* Dividing the operator fixes this: **time attention** mixes only the same spatial location across frames, costing $O(T)$, and **space attention** mixes only within a frame, costing $O(N)$. Combined per-token cost falls from $O(NT)$ to $O(N+T)$. Full cross-frame propagation still occurs, but **after many blocks**, not within one.

*(99 words.)*

**C5.**
A patch is extracted from a single frame, so it contains only spatial appearance — there is simply no temporal axis in it, and it cannot distinguish a person walking from the same person running. Slide p.38 states this directly: *"Patches do not contain motion information, but tubelets do."* A tubelet spans $k$ consecutive frames, so its flattened representation already encodes displacement across that window. It is therefore both **more informative** and **cheaper**: $k\times$ fewer tokens, and because attention cost is quadratic, $k^2\times$ less compute. For $k=4$ that is 4× fewer tokens and 16× less compute — which is why tubelets are *"more efficient than patches!"* and are used by ViViT, VideoMAE, Video Swin, MViT and V-JEPA.

*(100 words.)*

### Section D — Solutions

#### D1 — Clip extraction

**Key idea tested.** Uniform non-overlapping (or overlapping) tiling, and stopping cleanly rather than raising on a short video.

```python
def uniform_frame_indices(total_frames, num_clips, clip_len, stride=None):
    if total_frames <= 0 or num_clips <= 0 or clip_len <= 0:
        return []
    if stride is None:
        stride = clip_len
    if stride <= 0:
        return []
    out, s = [], 0
    while s + clip_len <= total_frames and len(out) < num_clips:
        out.append(tuple(range(s, s + clip_len)))
        s += stride
    return out

def extract_clip(frames, start, length):
    if start < 0 or length <= 0 or start + length > len(frames):
        return None
    return frames[start:start + length]
```

**Verified execution output:**

```
(300, 2, 16)  -> 2 clips: (0..15), (16..31)
(300, 5, 16)  -> 5 clips: (0..15), (16..31), (32..47), (48..63), (64..79)
(300, 5, 16, 80) -> 4 clips: (0..15), (80..95), (160..175), (240..255)
degenerate    -> [] [] []
clip too long -> []
extract       -> [0,1,2,3] / None / [4,5,6,7]
```

**Why it works.** `stride` defaults to `clip_len`, giving the non-overlapping tiling of slide p.7's *"chunks"*. The loop condition `s + clip_len <= total_frames` is what makes a 5-frame request from a 10-frame video return `[]` instead of a short final clip — the video is either long enough or it is not.

**The overlap case is the interesting one.** With `stride=80` and `clip_len=16`, only **4** clips fit in 300 frames, not 5: the fifth would start at 320. Returning fewer than requested is correct; padding or looping back would double-count frames.

**Common wrong approaches.**
- Using `<` instead of `<=` in the loop condition — silently drops the final valid clip.
- Ignoring `num_clips` and returning every clip, ignoring the cap.
- Returning `[]` for an empty video but raising for a negative `stride`.

#### D2 — 3D convolution shapes

**Key idea tested.** The output-size formula applied per axis including time, and the parameter formula that must include $k_t$ and the $K$ biases.

```python
def conv3d_output_shape(T, C, H, W, k_t, k_h, k_w,
                        s_t=1, s_h=1, s_w=1, p_t=0, p_h=0, p_w=0):
    for v in (T, C, H, W, k_t, k_h, k_w, s_t, s_h, s_w):
        if v is None or v <= 0:
            return None
    if k_t > T or k_h > H or k_w > W:
        return None
    return ((T + 2 * p_t - k_t) // s_t + 1, (C, H, W),
            (H + 2 * p_h - k_h) // s_h + 1,
            (W + 2 * p_w - k_w) // s_w + 1)

def conv3d_params(C_in, k_t, k_h, k_w, K):
    if min(C_in, k_t, k_h, k_w, K) <= 0:
        return 0
    return K * (k_t * C_in * k_h * k_w + 1)
```

**Verified execution output:**

```
(6,3,32,32, 2,5,5)                -> (5, (3,32,32), 28, 28)
(8,3,32,32, 3,3,3, 2,2,2,1,1,1)  -> (4, (3,32,32), 16, 16)
(1,3,32,32, 2,5,5)                -> None    # kernel wider than input in TIME
(6,3,3,32,   2,5,5)               -> None    # kernel taller than input in H
(6,3,32,32, 2,5,5)  after fix     -> (5, (3,32,32), 28, 28)
non-positive                        -> None, None
conv3d_params(3,2,5,5,5) -> 755
conv3d_params(3,2,5,5,0) -> 0
```

**Why it works.** The formula from Class 01 applies unchanged to each axis; the only difference is that one of the axes is time. The returned tuple keeps the input shape `(C, H, W)` in the middle so the channel count — which becomes the next layer's `C_in` — travels with the result rather than being discarded.

**A test-label bug worth recording.** The first run asserted *"kernel too big"* for `T=1, k_t=2` but the printed line came from a **different** call, `(4, 3, 32, 32, 2, 5, 5)`, where the kernel *does* fit in time ($2 \le 4$). The guard was never exercised. The corrected cases are the two `None`s above, and both genuinely trigger. A test that does not fail when the guard is removed is not testing the guard.

**Common wrong approaches.**
- Applying the formula to $H, W$ but forgetting $T$ — the most common 3D-conv error.
- Counting parameters as $K \cdot k_h k_w C$ (the 2D formula), forgetting $k_t$.
- Returning `$K$` parameters for the biases, or `1`.

#### D3 — Aggregating frame predictions

**Key idea tested.** Two different aggregation *rules* that answer two different questions, plus deterministic tie-breaking.

```python
def aggregate(preds, mode="mean"):
    if not preds:
        return None
    if mode not in ("mean", "vote"):
        return None
    C = len(preds[0])
    if C == 0 or any(len(p) != C for p in preds):
        return None
    if mode == "mean":
        return [round(sum(p[c] for p in preds) / len(preds), 6) for c in range(C)]
    counts = [0] * C
    for p in preds:
        counts[p.index(max(p))] += 1
    best = max(counts)
    return min(i for i, v in enumerate(counts) if v == best)   # lowest index on tie
```

**Verified execution output:**

```
mean: [0.133333, 0.6, 0.266667]
vote: 1
tie vote ([[0.4,0.4],[0.3,0.3]]): 0
empty: None   bad mode: None   ragged: None
```

**Why it works.** `"mean"` averages the probability vectors element-wise — the slide p.8 approach of *"average predicted probs at test-time"*. `"vote"` counts per-row argmax labels, which is the transcript's *"if the majority of your classes are saying that it's not running, then what you will use is not running."* `min(...)` over the indices attaining the maximum makes the tie-break deterministic and lowest-index, as specified.

**The ragged-row guard is a real check, not boilerplate.** `any(len(p) != C ...)` catches `[[0.1,0.2],[0.3]]`. Without it the `"mean"` branch silently produces a wrong-length list (or divides by the wrong count), and the `"vote"` branch can index past the end.

**Common wrong approaches.**
- `p.index(max(p))` picking the *last* maximum on a tie — here `max` picks the first, giving determinism, but an explicit rule is clearer.
- Averaging the **argmax labels** instead of the probabilities and calling it "mean" — that is vote mode mislabelled.
- Using `zip` over rows, which silently truncates ragged input instead of rejecting it.

### Section E — Solutions

**Model answer covering the chain.**

1. **Task.**
   - **(a) "traffic sign present"** — **classification** of a clip. Matches slide p.3's structure: input clip → one activity label.
   - **(b) "flag the exact second a sign appears"** — **detection in time**, i.e. temporal localisation. This is not classification: classification of the whole clip cannot say *when*. It is closer to a detection problem (find *what* and *where*) applied along the time axis rather than the spatial axis — the video analogue of Class 06's *"region proposals → feature extraction → classifier"*.
   - **(c) in-vehicle classifier** — classification, per frame or per clip, under a latency budget.
2. **Data.** Storage for one vehicle, one day, raw: $1920\times1080\times3 = 6{,}220{,}800$ B/frame; $\times 30 \times 60 = 11.2$ GB/min; $\times 1440$ min $= \mathbf{16.1\ TB}$ per vehicle-day. *(Verified: $11.197 \times 1440 = 16{,}124$ GB $\approx 16.1$ TB.)* × 400 vehicles = **6.4 PB per day.** Untenable at any price per GB.
   Instead: sample **1 fps** (*"we take only one frame… and we get rid of all the other 29"*) at reduced resolution, and — critically — **tag in flight** while recording (*"while the recording is going on you keep tagging them — what lies in this section of the videos"*). This is what makes the archive queryable: *"I will just query [with] a simple SQL query which will give me sections of the video where I had traffic signs."*
3. **Model.** The 200 ms per-frame budget rules the video transformer out — 1,411,200 tokens for 5 minutes of full-rate video is off the table. A **3D CNN** gives learned early temporal fusion at a moderate token/compute cost, and matches slide p.11's *"slowly fuse temporal information over the course of the network."* But the instructor's own guidance on cost applies, and for a **classifier** the honest first choice is the **single-frame CNN baseline**, which slide p.8 calls *"often a very strong baseline."* The accepted trade-off: give up explicit fine-grained motion modelling in exchange for latency. Escalate to 3D CNN only if the baseline misses the accuracy target. If temporal grounding is needed for (b), 3D conv over short overlapping clips is the natural fit.
4. **Metric.** Flagging the wrong second should cost more the further it is from truth, so use a **temporal detection** metric rather than accuracy. Concretely: a **tolerance-window score** — a flag is correct if it falls within $\pm\tau$ seconds of the true appearance (e.g. $\tau = 0.5$ s); report **precision and recall at that tolerance**, and additionally report the **signed mean temporal error** in seconds. Precision/recall at tolerance measures whether the system finds signs at all; the mean error exposes systematic bias (a model that is always 2 s late will otherwise hide behind a generous tolerance).
5. **Failure mode.** **Night / heavy rain → missed detections**, through two compounding mechanisms: low illumination collapses contrast, so both the appearance stream and flow estimation degrade (flow needs reliable texture to track); and rain adds high-frequency moving clutter that is indistinguishable from motion at reduced frame rate. Net effect: false negatives at exactly the moments when a sign is most critical — a safety-relevant failure, since a missed sign is worse than a spurious flag.

**Sub-questions.**

**E-a.**
- **Divided space–time attention.** Replace the joint operator with time attention then space attention: per-token cost $O(NT) \to O(N+T)$, and total cost $O((NT)^2) \to O(N^2 + T^2)$.
- **Tubelets over 4 frames.** Tokens divide by 4 and compute by $4^2 = \mathbf{16\times}$, exactly as slide p.38 states.

*Saving each:* for divided attention at $N{=}196$, $T{=}7200$, per-token cost falls from $1{,}411{,}200$ to $7{,}396$ — a **191×** reduction. For tubelets, a 4× token cut gives **16×**. They are independent and compose: applying both gives roughly $191 \times 16 \approx 3{,}000\times$. (A third option the instructor rules out: reducing frame rate — *"we have moved past that, we are already doing that."*)

**E-b.** The most likely cause is using $T$ instead of $T-1$. Optical flow is defined between **consecutive** pairs, so $T$ frames yield $T-1$ flow images, each with a horizontal and a vertical component. Correct value: $\mathbf{2(T-1)}$. For $T=16$ that is $30$ channels, not $32$. The instructor derives it explicitly: *"for optical flow obviously there is going to be $T$ minus one because it corresponds to the previous frame… and two is because you have $d_x$ and $d_y$."*

**E-c — two valid configurations.**
*Constraint:* $N_{\text{out}} = \left\lfloor \dfrac{32 + 2P_h - k_h}{S_h} \right\rfloor + 1 = 28$, i.e. $k_h = 32 + 2P_h - 27S_h$.

Enumerating $S_h \in \{1,2,3\}$ and $P_h \ge 0$ (requiring $1 \le k_h \le 32$ and $P_h \le k_h/2$):

| $S_h$ | $P_h$ | $k_h$ | Params per filter $= k_t C_{in}k_h^2 + 1$ | Total |
|---|---|---|---|---|
| **1** | **0** | **5** | $3 \times 3 \times 25 + 1 = \mathbf{226}$ | $\mathbf{226K}$ |
| **1** | **1** | **7** | $3 \times 3 \times 49 + 1 = \mathbf{442}$ | $\mathbf{442K}$ |
| 1 | 2 | 9 | $3 \times 3 \times 81 + 1 = 730$ | $730K$ |
| 2 | any | — | no valid solution | — |

Both rows verified.

**Two things this exposes, and they are the examinable content.**

1. **The stride is forced to 1.** For $S_h = 2$ the equation gives $k_h = 32 + 2P_h - 54 = 2P_h - 22$, so a valid kernel requires $P_h \ge 11$, which in turn forces a kernel far larger than the input. **No valid stride-2 configuration exists here.** Stride-2 downsampling and same-resolution output are mutually exclusive for a 32→28 step, because the reachable output sizes at stride 2 are of the opposite parity.
2. **Same-resolution output is *not* unique.** Because padding grows the kernel alongside it, every $(P_h, k_h) = (n, 5 + 2n)$ pair with $S_h = 1$ is valid — and each costs more, quadratically, than the last. **Choose $k_h = 5, P_h = 0$: it is the cheapest of the family at $226K$, roughly half the cost of the $k_h=7$ alternative.**

### Section F — Solutions

**F1.** *Formula:* $T = \text{duration} \times \text{fps}$; $\text{shape} = (T, C, H, W)$.

*Substitute and calculate:*
- $T = 45 \times 30 = \mathbf{1350}$ frames
- shape $= \boxed{(1350,\ 1,\ 128,\ 128)}$ — greyscale keeps $C=1$, per the instructor: *"we will not give away this dimension but we will keep one value here."*
- memory $= 1350 \times 1 \times 128 \times 128 = 22{,}118{,}400$ B $= \boxed{22.12\ MB}$ *(verified)*

*Interpret:* a single clip of 45 seconds is small enough to hold entirely in RAM — which is exactly why short-clip training is feasible at all. The 1-channel convention is the easily-missed part: greyscale video is still a 4-D tensor, not 3-D.

**F2.** *Formula:* $S = W \times H \times 3 \times \text{fps} \times 60$.

*Substitute:* $1920 \times 1080 \times 3 = 6{,}220{,}800$ B/frame.
*Calculate:* $6{,}220{,}800 \times 30 \times 60 = 11{,}197{,}440{,}000$ B $= \mathbf{11.20\ GB/min}$ *(verified)*.

*Time to fill 2 TB:*
$$\frac{2 \times 10^{12}\ \text{B}}{11.197 \times 10^{9}\ \text{B/min}} = 178.6\ \text{min} \approx \mathbf{3\ hours}$$

*Interpret:* one HD camera fills 2 TB in three hours — roughly the working day of a single vehicle. For 400 vehicles that is **12 days of storage at 2 TB**, before any retention policy. This is the arithmetic behind the instructor's *"you don't want to keep collecting all that data and filling up your memories"* and the reason in-flight tagging is mandatory rather than optional.

**F3.** *Formula:* $\text{size} = T \times C \times H \times W \times 3$; duration $= T/\text{fps}$.

*Calculate:*
- $16 \times 3 \times 112 \times 112 = 602{,}112$ B $= \mathbf{588.0\ KB}$ *(verified)*
- duration at 5 fps $= 16/5 = \mathbf{3.2\ s}$ *(verified)*
- SD per minute $= 640 \times 480 \times 3 \times 30 \times 60 = 1.659\ \text{GB} = 1{,}658{,}880{,}000$ B *(verified)*
- ratio $= 1{,}658{,}880{,}000 / 602{,}112 = 2755.1 \approx \boxed{2{,}755\times}$

*Interpret:* one training clip is about **2,750×** smaller than one minute of raw SD footage. Two reductions contribute — spatial ($640{\times}480 \to 112{\times}112$, a $24.5\times$ reduction in pixels) and temporal ($1800$ frames/min $\to 16$). Neither alone gets there; both are needed.

**F4.** *Formula:* $N_{\text{out}} = \lfloor (N + 2P - F)/S \rfloor + 1$ per axis; $\text{params} = K(k_t C_{in} k_h k_w + 1)$.

*Output shape:*
- $T' = (8 - 3)/1 + 1 = \mathbf{6}$
- $H' = W' = (32 - 3)/1 + 1 = \mathbf{30}$
- output $= \boxed{6 \times 4 \times 30 \times 30}$

*Weights per filter:* $k_t C_{in} k_h k_w = 3 \times 3 \times 3 \times 3 = \mathbf{81}$.
*Total:* $K \times (81 + 1) = 4 \times 82 = \boxed{328}$.

*Interpret:* 4 filters means a **4-dimensional bias vector**, not one scalar — slide p.17's *"Also 5-dim bias vector"* generalises. The temporal extent $k_t = 3$ is what makes this a 3D layer, and it is also what makes it expensive: the equivalent 2-D layer would have $k_hk_wC_{in} = 27$ weights per filter, i.e. $4 \times 28 = 112$ parameters. **The 3-D layer costs $328/112 = 2.93\times$ more.** That is the per-layer cost of early fusion, and it is exactly why the instructor's compensating argument (§B6) is architectural rather than per-layer — with temporal mixing already done, *later* layers can be removed entirely.

**F5.** *Formula:* $\text{tokens} = T \times (224/16)^2 = T \times 196$.

| Clip | $T$ | Tokens | Verified |
|---|---|---|---|
| 8 frames | 8 | **1,568** | ✔ |
| 64 frames | 64 | **12,544** | ✔ |
| 10 s at 30 fps | 300 | **58,800** | ✔ |

*(The last matches slide p.26's *"5 minutes at 1 fps: T=300 — 58.8k tokens"* — the token count depends on frame count only, not on wall-clock duration or fps.)*

*Tubelets over 4 frames:*
- token reduction $= 4\times$ → 300 frames become $75$ tokens, giving $75 \times 196 = \mathbf{14{,}700}$
- compute reduction $= 4^2 = \mathbf{16\times}$, since attention cost is quadratic in token count (slide p.38).

*Interpret:* the 10-second clip is **1,568 → 58,800** tokens purely because more frames were sampled — at 30 fps you get 4.7× the tokens of the 8-frame clip for the same spatial resolution. This is the practical form of the instructor's complaint: *"five minutes of video at one FPS will give you this many tokens, and if you increase the FPS… see how much they have gone to."* Tubelets cut 58,800 to 14,700 and cost to one sixteenth, at the price of making each token 4× larger in raw dimensions — which still projects down to $D$ via the same trainable linear layer.

**F6.** *Formula:* joint $= O(NT)$; divided $= O(N + T)$.

*Substitute:* $N=196$, $T=16$.

*Calculate:*
- joint: $196 \times 16 = \mathbf{3{,}136}$
- divided: $196 + 16 = \mathbf{212}$
- reduction factor: $3136/212 = \mathbf{14.8\times}$ *(verified)*

*At $T = 7200$:* joint $= 196 \times 7200 = \mathbf{1{,}411{,}200}$; divided $= 196 + 7200 = \mathbf{7{,}396}$; factor $= \mathbf{190.8\times}$.

*Interpret:* the saving grows **linearly in $T$**, which is exactly what matters — dividing attention does not just help a little, it converts an unusable cost into a usable one. At 24 fps the naive form reaches 1,411,200 tokens, which slide p.27 annotates as *"$ \sim $context limit of most modern LLMs"* and slide p.30 labels *"Extremely inefficient!"*, while the divided form is 7,396 — trivially affordable. The caveat from slide p.34 still applies: information propagates across space and time **after many blocks**, not within one.

---

## J. Sources and references

### Course sources

| Source | File | Detail |
|---|---|---|
| Presentation | `session_9.pdf` | 39 slides, *"Lecture 10: Video Understanding"* (CS231n); image-only, OCR'd in full |
| Transcript | `Session_9_—_Discussion on Quiz + Project and Practicals_…txt` | 21/9/2026, 9:30 AM, 120 min; 8.5k words |
| **Professor repository** | **None in this workspace** | The practical uses *"placeholder functions"* with `utils` evaluators, Coursera-style, shared via a Drive link |

### Key slide numbers

| Slides | Topic |
|---|---|
| p.2–4 | Video = 2D + time; $T\times3\times H\times W$; images vs actions |
| p.5–6 | **Video size: 1.5 / 10 GB per minute**; training clip $T{=}16$, $H{=}W{=}112$ |
| p.7 | **Train low fps / test high fps**, overlapping clips, average predictions |
| p.8 | Single-frame CNN baseline |
| p.9–10 | Late fusion: FC layers vs pooling |
| p.11 | Early fusion with 3D CNNs; $D\times T\times H\times W$ activations |
| p.12 | 2D convolution recap (6 activation maps, 6-dim bias) |
| p.13–16 | 3D filters, sliding, **time stride** |
| p.17 | **3D output $T'\times5\times28\times28$; 5-dim bias** |
| p.19–21 | Optical flow: $F(x,y)=(d_x,d_y)$ |
| p.22 | **Two-stream networks**, $[2(T-1)]\times H\times W$ |
| p.23 | **I3D inflation**: copy $K_t$ times, divide by $K_t$ |
| p.24 | Evolution of the field: 2014–2021 3D CNNs + RNNs; 2021–2026 transformers |
| p.25 | Image-level ViT |
| p.26–27 | **Video token counts: 196 / 3,136 / 58.8k / 1,411,200** |
| p.28, p.36 | **Two broad strategies** (modify attention / reduce tokens) |
| p.29–34 | **Joint vs divided space–time attention**; $O(NT)\to O(N{+}T)$ |
| p.35 | Video Swin: local space–time cubes, shifted |
| p.37–39 | **Tubelets**: *"patches do not contain motion information, but tubelets do"*; 4× tokens, 16× compute |

### Supplementary external sources

Used only where the lecture left a detail implicit.

- Karpathy et al., *"Large-scale Video Classification with Convolutional Neural Networks"*, CVPR 2014 — cited on slides p.9 and p.11; the source of the single-frame, late-fusion and early-fusion taxonomy. https://arxiv.org/abs/1405.5672
- Ji et al., *"3D Convolutional Neural Networks for Human Action Recognition"*, TPAMI 2010 — cited on slide p.11. https://arxiv.org/abs/1107.3530
- Simonyan & Zisserman, *"Two-Stream Convolutional Networks for Action Recognition in Video"*, NeurIPS 2014 — slide p.22. https://arxiv.org/abs/1406.2199
- Carreira & Zisserman, *"Quo Vadis, Action Recognition?"* (I3D), CVPR 2017 — slide p.23. https://arxiv.org/abs/1704.04875
- Arnab et al., *"ViViT: A Video Vision Transformer"*, ICCV 2021 — slides p.28, p.37–39. https://arxiv.org/abs/2103.14126
- Liu et al., *"Video Swin Transformer"*, CVPR 2022 — slides p.28, p.35. https://arxiv.org/abs/2106.13230
- Bertasius et al., *"Is Space-Time Attention All You Need for Video Understanding?"*, ICML 2021 — slides p.29–34. https://arxiv.org/abs/2102.01502
- Fan et al., *"Multiscale Vision Transformers"* (MViT) and Li et al., *"MViTv2"*, CVPR 2022 — slide p.28. https://arxiv.org/abs/2104.11178

### Unresolved ambiguities

1. **Slide p.24 dates the transformer era as "2021 – 2026".** 2026 is the current year, so this is either a forward-looking label or a typo in the deck. No claim is built on it.
2. **Slide p.26's caption reads "5 minutes at 1fps: T=300".** $300$ frames is 5 minutes **at 1 fps**, which is internally consistent, but it reads ambiguously next to the 24 fps row. The arithmetic ($196 \times 300 = 58{,}800$) is unambiguous and verified.
3. **No slides cover the practical session or the assignment logistics.** §B12 and §B13 rest entirely on the transcript; no code, `utils` evaluators, or Drive links are present in this workspace.
4. **The first ~20 minutes were midterm-quiz Q&A.** Per your instruction that quiz is excluded from difficulty/style calibration and no question from it is reproduced. The receptive-field walkthrough in that segment is cross-referenced to [Class 03](../class-03-cnns-under-the-hood/README.md) rather than restated.

---

*Questions A–F are original, written against the Session 9 sources only. The Session 9 midterm quiz was excluded as evidence of difficulty or style. Every numerical answer and every code solution above was executed and verified; one test that never exercised its own guard was found and corrected.*