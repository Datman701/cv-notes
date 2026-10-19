# Class 15 — Generative Computer Vision (Deep Generative Models)

> **Exam weight: Tier 1 for A/B/C and F.** This class builds one idea from the ground up: **model the data distribution $p(x)$, then sample from it.** It contains the only serious *maths* of the foundation-model block — the **ELBO**, the **reparameterization trick**, the **diffusion forward kernel**, and **noise prediction** — so expect a numerical or a derivation-flavoured MCQ. Every one of those formulae has a cheap, checkable number attached, and they are all re-derived below.
>
> **Sources:** `session_15_annotated.pdf` (40-slide annotated deck, has a text layer, *"Deep Generative Models / Generative Computer Vision"*) · `Session_15_—_Generative_Computer_Vision_….txt` (transcript).
> **Professor repository: none in this workspace.**

**Every §B concept carries an intuitive explanation *and* a concrete "▸ Walk through it".** The examiner's favourite discriminator in this class is *"what is fixed and what is learned?"* — in a VAE the encoder is learned; in a diffusion model the "encoder" is a **fixed** noise schedule. Every subsection ends by answering it.

---

## A. Lecture overview

**The one-sentence version.** A generative model learns $p(x)$ — the distribution your training images came from — so that **sampling** produces new, plausible $x$; the two great families this class studies are **VAEs** (learn a latent $\to$ decode) and **diffusion models** (destroy an image with noise, then learn to undo it), and diffusion is best understood as a **hierarchical VAE whose encoder is fixed**.

**The whole class in nine lines:**

```
  Discriminative   p(y | x)   image → label        (all of Classes 1–14)
  Generative       p(x | y)   label → image        (this class)

  Autoregressive   p(x)=∏ p(xi | x<i)   exact, but sequential/slow
  VAE              z ~ p(z);  decode p(x|z)
                   ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z))
  Hierarchical VAE z3 → z2 → z1 → x     (layout → parts → texture)

  Diffusion        FORWARD:  add noise (FIXED, no learning)
                   REVERSE:  predict the noise (LEARNED, one U-Net)
                   train:    pick t, add ε, predict ε, MSE

  The punchline     diffusion = hierarchical VAE with a frozen encoder
                    ⇒ no posterior collapse, but slow sampling
```

**The arc of the lecture.** (1) What it means to learn $p(x)$ and why it is useful (content, representation, art, restoration). (2) The landscape of six generative families. (3) The classific-vs-generative flip, and why independent pixels fail. (4) VAEs: latents, the ELBO, the reparameterization trick, hierarchies, posterior collapse. (5) Diffusion: forward process, diffusion kernel, reverse process, noise prediction, training and sampling. (6) Faster sampling and the Imagen case study.

---

## B. The ideas

### B1. Learning to generate data — what is actually being learned

**The intuition.** Discriminative learning models `p(y | x)`: given an image, which label. Generative learning models `p(x)`: *what images look like*. Train it on many examples (no labels needed — the data is its own supervision), then **sample** brand-new $x$. A good sample looks like the data but **is not a copy** of any training image.

**▸ Walk through it.** You never see the "true" distribution — you only hold samples from it. Concretely: if you ask for "a cat", there is some distribution over all cat-looking images; collecting a dataset is just drawing samples from it. A generative model tries to **fit** that distribution with a parametric function, then draws fresh samples. Classic examples on the slides: a generator producing images from random noise (GANs), and creating a plausible face from a mask or an edge map.

**The takeaway.** *Generative modelling = estimate $p(x)$ from samples, then sample. It is unsupervised in the sense that the training images are the supervision.*

---

### B2. Why generate? The four uses on the slides

**The intuition.** Generation is not just for making pretty pictures. A model that can *draw* a car must have internalised what cars *are* — and that knowledge is reusable.

**▸ Walk through it.**

- **Content generation — StyleGAN3.** A generator maps a noise vector $z$ to an image and is trained against a discriminator. StyleGAN3's contribution is being **alias-free**: fine details (hair, wrinkles, foam) move *with* the object rather than sticking to fixed pixel positions — which matters for video and animation.
- **Representation learning — DatasetGAN.** A generator that can draw cars and faces already "knows" their parts. Annotate only a **few dozen** generated images, train a small decoder on the generator's internal features, and you get an **unlimited labelled dataset** (image + segmentation mask) for free.
- **Artistic tools — GauGAN / SPADE.** Draw a **semantic map** (each colour a class: sky, mountain, snow, rock) and get a photorealistic image respecting that layout. The trick (**SPADE**) is that ordinary normalisation layers *wash out* the layout, so SPADE instead uses the map to predict a **per-pixel scale and shift** after every normalisation, letting the layout steer the generator at **every layer**.
- **Image-to-image — Palette.** Super-resolution (64×64 → 1024×1024), colorization, inpainting, restoration and outpainting are all treated as *one* problem: generate $y$ given an input $x$. The network and loss are **identical** for every task — only the (input, target) training pairs change, and the conditioning image is concatenated to the noisy image at every denoising step.

**The takeaway.** *Four uses: content (StyleGAN3, alias-free), labels (DatasetGAN), artistic control (GauGAN/SPADE), and any image-to-image task (Palette). Palette's point is that the *loss never changes* — only the data does.*

---

### B3. The landscape of six families

**The intuition.** All six model $p(x)$; they trade off **sample quality**, **training stability**, and **sampling speed**.

**▸ Walk through it.** The slide's pairs:

| family | strength | weakness |
|---|---|---|
| **GANs** | sharp & fast | unstable training (mode collapse) |
| **VAEs** | stable & fast | often blurry |
| **Autoregressive** | exact likelihood, expressive | slow **sequential** sampling |
| **Normalizing flows** | exact likelihood | restricted (must be invertible) |
| **Energy-based** | very flexible | costly MCMC sampling |
| **Diffusion** | high quality & **stable** | **many sampling steps** |

The historical ladder on the same slides: **VAE 2013 → GAN 2014 → PixelCNN 2016 → BigGAN 2019 → Imagen 2022**. And the "2022/2023: the year of generative modelling?" slide lists Imagen, Imagen Video, Parti, Phenaki (Google), DALL·E 2 (OpenAI), Stable Diffusion (open weights) — most are **diffusion**; Parti and Phenaki are token-based exceptions.

**The takeaway.** *Diffusion is the focus because it combines likelihood-style **stability** with GAN-level **quality**, and scales. The price is **many** sampling steps.*

---

### B4. The classific-vs-generative flip, and why pixels alone fail

**The intuition.** Classification asks "given this image, what label?". Generation asks the reverse: "given this label, make an image". The output space explodes from a few class numbers to **~200,000** correlated numbers.

**▸ Walk through it.** A label $y$ has $K$ possible values, so a softmax over $f(x)$ expresses $p(y|x)$ completely — a few numbers of uncertainty. Now flip it: to generate a $256\times256\times3$ image you must produce $256\times256\times3=\mathbf{196{,}608}$ numbers (the slide calls it "~200,000"), and they are **strongly correlated** (neighbouring pixels, object shapes). Predicting each pixel **independently** gives the *average* image — a blurry blob. The uncertainty here is not "which label" but a whole distribution over high-dimensional data.

**The takeaway.** *Generation is hard because $x$ is high-dimensional and correlated; independent pixels collapse to the mean. That is why we need either autoregression or latent variables.*

---

### B5. Autoregressive models

**The intuition.** Break one hard problem into many easy ones using the **chain rule**: predict one pixel (or token) at a time, each conditioned on everything already produced.

**▸ Walk through it.** $p(x\mid y)=\prod_i p(x_i\mid y, x_{<i})$. Each factor is just a **classification** over one pixel's value (e.g. 256 levels). Upside: **exact likelihood** and very expressive (PixelCNN for images, GPT for text). Downside: **sampling is sequential** — one network call *per dimension*, so a 196,608-pixel image needs ~196k forward passes.

**The takeaway.** *Autoregression is exact but slow to sample. The latent-variable approach exists precisely to generate all pixels in **parallel**.*

---

### B6. Latent variable models

**The intuition.** Behind each image are hidden factors the label does not capture — shape, angle, lighting, how broken the bar is. Model those as a latent $z$. Sampling then becomes: draw $z$, run the network once.

**▸ Walk through it.** Let the **prior** be simple, $p(z)=\mathcal N(0,I)$, and the **likelihood** also be simple, $p(x\mid z)=\mathcal N(\mu_\theta(z), \sigma^2)$ — but with the mean computed by a **deep network** of $z$. Because a different $z$ gives a different Gaussian mean, integrating over $z$ yields an **infinite mixture of Gaussians**: a very flexible $p(x)$. Compare with autoregression: here **all pixels are generated in parallel**, and their correlations come from **sharing the same $z$**. Drop $y$ and you get the **unconditional** version, where $z$ must capture everything — including which object it is — with no supervision: sample $z\sim\mathcal N(0,I)$ and decode with $\mu_\theta(z)$.

**The takeaway.** *Latent variable models generate in one shot; the correlations come from $z$, not from sequential conditioning. $p(z)$ is simple and chosen; $p(x|z)$ is simple but *deep* in $z$.*

---

### B7. Why maximum likelihood is intractable

**The intuition.** To train by maximum likelihood you need $p(x)=\int p(x\mid z)p(z)\,dz$, an integral over a very high-dimensional $z$ with a neural network inside. There is **no closed form**.

**▸ Walk through it.** The naive fix — estimate the integral by sampling $z$ from the prior — fails badly: for a given image $x$, almost every random $z$ decodes to something unrelated, so the estimate has **huge variance**. What you actually need is the **posterior** $p(z\mid x)$ — *which $z$ are likely for this image* — which is itself intractable. The slide's geometric picture is exactly this: in high-dimensional $z$-space, the per-image Gaussians sit in isolated islands; the gaps between islands decode to nothing.

**The takeaway.** *Both the marginal $p(x)$ and the posterior $p(z|x)$ are intractable; so we approximate the posterior instead of computing it. That approximation is what a VAE learns.*

---

### B8. VAEs and the ELBO

**The intuition.** Add an **encoder** that guesses the posterior: $q(z\mid x)$ gives a mean and variance for each image. Then optimise a **lower bound** on $\log p(x)$ instead of $\log p(x)$ itself.

**▸ Walk through it.** Encoder $q_\phi(z\mid x)$: image → $(\mu, \sigma)$. Decoder $p_\theta(x\mid z)$: $z$ → image. The objective is the **evidence lower bound**:

$$\mathrm{ELBO}=\mathbb E_{q(z\mid x)}\big[\log p(x\mid z)\big]-\mathrm{KL}\big(q(z\mid x)\,\Vert\,p(z)\big).$$

The first term says *reconstruct well*; the second says *keep codes close to the $\mathcal N(0,I)$ prior*. Both $q$ and $p$ are optimised **jointly** w.r.t. the ELBO. The bound is **tight when $q(z\mid x)$ equals the true posterior**, and the gap is exactly $\mathrm{KL}(q(z\mid x)\Vert p(z\mid x))$.

**The takeaway.** *ELBO = reconstruction − KL-to-prior. It makes the intractable log-likelihood optimisable, and the $\mathcal N(0,I)$ prior is what makes the latent space samplable and gap-free.*

---

### B9. The reparameterization trick

**The intuition.** $\mathbb E_{q(z\mid x)}[\cdot]$ needs a gradient that flows *through* a random sample — but you cannot backpropagate through "draw from a distribution". Rewrite the draw so randomness is a separate, parameter-free input.

**▸ Walk through it.** Instead of $z\sim\mathcal N(\mu,\sigma^2)$, write

$$z=\mu+\sigma\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I).$$

Now $\varepsilon$ is the only random part and it carries no parameters, so gradients flow to $\mu$ and $\sigma$ through ordinary backprop. Example: $\mu=0.5$, $\sigma=0.2$, $\varepsilon=1.5$ gives $z=0.5+0.2(1.5)=\mathbf{0.8}$.

**The takeaway.** *Reparameterization splits sampling into "parameterised mean/scale" × "fixed noise", which is the only reason VAEs train end-to-end with backprop.*

---

### B10. Hierarchical VAEs

**The intuition.** A single Gaussian $z$ is too simple to describe image structure. **Stack** latents so each level models a different scale.

**▸ Walk through it.** Make both the inference model and the generative model hierarchical: $z_3 \to z_2 \to z_1 \to x$ (and $x \to z_1 \to z_2 \to z_3$). Higher levels capture **global layout**, middle levels **object parts**, low levels **fine texture**. Examples: Ladder VAE, NVAE. This is the picture to keep in mind, because **a diffusion model has exactly this chain structure** — with $T$ levels instead of three.

**The takeaway.** *Hierarchies of latents give better likelihoods than a flat VAE. Remember the chain $z_3\!\to\!z_2\!\to\!z_1\!\to\!x$ — diffusion reuses it.*

---

### B11. VAE failure modes — posterior collapse

**The intuition.** The ELBO is an **information bottleneck**: it *pays* to ignore $z$ if the decoder is strong enough.

**▸ Walk through it.** If the decoder can model $x$ alone, the cheapest way to lower the loss is to set $q(z\mid x)\approx p(z)$ — making the **KL term zero** and shutting $z$ down entirely. That is **posterior collapse**: the latents transmit no information. It is a *dreaded bad local minimum*. Remedies: **KL annealing / warm-up** and **free bits**. Large models also make optimisation hard for the same reason.

**The takeaway.** *Posterior collapse is the VAE's signature failure. Diffusion sidesteps it entirely by **not learning an encoder at all** — the "encoder" is a fixed noise schedule.*

---

### B12. Diffusion — the core idea and the forward process

**The intuition.** Destroying an image with noise is easy and needs no learning. Undoing a *tiny* bit of noise is a simple, learnable task. Chain many tiny undo steps and pure noise becomes a realistic image.

**▸ Walk through it.** **Forward diffusion (fixed):** each step slightly shrinks the signal and adds a little Gaussian noise,

$$x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t .$$

In DDPM, $T=1000$ and $\beta_t$ rises **linearly from $10^{-4}$ to $0.02$**. There are **no learnable parameters** in the forward process. **Only the reverse process has learnable parameters.**

**The takeaway.** *Forward = fixed noise-adding; reverse = the only learned part. If asked "what is trained?", the answer is: the reverse denoiser, never the forward schedule.*

---

### B13. The diffusion kernel — jump straight to any $t$

**The intuition.** Gaussians compose: many small noise steps collapse into **one** Gaussian step, so $x_t$ can be drawn directly from $x_0$ — no loop.

**▸ Walk through it.** Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$. Then

$$q(x_t\mid x_0)=\mathcal N\!\Big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\Big),
\qquad
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon .$$

Here $\bar\alpha_t$ is the **fraction of the original signal variance left at step $t$**: near 1 at the start, near 0 at $t=T$. Example: $\bar\alpha_t=0.81$, $x_0=2$, $\varepsilon=1$ gives $x_t=0.9(2)+\sqrt{0.19}(1)=2.2359$.

**The takeaway.** *Because the kernel is closed-form, training is cheap: pick a random $t$, draw one $\varepsilon$, form $x_t$ in a single shot. No sequential forward loop is ever needed.*

---

### B14. The reverse process and noise prediction

**The intuition.** The true reverse step $q(x_{t-1}\mid x_t)$ is intractable (it needs the distribution of *all* noisy images). But when each forward step adds only a little noise, the reverse step is **approximately Gaussian** — so the network only has to learn its **mean**.

**▸ Walk through it.** Start from pure noise $x_T$ and apply the same learned step $T$ times. One network $\mu_\theta(x_t,t)$ is **shared across all steps**; $t$ is an input so the network knows how much noise to expect, and the variance $\sigma_t^2$ is **fixed** (e.g. $\beta_t$), not learned. Ho et al. (DDPM) re-parameterise the mean with a **noise-prediction network**:

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big).$$

So instead of predicting the denoised image, the network **predicts the noise $\varepsilon$ that was added**, and the loss becomes weighted mean-squared error between true and predicted noise. This is equivalent to **denoising score matching**: $\varepsilon_\theta$ points (up to scale) along $-\nabla\log q(x_t)$.

**The takeaway.** *Reverse step = subtract predicted noise. Predicting $\varepsilon$ rather than the image mean turns the whole objective into "add noise, predict it, minimise MSE".*

---

### B15. The training objective and its weighting

**The intuition.** Diffusion is a hierarchical VAE, so it has a negative ELBO. Its terms have clean interpretations, and — surprisingly — throwing away the "correct" weights makes samples *better*.

**▸ Walk through it.** The negative ELBO decomposes as $L_T + \sum_{t}L_{t-1} + L_0$, where:

- **$L_T$** — no parameters; ≈ 0 **by design**;
- **$L_{t-1}$** — make each learned reverse step match the true reverse step; tractable because it is **conditioned on the clean $x_0$**;
- **$L_0$** — the final decoding to pixels.

The likelihood-correct weight $\lambda_t$ is **very large for small $t$**, so training would obsess over almost-invisible noise. DDPM sets $\lambda_t=1$, which **improves sample quality** (at a small cost in log-likelihood): the model spends effort on high-noise steps that decide **global structure**. The result is the simple objective: *add noise, predict it, minimise MSE*.

**The takeaway.** *Three terms by design; $L_{t-1}$ is the trainable core and is tractable only because it is conditioned on $x_0$. $\lambda_t=1$ trades likelihood for perceptual quality.*

---

### B16. Training, sampling, architectures, and the VAE connection

**The intuition.** Training and sampling are mirror images; the network is a U-Net because input and output are both images.

**▸ Walk through it.** **Training:** pick a real image, a random $t$ and random noise; build $x_t$ in one shot; teach the network to predict the noise. Every iteration trains one random step. **Sampling:** start from pure noise; at each step subtract the predicted noise (the learned mean) and add a little fresh noise $\sigma_t z$; after $T$ steps you have a new $x_0$. **Architecture:** a **U-Net** with ResNet blocks and self-attention, because input and output are the same size (image in, noise map out); skip connections keep fine detail, the bottleneck/self-attention captures global context. The time input $t$ is embedded with **sinusoidal positional embeddings or random Fourier features** and fed to residual blocks by spatial addition or **adaptive group normalisation**.

**Connection to VAEs.** Diffusion **is** a hierarchical VAE with the inference model **fixed** to "add noise":

| | VAE | Diffusion |
|---|---|---|
| inference model | learned $q(z\mid x)$ | **fixed** noise schedule |
| latent dimension | smaller than data | **same as the data** |
| ELBO | one term | **decomposed per timestep** (fast to train) |
| depth | a few levels | can be **extremely deep (infinite)** |
| objective | ELBO | **reweighted** ELBO |

Because there is no encoder to learn, there is **no posterior collapse**. The price is that latents are as big as the image, so sampling is costly.

**The takeaway.** *Diffusion = hierarchical VAE with a frozen encoder. Benefits: easier optimisation, no posterior collapse, per-timestep ELBO, infinite depth. Cost: slow sampling. Latent diffusion (Stable Diffusion) fixes the cost by diffusing in a compressed space.*

---

### B17. Faster sampling, and the Imagen case study

**The intuition.** The bottleneck is that each of 10–1000+ steps is a full U-Net forward pass, so one image costs **1000 network evaluations** where a GAN costs **one**.

**▸ Walk through it.** **Progressive distillation** trains a student to match **two teacher steps in one**; the student then becomes the teacher, **halving** the step count each round (e.g. $1024\to512\to\dots\to4$ → 4–8 steps). Other routes: deterministic samplers such as **DDIM** (tens of steps), fast ODE solvers (**DPM-Solver**), **consistency models** (1–2 steps), and running diffusion in a **smaller latent space**. **Imagen** (Google, 2022) is the case study: a **frozen T5-XXL** text encoder reads the prompt, then a **cascade** of *pixel-space* diffusion models generates 64×64 → 256×256 → 1024×1024. Its headline choices: **no latent space, no quantization** — it diffuses directly in pixels — plus **classifier-free guidance** (train with and without text; at sampling, extrapolate from the unconditional prediction toward the text-conditioned one, trading diversity for prompt adherence) and **dynamic thresholding** so strong guidance does not saturate pixel colours. The key empirical finding: **scaling the text encoder improved results more than scaling the image U-Net**. Evaluation: FID on COCO, CLIP score for alignment, and **DrawBench** human ratings, where Imagen was preferred over DALL·E 2.

**The takeaway.** *Distillation/deterministic solvers/latent diffusion attack sampling cost; Imagen attacks quality with a frozen big text encoder + a pixel-space SR cascade. The memorable claim: the text encoder mattered more than the U-Net.*

---

## C. Numbers worth knowing

All re-derived by execution.

### C1. The classification-to-generation scale jump

| quantity | value |
|---|---|
| label space (classification) | $K$ classes — a few numbers |
| image $256\times256\times3$ | **196,608** numbers ("~200,000") |
| a $256$-level pixel | $\log_2 256=\mathbf{8}$ bits |
| bits for the whole image if pixels independent | $196{,}608\times8=\mathbf{1{,}572{,}864}$ bits |

### C2. DDPM's fixed schedule

| quantity | value |
|---|---|
| steps $T$ | **1000** |
| $\beta_1$ | $10^{-4}=\mathbf{0.0001}$ |
| $\beta_T$ | $\mathbf{0.02}$ |
| schedule | **linear** |
| mean $\beta$ | $\mathbf{0.01005}$ |
| $\sum_t \beta_t$ | $\approx\mathbf{10.05}$ |
| $\bar\alpha_T$ (exact product) | $\approx\mathbf{4.04\times10^{-5}}$ |
| $\sqrt{\bar\alpha_T}$ (signal left at $T$) | $\approx\mathbf{0.00635}$ |

### C3. VAEs and the ELBO

$$\underbrace{\mathbb E_{q(z\mid x)}[\log p(x\mid z)]}_{\text{reconstruct}}-\underbrace{\mathrm{KL}(q(z\mid x)\Vert p(z))}_{\text{stay near }\mathcal N(0,I)}$$

| quantity | value |
|---|---|
| reparameterization | $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$ |
| $\mathrm{KL}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))$ per dim | $\tfrac12(\mu^2+\sigma^2-1-2\ln\sigma)$ |
| example $\mu=1,\ \sigma=0.5$ | $\mathbf{0.81815}$ |
| example $\mu=0,\ \sigma=1$ | $\mathbf{0}$ |
| bound tight when | $q(z\mid x)=p(z\mid x)$; gap $=$ their KL |

### C4. Sampling cost

| model | network evaluations per image |
|---|---|
| GAN | **1** |
| diffusion, no distillation | $\approx T=\mathbf{1000}$ |
| diffusion, progressive distillation | **4–8** |
| halvings from 1024 to 4 | $\log_2(1024/4)=\mathbf{8}$ rounds |

### C5. Imagen (slides 62–65)

| quantity | value |
|---|---|
| text encoder | **frozen T5-XXL** |
| resolution cascade | **64 → 256 → 1024** |
| latent space | **none** (pixel-space) |
| guidance | classifier-free + dynamic thresholding |
| key finding | scaling the **text encoder** beat scaling the **U-Net** |

---

## D. Pictures

### D1. The six families at a glance

```
   model p(x) ─────────────────────────────────────────────
     │
     ├─ Autoregressive   ∏ p(xi|x<i)      exact, sequential
     ├─ Normalizing flow  invertible map  exact, restricted
     ├─ VAE               z → decode       stable, blurry
     ├─ GAN               G vs D game      sharp, unstable
     ├─ Energy-based      E(x) ∝ p(x)      flexible, slow MCMC
     └─ Diffusion         denoise chain    quality + stable, slow
```

### D2. Hierarchical VAE chain

```
        z3   (global layout)
        │
        z2   (object parts)
        │
        z1   (fine texture)
        │
        x    (image)

   inference  q(z|x):  x → z1 → z2 → z3
   generative p(x,z):  z3 → z2 → z1 → x
   diffusion has this SAME chain, but with T levels
   and the "encoder" is fixed noise-adding.
```

### D3. Diffusion forward and reverse

```
   x0 ──┬──► x1 ──► x2 ──► ... ──► xT          FORWARD (fixed)
        │   add β1      add β2        pure noise   no parameters
        │   never loops: xt from x0 in one shot
        │        xt = √(ᾱt) x0 + √(1−ᾱt) ε
        │
        └── training: pick t, ε; predict ε (MSE)

   xT ──► xT-1 ──► ... ──► x1 ──► x0            REVERSE (learned)
         one shared U-Net μθ(xt, t), t as input
         each step: subtract predicted noise,
                    add a little fresh noise σt z
```

### D4. VAE encoder/decoder

```
   x ──► [ encoder q(z|x) ] ──► μ, σ        (learned)
                                 │
                    z = μ + σ ⊙ ε,  ε~N(0,I)   ← reparameterization
                                 │
   x̂ ◄── [ decoder p(x|z) ] ◄────┘          (learned)

   ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z))
   posterior collapse ⇔ q(z|x) → p(z), KL → 0, z ignored
```

### D5. Imagen

```
   prompt "a brain riding a rocketship…"
        │
        ▼
   [ frozen T5-XXL text encoder ]
        │  text embedding
        ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  base 64×64  │──►│  SR  256×256 │──►│  SR 1024×1024│
   │  diffusion   │   │  diffusion   │   │  diffusion   │
   └──────────────┘   └──────────────┘   └──────────────┘
        pixel space throughout (no latent, no quantization)
        classifier-free guidance + dynamic thresholding
```

---

## E. Code

Section D of the exam asks you to *implement* a concept. For a generative class the natural targets are the **forward noising kernel**, the **VAE ELBO pieces**, and **classifier-free guidance / the DDPM mean**. Full specifications and verified outputs are in **Section D**; reference implementations are in the worked solutions.

**Four one-liners worth memorising.**

```
x_t = √(ᾱ_t) x0 + √(1−ᾱ_t) ε            (forward kernel)
z   = μ + σ ⊙ ε                          (reparameterization)
ELBO = recon_loglik − KL(q(z|x) ‖ p(z))  (VAE objective)
μ_θ(x_t,t) = (x_t − β_t/√(1−ᾱ_t) · ε_θ) / √α_t   (DDPM mean)
```

---

## F. Comparisons

| | **Autoregressive** | **VAE** | **GAN** | **Diffusion** |
|---|---|---|---|---|
| density | exact | lower bound | implicit | (weighted) lower bound |
| sampling | sequential, slow | one shot, fast | one shot, fast | **many steps** |
| quality | good | blurry | **sharp** | **SOTA** |
| stability | stable | stable | **unstable** | **stable** |
| encoder | — | learned $q(z\|x)$ | — | **fixed noise schedule** |
| key failure | slow | posterior collapse / blur | mode collapse | slow sampling |
| headline | PixelCNN, GPT | Kingma & Welling 2013 | StyleGAN3 | DDPM, Imagen |

**Decision line.** Use **autoregression** when exact likelihood matters; a **VAE** for a fast, stable latent space and cheap reconstruction; a **GAN** when you need sharp images in one pass; **diffusion** when sample quality and stability dominate and you can afford many steps (or a distilled/latent variant).

---

## G. Exam-focused revision sheet

- **Discriminative vs generative:** $p(y|x)$ vs $p(x|y)$; generation predicts **196,608** numbers per $256^2$ image.
- **Independent pixels → blurry mean**; that is why we need autoregression or latents.
- **Autoregressive:** $p(x|y)=\prod_i p(x_i|y,x_{<i})$; **exact** likelihood, **sequential** sampling.
- **Latent variable:** $p(z)=\mathcal N(0,I)$, $p(x|z)$ Gaussian with deep-net mean ⇒ infinite mixture of Gaussians; one-shot sampling.
- **Intractability:** $p(x)$ and $p(z|x)$ both intractable; sampling $z$ from the prior has huge variance.
- **ELBO:** $\mathbb E_q[\log p(x|z)]-\mathrm{KL}(q(z|x)\Vert p(z))$; tight when $q=$ posterior; gap $=\mathrm{KL}(q(z|x)\Vert p(z|x))$.
- **Reparameterization:** $z=\mu+\sigma\odot\varepsilon$ — why VAEs backprop.
- **Hierarchical VAE:** $z_3\to z_2\to z_1\to x$; Ladder VAE, NVAE.
- **Posterior collapse:** $q(z|x)\to p(z)$, KL → 0; fixes = KL annealing / free bits; diffusion avoids it by **fixing the encoder**.
- **Forward diffusion:** $x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\varepsilon$; $T=1000$, $\beta:10^{-4}\to0.02$; **no parameters**.
- **Diffusion kernel:** $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$, $\bar\alpha_t=\prod(1-\beta_s)$; lets you jump to any $t$.
- **Reverse:** $q(x_{t-1}|x_t)$ intractable ⇒ approximate Gaussian; one shared $\mu_\theta(x_t,t)$; variance **fixed**.
- **Noise prediction:** $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta)$; loss = MSE on $\varepsilon$; $=$ denoising score matching.
- **Three terms:** $L_T$ (no params, ≈0) + $L_{t-1}$ (match true reverse, tractable via $x_0$) + $L_0$ (pixel decode); $\lambda_t=1$ improves samples.
- **U-Net:** same size in/out; time $t$ embedded by sinusoids/Fourier, injected via add or adaptive group norm.
- **Diffusion = hierarchical VAE** with **fixed** inference: no posterior collapse, latent dim = data dim, per-step ELBO, infinitely deep.
- **Fast sampling:** distillation (1024→…→4), DDIM, DPM-Solver, consistency models, latent diffusion.
- **Imagen:** frozen **T5-XXL** + 64→256→1024 pixel-space cascade; classifier-free guidance; dynamic thresholding; **scaling the text encoder beat scaling the U-Net**; DrawBench.

---

## H. Practice Questions

> **Format mirrors the exam:** A single-correct (1 mark), B multi-select (2), C descriptive ≤100 words (2), D coding (20, attempt one), E case study (15), F numericals (2.5). Attempt before reading the solutions.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** A generative model differs from a discriminative model because it models:
A. $p(y\mid x)$  
B. $p(x)$ (or $p(x\mid y)$)  
C. Only the decision boundary  
D. The loss function  

**A2.** In a VAE, the ELBO consists of:
A. A reconstruction term plus a KL term to the prior  
B. Two discriminator losses  
C. A single MSE loss  
D. A reward and a penalty  

**A3.** The reparameterization trick is needed because:
A. It speeds up convolution  
B. Gradients cannot flow through a random sampling operation otherwise  
C. It makes the prior uniform  
D. It removes the decoder  

**A4.** In DDPM's *forward* process:
A. The network learns to add noise  
B. There are no learnable parameters  
C. Only the encoder is trained  
D. The noise schedule is learned by gradient descent  

**A5.** DDPM trains the network to predict:
A. The clean image $x_0$  
B. The class label  
C. The noise $\varepsilon$ that was added  
D. The segmentation mask  

**A6.** Posterior collapse means:
A. The decoder fails to reconstruct  
B. The latents transmit no information ($q(z\mid x)\to p(z)$)  
C. The KL term explodes  
D. The prior is learned  

**A7.** A diffusion model can be described as:
A. A GAN with many discriminators  
B. A hierarchical VAE with a fixed inference model  
C. An autoregressive model over pixels  
D. A normalizing flow  

**A8.** Imagen's key empirical finding was that:
A. A larger U-Net always wins  
B. Scaling the text encoder improved results more than scaling the image U-Net  
C. Latent space is required for photorealism  
D. Guidance hurts image quality  

### Section B — Multi-select (5 × 2 marks)

**B1.** Which are true of autoregressive generative models? Select all that apply.
A. They use the chain rule over dimensions  
B. They give exact likelihood  
C. Sampling is sequential and slow  
D. They require an encoder $q(z\mid x)$  
E. PixelCNN and GPT are examples  

**B2.** Which are true of the VAE objective? Select all that apply.
A. It maximises a lower bound on $\log p(x)$  
B. It contains a reconstruction term  
C. It contains a KL term pulling $q(z\mid x)$ toward the prior  
D. The bound is tight when $q(z\mid x)$ equals the true posterior  
E. It has no way to avoid posterior collapse  

**B3.** Which are true of the diffusion forward process? Select all that apply.
A. $T$ is typically 1000  
B. $\beta_t$ rises linearly from $10^{-4}$ to $0.02$  
C. It has no learnable parameters  
D. $x_t$ can be derived from $x_0$ in closed form  
E. It requires a U-Net to add noise  

**B4.** Which are true of diffusion *sampling* and speed? Select all that apply.
A. Naively it needs $T$ network evaluations per image  
B. Progressive distillation halves the step count each round  
C. DDIM is a deterministic sampler that uses fewer steps  
D. Latent diffusion runs the process in a compressed space  
E. A GAN also needs about 1000 evaluations per image  

**B5.** Which are true of Imagen? Select all that apply.
A. It uses a frozen T5-XXL text encoder  
B. It uses a 64 → 256 → 1024 cascade  
C. It uses classifier-free guidance  
D. It diffuses in a latent space with quantization  
E. It adds dynamic thresholding to avoid colour saturation  

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Explain why $p(x)$ is intractable for a latent-variable model and why sampling $z$ from the prior does not fix it.

**C2.** Write the ELBO and say in one clause what each term encourages.

**C3.** State the reparameterization trick and explain in one sentence why it is necessary.

**C4.** Define posterior collapse, give its cause, and name two remedies.

**C5.** Explain the claim "a diffusion model is a hierarchical VAE with a fixed encoder" by listing two benefits and one cost.

### Section D — Coding (20 marks — attempt ONE of the three)

Implement in **pure Python (standard library only)**. Return `None` on malformed input.

**D1. The forward noising kernel.**
- `beta_schedule(T, start=1e-4, end=0.02)` → the length-`T` linear list $\beta_t = \text{start}+(\text{end}-\text{start})\,t/(T-1)$, $t=0,\dots,T-1$; `None` if `T <= 0` or `not (0 < start <= end < 1)`.
- `alpha_bar(betas, t)` → $\prod_{s=1}^{t}(1-\beta_s)$; `None` if `betas` is empty or `t < 1` or `t > len(betas)`.
- `q_sample(x0, abar_t, eps)` → $\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ element-wise; `None` if lengths differ, are empty, or $\bar\alpha_t\notin(0,1]$.

```
beta_schedule(1000)[0]                     ->  0.0001
beta_schedule(1000)[-1]                    ->  0.02
alpha_bar(beta_schedule(1000), 1000)       ->  4.0358297653756754e-05
q_sample([2.0], 0.81, [1.0])               ->  [2.2358898943540675]
q_sample([2.0], 0.0, [1.0])                ->  None
```

**D2. VAE ELBO pieces.**
- `kl_gaussian(mu, logvar)` → the closed form $\sum_i \tfrac12(\mu_i^2 + e^{l_i} - 1 - l_i)$ for $\mathrm{KL}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))$, where `logvar` holds $\ln\sigma^2$; `None` if lengths differ or are empty.
- `vae_loss(recon, target, mu, logvar)` → mean-squared reconstruction error plus `kl_gaussian`; `None` on malformed input.

```
kl_gaussian([0.0],[0.0])                   ->  0.0
kl_gaussian([1.0],[math.log(0.25)])        ->  0.8181471805599453
vae_loss([1.0,2.0],[1.0,1.0],[1.0],[math.log(0.25)])
                                           ->  1.3181471805599454
```

**D3. Guidance and the DDPM mean.**
- `classifier_free_guidance(eps_uncond, eps_cond, scale)` → $\varepsilon_u+s(\varepsilon_c-\varepsilon_u)$ element-wise; `None` if lengths differ/empty or `scale < 0`.
- `ddpm_mean(x_t, eps, beta_t, abar_t)` → $\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\big)/\sqrt{1-\beta_t}$ element-wise; `None` if lengths differ/empty or ranges are invalid.

```
classifier_free_guidance([0.2],[0.8],4.0)                ->  [2.6000000000000005]
classifier_free_guidance([0.2],[0.8],1.0)                ->  [0.8]
ddpm_mean([1.0],[0.5],0.01,0.81)                         ->  [0.9935092408614442]
```

### Section E — Case study (15 marks)

> **Scenario.** An e-commerce company wants to generate **product photography**: given a text prompt such as `"a blue ceramic mug on a white table, studio lighting"`, produce a 1024×1024 catalogue image. They must generate ~200,000 images, the GPU budget is fixed, and the brand will reject any image with **wrong colours** or **wrong object counts**. They are choosing between a **GAN**, a **VAE**, and a **diffusion model**, and between a **pixel-space** and a **latent-space** diffusion model.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Problem.** Why would a blurry average-looking image be unacceptable, and which family guarantees that failure?
2. **Task.** Write the task formally ($p(\cdot\mid\cdot)$) and say what the model must produce.
3. **Data.** What supervision does a diffusion model need, and does it require paired text–image labels for the *text* conditioning specifically?
4. **Model.** Justify diffusion over GAN and VAE for this job, then choose pixel-space or latent-space.
5. **Metric.** Name one automatic metric and one human-evaluation approach.
6. **Failure mode.** Give two failures specific to the "wrong colours / wrong counts" requirement and one mitigation each.

**Sub-questions**

- **E-a.** The team asks why the model needs 1000 steps when a GAN needs one. Explain and name two ways to cut the step count.
- **E-b.** Guidance scale is turned up until colours look oversaturated. Name the technique that addresses exactly this and say what it does.
- **E-c.** The company later wants to *edit* an existing photo (remove a logo, extend the canvas). Explain how one model can do all of this.

### Section F — Numericals (8 × 2.5 marks)

**F1.** An image is $256\times256\times3$. Compute the number of values it contains. If each pixel took one of 256 levels independently, compute the bits needed (8 bits/pixel-channel) and the bytes.

**F2.** DDPM uses a linear schedule $\beta_1=10^{-4}$ to $\beta_{1000}=0.02$. Compute the mean $\beta$, the approximate sum $\sum\beta_t$, and hence the approximate $\bar\alpha_{1000}=e^{-\sum\beta_t}$ and $\sqrt{\bar\alpha_{1000}}$.

**F3.** For a 1-dimensional latent with $\mu=1$, $\sigma=0.5$, compute $\mathrm{KL}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))=\tfrac12(\mu^2+\sigma^2-1-2\ln\sigma)$. Then compute it for $\mu=0,\sigma=1$.

**F4.** Using the reparameterization trick $z=\mu+\sigma\varepsilon$ with $\mu=0.5$, $\sigma=0.2$, $\varepsilon=1.5$, compute $z$.

**F5.** A VAE has mean-squared reconstruction error $0.5$ and a KL of $0.81815$ with the loss defined as $\mathrm{MSE}+\mathrm{KL}$. Compute the loss. If training halves the KL term, by how much does the loss fall?

**F6.** At some timestep $\bar\alpha_t=0.81$, the clean pixel value is $x_0=2.0$ and the sampled noise is $\varepsilon=1.0$. Compute $x_t$. Then compute the reconstructed $x_0$ if a perfect network predicts $\varepsilon=1.0$ (i.e. $x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon)/\sqrt{\bar\alpha_t}$).

**F7.** A 1000-step sampler takes 50 ms per step per image. Compute the time per image and for 200,000 images. Then compute both again after progressive distillation to 4 steps.

**F8.** Classifier-free guidance uses $\hat\varepsilon=\varepsilon_u+s(\varepsilon_c-\varepsilon_u)$ with $\varepsilon_u=0.2$, $\varepsilon_c=0.8$. Compute $\hat\varepsilon$ for $s=1$, $s=4$ and $s=0$. With margins, what does each case correspond to?

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — $p(x)$ (or $p(x\mid y)$).** Generative models the data distribution; discriminative models the label distribution.

**A2. → A — A reconstruction term plus a KL term to the prior.** That is exactly the ELBO.

**A3. → B — Gradients cannot flow through a random draw otherwise.** $z=\mu+\sigma\odot\varepsilon$ moves the randomness out of the parameter path.

**A4. → B — There are no learnable parameters.** The forward process is a fixed schedule; only the reverse denoiser is trained.

**A5. → C — The noise $\varepsilon$ that was added.** The re-parameterised mean makes this equivalent to denoising score matching.

**A6. → B — Latents transmit no information.** The decoder models $x$ alone, so $q(z|x)\to p(z)$ and the KL term is zero.

**A7. → B — A hierarchical VAE with a fixed inference model.** The inference ("encoder") is the noise schedule.

**A8. → B — Scaling the text encoder beat scaling the U-Net.** Imagen's central empirical claim.

### Section B — Solutions

**B1. → A, B, C, E. Reject D.** Autoregressive models have no latent encoder.

**B2. → A, B, C, D. Reject E.** Remedies (KL annealing, free bits) exist, and diffusion avoids collapse; the loss is not unfixable.

**B3. → A, B, C, D. Reject E.** Noise-adding is arithmetic, not learned — no U-Net is involved in the forward process.

**B4. → A, B, C, D. Reject E.** A GAN needs **one** evaluation per image; the 1000-evaluation cost is diffusion's specific problem.

**B5. → A, B, C, E. Reject D.** Imagen deliberately uses **no** latent space and **no** quantization.

### Section C — Solutions

**C1.** $p(x)=\int p(x\mid z)p(z)\,dz$ integrates over a very high-dimensional $z$ with a neural network inside the integrand, so there is no closed form. Estimating it by drawing $z$ from the prior fails because for a given image almost every random $z$ decodes to something unrelated, giving an estimator with enormous variance. What is needed is the posterior $p(z\mid x)$ — which $z$ are likely for *this* $x$ — but that is intractable too, so VAEs learn an approximation $q(z\mid x)$ instead.

**C2.** $\mathrm{ELBO}=\mathbb E_{q(z\mid x)}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\Vert p(z))$. The first term encourages the decoder to **reconstruct the input** accurately; the second encourages the encoder's posterior to **stay close to the $\mathcal N(0,I)$ prior** so the latent space is well-behaved and samplable.

**C3.** Write $z=\mu+\sigma\odot\varepsilon$ with $\varepsilon\sim\mathcal N(0,I)$, instead of $z\sim\mathcal N(\mu,\sigma^2)$. This is necessary because a gradient cannot be backpropagated through a random sampling operation; by making $\varepsilon$ a parameter-free input, gradients reach $\mu$ and $\sigma$ through ordinary backprop.

**C4.** Posterior collapse is a bad local minimum where the latents carry no information: $q(z\mid x)\approx p(z)$, so the KL term is zero and the decoder ignores $z$. It happens because the ELBO is an information bottleneck — if the decoder is powerful enough to model $x$ alone, ignoring $z$ is the cheapest way to reduce the loss. Remedies include **KL annealing / warm-up** and **free bits**.

**C5.** In diffusion, the inference model is the **forward noise schedule**, which is fixed rather than learned, giving the same chain structure as a hierarchical VAE. **Benefits:** there is no encoder to train, so **posterior collapse cannot occur**, optimisation is easier, and the ELBO decomposes per timestep (cheap training). **Cost:** the latent variables have the **same dimension as the data**, so sampling requires many expensive steps.

### Section D — Solutions

**D1.**

```python
import math

def beta_schedule(T, start=1e-4, end=0.02):
    if T is None or T <= 0 or start is None or end is None:
        return None
    if not (0.0 < start <= end < 1.0):
        return None
    return [start + (end - start) * t / (T - 1) for t in range(T)] if T > 1 else [start]

def alpha_bar(betas, t):
    if betas is None or t is None or len(betas) == 0:
        return None
    if t < 1 or t > len(betas):
        return None
    p = 1.0
    for b in betas[:t]:
        p *= (1.0 - b)
    return p

def q_sample(x0, abar_t, eps):
    if x0 is None or abar_t is None or eps is None:
        return None
    if len(x0) != len(eps) or len(x0) == 0:
        return None
    if not (0.0 < abar_t <= 1.0):
        return None
    a = math.sqrt(abar_t)
    b = math.sqrt(1.0 - abar_t)
    return [a * x + b * e for x, e in zip(x0, eps)]
```

Verified outputs: `beta_schedule(1000)[0] → 0.0001`; `[-1] → 0.02`; `alpha_bar(…,1000) → 4.0358297653756754e-05`; `q_sample([2.0],0.81,[1.0]) → [2.2358898943540675]`; `q_sample([2.0],0.0,[1.0]) → None`.

**D2.**

```python
import math

def kl_gaussian(mu, logvar):
    if mu is None or logvar is None or len(mu) != len(logvar) or len(mu) == 0:
        return None
    total = 0.0
    for m, lv in zip(mu, logvar):
        total += 0.5 * (m * m + math.exp(lv) - 1.0 - lv)
    return total

def vae_loss(recon, target, mu, logvar):
    if recon is None or target is None or len(recon) != len(target) or len(recon) == 0:
        return None
    if mu is None or logvar is None or len(mu) != len(logvar):
        return None
    mse = sum((r - t) ** 2 for r, t in zip(recon, target)) / len(recon)
    kl = kl_gaussian(mu, logvar)
    if kl is None:
        return None
    return mse + kl
```

Verified outputs: `kl_gaussian([0.0],[0.0]) → 0.0`; `kl_gaussian([1.0],[log 0.25]) → 0.8181471805599453`; `vae_loss([1.0,2.0],[1.0,1.0],[1.0],[log 0.25]) → 1.3181471805599454`.

**D3.**

```python
import math

def classifier_free_guidance(eps_uncond, eps_cond, scale):
    if eps_uncond is None or eps_cond is None or scale is None:
        return None
    if len(eps_uncond) != len(eps_cond) or len(eps_uncond) == 0:
        return None
    if scale < 0:
        return None
    return [u + scale * (c - u) for u, c in zip(eps_uncond, eps_cond)]

def ddpm_mean(x_t, eps, beta_t, abar_t):
    if x_t is None or eps is None or beta_t is None or abar_t is None:
        return None
    if len(x_t) != len(eps) or len(x_t) == 0:
        return None
    if not (0.0 < beta_t < 1.0) or not (0.0 < abar_t <= 1.0):
        return None
    coef = beta_t / math.sqrt(1.0 - abar_t)
    return [(x - coef * e) / math.sqrt(1.0 - beta_t)
            for x, e in zip(x_t, eps)]
```

Verified outputs: `classifier_free_guidance([0.2],[0.8],4.0) → [2.6000000000000005]`; `…,1.0) → [0.8]`; `ddpm_mean([1.0],[0.5],0.01,0.81) → [0.9935092408614442]`.

### Section E — Solutions

**Model answer covering the chain.**

1. **Problem.** A catalogue image must look real; an "average" image is a blurry blob, which is exactly the **VAE** failure mode (and any independent-pixel likelihood model). A **GAN** avoids blur but is unstable to train and prone to mode collapse; **diffusion** gives high quality *and* stable training.
2. **Task.** Model $p(x\mid y)$ where $y$ is the text prompt and $x$ is the image; at inference, sample $\hat x\sim p(x\mid y)$ at 1024×1024. The model must produce a photorealistic image that **respects the prompt's objects, attributes and colours**.
3. **Data.** Only images are needed for the core generative objective; text conditioning needs **paired** text–image data, so a large cap-ended image–text corpus is used (Imagen uses a frozen T5-XXL encoder trained separately on text). No pixel labels or masks are required for generation.
4. **Model.** Diffusion, because quality and training stability both matter at 200,000 images. Prefer **latent-space** diffusion (Stable Diffusion) if the *fixed* GPU budget is the binding constraint, since it cuts sampling cost; choose **pixel-space** (Imagen) if quality and text fidelity dominate and budget allows the cascade. Here, with a fixed budget and huge volume, latent-space is the pragmatic pick.
5. **Metric.** Automatic: **FID** on a reference set for realism plus a **CLIP score** for prompt alignment. Human: a **DrawBench-style** preference study where raters compare prompts across models for colour, counting and spatial relations.
6. **Failure mode.** (i) **Wrong colours** — models over-trust the prompt text and under-use the guidance scale, or saturate under strong guidance. Mitigation: **classifier-free guidance** with **dynamic thresholding** so strong guidance does not clip colours. (ii) **Wrong object count** — the classic compositional/counting failure of text-to-image models. Mitigation: prompt engineering (explicit counts, "exactly three mugs") plus **rejection sampling** with a captioner/CLIP check and re-generation until the count is right.

**Sub-questions.**

**E-a.** Diffusion's reverse process is a **chain of small denoising steps**, each a full U-Net forward pass; DDPM uses $T=1000$ steps where a GAN uses one generator pass. Two ways to cut it: (1) **progressive distillation** — a student learns to match two teacher steps in one, then becomes the teacher, halving the count each round until 4–8 steps; (2) **deterministic/latent samplers** — DDIM, DPM-Solver, consistency models, or running diffusion in a compressed latent space (latent diffusion).

**E-b.** **Dynamic thresholding.** At high guidance scale the sampled value is pushed outside the valid pixel range and gets clipped, which saturates colours; dynamic thresholding re-scales the predicted value per step so its dynamic range stays stable, preserving colour fidelity under strong guidance.

**E-c.** This is exactly the **Palette** formulation: every image-to-image task (inpainting, outpainting, colourisation, restoration, logo removal) is "generate $y$ given input $x$", with the **same network and the same loss** — only the (input, target) training pairs change. The conditioning image is concatenated to the noisy image at **every** denoising step, so one multi-task model can handle all of them; a single model can even handle several tasks at once.

### Section F — Solutions

**F1.** *Formulas:* values $=H\cdot W\cdot C$; bits $=$ values $\times8$; bytes $=$ bits$/8$.

- Values: $256\times256\times3=\mathbf{196{,}608}$.
- Bits: $196{,}608\times8=\mathbf{1{,}572{,}864}$ bits.
- Bytes: $1{,}572{,}864/8=\mathbf{196{,}608}$ bytes.

*Interpret:* a single $256^2$ image is ~200k correlated numbers — three orders of magnitude more output than a $K$-way label, which is why independent-pixel generation collapses to the mean.

**F2.** *Formulas:* $\overline\beta=(\beta_1+\beta_T)/2$; $\sum\beta_t\approx T\overline\beta$; $\bar\alpha_T\approx e^{-\sum\beta_t}$.

- Mean: $(0.0001+0.02)/2=\mathbf{0.01005}$.
- Sum: $1000\times0.01005=\mathbf{10.05}$.
- $\bar\alpha_{1000}\approx e^{-10.05}=\mathbf{4.32\times10^{-5}}$.
- $\sqrt{\bar\alpha_{1000}}\approx\mathbf{0.00657}$.

*Interpret:* by $t=T$ only **0.66%** of the original signal amplitude survives — "pure noise" is quantitative. (The exact product $\prod(1-\beta_s)$, rather than the $e^{-\sum\beta}$ shortcut, gives $4.04\times10^{-5}$ and $\sqrt{\bar\alpha_T}=0.00635$; both round to the same statement.) The schedule is designed so $\bar\alpha_T\approx0$.

**F3.** *Formula:* $\mathrm{KL}=\tfrac12(\mu^2+\sigma^2-1-2\ln\sigma)$.

- $\mu=1,\sigma=0.5$: $\tfrac12(1+0.25-1-2\ln0.5)=\tfrac12(0.25+1.38629)=\mathbf{0.81815}$.
- $\mu=0,\sigma=1$: $\tfrac12(0+1-1-0)=\mathbf{0}$.

*Interpret:* the KL is a "price" for a posterior that departs from $\mathcal N(0,1)$; it is exactly zero only when the posterior **is** the prior — which is also the definition of posterior collapse.

**F4.** *Formula:* $z=\mu+\sigma\varepsilon$.

$$z=0.5+0.2(1.5)=\mathbf{0.8}.$$

*Interpret:* all the randomness sits in $\varepsilon$; $\mu$ and $\sigma$ are differentiable outputs of the encoder, so the sample $z$ remains attached to the computation graph.

**F5.** *Formula:* $\text{loss}=\mathrm{MSE}+\mathrm{KL}$.

- Loss $=0.5+0.81815=\mathbf{1.31815}$.
- Halving the KL: new loss $=0.5+0.409075=\mathbf{0.909075}$, a fall of $\mathbf{0.409075}$.

*Interpret:* the two terms are comparable in size, so early training is a tug-of-war between fitting the data and matching the prior — the tension that produces posterior collapse when the KL wins.

**F6.** *Formulas:* $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$; inverse $x_0=(x_t-\sqrt{1-\bar\alpha_t}\varepsilon)/\sqrt{\bar\alpha_t}$.

- Forward: $0.9(2.0)+\sqrt{0.19}(1.0)=1.8+0.435890=\mathbf{2.235890}$.
- Inverse with a perfect $\varepsilon$: $(2.235890-0.435890)/0.9=1.8/0.9=\mathbf{2.0}$.

*Interpret:* the kernel is exactly invertible when the noise is known — the whole training signal is "the network must recover the $\varepsilon$ we added". The inverse is the ideal one-shot denoise; practice needs many small steps because the network never knows $\varepsilon$ exactly.

**F7.** *Formulas:* time $=T\times t_{\text{step}}$; total $=N\times$ time.

- Per image (1000 steps): $1000\times0.05=\mathbf{50}$ s.
- 200,000 images: $200{,}000\times50=\mathbf{10{,}000{,}000}$ s $\approx\mathbf{115.7}$ days.
- Distilled (4 steps): $4\times0.05=\mathbf{0.2}$ s per image; $200{,}000\times0.2=\mathbf{40{,}000}$ s $\approx\mathbf{11.1}$ hours.

*Interpret:* sampling cost is the dominant practical obstacle; distillation changes a 4-month job into an overnight one (**250× fewer evaluations**).

**F8.** *Formula:* $\hat\varepsilon=\varepsilon_u+s(\varepsilon_c-\varepsilon_u)$.

- $s=1$: $0.2+1(0.6)=\mathbf{0.8}$.
- $s=4$: $0.2+4(0.6)=\mathbf{2.6}$.
- $s=0$: $0.2+0=\mathbf{0.2}$.

*Interpret:* $s=1$ recovers the plain conditional prediction; $s=0$ recovers the unconditional prediction; $s>1$ **extrapolates** past the conditional toward stronger prompt adherence at the cost of diversity — and large $s$ is what pushes values out of range, which is why Imagen adds dynamic thresholding.

---

## J. Sources

### Course sources

| source | what it provided |
|---|---|
| `Session_15_—_Generative_Computer_Vision_….txt` | the lecture narrative: the latent-space intuition, why sampling from a prior fails, the diffusion forward/reverse walkthrough, distillation for speed, and the pointer to Imagen |
| `session_15_annotated.pdf` (40 slides) | all figures, formulae and numbers; note this deck **has a text layer** and includes embedded EXPLANATION sections |

### Key slides

- **4–8** generating data, applications: StyleGAN3, DatasetGAN, GauGAN/SPADE.
- **9–12** the generative landscape and why diffusion is the focus.
- **13–17** AI art and image-to-image (Palette): super-resolution, colourisation, inpainting, outpainting.
- **19–23** VAE overview; classification vs generation; autoregressive vs latent-variable.
- **24–26** intractability of $p(x)$; VAEs and the ELBO; reparameterization.
- **27–28** hierarchical VAEs and the challenges (posterior collapse).
- **29–38** diffusion: forward process, diffusion kernel, reverse process, noise prediction, weighting, summary.
- **39** architectures (U-Net, time embeddings).
- **40** diffusion as a hierarchical VAE; faster sampling / distillation.
- **62–65** Imagen case study.

### Supplementary

- Kingma & Welling, *"Auto-Encoding Variational Bayes"* (VAE), 2013.
- Sohl-Dickstein et al., *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"*, 2015.
- Ho et al., *"Denoising Diffusion Probabilistic Models"* (DDPM), 2020.
- Song et al., *"Score-Based Generative Modeling through SDEs"*, 2021.
- Vahdat & Kautz, *"NVAE"*, 2020; Sønderby et al., *"Ladder VAEs"*, 2016.
- Salimans & Ho, *"Progressive Distillation for Fast Sampling of Diffusion Models"*, 2022.
- Saharia et al., *"Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"*, 2022.
- Karras et al., *"Alias-Free GAN"* (StyleGAN3), 2021; Park et al., *"SPADE"*, 2019.

### Unresolved

- The annotated deck's formula images (the exact DDPM objective and the $q(x_{t-1}\mid x_t)$ closed form) render as images without a text layer; the forms quoted here are the standard DDPM statements and match the transcript's verbal description.
- The slide says a $256\times256\times3$ image has "~200,000 numbers"; the exact value is **196,608**.
- Depth of the Imagen super-resolution cascade is described as "two super-resolution diffusion models (256², then 1024²)" with a **64×64** base; the intermediate conditioning resolutions between stages are not stated numerically.
- No slide gives wall-clock timings for sampling; the 50 ms/step figure used in F7 is a stated assumption for arithmetic, not a quoted benchmark.
