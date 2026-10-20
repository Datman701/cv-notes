# Formula Sheet — the 28 worth memorising

**What this is.** Only formulas that are (a) one line, (b) plug-in, and (c) likely to appear in Section F or as a 2-mark Section C. No derivations. **★ = if you learn nothing else, learn these.**

**How Section F is marked.** Write the formula → substitute → calculate → interpret (units). A correct answer with no formula can lose marks; a wrong answer with the right formula can still earn some.

**Symbols.** $N$ = input size, $K$ = kernel size, $S$ = stride, $P$ = padding, $D$ = dilation, $C$ = channels. 3D versions apply the same formula per axis. $f$ is in **pixels**, $B$ in **metres**, $Z$ in **metres**, disparity $d$ in **pixels**.

---

## 1. Shapes and counts (6)

### 1. ★ Convolution / pooling output

$$\boxed{\text{out}=\left\lfloor\frac{N+2P-K}{S}\right\rfloor+1}$$

3D: compute $T',H',W'$ separately with $K_t,S_t,P_t$ etc.

| $N$ | $K$ | $S$ | $P$ | out |
|---|---|---|---|---|
| 28 | 2 | 2 | 0 | **14** |
| 224 | 7 | 2 | 3 | **112** |
| 7 | 3 | 3 | 0 | **2** |
| 32 | 5 | 1 | 2 | **32** |

*Trap:* floor, not round. A window that would run past the edge simply does not fit.

### 2. Dilated (atrous) kernel

$$\boxed{K_{\text{eff}}=K+(K-1)(D-1)}$$

$K=3,D=2\Rightarrow K_{\text{eff}}=5$. Then use $K_{\text{eff}}$ in the output formula: $N=32,P=2,S=1\Rightarrow32$.

### 3. Transposed convolution output

$$\boxed{\text{out}=(N-1)S-2P+K\;[+\,\text{output\_padding}]}$$

$N=7,K=3,S=2,P=0\Rightarrow(6)(2)+3=\mathbf{15}$.

### 4. ★ Convolution parameter count

$$\boxed{\text{params}=(K_hK_wC_{\text{in}}+1)\,C_{\text{out}}}$$

3D: $(k_tk_hk_wC_{\text{in}}+1)C_{\text{out}}$.

- $3\times3$ kernel, 3 input channels, 64 filters: $(9\cdot3+1)\cdot64=\mathbf{1792}$.
- 5 filters of size $2\times5\times5$ with 3 input channels: $(2\cdot5\cdot5\cdot3+1)\cdot5=\mathbf{755}$.

*Trap:* one bias **per filter**, not per weight.

### 5. ★ Receptive field (stacked layers)

Start $r=1$, $j=1$. For each layer:

$$\boxed{r\leftarrow r+(k-1)j,\qquad j\leftarrow j\cdot s}$$

Layers $(k3,s1),(k3,s2),(k3,s2)$: $r=3\to5\to\mathbf{9}$, $j=1\to2\to4$.

Two $3\times3$ stride-1 convs: $r=1+2+2=\mathbf{5}$.

### 6. Total downsampling (output stride)

$$\boxed{\text{stride}_{\text{total}}=\prod_\ell s_\ell}$$

Three stride-2 layers: $2\cdot2\cdot2=\mathbf{8}$ (224 → 28).

---

## 2. Detection and segmentation metrics (5)

### 7. ★ IoU

$$\boxed{\text{IoU}=\frac{|A\cap B|}{|A\cup B|}=\frac{\text{inter}}{\text{area}_A+\text{area}_B-\text{inter}}}$$

$[0,0,10,10]$ vs $[5,5,15,15]$: inter $=25$, union $=100+100-25=175$, IoU $=\mathbf{0.142857}$.

Inside box $[2,2,8,8]$: inter $=36$, union $=100$, IoU $=\mathbf{0.36}$.

### 8. ★ Dice

$$\boxed{\text{Dice}=\frac{2|A\cap B|}{|A|+|B|}}$$

Same boxes: $50/200=\mathbf{0.25}$. For any pair, **Dice ≥ IoU** (Dice is the harmonic mean of precision and recall; IoU is not).

### 9. Precision, recall, F1

$$\boxed{P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}}$$

$TP=8,FP=2,FN=4$: $P=0.8$, $R=0.666667$, $F_1=\mathbf{0.727273}$.

### 10. mIoU and pixel accuracy

$$\boxed{\text{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c},\qquad \text{mIoU}=\frac1C\sum_c\text{IoU}_c}$$

$$\boxed{\text{pixel\_acc}=\frac{\sum_c TP_c}{\sum_c(TP_c+FN_c)}}$$

Classes $(8,1,1)$ and $(6,1,1)$: IoUs $0.8,0.75$; mIoU $=\mathbf{0.775}$; pixel acc $=14/16=\mathbf{0.875}$.

*Trap:* pixel accuracy's denominator is **labelled pixels** $\sum(TP+FN)$, not $\sum(TP+FP+FN)$.

### 11. Panoptic quality

$$\boxed{PQ=SQ\times RQ,\quad SQ=\overline{\text{IoU}}_{\text{TP}},\quad RQ=\frac{TP}{TP+\tfrac12FP+\tfrac12FN}}$$

IoUs $[0.8,0.6]$, $TP=2,FP=1,FN=1$: $SQ=0.7$, $RQ=2/(2+0.5+0.5)=0.666667$, $PQ=\mathbf{0.466667}$.

---

## 3. Image operations (4)

### 12. Gaussian kernel

$$\boxed{G(x)=\exp\!\left(\frac{-x^2}{2\sigma^2}\right)}$$

Sample, then divide by the sum. $\sigma=1$, taps $x=-1,0,1$: raw $[0.606531,1,0.606531]$, normalised $[0.274069,\mathbf{0.451863},0.274069]$.

### 13. Gradient magnitude and orientation

$$\boxed{|\nabla I|=\sqrt{I_x^2+I_y^2},\qquad \theta=\operatorname{atan2}(I_y,I_x)}$$

$I_x=3,I_y=4$: magnitude $=\mathbf{5}$, $\theta=\mathbf{53.13^\circ}$.

### 14. Bilinear interpolation

For fractional position $(dx,dy)$ inside a cell:

$$\boxed{I=(1-dx)(1-dy)I_{00}+dx(1-dy)I_{10}+(1-dx)dy\,I_{01}+dx\,dy\,I_{11}}$$

Centre of $\begin{bmatrix}0&10\\20&30\end{bmatrix}$ ($dx=dy=0.5$): $I=\mathbf{15}$.

### 15. Histogram equalisation

$$\boxed{s_k=\operatorname{round}\!\left((L-1)\frac{\text{CDF}(k)-\text{CDF}_{\min}}{N-\text{CDF}_{\min}}\right)}$$

$[[0,0,1],[1,2,2]]$, $L=8$: CDF $[2,4,6]$, $\text{CDF}_{\min}=2$, span $=4$ → $[[0,0,4],[4,7,7]]$.

*Trap:* use the **cumulative** count, and subtract the smallest non-zero CDF before scaling.

---

## 4. 2D / 3D geometry (7)

### 16. ★ Pinhole projection

$$\boxed{u=f_x\frac{X}{Z}+c_x,\qquad v=f_y\frac{Y}{Z}+c_y}$$

$f_x=1000,c_x=320,X=0.5,Z=10\Rightarrow u=\mathbf{370}$.

### 17. ★ Object size → depth

$$\boxed{Z=\frac{f\,H}{h}}$$

$f=1600,H=1.5\text{ m},h=60\text{ px}\Rightarrow Z=\mathbf{40\text{ m}}$. With $f=1000$: $Z=\mathbf{25\text{ m}}$.

### 18. ★ Disparity → depth

$$\boxed{Z=\frac{f\,B}{d}}$$

$f=720,B=0.54\text{ m},d=24\Rightarrow Z=\mathbf{16.2\text{ m}}$. At $d=23$: $Z=\mathbf{16.904\text{ m}}$.

### 19. ★ Depth error from disparity error

$$\boxed{\Delta Z=\frac{Z^2}{fB}\,\Delta d}$$

At $Z=16.2$, one pixel ($\Delta d=1$): $\Delta Z=\mathbf{0.675\text{ m}}$. Error grows as $Z^2$ — doubling depth quadruples the error.

### 20. Focal length from field of view

$$\boxed{f=\frac{W/2}{\tan(\text{FOV}/2)}}$$

$W=1280$, FOV $=60^\circ$: $f=640/\tan30^\circ=\mathbf{1108.51\text{ px}}$.

### 21. Triangulation (rectified stereo)

Once $Z$ is known from $Z=fB/d$:

$$\boxed{X=\frac{(u-c_x)Z}{f_x},\qquad Y=\frac{(v-c_y)Z}{f_y}}$$

$f=800,B=0.10,d=40\Rightarrow Z=2.00$ m. $u=520,c_x=640\Rightarrow X=\mathbf{-0.30}$ m; $v=300,c_y=360\Rightarrow Y=\mathbf{-0.15}$ m.

### 22. Homography and epipolar constraint

$$\boxed{x'=Hx,\qquad x'^{\top}Fx=0}$$

- $H$ is $3\times3$ up to scale → **8 DOF** → **4 point pairs**.
- $F$ is $3\times3$ up to scale and $\det F=0$ → **7 DOF** → **8-point** algorithm.
- $E=K'^{\top}FK$ (calibrated cameras).

---

## 5. Transformers (3)

### 23. ★ Scaled dot-product attention

$$\boxed{\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V}$$

Softmax is **row-wise**; subtract the row maximum first. With $d_k=4$ and score row $[0.5,0,0]$: weights $[\mathbf{0.451863},0.274069,0.274069]$.

*Trap:* divide by $\sqrt{d_k}$, not $d_k$.

### 24. Cosine similarity

$$\boxed{\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}}$$

$[1,2,3]$ vs $[2,4,6]$: $\mathbf{1.0}$; orthogonal: $\mathbf{0.0}$; zero vector: define as $0$.

### 25. ViT token count

$$\boxed{n_{\text{tokens}}=\frac{H}{P}\cdot\frac{W}{P}+1}$$

$224\times224$, $P=16$: $14\cdot14+1=\mathbf{197}$ (196 patches + CLS).

---

## 6. Losses and generative (3)

### 26. Cross-entropy and focal loss

$$\boxed{CE=-\log p_t,\qquad \text{focal}=-\alpha(1-p_t)^\gamma\log p_t}$$

$p_t=0.7$: $CE=\mathbf{0.356675}$. With $\alpha=0.25,\gamma=2$: focal $=\mathbf{0.008025}$.

### 27. VAE reparameterisation

$$\boxed{z=\mu+\sigma\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)}$$

$\mu=1,\sigma=0.5,\varepsilon=0.4\Rightarrow z=\mathbf{1.2}$.

### 28. Diffusion: forward jump, DDPM mean, guidance

$$\boxed{q(x_t\mid x_0)=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon}$$

$$\boxed{\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta\right)}$$

$$\boxed{x_{\text{guided}}=x_{\text{uncond}}+s\,(x_{\text{cond}}-x_{\text{uncond}})}$$

- $x_0=2,\bar\alpha_t=0.25,\varepsilon=1\Rightarrow x_t=\mathbf{1.866025}$.
- $\alpha=0.9,\beta=0.1,\bar\alpha=0.5,x_t=1,\varepsilon=0.1\Rightarrow\mu_t=\mathbf{1.039185}$.
- $x_{\text{cond}}=1,x_{\text{uncond}}=0.5,s=3\Rightarrow\mathbf{2.0}$.

---

## Not worth memorising

Skip these unless the question hands you the setup: ELBO derivation, KL-divergence algebra, DLT/8-point matrix construction, bundle adjustment, Canny's full four-stage pipeline, StyleGAN/GAN losses, FLOPs and activation-memory counts, attention complexity proofs.

**If a numerical gives you a formula you half-remember, write the closest correct one and substitute.** Section F awards working, not just the final number.
