# Class 12 — 3D Vision & Motion

> **Exam weight: Tier 1.** This is the class that turns Class 11's geometry into **numbers**: one inverse law, $Z=fB/d$, carries most of Section F, and the slides hand you fully worked examples. The second half (optical flow, structure from motion, visual odometry/SLAM) is Section A/B/C territory — recognition and one-sentence explanations.
>
> **Sources:** `Session_11_12.pdf` pages 37–70 (Modules 4 stereo, 6 triangulation + *"Stereo 3D Reconstruction"*) · `Session_12.pdf` (56 image-only slides: stereo-as-optimization, optical flow, structure from motion, visual SLAM) · `Session_12_—_3D_Vision_&_Motion_….txt` (transcript, 14k words).
> **Professor repository: none in this workspace.**

**Scope note.** The `Session_11_12.pdf` deck is one file for two classes. **Class 11 took pages 1–48** (projection, homography, calibration, epipolar). **Class 12 is pages 37–70 plus the whole of `Session_12.pdf`.** Epipolar geometry was taught in Class 11 and is only *referenced* here — see `../class-11-camera-geometry/README.md §B11`.

**Every concept below has a worked example underneath it.** The idea is: read the two-sentence intuition, then walk the numbers, then the takeaway. If you can do the walkthroughs, you can do Section F.

---

## A. Lecture overview

**The one-sentence version.** One camera loses depth because a pixel only fixes a *direction*; a **second view** restores depth because the *difference* between two views is inversely proportional to distance — and once you can move through many views, you recover the scene **and** the camera's own path.

**The whole class in five lines:**

```
  one camera         →  a pixel is a DIRECTION        (depth stays unknown)

  two cameras        →  the shift between them (disparity)
                        →  Z = f·B / d               (STEREO)

  two views,         →  intersect the two rays       (TRIANGULATION)
  unknown scene

  many views         →  recover the scene AND the camera poses
                        →  up to a global scale      (STRUCTURE FROM MOTION)

  many views,        →  recover the camera's motion as frames arrive
  online             →  pose after pose, then correct drift   (VISUAL ODOMETRY / SLAM)
```

**Prerequisites:** Class 11 (pinhole model, $u=f_xX/Z+c_x$, epipolar geometry, calibration), Class 02 (SIFT, RANSAC, Lowe ratio), Class 01 (matrix algebra).

**Difficulty: Intermediate, and the most arithmetically loaded class.** The algebra is **one inverse law** ($Z=fB/d$) plus **one small linear system** (2×2 for two rays, 4×4 for DLT). The conceptual load is **scale ambiguity** — the one idea that makes SFM different from everything before it.

**Where it sits.** Class 11 told you *how a camera works*. This class *uses two or more of them* to get depth, motion, and structure.

---

## B. The ideas

> **How to read this section.** Each idea is: **the intuition** (why it exists) → **▸ Walk through it** (a concrete example, numbers, in plain steps) → **the takeaway** (the one sentence to remember). The walkthroughs are the part you should be able to reproduce on blank paper.

### B1. Why one camera cannot give you depth

**The intuition.** A pixel is not a place in the world — it is a **ray**. Every 3-D point along that ray lands on the same pixel. The camera has thrown away *how far*.

Slide 37 says it three ways: *"One pixel = one ray. Depth along the ray is lost."* · *"Every point on the red ray lands on the same pixel."* · *"Small-and-near looks exactly like big-and-far."* And the slide is blunt that **"No single-image geometry can undo this."**

The slide even names the escape hatch: monocular depth networks *do* predict depth — not by geometry, but from **learned priors** about how big cars, people and roads usually are. *"(Hint: learned priors about object sizes and scenes.)"* That is a statistical prior, not a measurement.

**▸ Walk through it.** Hold a pen at arm's length and close one eye. Move the pen toward you until it is 10 cm from your face. It covers the same part of your vision as a **door across the room** did a second ago. Both fill the same set of pixels. The image is identical; the world is not.

$$\text{a pixel tells you } \frac{X}{Z}, \text{ not } X \text{ or } Z.$$

**The takeaway.** *A single image constrains direction, not distance. The missing quantity is exactly one number per pixel: $Z$.*

---

### B2. Disparity — the horizontal shift

**The intuition.** Your two eyes see the world from two slightly different places, so the **same object lands on different pixels** in each eye. *How different* is the signal.

Slide 38 gives the experiment: *"hold up a finger, close one eye, then the other. That jump is disparity."* Crucially, the slide adds the second half: a nearby **finger jumps a lot**, a distant **tree barely moves**. Your brain uses that jump size to sense distance — continuously, without thinking.

**▸ Walk through it.** A point appears at pixel **$u_L = 400$** in the left image and **$u_R = 330$** in the right image (slide 41). The *shift* is

$$d = u_L - u_R = 400 - 330 = \mathbf{70\ px}.$$

That is **disparity**, measured in pixels. After *rectification* (B6) the match is guaranteed to be on the **same row**, so the only unknown left is this horizontal shift.

**The takeaway.** *Disparity is the horizontal pixel shift of the same point between two views. It is the raw measurement; depth is a formula applied to it.*

---

### B3. The depth equation — from your eye to the pin-hole model

**The intuition.** Disparity is **large when the object is close** and **small when it is far**. So depth must be *inversely* proportional to disparity. That single sentence *is* the depth equation — everything else is finding the constant.

**▸ Walk through it.**

Place two copies of the pin-hole camera side by side, a distance $B$ apart (the **baseline**), both looking the same way. A point sits at $(X, Z)$ — $Z$ is how far in front, $X$ is how far to the side. Each camera applies Class 11's law, $x = fX/Z$, but the **right camera's world is shifted by $B$**:

$$x_L = \frac{f\,X}{Z} \qquad\qquad x_R = \frac{f\,(X - B)}{Z}$$

Subtract them:

$$d = x_L - x_R = \frac{fX}{Z} - \frac{f(X-B)}{Z} = \frac{fB}{Z}$$

Solve for $Z$:

$$\boxed{\,Z = \frac{f\,B}{d}\,}$$

Now put the slide's numbers in (slide 44). With $f = 720$ px, $B = 0.54$ m, and a measured $d = 24$ px:

$$Z = \frac{720 \times 0.54}{24} = \frac{388.8}{24} = \mathbf{16.2\ m}.$$

Sanity check the units: $f$ is in **pixels**, $B$ in **metres**, $d$ in **pixels** — the pixels cancel and you are left with metres. That is why $f$ here must be the *pixel* focal length from Class 11, not the physical lens number.

**The takeaway.** *$Z = fB/d$. Bigger disparity ⇒ closer. The constant is fixed by the camera (where $fB$ is a property of the rig).*

---

### B4. Near objects shift more — and why error grows like $Z^2$

**The intuition.** $d$ depends on $1/Z$, so the curve is a **hyperbola**: a huge change in $d$ near the camera, and a nearly flat, tiny $d$ far away. The nasty consequence is on the *far* end: out where $d$ is small, **one pixel of error means a large depth error**.

**▸ Walk through it.** Slide 42 plots $Z = fB/d$ with $f=720$, $B=0.54$:

| disparity $d$ | computation | depth $Z$ |
|---|---|---|
| 70 px (near) | $388.8 / 70$ | **5.6 m** |
| 8 px (far) | $388.8 / 8$ | **48.6 m** |

Then slide 44 does the error at 16.2 m: read the disparity as **23 px instead of 24** (a one-pixel mistake) and

$$Z = \frac{388.8}{23} = \mathbf{16.9\ m} \qquad\text{— off by } 0.7\text{ m from } 16.2.$$

A **single pixel** moved the answer 70 cm. The slide's rule is *"Error grows like $Z^2$."* Here is why, in one line — differentiate $Z = fB/d$:

$$\left|\frac{dZ}{dd}\right| = \frac{fB}{d^2} = \frac{(fB/d)^2}{fB} = \frac{Z^2}{fB} \qquad\Longrightarrow\qquad \Delta Z \approx \frac{Z^2}{fB}\,\Delta d.$$

At $Z = 16.2$: $\Delta Z \approx \frac{16.2^2}{720\times0.54}\times 1 = \frac{262.4}{388.8} = \mathbf{0.675\ m}$ per pixel. *(First-order; the exact answer above is 0.70 m — the formula is the linearisation.)*

**The takeaway.** *Depth resolution degrades with the square of distance. To see far, you need a larger $fB$ — a longer lens or a wider baseline.*

---

### B5. The baseline — and the trade-off you cannot escape

**The intuition.** $fB$ is the whole game: it sets how many pixels of disparity you get per unit of depth. A bigger baseline $B$ means **finer depth far away** — but also a **bigger blind zone** (the near region only one camera sees) and **harder matching** (the two views differ more).

**▸ Walk through it.** Take three different rigs (slide 39 scale):

| rig | baseline $B$ | at $Z = 50$ m, $d = fB/Z$ | depth error per px, $Z^2/(fB)$ |
|---|---|---|---|
| a phone-ish pair | 0.10 m | **1.44 px** | **34.7 m** |
| human eyes | ≈ 0.065 m | 0.94 px | 53.4 m |
| car rig (KITTI) | 0.54 m | **7.78 px** | **6.43 m** |
| wide survey rig | 1.00 m | **14.4 px** | **3.47 m** |

(with $f=720$ px). Look at the first row: at 50 m the phone pair produces less than **two pixels** of disparity. A one-pixel error there is a **35-metre** depth error — the answer is dominated by noise. Now look at the last row: a 1 m baseline raises the disparity to 14.4 px (1.85× the 0.54 m rig) and cuts the per-pixel error to **54%** of it.

But the slide's warning is the other half: *"Wider B → better depth far away, but a bigger blind zone and harder matching."* The instructor's transcript gives the mechanism: the two cameras see **less similar images**, so feature matching gets harder, and each camera's field of view leaves a near wedge the other cannot see.

**The takeaway.** *Baseline is a dial: turn it up for far depth, down for easy matching and a small blind zone. Every stereo system is a compromise on this dial.*

---

### B6. Rectification — turning a 2-D search into a 1-D search

**The intuition.** In general, the matching point lies somewhere on an **epipolar line** (Class 11, §B11) that is *tilted*. If you **warp both images** so the lines become **horizontal rows**, then the match can only be to the **left or right on the same row** — a one-dimensional search.

**▸ Walk through it.** Before rectification you would drag a template along a sloping line. After rectification, slide 41 shows: the point at row $v$ in the left image is matched by scanning **along row $v$ only** in the right image. From $u_L = 400$ to $u_R = 330$, disparity $70$ px — and you never leave the row.

This is the *"rectified special case"* from Class 11 slide p.48 in action: *"Lines become rows, so the match is just a horizontal shift: disparity."* It is also what makes dense stereo (a disparity at *every* pixel) computationally feasible — Class 11's $720\times$ saving, applied per pixel.

**The takeaway.** *Rectification doesn't add information; it removes a dimension from the search. Dense stereo is only practical on rectified pairs.*

---

### B7. Where stereo fails — and the left–right check

**The intuition.** Stereo is **only as good as the matching** (slide 46). Three failure modes, and one clever consistency check.

Slide 46 lists the three:

| Failure | Why the match is wrong |
|---|---|
| **Textureless areas** — white wall, sky, road | *"every patch looks the same, so any match is equally good"* |
| **Reflections & glass** | *"the mirror image seems to be behind the glass → wrong, too-far depth"* |
| **Occlusions** | *"some points are seen by only one camera — they have no match at all"* |

**▸ Walk through it** (slide 7's strategy). Slide 7 proposes a **left-to-right, then right-to-left** check: match a pixel in the left image to the right, then independently match it back from the right to the left. **If the two round-trips disagree, break the match.** This is the *left–right consistency* check, and it directly kills the occlusion failures:

> Point $P$ is visible only to the left camera (the right camera's view of it is blocked by a nearer object). The left-to-right search still "finds" *something* — a look-alike patch — so it produces a fake match. But that fake patch, searched right-to-left, points back to a **different** left pixel. The round trip fails ⇒ the match is discarded.

Slide 7's full strategy line: *"correspondences occur only along relevant epipolar line · represent points with local representation · color + surface smoothness; filters; some kind of deep feature."*

**The takeaway.** *Match quality is the ceiling on stereo. Occlusions have no true match, so consistency checks (left–right) are how you detect that and refuse, rather than inventing a wrong depth.*

---

### B8. Stereo as an optimisation problem

**The intuition.** The old recipe was *match first, then compute depth*. The trouble (transcript): *"this feature matching can be wrong and you should not directly just calculate the rays… there can be some pixels of difference which can introduce errors."* The rewrite **flips the order**: *choose a depth, project it, and ask how well it explains the second image.* Now the quality of a guess is a **number** you can minimise — and anything you can minimise, you can train with.

**▸ Walk through it** (slides 10–13, and the transcript's long version).

1. Pick a pixel $x_1$ in the left image. **Guess a depth** $Z = 3$ m.
2. Because the cameras are calibrated, you know exactly the **disparity** that depth implies: $d = fB/Z$. You now know exactly which pixel $x_2$ in the right image you are predicting.
3. **Look at the colour** there. Compare it with the colour at $x_1$:
$$\text{unary cost} = \big\| \text{colour}(x_1) - \text{colour}(x_2(d)) \big\|^2.$$
4. If the guess was right, the colours match and the cost is low. If it was wrong, the cost is high.
5. **Smoothness.** Pick a *neighbouring* pixel $x_1'$ and demand their depths agree: *"if I create an $x_1$ point, then $x$ point which is very close to $x_1$, they should have a similar depth."* The instructor's concrete version: *"if the depth here was 1 metre then here it can be 1.1 or 1.2 maybe but it cannot be suddenly 10."*
6. **Regions.** To avoid smoothing across real object edges (where depth *does* jump), slide 13 segments the image into **constant-colour regions** and fits each region a **linear disparity**:
$$d(x,y) = ax + by + c.$$
The chosen disparity for a region becomes a **vocabulary item**; the answer is a discrete optimisation with a **unary** term (does this disparity agree with the colour?) and a **binary** term (do neighbouring regions agree on their boundary?).

The instructor's summary of why this matters: the new formulation *"can be formulated in a way you can have a loss calculated, based on the loss you can do a backward propagation"* — the old formulation *"would not have worked with a neural network."*

**The takeaway.** *Invert the problem: guess depth → predict the second image → measure colour error + smoothness in regions. That makes stereo a loss you can optimise (or learn).*

---

### B9. Triangulation — two rays that should meet

**The intuition.** If you know both cameras completely ($K$, $R$, $t$) and you have the matched pixels $x \leftrightarrow x'$, then each pixel defines a **ray** in space, and the 3-D point is where the two rays **cross**.

Slide 57: *"Each pixel defines a ray. The ray holds every 3-D point that could have made that pixel. Two cameras → two rays. Where they cross is the 3-D point $P$."* Note the inputs: **calibration** (both $K,R,t$) *and* the **match** (from the epipolar search). Class 11's chain — *epipolar geometry → correspondence → depth* — completes here.

**▸ Walk through it — the ideal case** (slide 58). Write the two rays parametrically:

$$P = O + \lambda\,d \qquad\qquad P' = O' + \mu\,d'$$

Demand they meet, $O + \lambda d = O' + \mu d'$, and solve for the two distances $\lambda, \mu$. Then $P = O + \lambda d$.

There is a catch the slide flags immediately: this is **3 equations (x, y, z) for 2 unknowns** ($\lambda, \mu$). Normally there is **no exact solution at all** — so the "ideal case" is, well, ideal. **Which brings us to reality.**

**▸ Walk through it — the real case** (slide 59). In practice the rays **do not meet**; they pass each other with a small **gap**:

| source of the gap | slide's words |
|---|---|
| **Pixel noise** | *"Corners are found to ± a fraction of a pixel"* |
| **Calibration error** | *"K, R, t are estimates, not truth"* |
| **Quantization** | *"Pixels are discrete"* |

The fix (slide 60): **take the closest point between the two rays.**

1. Find the shortest segment $Q_1Q_2$ joining the rays — *"it is perpendicular to both."*
2. Its midpoint $\hat{P}$ is the estimate.

And the *better* standard (slide 60): **choose $\hat{P}$ whose reprojections land closest to the observed pixels** — minimise reprojection error, exactly as in Class 11's calibration. Same objective, new unknowns.

**The takeaway.** *Triangulation = intersect two rays. In reality the rays miss, so you take the closest point (or, better, the point that reprojects best).*

---

### B10. The math, kept simple — linear triangulation (DLT)

**The intuition.** "Find the closest point between two lines" is geometry. There is a slicker **linear** way: each view's projection equation is really **two** linear equations in the unknown 3-D point, so stack the equations from all views and solve.

**▸ Walk through it** (slide 61). Recall the camera matrix $P$ (3×4) from Class 11, with rows $p_1^\top, p_2^\top, p_3^\top$. For a pixel $(u,v)$ and unknown homogeneous point $X = (X,Y,Z,1)$:

$$x \simeq P X \;\Longrightarrow\; u = \frac{p_1^\top X}{p_3^\top X}, \quad v = \frac{p_2^\top X}{p_3^\top X}.$$

Cross-multiply to clear the fractions — **this is the trick that makes it linear**:

$$u\,(p_3^\top X) - p_1^\top X = 0 \qquad\qquad v\,(p_3^\top X) - p_2^\top X = 0.$$

Each view gives **two** equations. **Two views → 4 equations.** Stack them:

$$A\,X = 0, \qquad A \text{ is } 4\times4.$$

Since the true $X \neq 0$ solves $AX=0$, $X$ must be the **null vector** of $A$ — and the numerically stable way to get it is the **last singular vector** ("$X =$ last singular vector of $A$"), which is the answer even when noise makes $A$ slightly non-singular. Then divide by the fourth component to get $(X,Y,Z)$.

If you have a **rectified** stereo pair, you skip all of this — slide 61's shortcut:

$$Z = \frac{fB}{d}, \qquad X = \frac{(u - c_x)\,Z}{f}, \qquad Y = \frac{(v - c_y)\,Z}{f},$$

or in one call, `cv2.triangulatePoints(P1, P2, x1, x2)`.

**▸ Walk through it — the shortcut on real numbers.** With $f = 800$ px, $B = 0.10$ m, $c_x=640$, $c_y=360$, $u_L = 520$, $u_R = 480$, $v = 300$:

$$d = u_L - u_R = \mathbf{40\ px} \;\Rightarrow\; Z = \frac{800\times0.10}{40} = \mathbf{2.00\ m}$$
$$X = \frac{(520-640)\times 2.00}{800} = \frac{-240}{800} = \mathbf{-0.30\ m}, \qquad Y = \frac{(300-360)\times 2.00}{800} = \mathbf{-0.15\ m}.$$

**The takeaway.** *DLT = "each view contributes two linear equations; the point is the null vector of the stack." For rectified pairs, just use the three-line shortcut.*

---

### B11. Optical flow — pixels moving in time

**The intuition.** So far the two (or more) images were taken from **different places**. Optical flow is the same idea across **time**: a small camera movement produces image 2 from image 1, and you ask **where each pixel moved**. It is *"motion of pixels in the images"* (transcript).

**▸ Walk through it** (slides 15–21). Slide 15's key assumption: *"Assume we're moving rigidly in a stationary environment — then points will move along their epipolar lines, where the epipolar lines follow from the fundamental matrix, so from camera movement."* And slide 20 adds the crucial "how far":

> *"As we saw, **HOW FAR they move is determined by depth** — and by their movement!!!"*

Both terms matter. **Depth** because near things sweep across the image faster (the same disparity logic: they have a bigger $1/Z$), and **movement** because how the camera itself moves sets the direction of the flow.

The transcript's way of seeing it: move the camera **forward** and the image *zooms* — *"whatever was here zoomed in, so it moved outward."* Ride in a car and look out the side: *"the pixels going back."* Slide 21 shows the two-camera version: *"with respect to this image, you must have moved in this direction."*

**▸ Walk through it — why the depth term is the same as stereo.** In a plane, sitting on a train, glance at the fence posts. The nearest posts blur past; the distant hills crawl. That ordering *is* disparity — the near posts "jump" more between two instants, exactly as a near finger jumps more between two eyes. The instructor: *"when the object was placed very close, they move very quickly than it was placed far. Disparity comes into the picture."*

And the punchline — combining them: *"you are already able to calculate the optical flow in 2-D. Now you also have a depth parameter… now you can calculate this optical flow in 3-D."* Same smoothing trick as B8: *"for smoothing… you can segment the regions and then optical flow for a specific region should be similar."*

**The takeaway.** *Optical flow is "which way and how far did each pixel move between two times." How far = depth × camera motion. The depth part is disparity's twin.*

---

### B12. Structure from motion — when nothing is known

**The intuition.** Stereo assumed you *built* the rig and *measured* the baseline. Triangulation assumed you *calibrated* the cameras. **Structure from motion (SFM)** assumes neither — it takes a set of photos from a moving camera and recovers **both** the scene **and** the camera poses.

Slide 28 states it exactly:

> *"Given: $m$ images of $n$ fixed 3-D points… Problem: estimate $m$ projection matrices $P_i$ **and** $n$ 3-D points $X_j$ from the $mn$ correspondences $x_{ij}$."*

**▸ Walk through it — where it sits.** Slides 26 and 27 are the two things you already know, and SFM is the marriage of them:

| | what's known | what's estimated | slide |
|---|---|---|---|
| **Calibration** (Class 11) | known 3-D points + their pixels | the **camera** $K,R,t$ | slide 26 |
| **Triangulation** (this class) | known cameras + matched pixels | the **3-D point** | slide 27 |
| **SFM** | only the raw image matches $x_{ij}$ | **both** camera poses **and** 3-D points | slide 28 |

The instructor's clarification of what "known 3-D points" means in SFM: they are known *relative to each other* (*"if this is 0, then this must be some 1 0 0… this could be 0 1 0"*), but **their position relative to the cameras is unknown** — *"that is what we are estimating."* That distinction is the heart of the scale ambiguity that follows.

**The takeaway.** *SFM = estimate scene **and** camera poses from raw matches alone. It contains calibration and triangulation as special cases, and it needs many views.*

---

### B13. Is SFM uniquely solvable? No — and the ambiguity ladder

**The intuition.** SFM *looks* well-posed. It is not: the **same images can be produced by more than one scene.** Slide 29 asks the question with the **Necker cube** — a wireframe that your own eye flips between two equally valid 3-D interpretations, from *one* image. The slide answers: *"Is SFM always uniquely solvable?"* — **no.**

The cleanest statement is **scale**: slide 31 — *"If we scale the entire scene by some factor $k$ and, at the same time, scale the camera matrices by the factor of $1/k$, the projections of the scene points remain exactly the same."*

**▸ Walk through it — verify it yourself.** Take a camera at $C=(0,0,2)$, a point $X=(1,1,7)$, and $f = 1000$ px, $c_x = 320$. In the camera frame the point is at $(1,1,5)$:

$$u = 1000\times\frac{1}{5} + 320 = \mathbf{520\ px}.$$

Now **scale the whole world by $k=3$**: camera to $C=(0,0,6)$, point to $X=(3,3,21)$. Camera frame: $(3,3,15)$.

$$u = 1000\times\frac{3}{15} + 320 = 200 + 320 = \mathbf{520\ px}.$$

**Identical pixel.** The images cannot tell a 1 m scene from a 3 m scene. *(Verified.)*

Why does it happen? Because the camera equation divides by $Z$ — it depends only on **ratios** $X_c/Z_c$. Scaling camera and scene together cancels out of every ratio. The consequence (slide 31, verbatim): ***"Without a reference measurement, it is impossible to recover the absolute scale of the scene!"***

**▸ Walk through it — the ladder.** Adding knowledge removes ambiguity one rung at a time (slides 32–37):

| reconstruction is valid up to… | constraint that gets you there | transform $Q$ | free parameters |
|---|---|---|---|
| **Projective** | nothing | general 4×4, full rank | **15** |
| **Affine** | parallelism preserved | $\begin{bmatrix}A&b\\0&1\end{bmatrix}$ | **12** |
| **Similarity** | orthogonality preserved | $\begin{bmatrix}sR&t\\0&1\end{bmatrix}$ | **7** (3 rot + 3 trans + 1 scale) |
| **Euclidean** | as similarity, scale fixed | $R, t$ | **6** |

Slide 34 (affine): *"If we impose parallelism constraints, we can get a reconstruction up to an affine ambiguity."* Slide 36 (similarity): *"A reconstruction that obeys orthogonality constraints on camera parameters and/or scene… can be replaced by rotation and translation,"* with $R$ having *"3 independent parameters."* The slide's own summary of the arithmetic: *"now we use the number of parameters."*

The instructor's framing of why the ambiguity is *real* (transcript): *"If I just measure one bottle… if I know exactly how much is the dimension this bottle is, then basically I can get the true value. But till the time I don't know the exact value of any of the thing in this picture, I cannot come to a unique solution."* And how self-driving cars cheat: they **learn** the scale from **lidar** in the training data — *"we learn the scaling parameters also from the lidar."*

**The takeaway.** *From images alone the scene is recoverable only up to a similarity transform (7 free parameters) — and its **scale** is unknowable. A known-size object, a known baseline, or lidar/IMU is what pins it down.*

---

### B14. Bundle adjustment — optimise everything at once

**The intuition.** Estimating poses one frame at a time lets small errors **compound**. **Bundle adjustment** instead collects **all** the observations of a point across **many** frames and optimises the whole bundle jointly.

**▸ Walk through it** (transcript). The instructor's contrast: with just two frames you optimise a point *"from two perspectives."* With bundle adjustment you *"optimize for the single point at the same time from 10 number of frames… That is what it is called as bundle."* A single landmark is typically **visible in many frames — not just consecutive ones** — and every one of those sightings is a constraint.

The objective is the **reprojection error** you already know from Class 11, now summed over poses *and* points:

$$\min_{\{T_i\},\{X_j\}} \sum_{i,j} \big\| \pi(T_i\,X_j) - u_{ij} \big\|^2$$

where $\pi$ is the projection. The transcript's version: *"any pixel $p_k$ in any given image minus the reprojected image pixel value should be almost negligible."*

The catch — and the reason it is *offline*: *"computation time grows with the number of images."* So it is run in batches: *"after some m points, you can do the bundle adjustment for the previous one."*

**The takeaway.** *Bundle adjustment = joint reprojection-error minimisation over all poses and points. It is the accurate, expensive, offline refinement; every real SFM/VO/SLAM pipeline finishes with it.*

---

### B15. Visual odometry — the camera's own motion, in real time

**The intuition.** Put a camera on a moving robot. Between two frames the camera has undergone a **rigid motion** $T_k$ (a rotation and a translation). Estimate $T_k$ from the frame pair, then **chain** the motions to get the whole path. That is **visual odometry**.

Slide 39: *"Two camera positions at adjacent time instants $k-1$ and $k$ are related by the rigid body transformation $T_k$."* Slide 41: *"The main task of VO is to compute the relative transformations $T_k$ from images $I_k$ and $I_{k-1}$ and then to concatenate these transformations to recover the full trajectory."*

**▸ Walk through it — chaining.**

$$C_k = C_{k-1}\,T_k, \qquad C_0 = I$$

Each new frame contributes one $T_k$; multiplying them up gives the pose. Slide 41: *"VO recovers the path **incrementally, pose after pose**."*

Slide 40's framing of the two objects — and they are easy to confuse:

| | what it is | slide |
|---|---|---|
| **Transformation** $T_k$ | what happened **to the image** between two frames | *"the relative transformations"* |
| **Pose** $C_k$ | where the camera **is**, relative to the start | *"transformations of the camera w.r.t. the initial coordinate frame"* |

**▸ Walk through it — drift, the whole reason SLAM exists.** Slide 52: *"The errors introduced by each new frame-to-frame motion accumulate over time. This generates a drift of the estimated trajectory from the real one."* The instructor's picture: the robot is told to go 5 m, *"it will move at some angle and then only reach 4.8 meters"* — and every frame adds a little more. The uncertainty diagram (slide 52) makes it explicit: the pose uncertainty at $C_k$ is the uncertainty at $C_{k-1}$ **plus** the uncertainty of $T_k$ — they add up.

Slide 43 gives the reward for doing this well: VO is *"not affected by wheel slip"* (wheel odometry counts revolutions, and a slipping wheel lies), giving *"relative position error 0.1%–2%"*, and it works in **GPS-denied** places (underwater, indoor, aerial).

**The takeaway.** *VO = estimate frame-to-frame motion $T_k$, chain it into the path $C_k=C_{k-1}T_k$. Errors accumulate as drift, which is what loop closure and bundle adjustment repair.*

---

### B16. The VO pipeline — detection → matching → motion → refinement

**The intuition.** Four steps (slide 44): **feature detection**, **matching/tracking**, **motion estimation**, **local optimisation.** The interesting part is **step 3**, because there are three different ways to do it depending on what you have.

**▸ Walk through it** (slides 45–49):

1. **Feature detection** — *"extract high-contrast areas in the image"* (SIFT/ORB), giving features $f_k$ at time $k$.
2. **Matching / tracking** — `tracking` searches locally for each feature; `matching` detects independently in both images and pairs by descriptor (SIFT/ORB/SURF). Slide 46's distinction.
3. **Motion estimation** — compute $T_k$ from the two feature sets. **Three cases** (slide 47):

| type | what you feed it | minimal case |
|---|---|---|
| **2-D to 2-D** | two sets of 2-D image points | **5-point** correspondences |
| **3-D to 3-D** | two sets of 3-D points (from stereo pairs) | **3 non-collinear** correspondences |
| **3-D to 2-D** | 3-D points + 2-D image points | **PnP** (slide 51) |

4. **Local optimisation** — slide 48: *"An iterative refinement over last $m$ poses can be optionally performed after motion estimation."* That is a **mini bundle adjustment**.

**▸ Walk through it — the 2-D→2-D route end to end** (slide 49; this is the purest VO):

```
1. capture new frame I_k
2. extract + match features between I_{k-1} and I_k
3. compute the ESSENTIAL matrix E for the pair     ← Class 11: needs K
4. decompose E (SVD) into R and t
5. compute the relative scale and rescale t
6. concatenate:  C_k = C_{k-1} · T_k
7. repeat
```
Note step 3 uses the **essential** matrix, not the fundamental — slide 49's *"NOTE: the minimal-case solution involves 5-point correspondences"*, and the transcript's reason: *"we assume that the cameras are calibrated… that is why we are not computing fundamental matrix."*

**The takeaway.** *VO = detect, match, estimate motion (2D-2D / 3D-3D / 3D-2D), optionally refine. The route you pick is fixed by whether you have one moving camera or a stereo rig.*

---

### B17. VO vs SFM — same family, different clock

**The intuition.** They solve the same underlying problem, but **SFM is offline on unordered photos** and **VO is online on a stream**. Slide 53: *"VO is a particular case of SFM."* Slide 53: *"SFM is more general than VO and tackles 3-D reconstruction of both the structure and camera poses from **unordered** image sets."*

**▸ Walk through it.**

| | **Structure from Motion** | **Visual Odometry** |
|---|---|---|
| Input | unordered image set (a photo album) | ordered video stream, frame by frame |
| When | offline, after all images exist | **real time**, as each new frame arrives |
| Output | scene + all poses | the trajectory, incrementally |
| Optimisation | full bundle adjustment | fast per-frame; bundle adjustment optional/local |
| Slide | *"computation time grows with the number of images"* (53) | *"focuses on estimating the 3-D motion of the camera sequentially… and in real time"* (54) |

Slide 54's summary of the timing constraint: VO *"can't keep doing bundle adjustment because it takes a lot of time… you can basically use it if you want to… after every instance… but it takes time."*

**The takeaway.** *SFM is the batch problem, VO is the streaming special case. Same geometry; VO trades accuracy for latency.*

---

### B18. Visual SLAM — localise *and* map, and close the loop

**The intuition.** VO tracks the camera but **drifts**. **SLAM** = **S**imultaneous **L**ocalisation **A**nd **M**apping: as well as knowing where you are, you **build a map** — and once you have a map, **revisiting a place lets you correct the drift.** That correction is **loop closure**.

**▸ Walk through it** (slide 56, the 9-step stereo-VSLAM figure; slide 55 for the objective).

At each time step:

```
1. capture stereo images at t
2. stereo matching → disparity → depth → back-project each feature to a 3-D LANDMARK
3. robot moves to t+1: pose changes T_t → T_{t+1}
4. PROJECT the old 3-D landmarks into the new image  ← "prediction"
5. OBSERVE the same landmarks in the new image       ← the actual pixels
6. reprojection error = predicted pixel − observed pixel
7. UPDATE the camera pose (R,t) to reduce the error  ← e.g. PnP / nonlinear opt
8. UPDATE the map: triangulate new landmarks, place them in world coords
9. over many frames: BUNDLE ADJUSTMENT + LOOP CLOSURE
```

The objective is the same reprojection error (slide 55): *"minimise the squared Euclidean distance over all points"* between $\hat{u}_i = \pi(T_c X_i)$ and the observed $u_i$. Slide 55's setup is *"3-D-2D point correspondences… N 3-D points in world coordinates and their observed 2-D image points."*

**▸ Walk through it — the watch example.** The instructor's concrete demo of localisation: *"this watch becomes a landmark. Now if I move — let's say I thought I moved 1 metre but I may have moved anything — then related to that watch's position must have also changed."* You **predict** where the watch should appear given your believed motion; you **observe** where it actually is; **the difference corrects your belief.** That is step 6→7, every frame.

**Loop closure** (slide 56, *"when we revisit a place"*): a robot drives a big loop and returns to the start. VO's accumulated drift says it is 4 m off. But the camera **recognises the start** — the map has been here before — so the loop constraint **snaps the whole trajectory back**, *"reduces accumulated drift, produces a consistent camera trajectory and a global 3-D map."*

**▸ Walk through it — why you must filter moving objects.** The transcript's scenario: *"let's say you were moving next to me, I was also moving… you reached here, you looked, I was standing there — you will be confused: did I even move or not?"* A moving object breaks the assumption that the scene is static. The fix from this very lecture: **optical flow**. *"For static features, the optical flow is almost similar… but a dynamic object is also moving, that will be an anomaly. Its flow will be a little different, and then I can separate out what is static and what is dynamic."*

Slide 42's four environmental assumptions, all with a reason:

| assumption | why it is needed |
|---|---|
| Sufficient illumination | *"you need some kind of illumination to detect the features"* |
| Dominance of static scene | moving objects make the estimated motion wrong (filter via optical flow) |
| Enough texture | *"to uniquely identify the similar features"* — flat walls give no features |
| Sufficient scene overlap | *"for matching — if you don't have any similarity between two images, what will you do?"* |

**The takeaway.** *SLAM = VO + a persistent map + loop closure. Predicting landmarks and correcting on the observed pixels is the inner loop; recognising a revisited place is what removes long-term drift.*

---

## C. Numbers worth knowing

All verified by execution. **C1–C3 are the class.**

### C1. The two constants of this lecture

$$\boxed{\;Z = \frac{f\,B}{d}\;\;} \qquad\qquad \boxed{\;\Delta Z \approx \frac{Z^2}{fB}\,\Delta d\;}$$

| symbol | meaning | unit |
|---|---|---|
| $Z$ | depth of the point | metres |
| $f$ | focal length (pixels) — Class 11 | px |
| $B$ | baseline (distance between camera centres) | metres |
| $d$ | disparity $= u_L - u_R$ (after rectification) | px |
| $\Delta d$ | pixel error in the disparity | px |

Worked: $f=720$, $B=0.54$, $d=24$ → $Z = 388.8/24 = \mathbf{16.2\ m}$. At that depth one pixel costs $\Delta Z = 16.2^2/(720\times0.54) = \mathbf{0.675\ m}$.

### C2. Disparity ↔ depth (slide 42), $f=720$, $B=0.54$

| $d$ (px) | $Z$ (m) | note |
|---|---|---|
| 70 | **5.55** | near — small $d$ error is harmless |
| 40 | 9.72 | |
| 24 | **16.2** | the slide's headline example |
| 23 | 16.90 | a 1-px error at 16.2 m = **0.70 m** |
| 8 | **48.6** | far — 1 px is now metres |

The relationship is a **hyperbola**: $Z \propto 1/d$. Equal steps in $d$ are wildly unequal steps in $Z$.

### C3. The three worked examples the slides *give you*

| slide | $f$ | $B$ | measurement | result |
|---|---|---|---|---|
| 41 | — | — | $u_L=400$, $u_R=330$ | $d = \mathbf{70\ px}$ |
| 44 | 720 | 0.54 | $d=24$ | $Z = \mathbf{16.2\ m}$; a 1-px slip ($d{=}23$) → $\mathbf{16.9\ m}$ |
| 56 (triangulate) | 800 | 0.10 | $u_L{=}520,u_R{=}480,v{=}300$ | $d{=}40$, $Z{=}\mathbf{2.00\ m}$, $X{=}\mathbf{-0.30\ m}$, $Y{=}\mathbf{-0.15\ m}$ |

The triangulation row also gives the error rule: *"a 1-pixel error costs 5 cm at 2 m but 20 cm at 4 m"* — check: $Z^2/(fB) = 4/80 = 0.05$ m at 2 m, $16/80 = 0.20$ m at 4 m. ✔

### C4. Ambiguity ladder — the DOF you keep removing

| reconstruction valid up to | $Q$ shape | free parameters |
|---|---|---|
| **Projective** | general $4\times4$ | $16-1 = \mathbf{15}$ |
| **Affine** | $\begin{bmatrix}A&b\\0&1\end{bmatrix}$ | $9+3 = \mathbf{12}$ |
| **Similarity** | $\begin{bmatrix}sR&t\\0&1\end{bmatrix}$ | $1+3+3 = \mathbf{7}$ |
| **Euclidean** | $[R\mid t]$ | $3+3 = \mathbf{6}$ |

### C5. Baselines and minimal cases

| quantity | value | source |
|---|---|---|
| Human eyes | $B \approx 6.5$ cm | slide 39 |
| Car stereo rig | tens of cm; **KITTI ≈ 54 cm** | slide 39 |
| VO relative position error | **0.1% – 2%** | slide 43 |
| SIFT descriptor | 128 numbers, Lowe ratio < 0.75 | reconstruction slides |
| minimal 2-D→2-D | **5-point** | slide 49 |
| minimal 3-D→3-D | **3 non-collinear** | slide 50 |
| minimal 3-D→2-D | **PnP** | slide 51 |
| Middlebury stereo benchmark | `vision.middlebury.edu/stereo` | slide 14 |

### C6. DLT shapes

| views | equations (2 per view) | $A$ shape | solve |
|---|---|---|---|
| 2 | 4 | $4\times4$ | $X$ = null vector (last singular vector) |
| 3 | 6 | $6\times4$ | over-determined; least-squares null vector |
| $n$ | $2n$ | $2n\times4$ | as above |

---

## D. Pictures

### D1. One camera loses depth — a pixel is a ray

```
        every point on this ray
        lands on the SAME pixel
              ╲ │ ╱
       ●       ● │ ●        ← far, large
        ╲      ●│●          ← middle
         ╲      ●           ← near, small
          ╲     │
           ╲    │
    ────────●───┴────────   image plane
             one pixel

    ● near-small  and  ● far-large  →  IDENTICAL PIXELS
    ⇒  no single-image geometry can undo this
```

### D2. Disparity and the depth equation

```
    LEFT eye                 RIGHT eye
    (what L sees)            (what R sees)

        finger at uL=400         finger at uR=330
        ────●────────            ──────●──────

    d = uL − uR = 400 − 330 = 70 px

    after rectification, the match is on the SAME ROW:
    you only ever search LEFT↔RIGHT — a 1-D search.


                 P  (X, Z)
                ╱ ╲
               ╱   ╲
              ╱     ╲
      xL ●───╱───────╲───● xR
         │  ╱         ╲  │
      f  │ ╱           ╲ │  f
         │╱             ╲│
    ─────OL──────────────OR─────
              B (baseline)

    similar triangles:  d = fB/Z   ⇒   Z = fB/d
    big d  ⇒  near.      small d  ⇒  far.
```

### D3. Why error grows like $Z^2$

```
   Z = fB/d   is a HYPERBOLA

   depth
   Z (m)
    │●
    │ ●                        equal steps in d...
    │  ●
    │   ●
    │     ●  ●  ●  ●  ●  ●  ●  ...are tiny steps in Z here
    └─────────────────────────── disparity d (px)

   dZ/dd = Z²/(fB)     ← the LOCAL slope
   near the camera: slope small   → 1 px ≈ nothing
   far away:        slope huge    → 1 px ≈ metres

   f=720,B=0.54:  @16.2 m, 1 px = 0.68 m
                  @48.6 m, 1 px ≈ 6 m
```

### D4. Triangulation — rays that miss

```
   IDEAL                        REAL (noisy pixels,
   (rays meet)                  calibration error, quantization)

        P                              gap
        ●                          Q1 ●
       ╱ ╲                             │   ← shortest segment,
      ╱   ╲                            │     ⊥ to BOTH rays
     ╱     ╲                        Q2 ●
    ╱       ╲                       ╱    ╲
   ╱         ╲                     ╱      ╲
  O1         O2                  O1        O2

   3 eqs (x,y,z), 2 unknowns      the rays pass each other;
   O + λd = O' + μd'              take the MIDPOINT of the gap
   → usually NO exact solution    (or, better, the point whose
                                    reprojections are closest)
```

### D5. DLT — "each view gives two equations"

```
   one pixel (u,v) in view i:

        u·(p3ᵀX) − p1ᵀX = 0     ← cross-multiply to LINEARISE
        v·(p3ᵀX) − p2ᵀX = 0

   2 views → 4 equations → A X = 0,  A is 4×4

   X = (X, Y, Z, 1)  =  the NULL VECTOR of A
                     =  its LAST SINGULAR VECTOR

   rectified special case (skip all of the above):
        Z = fB/d ,  X = (u−cx)Z/f ,  Y = (v−cy)Z/f
```

### D6. SFM ambiguity — the ladder

```
   what you still don't know, going down:

   PROJECTIVE   Q any 4×4           15 params   "shape is wrong"
        │  add: parallelism preserved
        ▼
   AFFINE       Q = [A b; 0 1]      12 params   "parallel lines stay parallel"
        │  add: orthogonality preserved
        ▼
   SIMILARITY   Q = [sR t; 0 1]      7 params   "shape is RIGHT, size unknown"
        │  add: a known length / baseline / lidar
        ▼
   EUCLIDEAN    R, t                 6 params   metric truth

   ★ scale k:  scale the whole scene by k
                scale the cameras  by 1/k
                → every image is IDENTICAL
```

### D7. Visual odometry — chain the motions

```
   frame 0      frame 1        frame 2        frame 3
     │            │              │              │
     ▼            ▼              ▼              ▼
   pose C0  ─T1→ C1  ─T2→  C2  ─T3→  C3

   Ck = C(k−1) · Tk          ← "pose after pose"

   drift:  each Tk has a little error;
           errors ADD UP along the chain
           →─────────────
                ╲  estimated path wanders off
                 ╲
                  ✗   true path
```

### D8. SLAM — predict, observe, correct; and loop closure

```
   INNER LOOP (every frame):
      3-D landmark ──project──▶ predicted pixel
                                     │  compare
      observed pixel ──────────────────┘
                │
                ▼  reprojection error
        update pose (R,t)   ← minimise over many landmarks

   LOOP CLOSURE:
      start ●──────────────╮
            │  VO drift →  │   the path has drifted:
            │              │   "I think I'm 4 m off"
            ╰──────────────╯
                  ● end     camera RECOGNISES the start
      → snap trajectory back → drift removed, ONE global map
```

---

## E. Code

> No instructor code exists in this workspace. These are **illustrative** and were **not executed** — numpy/OpenCV are unavailable here. The *Section D* solutions in §H are pure-Python and **were** executed, with real output pasted in.

### E1. Depth from a rectified stereo pair

```python
# rectified rig: fx == fy, same row for matches
Z = fx * B / disparity                     # disparity = uL - uR, in pixels
X = (uL - cx) * Z / fx
Y = (v  - cy) * Z / fx                     # note: same f for Y in the rectified case
```

**The division is by the measured disparity** — `disparity` must be > 0. Zero disparity means infinite depth (a parallel ray); guard it.

### E2. Dense disparity

```python
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)
disp   = stereo.compute(left, right)       # every pixel, not just features
XYZ    = cv2.reprojectImageTo3D(disp, Q)   # Q from cv2.stereoRectify -> (X,Y,Z) per pixel
```

`reprojectImageTo3D` is slide 61's shortcut applied per pixel. `disp` is in **fixed-point** (divide by 16) in some OpenCV versions — a classic silent bug.

### E3. Triangulation

```python
# rectified shortcut
Z = fx * B / (uL - uR)
# general (not rectified): intersect the rays
X = cv2.triangulatePoints(P1, P2, xL, xR)  # homogeneous 4-vector; divide by [3]
```

### E4. Visual odometry, 2-D→2-D, one line at a time

```python
E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)  # slide 49 step 3
_, R, t, mask = cv2.recoverPose(E, p1, p2, K)                       # step 4
T_k = np.eye(4); T_k[:3,:3] = R; T_k[:3,3] = t.ravel()              # step 5 (scale = 1)
C_k = C_{k-1} @ T_k                                                 # step 6 — C_k
```

`recoverPose` returns $t$ **up to scale** — that is slide 49's *"compute relative scale and rescale t"* step made visible. Without an external reference (stereo baseline, lidar, IMU) the scale is a free parameter — exactly B13.

### E5. Bugs worth memorising

| Symptom | Cause |
|---|---|
| Depth blows up / negative | disparity ≤ 0 (mis-match or occlusion) not guarded |
| Depth off by a constant factor across the whole image | `f` in **metres** (physical lens) instead of **pixels** |
| Works near, wrong far | pixel error amplified by $Z^2/(fB)$ — nothing wrong with the code, wrong baseline for the task |
| VO trajectory curves/rotates physically wrongly | $T_k$ chained in the wrong order ($C_k T$ vs $T C_k$) |
| VO scale is right for one clip, wrong for the next | scale fixed by an external reference; you relied on `recoverPose`'s arbitrary `t` |
| Dense disparity has horizontal streaking | rectification stale/invalid — the epipolar rows do not actually line up |

---

## F. Comparisons

**Stereo vs triangulation vs structure from motion — what each assumes**

| | **Stereo** | **Triangulation** | **SFM** |
|---|---|---|---|
| Known | $K$, **$B$**, rectified | both cameras' $K,R,t$ | **nothing** but matches |
| Estimated | depth $Z$ (per pixel) | one 3-D point | scene **and** poses |
| Core formula | $Z = fB/d$ | intersect two rays (DLT) | bundle adjustment |
| Scale? | **known** (from $B$) | **known** (from $K,R,t$) | **unknown** |
| Slide | 44 | 57–61 | 28–37 |

**The connection, in one line:** stereo assumes a *known baseline*; triangulation assumes *known cameras*; SFM assumes *neither*, and pays for it with the scale ambiguity.

**VO vs SFM (slides 53–54)**

| | SFM | VO |
|---|---|---|
| Input | unordered images | video stream, in order |
| Timing | offline | **real time** |
| Optimisation | full bundle adjustment | per-frame; BA optional |
| Relationship | general problem | **special case of SFM** |

**The three motion-estimation routes (slide 47)**

| route | input | minimal case | when |
|---|---|---|---|
| 2-D → 2-D | two 2-D point sets | 5-point | single calibrated camera, moving |
| 3-D → 3-D | two 3-D point sets | 3 non-collinear | **stereo** rig moving |
| 3-D → 2-D | 3-D points + 2-D points | PnP | have a map, want the pose |

**The three stereo failure modes (slide 46)**

| failure | mechanism | symptom |
|---|---|---|
| Textureless | *"any match is equally good"* | random/garbage depth: sky, road, walls |
| Reflections/glass | *"mirror image seems behind the glass"* | depth **too far** through the glass |
| Occlusions | *"seen by only one camera"* | holes; fixed by left–right consistency (B7) |

**Fundamental vs essential (recap of Class 11 §B11 — VO uses the second)**

| | $F$ | $E$ |
|---|---|---|
| Raw pixels / calibrated | raw | $K^{-1}x$ |
| Needs $K$? | no | **yes** |
| Gives | epipolar lines | pose $R, t$ (scale unknown) |

---

## G. Exam-focused revision sheet

**Definitions**
- **Disparity** $d$ — the horizontal pixel shift of a matched point between a rectified stereo pair, $d = u_L - u_R$.
- **Baseline** $B$ — the distance between the two camera centres.
- **Depth equation** — $Z = fB/d$; the inverse law that converts disparity to metric depth.
- **Rectification** — warping both images so epipolar lines become horizontal rows.
- **Triangulation** — recovering a 3-D point by intersecting the rays of a matched pixel pair.
- **DLT** — linear triangulation; the point is the null vector of the stacked projection equations.
- **Optical flow** — the apparent motion of pixels between two frames.
- **Structure from motion** — recovering scene *and* camera poses from matches alone.
- **Bundle adjustment** — joint reprojection-error minimisation over all poses and points.
- **Visual odometry** — estimating the camera's motion frame-to-frame, in real time.
- **SLAM** — simultaneous localisation and mapping; VO plus a persistent map plus loop closure.
- **Scale ambiguity** — from images alone, the absolute size of the reconstruction is unknowable.

**The formulas**

$$Z = \frac{f\,B}{d} \qquad\qquad \Delta Z \approx \frac{Z^2}{fB}\,\Delta d \qquad\qquad X = \frac{(u-c_x)Z}{f},\quad Y=\frac{(v-c_y)Z}{f}$$

$$\text{DLT:}\quad u\,(p_3^\top X) - p_1^\top X = 0,\quad v\,(p_3^\top X)-p_2^\top X=0 \;\Rightarrow\; A X = 0$$

$$\text{VO:}\quad C_k = C_{k-1}\,T_k, \qquad \text{BA:}\quad \min_{\{T_i\},\{X_j\}}\sum_{i,j}\big\|\pi(T_iX_j)-u_{ij}\big\|^2$$

**Ambiguity ladder — memorise**

| up to | params |
|---|---|
| Projective | 15 |
| Affine | 12 |
| Similarity | 7 |
| Euclidean | 6 |

**Traps — these lose marks**
1. **Disparity sign.** $d = u_L - u_R$ in a *rectified* pair; without rectification the two pixels are not on a row and "disparity" is undefined.
2. **$f$ in pixels, not metres.** $Z = fB/d$ only closes dimensionally with the pixel focal length.
3. **$Z^2$, not $Z$.** Depth error per pixel is proportional to the *square* of depth. Doubling distance quadruples the error.
4. **Wider baseline is not free.** Better far depth, but a bigger blind zone and harder matching.
5. **Thinking SFM gives absolute size.** It gives shape; scale needs a reference (slide 31).
6. **Confusing $T_k$ (a motion) with $C_k$ (a pose).** $C_k = C_{k-1}T_k$.
7. **Assuming the scene is static in VO.** Moving objects corrupt the motion estimate — filter with optical flow.
8. **Using the fundamental matrix when calibrated.** VO uses the **essential** matrix (5-point).
9. **Treating bundle adjustment as real-time.** Its cost grows with the number of images.

**Checklist**
- [ ] Derive $Z = fB/d$ from the two projection equations
- [ ] Compute depth, and the depth error for a 1-px disparity slip, at two distances
- [ ] Compute $(X,Y,Z)$ from $(u_L,u_R,v,f,B,c_x,c_y)$
- [ ] Explain the baseline trade-off with numbers
- [ ] Explain why rays don't meet and what you do instead
- [ ] Write the two DLT equations per view and say what $X$ is
- [ ] State the scale ambiguity and give a numerical demonstration
- [ ] List the ambiguity ladder with parameter counts
- [ ] Distinguish SFM / VO / SLAM in one line each
- [ ] Explain VO drift and how loop closure fixes it
- [ ] Name the three stereo failure modes and the left–right fix

---

## H. Practice Questions

> Mirrors `session16.pdf`: **A** single-correct MCQ · **B** multi-select · **C** descriptive ≤100 words · **D** coding, attempt 1 of 3 · **E** case study · **F** numericals.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** In a rectified stereo pair a matched point sits at $u_L=400$ and $u_R=330$. What is the disparity?

- A. $-70$ pixels
- B. $70$ pixels
- C. $365$ pixels
- D. $730$ pixels

**A2.** A point has disparity $d$. If the same point moved so its disparity **doubled**, what happened to its depth?

- A. It doubled
- B. It halved
- C. It quadrupled
- D. It was unchanged

**A3.** Why does the depth error caused by a one-pixel disparity error grow in proportion to $Z^2$?

- A. Because $\Delta Z = Z^2/(fB)\,\Delta d$
- B. Because disparity is measured modulo the image width
- C. Because the focal length scales with distance
- D. Because the baseline changes with depth

**A4.** What is the principal cost of increasing a stereo rig's baseline?

- A. The focal length must also increase
- B. The blind zone grows and matching gets harder
- C. The disparity decreases at every depth
- D. Rectification becomes impossible

**A5.** Two rays from a matched stereo point do not intersect because of pixel noise and calibration error. What is the standard estimate of the 3-D point?

- A. The intersection of the two rays
- B. The closest point between the two rays
- C. Either ray's endpoint, chosen at random
- D. The midpoint between the two camera centres

**A6.** In linear triangulation (DLT) for a point seen in two views, the unknown 3-D point is found as:

- A. The sum of the two camera matrices
- B. The null vector of a $4\times4$ matrix $A$
- C. The inverse of the fundamental matrix
- D. The average of the two matched pixels

**A7.** Two reconstructions of the same images produce point clouds identical in shape but scaled by a factor of 3. Which statement is correct?

- A. Only the smaller one is physically possible
- B. Both reproject to exactly the same images
- C. The scale difference is visible as a colour shift
- D. The larger one has a larger baseline

**A8.** Which is true of visual odometry relative to structure from motion?

- A. VO is a real-time special case of SFM
- B. VO needs unordered photo collections
- C. VO always runs full bundle adjustment
- D. VO recovers absolute scale without a reference

### Section B — Multi-select (5 × 2 marks)

**B1.** Which statements about **stereo depth** are correct? Select all that apply.

- A. $Z = fB/d$ requires $f$ in pixels
- B. A larger baseline improves depth resolution far away
- C. Depth error per pixel grows as $Z^2$
- D. Disparity is independent of the focal length
- E. Rectification makes the match a horizontal shift on a row

**B2.** Which are **failure modes of stereo matching**? Select all that apply.

- A. Textureless regions such as sky and white walls
- B. Reflections that appear behind glass
- C. Occlusions where a point is seen by one camera only
- D. Using a rectified image pair
- E. Left–right consistency disagreement

**B3.** Which are **true of triangulation**? Select all that apply.

- A. Each pixel defines a ray of possible 3-D points
- B. Two rays from a matched pair generally meet exactly
- C. The closest point between the rays is a standard estimate
- D. Each view contributes two linear equations in DLT
- E. Triangulation needs the match *and* the calibration

**B4.** Which are **correct about structure-from-motion ambiguity**? Select all that apply.

- A. Scaling the scene by $k$ and the cameras by $1/k$ keeps every image identical
- B. Without a reference measurement the absolute scale is unrecoverable
- C. Imposing parallelism gives a reconstruction up to an affine ambiguity
- D. A similarity ambiguity leaves 7 free parameters
- E. The ambiguity is removed by using more images of the same scene

**B5.** Which are **correct about visual odometry and SLAM**? Select all that apply.

- A. VO estimates frame-to-frame motion and chains it into a trajectory
- B. Per-frame errors accumulate, producing drift
- C. Loop closure uses a revisited place to reduce accumulated drift
- D. Bundle adjustment cost grows with the number of images
- E. VO requires a static environment to estimate motion reliably

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why can a single camera not recover depth, and what does a second view add?

**C2.** Explain, without deriving it, why the depth error from a one-pixel disparity error grows like $Z^2$.

**C3.** In practice the two rays of a triangulation do not intersect. Why, and what is done about it?

**C4.** What is the scale ambiguity in structure from motion, and what kinds of information remove it?

**C5.** Distinguish visual odometry from structure from motion, and say what loop closure adds.

### Section D — Coding (20 marks — attempt ONE of the three)

**Pure Python, standard library only.** Provide a reusable function. Handle empty and boundary inputs.

---

**Option D1 — Stereo depth and its error**

```python
def depth_from_disparity(f, B, d): ...
def disparity_from_depth(f, B, Z): ...
def depth_error(f, B, Z, dd): ...
```

- `f`, `B`, `d`, `Z`, `dd` are numbers. `d` and `dd` are in pixels; `Z` and `B` in metres.
- `depth_from_disparity` returns $fB/d$ rounded to 6 dp.
- `disparity_from_depth` returns $fB/Z$ rounded to 6 dp.
- `depth_error` returns the first-order error $Z^2/(fB)\times dd$ rounded to 6 dp.
- Return `None` for any input that is not a positive number (and for `None`).

Worked examples:

```
depth_from_disparity(720, 0.54, 24)   ->  16.2
depth_from_disparity(720, 0.54, 23)   ->  16.904348
disparity_from_depth(720, 0.54, 10)   ->  38.88
depth_error(720, 0.54, 16.2, 1)       ->  0.675

depth_from_disparity(720, 0.54, 0)    ->  None
depth_from_disparity(720, -0.54, 24)  ->  None
depth_from_disparity(None, 0.54, 24)  ->  None
```

**Option D2 — Triangulate two rays**

```python
def triangulate_rays(O1, d1, O2, d2): ...
```

- `O1`, `O2`: 3-D origins; `d1`, `d2`: 3-D direction vectors (need not be unit).
- Returns the midpoint of the shortest segment between the two rays, rounded to 6 dp, i.e. the estimate of the 3-D point.
- Returns `None` if any argument is malformed (not length 3), if either direction is the zero vector, or if the rays are **parallel** (no unique closest pair).

Worked examples:

```
O1=(0,0,0) d1=(0,0,1)   O2=(1,0,5) d2=(0,-1,0)   ->  (0.5, 0.0, 5.0)
O1=(0,0,0) d1=(2,0,6)   O2=(4,0,0) d2=(-2,0,6)   ->  (2.0, 0.0, 6.0)   # rays meet exactly
O1=(0,0,0) d1=(0,0,1)   O2=(1,0,0) d2=(0,0,1)    ->  None              # parallel
O1=(0,0,0) d1=(0,0,0)   O2=(1,0,0) d2=(0,0,1)    ->  None              # zero direction
triangulate_rays((0,0),(0,0,1),(1,0,0),(0,0,1))  ->  None              # malformed
```

**Option D3 — Apply and compose camera transforms (the VO step)**

```python
def apply_transform(T, X): ...
def compose(A, B): ...
def chain_transforms(transforms): ...
```

- `T`, `A`, `B`: 4×4 matrices (lists of lists). `X`: a 3- or 4-vector.
- `apply_transform` treats `X` as a homogeneous point, applies `T`, and returns the dehomogenised 3-vector rounded to 6 dp; `None` if malformed or the homogeneous divisor is 0.
- `compose(A, B)` returns the 4×4 matrix product $A\cdot B$ (rows rounded to 6 dp); `None` if malformed.
- `chain_transforms(transforms)` returns the list of cumulative poses starting with the identity: `[I, C1, C2, ...]` where `C_k = C_{k-1} · transforms[k-1]`.

Worked examples:

```
T1 = translate x by 1 ; T2 = translate z by 2
apply_transform(T1, (0,0,0))          ->  (1.0, 0.0, 0.0)
compose(T1, T2)                       ->  [[1,0,0,1],[0,1,0,0],[0,0,1,2],[0,0,0,1]]
chain_transforms([T1, T2])            ->  [I, T1, compose(T1,T2)]   (3 poses)

Rz90 = rotation 90° about Z
apply_transform(Rz90, (1,0,0))        ->  (0.0, 1.0, 0.0)
apply_transform([[1,0],[0,1]], (0,0,0)) ->  None
```

### Section E — Case study (15 marks)

> **Scenario.** An ADAS team fits a passenger car with a **rectified stereo camera** for forward obstacle detection. The rig has $f = 700$ px, baseline $B = 0.54$ m (KITTI-like), $c_x = 640$, $c_y = 360$ on a 1280×720 image. A pedestrian is matched at $u_L = 700$, $u_R = 690$, $v = 400$. The team is deciding whether to double the baseline for a next-generation rig.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Why is *stereo* the right tool here, rather than a single calibrated camera? Give the one-line geometric reason.
2. **Data.** What must be known *before* any depth can be computed, and which of those quantities is intrinsic vs a property of the rig?
3. **Model.** Compute the pedestrian's depth $Z$ and its $(X, Y)$ from the match above. State the equation chain.
4. **Metric.** State the depth error for a one-pixel disparity error at that depth, and compute the *ratio* of error at 30 m versus 10 m.
5. **Failure mode.** At 60 m the disparity is only about 6 px, so a one-pixel error is already metres of error. Explain what the system must do and why.

**Sub-questions**

- **E-a.** The team doubles the baseline to 1.08 m. Compute the new disparity at 30 m and the new 1-px depth error. State two costs of the change.
- **E-b.** A monocular depth network is proposed as a cheaper alternative. Explain what information class it uses and why its output is not directly usable for a metric safety threshold.
- **E-c.** The car also runs visual odometry. Explain why its trajectory drifts, what fixes it, and why stereo VO (3-D→3-D) has less of a scale problem than monocular (2-D→2-D) VO.

### Section F — Numericals (8 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** A stereo rig has $f=720$ px and $B=0.54$ m. Compute the depth for a measured disparity of $24$ px, and then for $23$ px. State the depth error of a one-pixel slip. Then find the disparity that corresponds to a depth of $10$ m.

**F2.** A rectified pair has $f=800$ px, $B=0.10$ m, $c_x=640$, $c_y=360$, $u_L=520$, $u_R=480$, $v=300$. Compute $d$, $Z$, $X$ and $Y$. Then use $\Delta Z \approx Z^2/(fB)\Delta d$ to find the depth error of a one-pixel error, and repeat at $Z=4$ m.

**F3.** A rig has $f=720$ px. At $Z=50$ m, compute the disparity for baselines $B = 0.10$ m, $0.54$ m and $1.00$ m, and the one-pixel depth error for each. State the trade-off the numbers illustrate.

**F4.** Ray 1 starts at $O_1=(0,0,0)$ with direction $d_1=(0,0,1)$. Ray 2 starts at $O_2=(1,0,5)$ with direction $d_2=(0,-1,0)$. Find the closest point between the rays, and state what it represents. Then state what changes if $d_2$ becomes parallel to $d_1$.

**F5.** A camera at $C=(0,0,2)$ sees a point $X=(1,1,7)$ with $f=1000$ px and $c_x=320$. Compute the pixel $u$. Now scale the *entire* world — camera and point — by $k=3$, and recompute $u$. State the name and consequence of this property.

**F6.** A rectified stereo pair at row $v$ has matches at $u_L=400$ and $u_R=330$ with $f=720$ px, $B=0.54$ m. Compute the disparity and the depth. State how many candidate positions the matching search must examine per pixel in a 720×1280 image, along the epipolar line, and the reduction factor.

**F7.** A robot using visual odometry drives a 400 m loop. If the relative position error is 1%, compute the accumulated drift at the end of the loop. State what loop closure does to this number and name the process that refines it offline.

**F8.** State the number of free parameters in each rung of the SFM ambiguity ladder — projective, affine, similarity and Euclidean — and explain, in one sentence each, what constraint takes you from one rung to the next.

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — 70 pixels.**

$d = u_L - u_R = 400 - 330 = 70$ px. Slide 41 shows exactly this. *(Verified.)*

- A has the sign flipped ($d$ is defined $u_L-u_R$).
- C and D are the average and the sum — neither is disparity.

**A2. → B — It halved.**

$Z = fB/d$: doubling $d$ **halves** $Z$ — the inverse law from slide 44. Sanity check with the slide's own table: $d=8$ → 48.6 m, $d=16$ → 24.3 m, $d=70$ → 5.55 m. *(Verified.)*

- A has the sign backwards (that would need $Z\propto d$).
- C would be true for the *error*, not the depth.
- D contradicts the whole premise.

**A3. → A — Because $\Delta Z = Z^2/(fB)\,\Delta d$.**

Differentiating $Z=fB/d$ gives $\lvert dZ/dd\rvert = fB/d^2$; substituting $d = fB/Z$ makes this $Z^2/(fB)$. So the sensitivity to a pixel of disparity error grows as the **square** of depth — slide 44: *"Error grows like $Z^2$."*

- B, C and D are all false: disparity is not modular, $f$ does not scale with distance, and the baseline is fixed.

**A4. → B — The blind zone grows and matching gets harder.**

Slide 39: *"Wider $B$ → better depth far away, but a bigger blind zone and harder matching."* The transcript adds the mechanism — the two views become less similar, so *"feature matching will become difficult."*

- A is false — $f$ is a property of the lens, independent of $B$.
- C is backwards — larger $B$ gives **larger** disparity ($d=fB/Z$).
- D is false; rectification is a calibration step and is unaffected by $B$'s size.

**A5. → B — The closest point between the two rays.**

Slide 60: *"Find the shortest segment $Q_1Q_2$ joining the rays — it is perpendicular to both. Its midpoint $\hat{P}$ is the estimate."* The slide notes the *better* standard is the point whose reprojections are closest, but the closest-point-between-rays is the named geometric estimate.

- A assumes the rays meet, which slide 59 says they generally do not.
- C and D are not estimates of the point.

**A6. → B — The null vector of a $4\times4$ matrix $A$.**

Slide 61: two views give four equations $\Rightarrow$ *"$A X=0$, $A$ is $4\times4$"* and *"$X=$ last singular vector of $A$."*

- A, C and D are not part of DLT.

**A7. → B — Both reproject to exactly the same images.**

Slide 31: scale the scene by $k$ and the cameras by $1/k$ and *"the projections of the scene points remain exactly the same."* Verified in F5: $u=520$ px both before and after $k=3$.

- A is false — the images cannot distinguish them, which is the *point*.
- C and D are inventions.

**A8. → A — VO is a real-time special case of SFM.**

Slide 53: *"VO is a particular case of SFM."* Slide 54: VO *"focuses on estimating the 3-D motion of the camera sequentially… and in real time."*

- B is backwards — *unordered* image sets are SFM's input.
- C is false — slide 54: bundle adjustment *"is optional"* for VO.
- D is false — VO's monocular form recovers $t$ only *up to scale* (slide 49).

### Section B — Solutions

**B1. → A, B, C, E. Reject D.**

- **A** ✔ $Z = fB/d$ closes dimensionally only with the **pixel** focal length (slide 44 uses $f=720$ px).
- **B** ✔ slide 39: *"Wider B → better depth far away."*
- **C** ✔ slide 44: *"Error grows like $Z^2$."*
- **D** ✗ **false** — $d = fB/Z$ is *linear* in $f$. A longer lens increases disparity, which is why $fB$ is the "strength" of the rig.
- **E** ✔ slide 41: *"After rectification the match is on the same row… the only unknown left is a horizontal shift: the disparity."*

**B2. → A, B, C. Reject D, E.**

This is slide 46's list, with two designed inverses.
- **A** ✔ *"White walls, sky, road: every patch looks the same."*
- **B** ✔ *"The mirror image seems to be behind the glass → wrong, too-far depth."*
- **C** ✔ *"Some points are seen by only one camera — no match at all."*
- **D** ✗ rectification is what *enables* matching (B6), not a failure.
- **E** ✗ the left–right check is the **fix** for occlusion failures (slide 7), not a cause.

**B3. → A, C, D, E. Reject B.**

- **A** ✔ slide 57: *"Each pixel defines a ray. The ray holds every 3-D point that could have made that pixel."*
- **B** ✗ slide 58: *"Three equations (x, y, z), two unknowns → normally no exact solution."*
- **C** ✔ slide 60: *"Take the closest point between the two rays."*
- **D** ✔ slide 61: *"each view gives two equations"* ($u$ and $v$).
- **E** ✔ slide 57 lists both: *"We need: both cameras' K, R, t (calibration) and the match $x\leftrightarrow x'$."*

**B4. → A, B, C, D. Reject E.**

- **A** ✔ slide 31, verbatim.
- **B** ✔ slide 31: *"Without a reference measurement, it is impossible to recover the absolute scale of the scene!"*
- **C** ✔ slide 34: *"If we impose parallelism constraints, we can get a reconstruction up to an affine ambiguity."*
- **D** ✔ slide 36 plus the parameter count: 3 (rotation) + 3 (translation) + 1 (scale) = **7**.
- **E** ✗ **more images do not remove the ambiguity.** The ambiguity is a *symmetry* of the projection: every image is invariant to it, no matter how many you take. Only an external **measurement** (known size, baseline, lidar, IMU) fixes scale.

**B5. → A, B, C, D, E. All five.**

- **A** ✔ slides 41, 39.
- **B** ✔ slide 52: *"The errors introduced by each new frame-to-frame motion accumulate over time."*
- **C** ✔ slide 56: loop closure *"reduces accumulated drift."*
- **D** ✔ slide 53: *"whose computation time grows with the number of images."*
- **E** ✔ slide 42's assumption *"dominance of static scene over moving objects"* — the transcript's "did I even move?" confusion.

### Section C — Solutions

*Each ≤100 words, three beats: define → reason → stop.*

**C1.**
A pixel does not identify a point in space — it identifies a **ray**. Every 3-D point along that ray projects to the same pixel, so the image fixes the direction $X/Z$ but says nothing about the distance $Z$. Slide 37 puts it plainly: *"One pixel = one ray. Depth along the ray is lost."* The ambiguity is exact, not merely noisier: *"small-and-near looks exactly like big-and-far."* A second view adds a **second ray** through the same point from a different centre; the two rays intersect at the point, so the point is now determined. The information the second view adds is precisely the **disparity** — the difference between the two projections — which is inversely proportional to depth.

*(99 words.)*

**C2.**
Because $d = fB/Z$, so $Z = fB/d$ is an **inverse** relationship. Differentiating gives $\lvert dZ/dd\rvert = fB/d^2$, and substituting $d=fB/Z$ turns this into $Z^2/(fB)$: the local sensitivity of depth to a pixel of disparity grows as the **square** of depth. Intuitively, a fixed pixel error is a fixed *fraction* of the disparity, and the disparity itself shrinks as $1/Z$, so the relative depth error grows as $Z$ and the absolute error as $Z^2$. Slide 44: *"Off by 1 px… 16.9 m. Error grows like $Z^2$."* Doubling the distance **quadruples** the error.

*(100 words.)*

**C3.**
Because the matched pixels are **noisy** and the camera parameters are **estimates**. Slide 59 lists the three causes: pixel noise (*"corners are found to ± a fraction of a pixel"*), calibration error (*"$K,R,t$ are estimates, not truth"*), and quantization (*"pixels are discrete"*). Each perturbs a ray slightly, and a small angular error in one ray displaces it by a distance proportional to depth, so the two rays pass each other with a gap rather than crossing. Slide 58 also notes the ideal case is over-determined — *"three equations… two unknowns"* — so no exact solution exists in general. The standard fix (slide 60) is to take the **shortest segment joining the rays and use its midpoint**; better still, choose the point whose **reprojections** best match the observed pixels, minimising reprojection error exactly as in calibration.

*(100 words.)*

**C4.**
It is the fact that reconstruction from images is only defined up to a **global similarity transform**, so the **absolute size** of the scene is unknowable. Slide 31: *"If we scale the entire scene by some factor $k$ and… scale the camera matrices by $1/k$, the projections remain exactly the same."* The images are blind to scale because projection depends only on ratios $X_c/Z_c$, which cancel. Three things remove it: a **known physical size** in the scene (a surveyed object), a **known baseline** (a calibrated stereo rig), or an **external sensor** — IMU, GPS, or lidar, which self-driving systems use to *learn* the scale, as the lecture notes. More images alone never remove it: the ambiguity is a symmetry, not noise.

*(100 words.)*

**C5.**
SFM reconstructs **both** scene and camera poses from an **unordered** image set, offline, and refines everything with full **bundle adjustment** — slide 53: *"computation time grows with the number of images."* VO is *"a particular case of SFM"* (slide 53) that runs on an **ordered** stream and estimates the camera's motion **sequentially, in real time** (slide 54), refining only locally because it cannot afford full bundle adjustment per frame. Because VO chains frame-to-frame motions, its errors **accumulate** into drift (slide 52). **Loop closure** adds the one thing frames alone cannot supply: recognising a **revisited place** constrains the whole trajectory at once, *"reduces accumulated drift,"* and yields *"a consistent camera trajectory and a global 3-D map"* (slide 56).

*(100 words.)*

### Section D — Solutions

#### D1 — Stereo depth and its error

**Key idea tested.** The inverse law, its inverse, and the $Z^2$ sensitivity.

```python
def depth_from_disparity(f, B, d):
    if f is None or B is None or d is None:
        return None
    if f <= 0 or B <= 0 or d <= 0:
        return None
    return round(f * B / d, 6)

def disparity_from_depth(f, B, Z):
    if f is None or B is None or Z is None:
        return None
    if f <= 0 or B <= 0 or Z <= 0:
        return None
    return round(f * B / Z, 6)

def depth_error(f, B, Z, dd):
    if None in (f, B, Z, dd):
        return None
    if f <= 0 or B <= 0 or Z <= 0:
        return None
    return round(Z * Z / (f * B) * dd, 6)
```

**Verified execution output:**

```
depth_from_disparity(720, 0.54, 24)   ->  16.2
depth_from_disparity(720, 0.54, 23)   ->  16.904348
disparity_from_depth(720, 0.54, 10)   ->  38.88
depth_error(720, 0.54, 16.2, 1)       ->  0.675
depth_from_disparity(720, 0.54, 0)    ->  None
depth_from_disparity(720, -0.54, 24)  ->  None
depth_from_disparity(None, 0.54, 24)  ->  None
```

**Why it works.** `depth_from_disparity` is slide 44 verbatim: $Z=fB/d$. `disparity_from_depth` inverts it, $d=fB/Z$ — the form you use to *design* a rig (F3). `depth_error` is the linearised sensitivity $\Delta Z = Z^2/(fB)\Delta d$ from B4. Note `dd` is **not** required positive: a negative `dd` correctly returns a negative error (depth moved closer), so it is not guarded.

**Guarding.** The `<= 0` check rejects the two silent killers: `d = 0` (a parallel ray → infinite depth, `ZeroDivisionError`) and negative inputs. The `None` checks come **first**, because `None <= 0` raises a `TypeError` in Python 3.

**Common wrong approaches.**
- Swapping $f$ and $B$ — they are both factors of $fB$ so it "works" here, but you will get the units wrong when you substitute a physical focal length.
- Computing the *exact* error `f*B/23 - f*B/24` inside `depth_error` — correct numerically but it is a *different function*; the question asks for the first-order $Z^2/(fB)$ rule.
- Forgetting to guard `d = 0` — passes every test except the degenerate one.

---

#### D2 — Triangulate two rays

**Key idea tested.** The closest point between two (usually skew) rays — the geometric core of triangulation (slide 60).

```python
import math

def _dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _mul(a, s): return (a[0]*s, a[1]*s, a[2]*s)

def triangulate_rays(O1, d1, O2, d2):
    pts = [O1, d1, O2, d2]
    if any(p is None for p in pts):
        return None
    if any(not hasattr(p, '__len__') or len(p) != 3 for p in pts):
        return None
    n1 = math.sqrt(_dot(d1, d1)); n2 = math.sqrt(_dot(d2, d2))
    if n1 == 0 or n2 == 0:
        return None
    d1 = _mul(d1, 1/n1); d2 = _mul(d2, 1/n2)      # unit directions
    a = _dot(d1, d1); b = _dot(d1, d2); c = _dot(d2, d2)
    w0 = _sub(O1, O2); dd = _dot(d1, w0); e = _dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:                         # parallel rays
        return None
    lam = (b * e - c * dd) / denom
    mu  = (a * e - b * dd) / denom
    P1 = _add(O1, _mul(d1, lam)); P2 = _add(O2, _mul(d2, mu))
    mid = ((P1[0]+P2[0])/2, (P1[1]+P2[1])/2, (P1[2]+P2[2])/2)
    return tuple(round(v, 6) for v in mid)
```

**Verified execution output:**

```
O1=(0,0,0) d1=(0,0,1)  O2=(1,0,5) d2=(0,-1,0)  ->  (0.5, 0.0, 5.0)
O1=(0,0,0) d1=(2,0,6)  O2=(4,0,0) d2=(-2,0,6)  ->  (2.0, 0.0, 6.0)   (rays meet)
parallel rays                                  ->  None
zero direction                                 ->  None
malformed (len 2)                              ->  None
```

**Why it works.** Write the rays as $O_1+\lambda d_1$ and $O_2+\mu d_2$. The shortest segment is perpendicular to **both**, which gives two linear conditions:

$$d_1\cdot(w_0 + \lambda d_1 - \mu d_2) = 0, \qquad d_2\cdot(w_0 + \lambda d_1 - \mu d_2) = 0, \qquad w_0 = O_1 - O_2.$$

Collecting terms: $\lambda a - \mu b = -d$ and $\lambda b - \mu c = -e$ with $a=d_1\!\cdot\!d_1$, $b=d_1\!\cdot\!d_2$, $c=d_2\!\cdot\!d_2$, $d=d_1\!\cdot\!w_0$, $e=d_2\!\cdot\!w_0$. Solving the 2×2 system gives $\lambda=(be-cd)/(ac-b^2)$ and $\mu=(ae-bd)/(ac-b^2)$. With **unit** directions $a=c=1$ and the determinant is $1-b^2$, which vanishes exactly when $b=\pm 1$, i.e. when the rays are **parallel** — the `denom` guard.

**Worked example by hand.** $O_1=(0,0,0)$, $d_1=(0,0,1)$, $O_2=(1,0,5)$, $d_2=(0,-1,0)$. Already unit. $w_0=(-1,0,-5)$; $b=0$; $d=d_1\!\cdot\!w_0=-5$; $e=d_2\!\cdot\!w_0=0$. So $\lambda=(0\cdot0-1\cdot(-5))/1=5$ and $\mu=(1\cdot0-0)/1=0$. Then $P_1=(0,0,5)$, $P_2=(1,0,5)$, midpoint $=\mathbf{(0.5,0,5)}$. ✔ The gap is the unit distance in $x$ between the two lines; the estimate splits it.

**Normalisation matters.** The formula assumes **unit** directions. Without dividing by $n_1,n_2$ the recovered $\lambda,\mu$ would be scaled wrongly — the numbers would look plausible and be wrong, which is the worst kind of bug.

**Common wrong approaches.**
- Returning $P_1$ or $P_2$ instead of the **midpoint** — slide 60 says the estimate is the midpoint of the shortest segment.
- Skipping normalisation (above).
- Not guarding the parallel case — `denom ≈ 0` produces a division-by-near-zero and enormous coordinates rather than `None`.

---

#### D3 — Apply and compose camera transforms

**Key idea tested.** The VO building block: $C_k = C_{k-1}T_k$ (slides 39–41).

```python
def apply_transform(T, X):
    if T is None or len(T) != 4 or any(len(r) != 4 for r in T):
        return None
    if X is None or len(X) not in (3, 4):
        return None
    x = list(X) + ([1.0] if len(X) == 3 else [])
    y = [sum(T[r][c] * x[c] for c in range(4)) for r in range(4)]
    if y[3] == 0:
        return None
    return tuple(round(y[i] / y[3], 6) for i in range(3))

def compose(A, B):
    if A is None or B is None:
        return None
    if len(A) != 4 or len(B) != 4:
        return None
    if any(len(r) != 4 for r in A) or any(len(r) != 4 for r in B):
        return None
    return [[round(sum(A[i][k] * B[k][j] for k in range(4)), 6) for j in range(4)]
            for i in range(4)]

def chain_transforms(transforms):
    C = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    out = [C]
    for T in transforms:
        C = compose(C, T)
        if C is None:
            return None
        out.append(C)
    return out
```

**Verified execution output:**

```
T1 = translate x by 1 ; T2 = translate z by 2
apply_transform(T1, (0,0,0))    ->  (1.0, 0.0, 0.0)
compose(T1, T2)                 ->  [[1,0,0,1],[0,1,0,0],[0,0,1,2],[0,0,0,1]]
chain_transforms([T1,T2])       ->  3 poses; C0 = I ; final translation (1,0,2)
apply_transform(Rz90, (1,0,0))  ->  (0.0, 1.0, 0.0)
apply_transform([[1,0],[0,1]], (0,0,0))  ->  None
```

**Why it works.** `apply_transform` is a homogeneous 4×4 multiply followed by dehomogenisation — Class 11's "rotate, translate, then divide," in one matrix. `compose(A,B)` is ordinary matrix multiplication, and `chain_transforms` applies it left-to-right, $C_k = C_{k-1}T_k$ — slide 41's *"concatenate these transformations to recover the full trajectory."* The list starts with the identity because that is $C_0$, the pose at $k=0$ (slide 40: *"the camera pose at the instant $k=0$, which can be arbitrarily set by the user"*).

**Order is the whole point.** `compose(A,B)` returns $A\cdot B$, so `chain_transforms([T1,T2])` computes $C_1 = I\,T_1 = T_1$ then $C_2 = T_1T_2$. Because these are all pure translations here they commute, so the translation-only test would **not** catch a reversed order — that is why `apply_transform` is tested separately with a **rotation**, where order matters. `Rz90·(1,0,0)ᵀ = (0,1,0)ᵀ` is the visible proof.

**Common wrong approaches.**
- Dehomogenising with the wrong component, or forgetting to at all — a rigid transform has last row `[0,0,0,1]`, so `y[3]=1` and the bug is invisible on translation-only tests.
- `chain_transforms` returning only the final pose, not the list including the identity.
- Building the product as element-by-element rather than a real matrix product — fine for diagonal matrices, silently wrong for rotations.

### Section E — Solutions

**Model answer covering the chain.**

1. **Task.** **Stereo**, because the system needs *metric* depth in a single time step. A single calibrated camera fixes only the **direction** to the pedestrian — it gives $X/Z$, never $Z$ (B1). Two cameras separated by a known baseline convert the pixel **disparity** into distance by $Z=fB/d$. That is a geometric measurement, not a learned guess, which is exactly what a safety function needs.
2. **Data.** Before any depth can be computed you need: the **intrinsics** $K$ ($f, c_x, c_y$) and the **rectification** that makes the pair row-aligned (isolating a single horizontal shift). Intrinsic: $f, c_x, c_y$ — fixed properties of the camera, from Class 11's checkerboard calibration. Rig property, not intrinsic: the **baseline** $B$ — a geometric fact about how the two cameras are mounted, measured once. Both must exist, or $fB/d$ has a free parameter.
3. **Model.** The equation chain is
$$d = u_L - u_R \;\to\; Z = \frac{fB}{d} \;\to\; X = \frac{(u_L-c_x)Z}{f},\quad Y = \frac{(v-c_y)Z}{f}.$$
With $u_L=700$, $u_R=690$: $d = 10$ px, so
$$Z = \frac{700\times0.54}{10} = \mathbf{37.8\ m}, \qquad X = \frac{(700-640)\times37.8}{700} = \mathbf{3.24\ m}, \qquad Y = \frac{(400-360)\times37.8}{700} = \mathbf{2.16\ m}.$$
4. **Metric.** One pixel of disparity error at this depth:
$$\Delta Z = \frac{Z^2}{fB}\Delta d = \frac{37.8^2}{700\times0.54}\times 1 = \frac{1428.8}{378} = \mathbf{3.78\ m}.$$
That is a **10%** depth error from a single pixel — at 37.8 m the pedestrian's position is known only to ±3.8 m. The ratio at 30 m versus 10 m: $\Delta Z \propto Z^2$, so the ratio is $(30/10)^2 = \mathbf{9\times}$ worse at 30 m than at 10 m.
5. **Failure mode.** At 60 m, $d = fB/Z = 378/60 = \mathbf{6.3\ px}$ — not below a pixel yet, but the *error* is $\Delta Z = 3600/378 = 9.5$ m, i.e. **16%**. Beyond about 200 m the disparity approaches one pixel and sub-pixel interpolation is all that stands between you and metres of error. The system must therefore **(a) refuse to report depth beyond a range where $d$ is large enough relative to the measurement noise**, and **(b) rely on temporal filtering / tracking** (the object is seen in many frames, so disparity can be averaged) rather than trusting a single frame. Reporting a confident 60 m when $d\approx 6$ px is the failure mode; the honest output is a large uncertainty.

**Sub-questions.**

**E-a — doubling the baseline.** New $B=1.08$ m at 30 m: $d = 700\times1.08/30 = \mathbf{25.2\ px}$ (was 12.6 px). New one-pixel depth error:
$$\Delta Z = \frac{Z^2}{fB}\Delta d = \frac{900}{700\times1.08} = \mathbf{1.19\ m}$$
(against $900/378 = 2.38$ m before — **halved**). But two costs: **(i) a larger blind zone** — the near wedge seen by only one camera grows in proportion to $B$ (slide 39: *"a bigger blind zone"*), so obstacles very close to the bumper lose depth; and **(ii) harder matching** — the two views differ more, so texture that matched at 0.54 m may not at 1.08 m (the transcript: *"if they are far, the views will be different. The feature matching will become difficult"*), and occlusion of the near-field increases.

**E-b — why monocular depth is not a metric safety input.** A monocular network uses **learned priors** — slide 37's own hint: *"learned priors about object sizes and scenes."* It has seen many cars and roads and learned typical sizes/proportions, so it can output a plausible depth from *appearance* alone. But that is **statistical**, not geometric: the same image is consistent with a small object close and a large object far (B1), and the network resolves it by what it learned, not by measuring. Consequences for a safety threshold: it can be confidently wrong on an **out-of-distribution** object (a ladder, a child, an animal it has rarely seen), it has no principled uncertainty to compare against a 3 m threshold, and it cannot be validated by a geometric argument — only by endless empirical testing. Stereo, by contrast, gives a depth whose *error is computable* from $Z$, $f$, $B$ and pixel noise. For a stop-at-3-m function, that difference is the whole argument.

**E-c — why VO drifts, and why stereo VO scales better.** VO estimates each frame-to-frame $T_k$ from noisy matches, then chains them: $C_k = C_{k-1}T_k$. Every error in any $T_k$ is **carried forward forever** — slide 52: *"the errors introduced by each new frame-to-frame motion accumulate over time… a drift of the estimated trajectory from the real one."* The fix is to recognise a **revisited place** (loop closure), which constrains the trajectory globally and *"reduces accumulated drift"* (slide 56), with **bundle adjustment** refining it offline. On scale: monocular 2-D→2-D VO recovers $E$, then $R$ and $t$ by decomposition — but $t$ comes out only **up to scale** (slide 49's *"compute relative scale and rescale t"*), because a scaled-up scene with proportionally scaled cameras gives identical images (B13). Stereo VO (slide 50) instead **triangulates each pair** to actual 3-D points using the **known baseline**, so the landmarks are metric from the first frame and $T_k$ is in metres — the scale is fixed by a measurement (B) rather than left as a free parameter.

### Section F — Solutions

**F1.** *Formula:* $Z = fB/d$.

$$d=24:\quad Z = \frac{720\times0.54}{24} = \frac{388.8}{24} = \mathbf{16.2\ m}$$
$$d=23:\quad Z = \frac{720\times0.54}{23} = \frac{388.8}{23} = \mathbf{16.9043\ m}$$

*Depth error of a one-pixel slip:* $16.9043 - 16.2 = \mathbf{0.70\ m}$ (the slide's rounding gives 16.9 m). The first-order rule gives $Z^2/(fB) = 16.2^2/388.8 = 0.675$ m — close, and the reason the slide says *"Error grows like $Z^2$."*

*Disparity for $Z=10$ m:*
$$d = \frac{fB}{Z} = \frac{388.8}{10} = \mathbf{38.88\ px}$$

*Interpret:* at 16 m, one pixel already costs 70 cm — this is the *"pixel error is amplified with distance"* point in numbers. Note the asymmetry: to measure a **near** object you need **many** pixels of disparity, which is easy; to measure a **far** object you have almost none, which is where stereo hurts.

**F2.** *Formula:* $d=u_L-u_R$; $Z=fB/d$; $X=(u_L-c_x)Z/f$; $Y=(v-c_y)Z/f$.

$$d = 520-480 = \mathbf{40\ px}, \qquad Z = \frac{800\times0.10}{40} = \mathbf{2.00\ m}$$
$$X = \frac{(520-640)\times2.00}{800} = \frac{-240}{800} = \mathbf{-0.30\ m}, \qquad Y = \frac{(300-360)\times2.00}{800} = \mathbf{-0.15\ m}$$

*One-pixel error at 2 m:*
$$\Delta Z \approx \frac{Z^2}{fB}\Delta d = \frac{4}{80}\times1 = \mathbf{0.05\ m} = 5\ \text{cm}$$

*At 4 m* ($d = 80/4 = 20$ px):
$$\Delta Z \approx \frac{16}{80}\times1 = \mathbf{0.20\ m} = 20\ \text{cm}$$

*Interpret:* doubling the distance **quadruples** the error, exactly as the slide's *"5 cm at 2 m but 20 cm at 4 m"* states. This is why $fB$ is called the strength of the rig and why a wide-baseline rig is mandatory for long-range work. Note also both $X$ and $Y$ are negative — the point is left of and above the optical axis (remember $v$ grows *downward*, so $v<c_y$ means above).

**F3.** *Formula:* $d = fB/Z$ and $\Delta Z \approx Z^2/(fB)\Delta d$, with $Z=50$ m, $f=720$ px.

| $B$ | $d = 720B/50$ | $\Delta Z$ for 1 px $= 2500/(720B)$ |
|---|---|---|
| 0.10 m | $\mathbf{1.44\ px}$ | **34.72 m** |
| 0.54 m | $\mathbf{7.78\ px}$ | **6.43 m** |
| 1.00 m | $\mathbf{14.40\ px}$ | **3.47 m** |

*Interpret:* the pattern is stark. The narrow rig produces less than **1.5 px** of disparity at 50 m — barely one pixel of signal, so a one-pixel error is a **35-metre** depth error, useless for a safety function. The wide rig produces **14.4 px** and cuts the error to **3.47 m** (a factor of 10). This is the quantitative form of slide 39's *"Wider B → better depth far away."* The **trade-off** the numbers set up: the wide rig also has a **larger blind zone** (the near region seen by only one camera scales with $B$) and **harder matching** (the two views are less similar). So baseline is chosen to fit the *range* the system must cover — 1 m for far-field, centimetres for near-field grasp.

**F4.** *Formula:* closest point between two rays — the midpoint of the shortest segment joining $O_1+\lambda d_1$ and $O_2+\mu d_2$, where the segment is perpendicular to both.

Rays are already unit: $d_1=(0,0,1)$, $d_2=(0,-1,0)$. $w_0=O_1-O_2=(-1,0,-5)$.
$$b=d_1\!\cdot\!d_2=0,\quad d=d_1\!\cdot\!w_0=-5,\quad e=d_2\!\cdot\!w_0=0$$
$$\lambda=\frac{be-cd}{ac-b^2}=\frac{0-1(-5)}{1}=5, \qquad \mu=\frac{ae-bd}{ac-b^2}=\frac{1(0)-0}{1}=0$$

So the closest points are $P_1=O_1+5d_1=(0,0,5)$ and $P_2=O_2+0=(1,0,5)$, giving

$$\hat{P} = \frac{P_1+P_2}{2} = \mathbf{(0.5,\ 0,\ 5.0)}.$$

*(Verified by execution.)*

*What it represents:* the **estimated 3-D point** from the two matched rays — here the rays are skew (ray 2 runs along $-Y$ at $x=1$, $z=5$; ray 1 is the $z$-axis), so they never meet, and the estimate splits the one-unit gap in $x$.

*If $d_2$ becomes parallel to $d_1$:* then $b = d_1\!\cdot\!d_2 = \pm1$, so $ac-b^2 = 1-1 = 0$ and the system is **singular**. There is no unique closest pair — every point on one ray is equidistant from the other — and the correct behaviour is to return **`None`**, not a garbage coordinate. (Verified: parallel input → `None`.)

**F5.** *Formula:* $u = f\,X_c/Z_c + c_x$, with $X_c = X - C$ (camera at $C$).

*Unscaled:* $X_c = (1-0,\,1-0,\,7-2) = (1,1,5)$.
$$u = 1000\times\frac{1}{5} + 320 = 200 + 320 = \mathbf{520\ px}$$

*Scaled by $k=3$:* camera $C'=(0,0,6)$, point $X'=(3,3,21)$, so $X_c'=(3,3,15)$.
$$u = 1000\times\frac{3}{15} + 320 = 200 + 320 = \mathbf{520\ px}$$

*Name and consequence:* this is the **scale ambiguity** of structure from motion (slide 31). Scaling the *entire world* — cameras and points together — leaves every image coordinate **identical**, because the projection depends only on ratios $X_c/Z_c$ in which the factor $k$ cancels. The consequence, verbatim from the slide: *"Without a reference measurement, it is impossible to recover the absolute scale of the scene!"* The reconstruction's **shape** is correct; its **size** is unknowable from images alone, and needs a known object size, a known baseline, or an external sensor (IMU/GPS/lidar).

**F6.** *Formula:* $d = u_L-u_R; \quad Z = fB/d$.

$$d = 400-330 = \mathbf{70\ px}$$
$$Z = \frac{720\times0.54}{70} = \frac{388.8}{70} = \mathbf{5.5543\ m}$$

*(This matches slide 42's near-object point at $d=70$ → ≈5.6 m.)*

*Search size* (Class 11 slide p.47, reused): with a 720×1280 image,
$$\text{whole-image search} = 720\times1280 = \mathbf{921{,}600\ candidates}$$
$$\text{along the epipolar line} = 1\times1280 = \mathbf{1{,}280\ candidates}$$
$$\text{reduction} = \frac{921{,}600}{1{,}280} = \mathbf{720\times}$$

*Interpret:* after rectification the epipolar line is a **row**, so the match is a pure horizontal shift on that row — the search is one-dimensional. The line length is the image width, 1280 px, which is where the second number comes from. This is what makes dense stereo (F2's disparity at *every* pixel) computationally feasible at all.

**F7.** *Formula:* $\text{drift} = \text{relative error} \times \text{distance}$ (accumulated along the chain of $T_k$).

$$\text{drift} = 1\% \times 400\ \text{m} = \mathbf{4\ m}$$

*Interpret:* after a 400 m loop the robot believes it is at the start but is **4 m** off — its map and its position have drifted apart at the rate VO's slide 43 quotes (*"relative position error 0.1%–2%"*; 1% is mid-range). *Loop closure* (slide 56): when the robot **recognises that it has returned to a place already in the map**, that recognition is a hard constraint tying the end of the trajectory to the beginning, so the optimiser **distributes the 4 m of error back along the whole loop** and the trajectory snaps shut — *"reduces accumulated drift… produces a consistent camera trajectory and a global 3-D map."* The offline refinement that performs the global optimisation is **bundle adjustment** (B14), which jointly re-optimises every pose and landmark.

**F8.** *The ladder* (slides 32–37), with the constraint at each rung:

| rung | free parameters | what constraint unlocks the **next** rung |
|---|---|---|
| **Projective** | **15** ($Q$ general 4×4, up to scale) | impose **parallelism** → affine |
| **Affine** | **12** ($A_{3\times3}$ + $b_{3\times1}$) | impose **orthogonality** of the axes → similarity |
| **Similarity** | **7** ($sR$: 3 rot + 1 scale, + $t$: 3) | fix the **scale** with a reference measurement → Euclidean |
| **Euclidean** | **6** ($R$: 3, $t$: 3) | — |

*One sentence each:*
- *Projective → affine:* the constraint is that **parallel lines must stay parallel**, which removes the projective terms that make them converge.
- *Affine → similarity:* the constraint is that mutually **orthogonal axes must stay orthogonal** (and lengths scale uniformly), which forces $A$ from a general $3\times3$ down to $sR$.
- *Similarity → Euclidean:* the only remaining freedom is the **global scale** $s$; a known physical length — a surveyed object, a known baseline, or lidar/IMU data — pins it, leaving the metric rotation and translation.

*Interpret:* each rung is a **symmetry of the projection** that images cannot see. More images do **not** help (B4); only extra **information** — a geometric assumption or an external measurement — does. This is why the ladder is the standard way to describe what a reconstruction does and does not know.

---

## J. Sources

### Course sources

| Source | File | Detail |
|---|---|---|
| Presentation A | `Session_11_12.pdf` pages 37–70 | Modules 4 (Stereo), 6 (Triangulation) + *"Stereo 3D Reconstruction"*; text layer |
| Presentation B | `Session_12.pdf` slides 38–93 | image-only, rotated 90°; stereo-as-optimisation, optical flow, **structure from motion**, **visual SLAM** |
| Transcript | `Session_12_—_3D_Vision_&_Motion_….txt` | 28/9/2026, 9:30 AM, 120 min; 14k words |
| **Professor repository** | **None in this workspace** | — |

### Key slides

| Slides | Topic |
|---|---|
| 37–39 | One pixel = one ray; two eyes; **baseline** (human 6.5 cm, KITTI 54 cm) |
| 41–42 | **Disparity** $u_L-u_R=70$; **near shifts more**; $Z=fB/d$ plot |
| 44–46 | **The depth equation** (derivation + $f{=}720,B{=}0.54,d{=}24\to16.2$ m); depth map; where stereo struggles |
| S12 · 6–13 | **Stereo as optimisation**: pragmatics, occlusions, left–right check, **linear disparity per region** $d=ax+by+c$ |
| S12 · 15–22 | **Optical flow**: rigid motion, along epipolar lines, "how far = depth × movement" |
| S12 · 23–37 | **Structure from motion**: problem, Necker cube, **scale/projective/affine/similarity ambiguity**, DOF ladder |
| 57–61 | **Triangulation**: two rays, ideal vs real, closest point, **DLT** ($A$ 4×4), rectified shortcut |
| S12 · 38–43 | **Visual odometry** formulation, $T_k$, $C_k$, assumptions, advantages (0.1–2%) |
| S12 · 44–52 | **VO pipeline**: detection, matching/tracking, motion (2D-2D/3D-3D/3D-2D), local optimisation, **drift** |
| S12 · 53–56 | **VO vs SFM**; pose from correspondences; **stereo VSLAM** 9-step loop; **bundle adjustment + loop closure** |
| *reconstruction* | Full stereo→depth→mesh flow (SIFT, Lowe 0.75, RANSAC F, SGBM, ICP, Poisson) |

### Supplementary

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed.) — triangulation, DLT, the ambiguity hierarchy, all of Modules 4–6. https://www.robots.ox.ac.uk/~vgg/hzbook/
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — stereo, optical flow, structure from motion, SLAM chapters. https://www.cs.cmu.edu/~./szeliski/Book/
- Middlebury Stereo Datasets and Evaluation — the benchmark slide 14 names as *"the best known, by a long way."* https://vision.middlebury.edu/stereo/
- Scharstein & Szeliski, "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms" — the standard dense-stereo reference. https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf
- Scaramuzza & Fraundorfer, "Visual Odometry" tutorial series (IEEE RAM, 2011) — the canonical VO survey; matches the slide 44–54 pipeline. https://rpg.ifi.uzh.ch/docs/RAM11_Scaramuzza.pdf
- Milford / Mur-Artal ORB-SLAM papers — the modern feature-based visual SLAM reference. https://arxiv.org/abs/1502.00956
- OpenCV `triangulatePoints`, `reprojectImageTo3D`, `findEssentialMat`, `recoverPose`, `StereoSGBM` docs. https://docs.opencv.org/4.x/

### Unresolved

1. **`Session_12.pdf` is image-only**, so slide titles are OCR-derived. Text is legible at 150 DPI; two slides (S09, S17–S19) carry no extractable text and are figures only. The numbered page tags on the figures (e.g. 37–61) belong to the *merged* `Session_11_12.pdf`, not to `Session_12.pdf`'s own 56-page numbering — page references in §J use the figure's printed tag.
2. **The lecture transcript's audio is partly degraded** (the first two minutes and the closing attendance are garbled). The substantive technical passages — disparity, baseline trade-off, the optimisation reformulation, occlusion/left–right, the ambiguity ladder, VO drift, loop closure — are all legible and were used directly.
3. **The deck's "Module 7 — Putting it together"** announced on the roadmap is never delivered; its material appears as the *Stereo 3D Reconstruction* pages (S12 p.1/5–5/5) at the end of `Session_11_12.pdf`.
4. **The transcript's aside on the fluid-dynamics Millennium-prize question** (navier–Stokes regularity) is context, not examinable CV content; it is not reproduced in the guide.

---

*Questions A–F are original, written against the Session 12 sources. Every numerical answer and all three code solutions were executed and verified; the scale-ambiguity demonstration (F5) and the ray-intersection example (F4/D2) were checked numerically, and the guide was built example-first so that each concept in §B carries a worked walkthrough.*
