# Class 11 — Camera Geometry

> **Exam weight: Tier 1, and the highest remaining value.** This is the one lecture where a single formula carries 20 marks. Section F here is pure arithmetic on three ideas — **projection**, **$Z = fH/h$**, and **DOF counting** — and the slides hand you worked numbers for all three. If you can do this class, Class 12 falls out of it.
>
> **Sources:** `Session_11_12.pdf` pages 1–48 (*"From Pixels to Points"*) · `Session_11_—_Camera_Geometry_….txt` (transcript, 11k words).
> **Professor repository: none in this workspace.**

**Scope note.** The deck is one file covering Sessions 11 and 12. This class is **pages 1–48**: Modules 0–3 (why 3D, projection, homography, calibration) plus Module 5 (epipolar geometry), which the instructor taught here as well — he says *"this will come in your exams."* Session 12 picks up with stereo and triangulation.

---

## A. Lecture overview

**The one-sentence version.** A photo throws away one thing — **depth**. Everything in this class is about getting it back, and the whole lecture is one idea in three stages:

```
  a 3D point  ──divide by Z──▶  a 2D pixel      (forward:  how images are formed)
  a 2D pixel  ──needs K, R, t▶  a 3D point      (inverse:  recovering 3D)
```

**Prerequisites:** Class 01 (matrix algebra), Class 02 (SIFT/RANSAC/homography), Class 06 (IoU).

**Difficulty: Intermediate.** The algebra is one division and one 3×3 multiply. The conceptual load is understanding **what each parameter means physically** and **why two frames exist**.

**Where it sits.** Class 10 ended at pixels. This class restores depth. Class 12 uses it for stereo, structure-from-motion and SLAM.

---

## B. The ideas

### B1. Why depth is the thing you lose

Slide p.2 makes the argument in one picture: **project · 3D → 2D**. A photo flattens the 3D world onto a 2-D sensor, and *"depth is the dimension that gets lost."*

The instructor's version (transcript): *"These rays falling on the object are reflected and then they are falling on a plane which I call the image… **I don't know where exactly in 3D they were coming from. So I lost the dimension of depth.**"*

**The consequence, stated as the whole motivation** (slide p.11): a toy car held close and a real car far away **appear identical**. Appearance can lie. So the rule *"bigger means closer"* is not a shortcut you can rely on — it needs a second observation.

The ladder of perception (slide p.4) is the cleanest summary of the course so far:

| | Question | Output |
|---|---|---|
| Classification | What is in the image? | "car" |
| 2D detection | What, and where in the image? | a pixel box |
| 3D localization | What, and **where in the world**? | position, size, heading |

> *"Each step adds geometry. The last one needs depth — which a single image does not give you directly."*

**The instructor's own demonstration of why two viewpoints are needed** is worth trying: close one eye and try to touch-match your two index fingers, or grab an object you already know the location of — *"you will be finding it very difficult."* With both eyes it is effortless, because you are doing **triangulation** continuously (Session 12).

### B2. The pinhole camera — an idealisation you can actually solve

**The problem.** Real lenses bend light in complicated ways. To get an equation, block every ray except those through one tiny hole.

**Mechanism (slide p.9):**
- each scene point → **exactly one** image point (the ray through the hole is unique);
- the image comes out **upside-down**;
- image size depends on **$f$**, the hole-to-sensor distance.

**The instructor's physical account (transcript)** is the intuition to keep: the pupil is a hole that expands and shrinks to control how much light enters; a camera's aperture is the same idea. He then models the camera as the **simplest** one — a pinhole — *"because it is very simpler compared to other types of camera… and then other cameras basically work in similar way but then they have little different projection formulas."*

**The $f$ intuition (transcript):** *"image size depends on $f$ as it is nothing but the distance between the plane and the opening section… So if I increase the $f$, what will happen with image size? … **$F$ controls the size of image.**"*

**The $f$-increase thought experiment** the slide asks for: if you move the image plane further from the hole, the same ray spreads over a **larger** image. So a long focal length magnifies — which is exactly the tradeoff slide p.15 shows: **short $f$ → wide view · long $f$ → narrow view**.

**The virtual-plane trick (slide p.9):** the real image is inverted. *"Trick: imagine a virtual plane in front of the hole — upright image, same math."* Nobody actually uses an inverted sensor; this avoids a sign flip in every formula downstream.

### B3. Similar triangles — the entire projection law

This is the one derivation you must be able to redo from memory.

**Setup (slide p.10).** Two triangles share the angle at the optical centre $O$: the big one through the 3-D point $P=(X,Y,Z)$, the small one through its image $p=(x,y)$ on the image plane at distance $f$.

Because they are similar, the ratios of corresponding sides are equal:

$$\frac{y}{f} = \frac{Y}{Z} \quad\Longrightarrow\quad y = f\cdot\frac{Y}{Z}, \qquad x = f\cdot\frac{X}{Z}$$

> **The insight that makes it stick:** *"Perspective is one division by depth $Z$. No matrices yet."*

**Intuition.** $Z$ is in the denominator. Push an object twice as far away and **every** coordinate in its image halves — including its offset from the centre. That single fact is perspective.

**Slide p.11 puts numbers on it:** the same car at $Z=40$ m appears 60 px; at 20 m, 120 px; at 10 m, 240 px. Halving the distance doubles the size. The implied constant is $f\cdot H = 2400$ px·m, so with a 1.5 m car, $f = 1600$ px. *(All three verified.)*

**Two consequences the slide draws out:**
- **Parallel lane lines converge** at a *vanishing point* — the visual signature of the same $1/Z$ law.
- Because size $\propto 1/Z$ **and** you cannot tell a toy from a real car, **a single image cannot give you depth**. This is where the lecture loops back to B1.

### B4. Three coordinate frames — and why a world frame is not optional

Slide p.12 sets up the pipeline, then slides p.12–16 define each frame:

| Frame | Unit | Origin | Moves with… |
|---|---|---|---|
| **World** | metres | a scene-fixed point you choose once | the scene |
| **Camera** | metres | the camera centre $C$ | **the camera** |
| **Image** | pixels | the **top-left pixel** | — |

> *"A point travels: world → camera (rigid motion) → image (projection)."*

**Why we need a world frame — the instructor's live Q&A** (transcript). Students correctly observed that you could just work in camera coordinates. He pushed for the reason:

> *"the camera might be in motion and if you don't know the relative motion… it becomes difficult to track what you are actually seeing. **That is why you need a world reference frame.** With respect to this world reference frame you are keeping track of the relative motion of the camera itself."*

And the demonstration: a point 5 m from the camera, then 8 m from it because **the camera moved**, not the object. From the camera frame the distance changes; **from the world frame the object is exactly where it always was.** You cannot compute the camera's relative motion unless there is something that does not move to measure it against.

**Camera frame conventions (slide p.13 / transcript).** Origin at the camera (the pinhole centre); **$Z$ points where the camera looks** (the optical axis); **X right, Y down** — *"the OpenCV convention."* The image plane sits at $Z = f$, and the axis pierces it at the **principal point**.

> *"This is a convention used, you can use anything but generally this is what is used. If you use something else you can create confusion. That is why all the libraries and implementation that you will use in Python, this is the general convention used."*

**Image frame (transcript):** *"the first pixel that you have becomes your origin, then in this direction you have $u$ and then in the vertical direction you have $v$."*

### B5. Intrinsics $K$ — the camera's personality

Slide p.14 converts the similar-triangle result from **metres to pixels**:

$$u = f_x\cdot\frac{X}{Z} + c_x, \qquad v = f_y\cdot\frac{Y}{Z} + c_y$$

The structure is unchanged — **scale by $f$, divide by $Z$, then shift to where the pixel grid starts.**

| Parameter | Meaning | Slide |
|---|---|---|
| $f_x, f_y$ | focal length **in pixels**; magnification. Longer $f$ → narrower view, bigger objects. **Two values because pixels may not be square.** | p.15 |
| $c_x, c_y$ | principal point — where the optical axis hits the sensor. *"Usually near, but not exactly at, the image center."* | p.15 |

**Why two focal lengths — the instructor's answer** (transcript):

> *"when your aperture is not exactly circular and it allows different ratio of rays in one direction and different ratio of rays in another direction you will have different values… it doesn't need to be circular opening, it can be a little elliptical opening. **It allows different field of views in each direction.** That is how many times you see that the image that you get is not exactly square but rectangular. That is why you have different $c_x$ and $c_y$… if it was square, then basically $c_x$ and $c_y$ could have been the same, and then $f_x$ and $f_y$ could have been the same."*

**And why they're called intrinsic** (transcript): *"intrinsic parameters of camera… means properties of self. What are the properties of the camera? Focal length — property of a camera. $C_x$, $C_y$ — properties of the camera."*

> **Intrinsics belong to the camera. Move the camera and they do not change** (slide p.15).

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

**The $c_x, c_y$ baseline the instructor gives** (transcript): *"if your resolution is 200×200 then your central pixel is 100×100."* Useful when you have no calibration at all — use the image centre as a first approximation.

### B6. Extrinsics $[R\mid t]$ — where the camera is

$$X_{cam} = R\,X_{world} + t$$

| | Meaning | DOF |
|---|---|---|
| $R$ (3×3) | rotation — roll, pitch, yaw | **3** |
| $t$ (3×1) | translation | **3** |

> **Extrinsics change every time the camera moves** (slide p.16) — this is the contrast that defines the two parameter sets.

**Why $R$ has only 3 degrees of freedom, not 9.** This is a free-mark point and the instructor raises it explicitly (*"Can someone tell me anything about the rotation matrix?"*). $R$ is constrained twice over (transcript):

1. Its columns are **orthogonal** — *"when you multiply [by its] transpose it gives you zero"* → $R^T R = I$.
2. Its columns are **unit vectors** — *"the three vectors there are all unit vectors… when you do the multiplication with itself, it will give you 1."*

**Verified:** for $R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$, $R^T R = I$ exactly. Those two constraints are also what make calibration tractable — *"based on this constraint it becomes easier to solve."*

### B7. The complete camera model

Slide p.17 packages all three steps into one equation:

$$\boxed{\ \mathbf{x} \simeq K\,[R\mid t]\,\mathbf{X}\ } \qquad \text{equivalently } \quad \mathbf{x} \simeq P\,\mathbf{X},\qquad P = K[R\mid t]$$

$$\underbrace{(X,Y,Z)}_{\text{world}} \xrightarrow{\;\text{rotate + move}\;} \underbrace{(X_c,Y_c,Z_c)}_{\text{camera}} \xrightarrow{\;\text{scale, shift, } \div Z\;} \underbrace{(u,v)}_{\text{pixels}}$$

**What $\simeq$ means (slide p.17):** *"equal up to scale: divide by the third coordinate — that is the ÷Z."* In homogeneous coordinates the division is postponed to the end, which is why no division appears in the matrix.

**$P$ is 3×4, not 4×4.** Columns 1–3 of $K$ scale and shift; column 4 is $Kt$, the projection of the origin.

**Verified numerically:** with $K=\begin{bmatrix}1000&0&320\\0&1000&240\\0&0&1\end{bmatrix}$, $R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$, $t=(1,2,3)$, and $X=(1,2,5)$:
$X_c = (-1, 3, 8)$ → $u = 1000(-1)/8 + 320 = \mathbf{195}$, $v = 1000(3)/8 + 240 = \mathbf{615}$.

**Crucially, $\mathbf{x}\simeq P\mathbf{X}$ is the *forward* equation.** One point in, one pixel out. It is **not invertible** — which is the whole problem the rest of the class solves.

### B8. Homography — the same model, specialised to a plane

**Motivation (slide p.19).** A bird's-eye view. Parking lines are parallel and evenly spaced in reality; in the camera view they converge. *"One matrix undoes the perspective — **because the ground is flat**."*

**The derivation (slides p.20–21).** Take the full camera model and impose the planar constraint. Every point on the plane satisfies $Z = 0$, so the third column of $R$ drops out of the multiplication entirely and the result collapses to a single 3×3 matrix:

$$\mathbf{x}' \simeq H\,\mathbf{x}$$

The instructor worked this live (transcript) and is emphatic that it is a **consequence**, not a new idea:

> *"I directly introduced homography to you but **it is getting derived from the same set of intrinsic equations that we have for the projection. And that is why it works in a plane, because $Z$ is zero.**"*

> *"Now, previously [in SIFT] you would use RANSAC: take 4 points in this image, 4 in the other, compute H. If it is right, many of the 100 matches will be explained; if not, most won't. RANSAC picks a different set and tries again."*

**Why it matters here:** homography needs **8 parameters**, calibration needs **10** — *"before you got rid of few of the parameters and then you had 8 parameters remaining in the homography. Now everything needs to be predicted to get you an actual projection."* The planar assumption is exactly what you give up to recover depth.

### B9. Why exactly 4 points — a DOF count you can reproduce

Slide p.23 gives the counting argument in a table:

| | Count |
|---|---|
| Unknowns in $H$ | $9 - 1$ (scale) $= 8$ |
| Equations per point pair | $2$ (one for $u$, one for $v$) |
| **Point pairs needed** | $8 \div 2 = \mathbf{4}$ |

**Condition (slide p.23):** *"no three of the four points on one line."* Verified: feeding four collinear points to my fitter returns `None`, because the 8×8 system becomes singular.

**In practice (slide p.23, transcript):** *"Use many matches and RANSAC to ignore bad ones"* → `cv2.findHomography(src, dst, cv2.RANSAC)`.

**What $H$ is actually worth in money terms** (transcript): the document-scanner app you use every few days. *"It predicts the corners… it already knows what adjustment it needs to make to make it an A4 size document… it calculates the homographic matrix to go from this document to this. **It is very simple — all maths inside the app that you are using.**"*

**Applications (slide p.25):** document scanning · inverse perspective mapping (converging lane lines → parallel, measurable lanes) · surround-view BEV from four fisheye cameras. *"All three assume the same thing: the content lies on one plane."*

### B10. Calibration — estimating $K$, $R$, $t$

**Why, when the box prints a focal length?** The instructor answers this live (transcript):

> *"the manufacturers are painting the same thing because they are mass manufacturing… but then there can be **manufacturing defects** where your $f$ might be slightly different than what they are printing. **Over time the camera's focal length might be changing** because there can be changes in its design itself with rough use."*

**Why it matters — the slide's own worked example (p.28).** A car 1.5 m tall appears 60 px. How far is it? From $Z = \frac{fH}{h}$:

| assumed $f$ | computed $Z$ |
|---|---|
| 1000 px (calibrated) | **25 m** |
| 800 px (guessed) | **20 m** |

The 5 m error, *"at 50 km/h — **the difference between braking in time and not**"*: $5 \div 13.89 = \mathbf{0.36\ s}$. *(All verified.)* **Every 3D measurement inherits the calibration error** — this is why $f$ is not a detail.

**The problem (slide p.29):** estimate $K$, $R$, $t$ from known 3-D points and their 2-D image correspondences.

| | Unknowns | DOF |
|---|---|---|
| Intrinsic | $f_x, f_y, c_x, c_y$ | 4 |
| Extrinsic | $R$ (3), $t$ (3) | 6 |

Plus distortion coefficients — the instructor notes *"every camera by default has some kind of distortion"* — but *"usually this step can be skipped for pinhole camera because the distortion in pinhole camera is generally less."*

**Why a checkerboard (slide p.30) — four reasons, all worth memorising:**
1. **Known geometry** — *"measure the square size once (e.g. 25 mm) and every corner position is known."*
2. **Precise corners** — *"black–white X-junctions can be located to a fraction of a pixel."*
3. **Easy to find automatically** — *"high contrast, regular pattern, cheap to print."*
4. **Flat** — *"all corners share $Z = 0$ — this simplifies the math (see homography)."*

**The three sub-steps (slide p.31):**
- **Detect** the grid → `cv2.findChessboardCorners(img, (8, 5))`
- **Order** consistently — *"the first corner in this image will be similar to the first corner of the second image taken from some other perspective."* This **replaces SIFT feature detection and matching entirely**: *"you have got rid of performing feature detection… using SIFT, you have got rid of feature matching in multiple images."*
- **Refine** to sub-pixel → `cv2.cornerSubPix(...)`, giving *"40 pixel positions, in a known order."*

> **Pitfall the instructor flags:** *"make sure that in each of the images that you are using, **whole of the checkerboard is visible**. If you take a partial board, your **ordering will fail**… in a partial image the first index might correspond to some middle checkerboard point."*

**The 3-D coordinates (slide p.32):** put the world origin on the board, so every corner has $Z = 0$, and corner $(i,j)$ sits at $(i\,s,\; j\,s,\; 0)$ in units of squares — multiply by $s = 25$ mm: $(0,0,0), (25,0,0), (50,0,0), (0,25,0), (0,50,0), (0,75,0), (125,75,0)$. *(Verified.)*

**The optimisation (transcript, in order):**
1. detect + order + refine corners across many images of the same board;
2. each image gives a homography between the plane and the image — *"all points lie on the $Z=0$ plane"* — and because $Z=0$ *"you can first solve for these two parameters… $R_3$ drops out"*;
3. each $H$ yields equations for $K$, $R_1$, $R_2$, $t$;
4. stack equations across images and solve, using $R^T R = I$ as the constraint;
5. solve distortion, if modelled;
6. **measure the residual.**

**Reprojection error (slide p.36) — the objective:**
$$E = \sum_i \left\|\mathbf{x}_i - \hat{\mathbf{x}}_i\right\|^2, \qquad \hat{\mathbf{x}}_i = K[R\mid t]\mathbf{X}_i$$

1. take a known 3-D corner $X$; 2. project it with the current $K,R,t$ → predicted pixel $\hat{x}$; 3. compare with the detected pixel $x$.

> **Rule of thumb (slide p.36):** *"RMS error well under ~0.5 px is good. Much larger → blurry photos, wrong corners, or a bad distortion model."*

### B11. Epipolar geometry — geometry tells you *where to look*

**The problem (slide p.38, transcript).** In SIFT you had thousands of keypoints per image and matched them exhaustively — *"computationally expensive… too many of it."* If you know where the cameras are, geometry constrains the search.

**Build-up, one step at a time:**

1. **Two camera centres $O$, $O'$ and a 3-D point $P$.** *"For a line you need 2 points and for a plane you need 3 points"* (transcript).
2. **The epipolar plane** (slide p.40) is the plane through $O$, $O'$ and $P$. *"It slices each image along a line called epipolar lines."*
3. **The epipolar line $l'$** (slide p.41) is the key result:

> *"I have lost the depth information. Now I don't know how far away this pen is… **but given a ray which intersects this pen, the pen is exactly somewhere on this ray.** I don't know how far, but I know the ray direction… **if I take the projection of these $P_1, P_2, P_3, P_4$ points on the second image, they are all falling on one line.**"*

The point could be $P_1$, $P_2$, $P_3$… along the ray. Each projects somewhere in the right image. **All of those projections fall on one line.** So search along $l'$, not the whole image.

4. **The epipole $e'$** (slide p.42) is where the other camera's centre appears in this image, and *"every epipolar line passes through it."* Slide p.43 gives three cases: general — lines meet at $e'$, *"often outside the image"*; **rectified stereo** — lines are parallel rows, $e'$ at infinity; **driving forward** — $e'$ at the centre, the *"focus of expansion."*

**Two matrices (slide p.46):**

| | **Fundamental $F$** | **Essential $E$** |
|---|---|---|
| Works on | raw pixels | normalised coords, $K^{-1}\mathbf{x}$ |
| Needs $K$? | **No** | **Yes** (calibrated) |
| DOF | **7** | **5** |
| Gives you | epipolar lines | relative pose: $R$ and $t$ (t up to scale) |
| Estimate from | **≥8 matches** (8-point algorithm), `cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)` | $E = K^T F K$ |

> Slide p.46: *"The Fundamental Matrix only tells you how pixels correspond on a screen, but the **Essential Matrix tells you how the physical camera moved** in the 3D world."*

**The payoff (slide p.47):** on a 720×1280 image, exhaustive search is **921,600 candidates**; along the epipolar line it is **≤1,280**.

$$\frac{921{,}600}{1{,}280} = \mathbf{720\times} \text{ fewer}$$

*(Both verified.)* Slide p.48: *"faster, and far fewer look-alike mistakes."*

**The division of labour (slide p.48):** *"**Geometry says where to look. Appearance says what matches.** The epipolar constraint restricts the match to a line. Compare patches (or learned features) along that line."* And the chain that ties the lecture together: **epipolar geometry → correspondence → depth.**

**The rectified special case:** *"Lines become rows, so the match is just a horizontal shift: disparity."* That is the entry point to Session 12.

---

## C. Numbers worth knowing

All verified. **These five are the class.**

### C1. The three constants of this lecture

$$\boxed{u = f_x\frac{X}{Z} + c_x \qquad v = f_y\frac{Y}{Z} + c_y \qquad Z = \frac{f\,H}{h}}$$

| Symbol | Meaning | Unit |
|---|---|---|
| $X,Y,Z$ | point in **camera** frame | metres |
| $u,v$ | pixel coordinates | pixels |
| $f_x,f_y$ | focal length | **pixels** |
| $c_x,c_y$ | principal point | pixels |
| $H$ | real object size | metres |
| $h$ | apparent size | pixels |

Worked: $f_x = 1000$, $c_x = 320$, $X = 0.5$, $Z = 10$ → $u = 1000(0.5)/10 + 320 = \mathbf{370}$.

### C2. Perspective: size $\propto 1/Z$ (slide p.11)

| $Z$ | apparent size | $Z \times \text{size}$ |
|---|---|---|
| 40 m | 60 px | 2400 |
| 20 m | 120 px | 2400 |
| 10 m | 240 px | 2400 |

Constant $fH = 2400$. With $H = 1.5$ m → $f = \mathbf{1600\ px}$.

### C3. Calibration error → a time you do not have (slide p.28)

$Z = \dfrac{fH}{h}$, $H = 1.5$ m, $h = 60$ px:

$$f = 1000 \Rightarrow Z = \frac{1000 \times 1.5}{60} = \mathbf{25\ m} \qquad f = 800 \Rightarrow Z = \mathbf{20\ m}$$

$$\Delta t = \frac{25 - 20}{50/3.6} = \frac{5}{13.89} = \mathbf{0.36\ s}$$

### C4. Degrees of freedom

| Object | Raw | Constraint | **DOF** | Notes |
|---|---|---|---|---|
| Homography $H$ | 9 | scale | **8** | 4 point pairs |
| Camera $(R,t)$ | 12 | $R^TR = I$ | **6** | |
| $K$ | — | — | **4** | $f_x,f_y,c_x,c_y$ |
| **Calibration total** | — | — | **10** | + distortion |
| Fundamental $F$ | — | — | **7** | from ≥8 matches |
| Essential $E$ | — | — | **5** | $E = K^T F K$ |

### C5. Epipolar search reduction (slide p.47)

$$720 \times 1280 = \mathbf{921{,}600} \quad\longrightarrow\quad 1 \times 1280 = \mathbf{1{,}280}$$
$$\text{reduction} = \frac{921{,}600}{1{,}280} = \mathbf{720\times}$$

---

## D. Pictures

### D1. The pinhole camera

```
            P (X,Y,Z)                    the virtual-plane trick:
              •                            real image is INVERTED,
             /|                           so imagine the plane in FRONT
            / |                          to get an upright image —
           /  |                          same maths, no sign flips.
          /   |
   image ──────•──────
   plane    f/  |  f
      x        /   |  y
              /    |
             /     |
            /______|_____
             ↙  pinhole  ↘

   ONE point → ONE image point.  That's the whole idea.
```

### D2. Similar triangles

```
              P (X,Y,Z)
              ●
             /|\
            / | \
           /  |  \
          /   |   \
    ─────●────|────●─────   small triangle
      p=(x,y)      image plane at distance f
        /|
       / |
      /  |
     /   |
    ─────●─────           O = optical centre (the pinhole)

     x / f  =  X / Z        ⇒   x = f · X / Z
     y / f  =  Y / Z        ⇒   y = f · Y / Z

         Z is in the DENOMINATOR.
         Twice as far  ⇒  half the size, half the offset.
```

### D3. Three frames, one journey

```
   WORLD FRAME               CAMERA FRAME              IMAGE FRAME
   metres, scene-fixed       metres, moves with        pixels, origin at
                             the camera                top-left

      Yw                         Yc                       v
       ↑  [R | t]                 ↑  project (÷Z)          ↑
   ────┼──→ Xw   ───────────▶  ────┼──→ Xc   ─────────▶   ──┼──→ u
       Z                         Z                        (0,0) top-left

   "the object is exactly          "from here it is          "here it is,
    where it always is"             5 m away"                  row 320"

   WHY BOTH?  a point goes 5 m → 8 m from the CAMERA because the
   CAMERA MOVED. In the world frame it never moved. Without a fixed
   frame you cannot compute the camera's own motion.
```

### D4. Perspective — why one photo is ambiguous

```
   REAL WORLD                 IMAGE
   ┌──────────┐            ┌──────────┐
   │   car    │  Z = 40 m  │  ▓▓     │  60 px
   └──────────┘            └──────────┘

   ┌──────────┐            ┌──────────┐
   │   car    │  Z = 20 m  │  ▓▓▓▓   │  120 px
   └──────────┘            └──────────┘

   ┌──────────┐            ┌──────────┐
   │  ▓▓▓▓▓   │  Z = 10 m  │  ▓▓▓▓▓▓ │  240 px
   └──────────┘            └──────────┘

   ★ same pixels = a TOY held close, or a real car far away.
     "bigger ⇒ closer" is not evidence. You need a SECOND VIEW.
```

### D5. Homography = camera model with Z = 0

```
   FULL CAMERA MODEL                     IMPOSE THE PLANE (Z = 0)
   ┌───────────┐                         ┌───────────┐
   │       K   │                         │           │
   │  [ R | t]│   3 × 4                 │     H     │   3 × 3
   └───────────┘                         │           │
        x ≃ P X                             x' ≃ H x

   ...because the 3rd column of R multiplies Z = 0,
   it contributes nothing. 9 params collapse to a homography.
   That is WHY homography only works on planes.
```

### D6. The epipolar constraint

```
        P
       /|\
      / | \            O and O' plus P define the
     /  |  \           EPIPOLAR PLANE
    ──x─┼───l'────     it cuts each image in a LINE
   left │   right     x and x' must both lie in this plane
       │  ╲
       │   ╲_________
   left image      right image

   Without geometry: search all 921,600 pixels.
   With geometry:    search along l′ — ≤ 1,280.

   P could be P1, P2, P3 … along the ray through x.
   All of them project onto the SAME line l′.
   ── every epipolar line passes through the epipole e′ ──
```

---

## E. Code

> No instructor code exists in this workspace. These are **illustrative** and were **not executed** — numpy/PyTorch are unavailable here. The *Section D* solutions in §H are pure-Python and **were** executed, with real output pasted in.

### E1. The forward model in practice

```python
K = np.array([[fx, 0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
R = ...   # 3x3 rotation, world -> camera
t = ...   # 3-vector translation

Xc = (R @ X_world.T).T + t                    # (N,3) rigid motion, no division
uv_h = (K @ Xc.T).T                          # (N,3) still homogeneous
uv   = uv_h[:, :2] / uv_h[:, 2:3]            # the ÷ Z happens HERE
```

**Everything is one 3×4 multiply, then one element-wise division.** The division must be **after** the multiply — dividing early loses the perspective coupling and gives wrong results.

### E2. Recovering intrinsics

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # list of (N,3) float32, Z == 0
    imgpoints,   # list of (N,1,2) float32, sub-pixel
    imageSize, None,
    flags=cv2.CALIB_FIX_ASPECT_RATIO)         # force fx == fy
```

`CALIB_FIX_ASPECT_RATIO` and `CALIB_ZERO_TANGENT_DIST` are the usual starting point — square pixels, no lens-distortion term — which reduces what you have to estimate and reduces overfitting on a small calibration set.

### E3. Homography

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
#                                   ^^^^^^ ^  3.0 px inlier threshold
```

The three-argument form is slide p.23's advice (*"Use many matches and RANSAC"*) in code. `mask` tells you which matches were inliers — **always inspect it**. A homography fit with 8% inliers is not a homography.

### E4. Epipolar geometry

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
E  = K.T @ F @ K                    # essential matrix (calibrated)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, mask)
#                                        ^ recoverPose, not recoverHomography:
#                                          it fixes t's scale using cheirality
```

The scale ambiguity is unavoidable from two views alone — that's why Session 12 needs a known baseline or motion.

### E5. Bugs worth memorising

| Symptom | Cause |
|---|---|
| Everything collapses toward the principal point | divided by $Z$ **before** the $K$ multiply |
| Trajectories curve when they shouldn't | $R$ not orthonormal — you built it from Euler angles without re-orthonormalising |
| Homography fits perfectly in-sample, fails on new images | no RANSAC, or too few points |
| Projection wrong only on one side of the image | $c_x,c_y$ wrong (principal point off-centre), or image was cropped/flipped after calibration |
| `recoverPose` returns garbage | $F$ estimated from too few points, or correspondences are largely wrong |

---

## F. Comparisons

**Intrinsic vs extrinsic — the distinction that gets asked every year**

| | Intrinsics $K$ | Extrinsics $[R\mid t]$ |
|---|---|---|
| What | $f_x,f_y,c_x,c_y$ | $R$ (3), $t$ (3) |
| Answers | *what camera is this* | *where is this camera* |
| Units | pixels | metres / rotation |
| DOF | 4 | 6 |
| Changes when camera moves? | **No** (slide p.15) | **Yes** (slide p.16) |
| Estimated by | checkerboard calibration | calibration (per image/view) |

**Forward vs inverse — what each equation can do**

| | $\mathbf{x}\simeq K[R\mid t]\mathbf{X}$ | $Z = \frac{fH}{h}$ |
|---|---|---|
| Direction | 3D → 2D | 2D → 3D |
| Inputs | world point + $K,R,t$ | apparent size + $f$ |
| Exact? | yes (given correct params) | only if the object's **real** size is known |
| Degenerate | — | toy car vs real car (slide p.11) |

**Fundamental vs essential (slide p.46)**

| | $F$ | $E$ |
|---|---|---|
| Works on | raw pixels | normalised $K^{-1}\mathbf{x}$ |
| Needs $K$ | no | **yes** |
| DOF | **7** | **5** |
| Yields | epipolar lines | relative pose $R$, $t$ (scale unknown) |
| Estimate from | ≥8 matches | $E = K^T F K$ |

**Three frames — pick the right one**

| Frame | Use it when you need |
|---|---|
| World | absolute position; **camera motion over time** |
| Camera | everything relative to the current view |
| Image | pixels: which column, which row |

**Why the planar assumption is a shortcut and a trap**

| | Works | Fails when |
|---|---|---|
| Homography | content on one plane; purely rotating camera | objects have **height** (slide p.26: height is *"smeared along the ground"*) |
| Full model $K[R\mid t]$ | always — at the cost of 10 parameters | never, if the parameters are known |

---

## G. Exam-focused revision sheet

**Definitions**
- **Intrinsics** — the camera's own properties; unchanged by moving it.
- **Extrinsics** — the camera's pose; changes every frame.
- **Principal point** $(c_x,c_y)$ — where the optical axis meets the sensor.
- **Focal length $f$** — hole-to-image-plane distance, expressed in **pixels**.
- **Homography** — the projection restricted to a plane; 8 DOF.
- **Epipolar plane/line/epipole** — plane through both camera centres and a point; its image trace; where the other centre projects.

**The three formulas**

$$u = f_x\frac{X}{Z} + c_x \qquad v = f_y\frac{Y}{Z} + c_y \qquad Z = \frac{fH}{h}$$
$$\mathbf{x} \simeq K[R\mid t]\mathbf{X},\qquad K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix},\quad P = K[R\mid t]\ \ (3\times4)$$
$$E = \frac{\sum\| \mathbf{x}_i - \hat{\mathbf{x}}_i\|^2}{n},\qquad \hat{\mathbf{x}}_i = K[R\mid t]\mathbf{X}_i$$

**DOF counts — memorise these**

| | DOF | How to derive |
|---|---|---|
| $H$ | **8** | $9 - 1$ scale; $8/2 = 4$ point pairs |
| $K$ | **4** | $f_x,f_y,c_x,c_y$ |
| $(R,t)$ | **6** | 3 + 3; $R^TR = I$ removes 3 from 9 |
| Calibration | **10** | $4 + 6$ |
| $F$ / $E$ | **7 / 5** | slide p.46 |

**Traps — these lose marks**
1. **Dividing by $Z$ early.** The ÷ must come after $K[R|t]$.
2. **Using $f$ in metres where the formula wants pixels.** In $u = f_x X/Z$, $f_x$ is in **pixels**.
3. **Assuming $f_x = f_y$ or $c_x = c_y$.** Only true for a perfectly square sensor with square pixels.
4. **Counting 9 parameters in $H$.** Scale is unrecoverable → **8**.
5. **Believing a single image gives depth.** Size $\propto 1/Z$ is confounded by object size (slide p.11).
6. **Applying a homography to non-planar content.** Heights get smeared (slide p.26).
7. **Confusing $F$ (7 DOF, pixels) with $E$ (5 DOF, calibrated, gives pose).**

**Checklist**
- [ ] Derive $x = fX/Z$ from similar triangles
- [ ] Compute $u,v$ from $X,Y,Z,f_x,f_y,c_x,c_y$
- [ ] Compute $Z$ from apparent size; compute the error from a wrong $f$
- [ ] State which parameters are intrinsic vs extrinsic, and what changes when the camera moves
- [ ] Write $K$, $P$ and say what $\simeq$ means
- [ ] Justify $R^TR = I$ and why $R$ has 3 DOF
- [ ] Count DOF for $H$, $K$, $(R,t)$, $F$, $E$ and the point counts they need
- [ ] Explain the epipolar constraint in one sentence and quantify the saving
- [ ] List the four reasons for a checkerboard
- [ ] Describe the calibration loop and define reprojection error

---

## H. Practice Questions

> Mirrors `session16.pdf`: **A** single-correct MCQ · **B** multi-select · **C** descriptive ≤100 words · **D** coding, attempt 1 of 3 · **E** case study · **F** numericals.

### Section A — Single-correct MCQ (8 × 1 mark)

**A1.** In the pinhole model, what happens to a point's image coordinates when you double its distance from the camera?

- A. They double
- B. They halve
- C. They stay the same
- D. They change by a constant offset

**A2.** A homography has 9 matrix entries. Why do we say it has 8 degrees of freedom?

- A. One entry is always exactly zero
- B. The matrix is defined only up to overall scale
- C. The determinant is constrained to equal one
- D. One row is constrained to be orthogonal

**A3.** Which parameter set changes when a camera is moved without changing its lens?

- A. $f_x$ and $f_y$
- B. $c_x$ and $c_y$
- C. $R$ and $t$
- D. Neither $R$ nor $t$

**A4.** A camera matrix $P = K[R\mid t]$ multiplies a 3×1 world point. What is the shape of $P$?

- A. 3×3
- B. 3×4
- C. 4×3
- D. 4×4

**A5.** You project a world point using $K[R|t]$ but divide by $Z_c$ *before* multiplying by $K$. What breaks?

- A. Nothing, the operations commute
- B. The intrinsics get applied in the wrong order
- C. The rotation matrix becomes non-orthonormal
- D. The image will be mirrored horizontally

**A6.** Which matrix requires you to have calibrated the camera before you can compute it?

- A. The fundamental matrix $F$
- B. The homography $H$
- C. The essential matrix $E$
- D. The permutation matrix

**A7.** A 1.5 m car appears 60 px tall. You estimate $Z$ using a focal length 20% smaller than the true value. What happens?

- A. $Z$ is overestimated by 20%
- B. $Z$ is underestimated by 20%
- C. $Z$ is unaffected
- D. $Z$ becomes negative

**A8.** Why must the whole checkerboard be visible in every calibration image?

- A. Partial views produce fewer than 8 equations
- B. Partial views break the corner ordering used to match images
- C. Partial views make $Z$ non-zero at the corners
- D. Partial views corrupt the focal length measurement

### Section B — Multi-select (5 × 2 marks)

**B1.** Which statements about the **intrinsic matrix $K$** are **correct**? Select all that apply.

- A. $f_x$ and $f_y$ are measured in pixels
- B. $f_x$ equals $f_y$ whenever pixels are perfectly square
- C. $c_x$ and $c_y$ must be exactly at the image centre
- D. $K$ is unchanged if the camera is moved
- E. $K$ has four degrees of freedom

**B2.** Which are **true** of a rotation matrix $R$ used in the camera model? Select all that apply.

- A. It is orthonormal
- B. It is symmetric
- C. It has determinant $+1$
- D. It has nine degrees of freedom
- E. It maps world coordinates into the camera frame

**B3.** Which are **valid reasons** to use a checkerboard for calibration? Select all that apply.

- A. Its square size can be measured once, so all corner positions are known
- B. Corners can be located to sub-pixel accuracy
- C. Every corner lies in a single plane, so $Z=0$
- D. It works without knowing the physical size of the board
- E. Its high contrast makes corners easy to find automatically

**B4.** Which statements about **homography** are **correct**? Select all that apply.

- A. It relates two views of the same plane
- B. It remains exact for objects at different heights above that plane
- C. It is exact for a purely rotating camera
- D. It needs exactly four point correspondences to be determined
- E. It is unaffected by an incorrect focal length

**B5.** Which are **correct** about **epipolar geometry**? Select all that apply.

- A. It constrains a match to lie on a line rather than anywhere in the image
- B. It requires the focal length to estimate $F$
- C. Every epipolar line passes through the epipole
- D. In rectified stereo the epipolar lines are horizontal rows
- E. It reduces a 720×1280 exhaustive search to roughly 1,280 candidates

### Section C — Descriptive (5 × 2 marks, ≤100 words each)

**C1.** Why can a single image not determine the distance to an object, even when the object is clearly visible?

**C2.** Why do we need both a world frame and a camera frame?

**C3.** Why does a homography only work when the content lies on a plane?

**C4.** Why does the focal length need to be calibrated rather than read off the camera's specification?

**C5.** What does the epipolar constraint tell you, and what does it not tell you?

### Section D — Coding (20 marks — attempt ONE of the three)

**Pure Python, standard library only.** Provide a reusable function. Handle empty and boundary inputs.

---

**Option D1 — World point to pixel**

```python
def project(K, R, t, X): ...
```

- `K`: 3×3 intrinsics. `R`: 3×3 rotation. `t`: 3-element translation.
- `X`: list of `(X, Y, Z)` **world** points.
- Returns a list of `(u, v)` rounded to 6 dp, applying `R`, then `t`, then `K`, then dividing by the third coordinate.
- Returns `None` for a malformed `K`/`R`/`t`, a point that is not length 3, or a point whose camera-frame $Z_c = 0$.
- An empty `X` returns `[]`.

Worked examples:

```
K = [[1000,0,320],[0,1000,240],[0,0,1]]
R = [[1,0,0],[0,1,0],[0,0,1]];  t = [0,0,0]
project(K,R,t,[(0,0,10),(0.5,0,10)])      ->  [(320.0, 240.0), (370.0, 240.0)]

R = [[0,-1,0],[1,0,0],[0,0,1]];  t = [1,2,3]
project(K,R,t,[(1,2,5)])                 ->  [(195.0, 615.0)]

project(K,R,t,[])                        ->  []
project([[1,2]],R,t,[(0,0,1)])          ->  None
project(K,R,t,[(0,0,0)])                 ->  None      # Zc == 0
```

**Option D2 — Fit a homography from four correspondences**

```python
def fit_homography(src, dst): ...
def apply_h(H, p): ...
```

- `src`, `dst`: lists of exactly **four** `(x, y)` pairs.
- Returns the 3×3 `H` normalised so `H[2][2] == 1`, or `None` if the system is singular (e.g. four collinear points) or the input is malformed.
- `apply_h` maps a point through `H`, returning `None` if the divisor is zero.

Worked examples:

```
src = [(0,0),(400,0),(400,300),(0,300)]
dst = [(0,0),(600,0),(600,800),(0,800)]

fit_homography(src,dst)
  ->  [[1.5, 0.0, 0.0],
       [0.0, 2.666667, 0.0],
       [0.0, 0.0, 1.0]]
apply_h(H, (400,300))        ->  (600.0, 800.0)

fit_homography(src[:3], dst[:3])                     ->  None
fit_homography([(0,0)]*4, [(1,1)]*4)                 ->  None     # collinear
```

**Option D3 — Reprojection error**

```python
def reprojection_error(K, R, t, correspondences): ...
```

- `correspondences`: list of `((X, Y, Z), (u, v))` — a known 3-D point and its detected pixel.
- Returns the RMS pixel error rounded to 6 dp; `None` for empty input or invalid input.

Worked examples:

```
reprojection_error(K, R, [0,0,0], [((0,0,10),(320.0,240.0))])   ->  0.0
reprojection_error(K, R, [0,0,0], [((0,0,10),(321.0,240.0))])   ->  1.0
reprojection_error(K, R, [0,0,0], [((0,0,10),(321.0,243.0))])   ->  3.162278
reprojection_error(K, R, [0,0,0], [])                          ->  None
```

### Section E — Case study (15 marks)

> **Scenario.** A warehouse-automation startup is retrofitting a forklift with a forward-facing camera. A safety function must stop the forklift if **anything is closer than 3 m**. The camera's datasheet states a focal length of 1200 px for a 1920×1080 sensor. In testing, a known 1.0 m pallet at a surveyed 6.0 m measures 200 px tall — but the forklift's stops are firing 1.2 s late at full speed.

Answer the chain — **Problem → task → data → model → metric → failure mode** — then the sub-questions.

1. **Task.** Which of projection, calibration, homography or stereo does this system actually need? Justify.
2. **Data.** From the pallet measurement, back out the true focal length and compare with the datasheet. State the depth error that explains the late stop.
3. **Model.** State the exact chain of equations from a detected pixel bounding box to a metric distance. Which parameters are intrinsic and which are extrinsic here?
4. **Metric.** How do you verify the calibration is good enough before shipping? State the threshold.
5. **Failure mode.** The forklift travels down a ramp. Which assumption of the pinhole model breaks first?

**Sub-questions**

- **E-a.** The team proposes replacing the calibration with a homography from the floor. Explain precisely what that would and would not give them, using the height-smearing argument.
- **E-b.** A second camera is added at the front-left. State what the epipolar constraint changes about the matching problem, with numbers.
- **E-c.** Two engineers disagree about the ordering of $R$, $t$ and the division by $Z$. Give the correct order and the symptom of each wrong ordering.

### Section F — Numericals (8 × 2.5 marks)

> Show all four steps: **formula → substitute → calculate → interpret.**

**F1.** A camera has $f_x = 1200$, $f_y = 1180$, $c_x = 954$, $c_y = 542$. Project the camera-frame points $(0.5, 0, 12)$ and $(-0.9, 0.3, 7.5)$. Then state what changes if $c_x$ is set to its approximation (image centre).

**F2.** A car 1.5 m tall is measured at 60 px. Compute its distance for $f = 1000$ px and for $f = 800$ px. Then compute the time gap at 50 km/h, and state the real-world consequence of that gap.

**F3.** A homography maps a document's corners $(0,0),(400,0),(400,300),(0,300)$ onto a flat A4-like rectangle $(0,0),(600,0),(600,800),(0,800)$. State why 4 points are needed, compute the homography, and verify it by applying it to all four corners. Then state what happens if the four points are collinear.

**F4.** An encoder has output stride 32 on a 1920×1080 image. Compute the feature map size, then at output stride 16. Then compute the camera-frame coordinates and pixel coordinates of the world point $(2.0, -1.0, 20.0)$ given $R = I$, $t = (0,0,0)$, $f_x=f_y=1000$, $c_x=960$, $c_y=540$.

**F5.** For the checkerboard calibration, count the unknowns for $K$, $R$ and $t$. Then count how many correspondences a single view supplies with an 8×5 corner grid, and state why many views are still needed.

**F6.** On a 720×1280 image, compute the number of candidate positions for exhaustive correspondence search per pixel, and the number along an epipolar line. State the reduction factor and explain what supplies the second number.

**F7.** Calibration converges with a reprojection RMS of 0.42 px on the checkerboard set. State whether the slide's rule of thumb accepts it, name three possible causes if a second dataset later gives 1.8 px, and give the RMS for a correspondence that is off by $(0.6, -0.8)$ px.

**F8.** Compute the pixel coordinates of a point at camera-frame $(-0.5, 0.25, 8)$ using $K$ with $f_x = 900$, $f_y = 900$, $c_x = 640$, $c_y = 360$. Then repeat with a rotation of 90° about $Z$ and $t = (0,0,0)$, and state which stage of the pipeline the rotation affected.

---

## Worked Solutions — Attempt the Questions First

### Section A — Solutions

**A1. → B — They halve.**

$u = f_x X/Z + c_x$: doubling $Z$ halves the $f_xX/Z$ term. Slide p.10: *"Perspective is one division by depth Z."* Slide p.11's table makes it concrete — 40 m→60 px, 20 m→120 px, 10 m→240 px.

- A and D contradict the $1/Z$ law. **C** is the trap: it is exactly what makes depth *unobservable* from one image (slide p.11).

**A2. → B — The matrix is defined only up to overall scale.**

Slide p.21: *"9 entries, but only defined up to scale → 8 degrees of freedom."* $2H$ and $H$ describe the identical mapping.

- A is false — nothing forces an entry to zero.
- C is false — $H$ is generally not a rotation; its determinant is unconstrained.
- D is false — $H$ is not orthonormal in general.

**A3. → C — $R$ and $t$.**

Slide p.15: *"Intrinsics belong to the camera. Move the camera — they don't change."* Slide p.16: *"Extrinsics change every time the camera moves."*

- A and B are intrinsic, so unchanged.
- D is wrong — $(R,t)$ **is** the pose, and it definitely changes.

**A4. → B — 3×4.**

$K$ is 3×3 and $[R\mid t]$ is 3×4, so $P = K[R|t]$ is **3×4**. Slide p.17 calls it *"the 3 × 4 camera matrix."* *(Verified: my `project` builds a 3×4 and matches the hand-computed $(195, 615)$.)*

**A5. → B — The intrinsics get applied in the wrong order.**

Slide p.14's diagram orders it explicitly: $\frac{f_x X}{Z} \to +\ c_x$, i.e. **scale → divide → shift**. Dividing first removes the perspective coupling, and the subsequent $f_x$ and $+c_x$ then act on an already-collapsed coordinate.

- A is false — division and matrix multiplication do **not** commute here; the ÷ must be applied to the *combined* homogeneous coordinate.
- C is unrelated — $R$ is unaffected by anything after it.
- D is a red herring; a mirrored image would need $c_x \to W - c_x$, which is a different bug.

**A6. → C — The essential matrix $E$.**

Slide p.46: $E$ works on *"Normalized coords, $K^{-1}\mathbf{x}$"* and *"Needs K? Yes (calibrated)."*

- A is false — $F$ works on raw pixels and *"Needs K? No."*
- B and D are unrelated.

**A7. → B — $Z$ is underestimated by 20%.**

$Z = \frac{fH}{h}$ is **linear in $f$**, so using $f' = 0.8f$ gives $Z' = 0.8Z$ — 20% **under**. This is slide p.28's point exactly: 1000 px → 25 m, 800 px → 20 m. *(Verified: 20/25 = 0.8.)*

- A has the sign backwards; C is false; D is absurd since $Z$ is a positive scale of $f$.

**A8. → B — Partial views break the corner ordering used to match images.**

The instructor is explicit (transcript): *"make sure that in each of the images that you are using, whole of the checkerboard is visible… **if you take a partial board, your ordering will fail.** In a partial image the first index might correspond to some middle checkerboard point."*

- A is wrong — a partial view of an 8×5 board still supplies far more than 8 equations.
- C is wrong — all corners are still coplanar ($Z=0$) in a partial view.
- D is a consequence of B, not the reason.

### Section B — Solutions

**B1. → A, B, D, E. Reject C.**

- **A** ✔ slide p.15: *"focal length in pixels."*
- **B** ✔ *"Two values because pixels may not be square"* (p.15) and the transcript: *"if it was square… $f_x$ and $f_y$ could have been the same."*
- **C** ✗ *"Usually near, but **not exactly** at, the image center"* (p.15).
- **D** ✔ slide p.15 — the defining property of intrinsics.
- **E** ✔ $f_x,f_y,c_x,c_y$.

**B2. → A, C, E. Reject B, D.**

- **A** ✔ the transcript's derivation: columns are orthogonal, so *"when you multiply [by its] transpose it gives you zero"* → $R^TR = I$.
- **B** ✗ a rotation matrix is **not** symmetric; $R \ne R^T$ in general ($R^T = R^{-1}$, not $R$).
- **C** ✔ $\det R = +1$ follows from orthonormality plus orientation preservation — this is what makes triangulation sign-checkable.
- **D** ✗ the two constraints cut 9 down to **3** DOF — roll, pitch, yaw (slide p.16).
- **E** ✔ slide p.16: $X_{cam} = R\,X_{world} + t$.

**B3. → A, B, C, E. Reject D.**

This is slide p.30's list verbatim — all four reasons, with D as the designed inverse:
- **A** ✔ *"Measure the square size once (e.g. 25 mm) and every corner position is known."*
- **B** ✔ *"Black–white X-junctions can be located to a fraction of a pixel."*
- **C** ✔ *"Flat — all corners share $Z = 0$."*
- **E** ✔ *"High contrast, regular pattern, cheap to print."*
- **D** ✗ the opposite is true — knowing the physical size is **precisely why** the board works.

**B4. → A, C, D. Reject B, E.**

- **A** ✔ slide p.21: *"Two views of the same plane are related by one 3 × 3 matrix H."*
- **B** ✗ slide p.26: *"H is exact only for points on the plane (or a purely rotating camera). **Anything with height gets smeared.**"*
- **C** ✔ same slide — a pure rotation keeps every point at effectively constant depth.
- **D** ✔ slide p.23: $8 \div 2 = 4$, with the non-collinearity condition.
- **E** ✗ a wrong focal length breaks the geometry $H$ assumes — *"the world is not flat"* also includes *"a purely rotating camera"* as the only exception.

**B5. → A, C, D, E. Reject B.**

- **A** ✔ slide p.48: *"The epipolar constraint restricts the match to a line."*
- **B** ✗ inverse of slide p.46 — $F$ is precisely the one that *"Needs K? **No**."*
- **C** ✔ slide p.42: *"The epipole $e'$… **Every epipolar line passes through it.**"*
- **D** ✔ slide p.43: rectified stereo gives *"lines are parallel rows, $e'$ at infinity."*
- **E** ✔ slide p.47 — 921,600 → 1,280, and *"720× fewer candidates per pixel."* *(Verified: 921600/1280 = 720.)*

### Section C — Solutions

*Each ≤100 words, three beats: define → reason → stop.*

**C1.**
From the pinhole model, a point's image position depends on $X/Z$ and $Y/Z$ — so the image constrains the **direction** from the camera but not the distance along that ray. Slide p.11 makes the ambiguity explicit: image size is $fH/Z$, which depends on the product $H \cdot Z$. A toy car held close and a real car far away produce identical pixels, so a single image cannot distinguish *"small $H$, small $Z$"* from *"large $H$, large $Z$."* Slide p.4 states the consequence: *"the last one needs depth — which a single image does not give you directly."* Resolving it requires a second viewpoint, prior knowledge of size, or another depth cue.

*(99 words.)*

**C2.**
Because the camera moves. Consider a point 5 m from the camera, then 8 m from it — not because the object moved, but because the camera did. In the **camera frame** the distance appears to change; in the **world frame** the object is exactly where it always was, so the change is attributable entirely to camera motion. The instructor states it directly: *"the camera might be in motion and if you don't know the relative motion… it becomes difficult to track what you are actually seeing. That is why you need a world reference frame."* You cannot measure a frame's own motion against itself; you need something fixed to measure against. The world frame supplies that, and it is also what lets a reconstruction from many views accumulate into a map.

*(99 words.)*

**C3.**
Because the homography is not a new model — it is the camera model with the planar constraint substituted in. Take $\mathbf{x} \simeq K[R|t]\mathbf{X}$ and impose $Z = 0$ for every point. The third column of $R$ then multiplies zero and contributes nothing, so the 3×4 matrix collapses to a single 3×3 matrix $H$ with 8 free parameters. The instructor derives it live: *"it is getting derived from the same set of intrinsic equations… that is why it works in a plane, because $Z$ is zero."* Off the plane, $Z \ne 0$ and the dropped column matters again — slide p.26's *"height smeared along the ground"* is exactly that missing term reappearing as error.

*(98 words.)*

**C4.**
Because the printed value is nominal, not measured. The instructor's reasons: *"manufacturing defects where your $f$ might be slightly different than what they are printing on the box,"* and *"over time the camera's focal length might be changing… with rough use."* Slide p.28 quantifies the cost — a 1.5 m car at 60 px gives $Z = 25$ m with $f = 1000$ but **20 m** with $f = 800$, a 5 m error worth **0.36 s at 50 km/h**, which the slide calls *"the difference between braking in time and not."* Since $Z = fH/h$ is linear in $f$, focal-length error passes through **undiminished** into depth, so every downstream measurement inherits it.

*(97 words.)*

**C5.**
It tells you **where** to look, not **what** matches. Given pixel $x$ in the left image, the true 3-D point could be $P_1$, $P_2$, $P_3$… anywhere along the ray through $x$ — that ambiguity is exactly the depth you have lost. But all of those candidates project onto a single line $l'$ in the right image, because they all lie in the epipolar plane through $O$, $O'$ and $P$. So the match is restricted to one line: slide p.47 reduces a 921,600-candidate search to 1,280. Appearance still decides *which* point on that line is correct — slide p.48: *"Geometry says where to look. Appearance says what matches."* The line gives a constraint, not a correspondence.

*(99 words.)*

### Section D — Solutions

#### D1 — World point to pixel

**Key idea tested.** The correct order — rigid motion, then intrinsics, then divide.

```python
def project(K, R, t, X):
    if not K or len(K) != 3 or any(len(r) != 3 for r in K):
        return None
    if not R or len(R) != 3 or any(len(r) != 3 for r in R):
        return None
    if t is None or len(t) != 3:
        return None
    if not X:
        return []
    out = []
    for p in X:
        if p is None or len(p) != 3:
            return None
        Xc = sum(R[0][k] * p[k] for k in range(3)) + t[0]
        Yc = sum(R[1][k] * p[k] for k in range(3)) + t[1]
        Zc = sum(R[2][k] * p[k] for k in range(3)) + t[2]
        if Zc == 0:
            return None
        out.append((round(K[0][0] * Xc / Zc + K[0][2], 6),
                    round(K[1][1] * Yc / Zc + K[1][2], 6)))
    return out
```

**Verified execution output:**

```
identity R,t=0 : [(320.0, 240.0), (370.0, 240.0)]     expected [(320.0,240.0),(370.0,240.0)]
R90, t=(1,2,3) : [(195.0, 615.0)]                    expected [(195.0,615.0)]
empty -> []   bad K -> None   Zc=0 -> None
```

**Why it works.** Each of $X_c,Y_c,Z_c$ is a dot product with a row of $R$ plus one component of $t$ — that is $X_c = RX_{world} + t$, exactly slide p.16. Then $u = K_{00}X_c/Z_c + K_{02}$ is slide p.14. The division is **last**, because $K[R|t]\mathbf{X}$ produces a homogeneous coordinate whose third component carries $Z_c$.

**The rotation case, by hand.** $X = (1,2,5)$, $R = \begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$, $t = (1,2,3)$:
$X_c = (0\cdot1 - 1\cdot2 + 0\cdot5 + 1,\ 1 + 0 + 0 + 2,\ 0+0+5+3) = (-1, 3, 8)$.
$u = 1000(-1)/8 + 320 = -125 + 320 = \mathbf{195}$ ✔
$v = 1000(3)/8 + 240 = 375 + 240 = \mathbf{615}$ ✔

**Common wrong approaches.**
- Dividing the world point by $Z$ *before* applying $R$ — the subject of A5.
- Using $K$'s off-diagonal zeros but reading $c_x$ as `K[0][1]` — it lives at `[0][2]`.
- Letting `Zc` go negative (a point behind the camera) and returning a mirrored pixel instead of rejecting it.

---

#### D2 — Fit a homography from four correspondences

**Key idea tested.** The $8\times8$ linear system implied by the DOF count.

```python
def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]

def fit_homography(src, dst):
    if src is None or dst is None or len(src) != 4 or len(dst) != 4:
        return None
    if any(len(p) != 2 for p in src) or any(len(p) != 2 for p in dst):
        return None
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = _solve(A, b)
    if h is None:
        return None
    return [h[0:3], h[3:6], h[6:8] + [1.0]]

def apply_h(H, p):
    d = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2]
    if d == 0:
        return None
    return (round((H[0][0]*p[0] + H[0][1]*p[1] + H[0][2]) / d, 6),
            round((H[1][0]*p[0] + H[1][1]*p[1] + H[1][2]) / d, 6))
```

**Verified execution output:**

```
H = [[1.5, 0.0, 0.0], [0.0, 2.666667, 0.0], [0.0, 0.0, 1.0]]
apply_h to each src -> [(0.0,0.0), (600.0,0.0), (600.0,800.0), (0.0,800.0)]   == dst
fit_homography(src[:3], dst[:3]) -> None
fit_homography(four collinear)    -> None
```

**Why it works.** Each correspondence gives **two** linear equations (slide p.23's *"2 (one for u, one for v)"*), so four pairs give exactly the 8 equations needed for $H$'s 8 DOF — with $h_{33}$ fixed to 1 to absorb the scale freedom. The returned matrix is a pure **anisotropic scale**: $400 \to 600$ gives 1.5, and $300 \to 800$ gives 2.666667, exactly as the round-trip confirms.

**The collinear case is the condition from slide p.23.** Four points on a line carry only 2 independent dimensions, so the 8×8 matrix is singular and `_solve` returns `None` via its pivot check — which is precisely *"no three of the four points on one line."*

**A bug worth recording.** My first version had a guard `if x == 0 and y == 0: return None`, intended to catch degenerate input. It rejected the **legitimate** first corner of a document and made every real call fail. A "degenerate" check must test the *system*, not an individual coordinate.

**Common wrong approaches.**
- Building one equation per pair instead of two — the system is then under-determined.
- Not normalising scale, giving a matrix that looks wrong despite mapping correctly.
- Using RANSAC here instead of an exact fit — with exactly four points and no outliers there is nothing to reject; RANSAC needs surplus matches (slide p.23).

---

#### D3 — Reprojection error

**Key idea tested.** RMS in **pixels**, over the whole calibration set.

```python
def reprojection_error(K, R, t, correspondences):
    if not correspondences:
        return None
    got = project(K, R, t, [c[0] for c in correspondences])
    if got is None:
        return None
    se = 0.0
    for g, (_, obs) in zip(got, correspondences):
        se += (g[0] - obs[0]) ** 2 + (g[1] - obs[1]) ** 2
    return round(math.sqrt(se / len(correspondences)), 6)
```

**Verified execution output:**

```
exact            -> 0.0
+1 px in u       -> 1.0
off by (1, 3) px -> 3.162278
empty            -> None
```

All three reproduce by hand: a 1-px horizontal error gives RMS $1$; a $(1,3)$ px error gives $\sqrt{\frac{1^2+3^2}{1}} = \sqrt{10} = 3.162278$ ✔.

**Why it works.** It is slide p.36's $E = \sum\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$ normalised by $n$ and square-rooted, so the result is **in pixels** and directly comparable to the *"well under ~0.5 px"* rule of thumb. Averaging rather than summing is what makes it a per-correspondence measure that is independent of how many corners you detected.

**Common wrong approaches.**
- Returning the sum instead of the RMS — it grows with the number of corners, so it is not comparable to the 0.5 px threshold.
- Computing the error in **world** units (3-D distance between true and reconstructed points) — that conflates calibration error with depth error, and needs the very extrinsics being estimated.
- Checking it only on the calibration images — slide p.36's warning (*"blurry photos, wrong corners, or a bad distortion model"*) is exactly why held-out images matter.

### Section E — Solutions

**Model answer covering the chain.**

1. **Task.** **Projection + calibration**, not stereo. The forklift is monocular, so it has **one** view and no parallax; the distance must come from the pinhole model, $Z = \frac{fH}{h}$, applied to a known-size object. Calibration supplies $f$ and the principal point. **Homography is the wrong tool** (E-a) and **stereo is unavailable** with one camera. What the system actually needs is *monocular depth-from-known-size*.
2. **Data.** $Z = \frac{fH}{h} = \frac{f \times 1.0}{200}$. The datasheet says $f = 1200$ px, which predicts $Z = 6.0$ m — matching the surveyed value. So the datasheet figure is *consistent with this one measurement*, but a single measurement cannot separate $f$ from $H$; the inconsistency must come from elsewhere in the chain (detector box tightness, lens variant, or the sensor being cropped in software). **A second independent object height, or a multi-view checkerboard, is required to pin $f$.** The honest answer is that the data is insufficient to conclude $f$ is wrong — this is worth stating, and worth a 2-mark question.
3. **Model.** From a detected bounding box: measure apparent height $h$ in pixels → invert $Z = \frac{fH}{h}$ → compare with the safety threshold. **Intrinsic:** $f$, $c_x$, $c_y$ — fixed for the camera, obtained once by calibration. **Extrinsic:** $R$, $t$ — the forklift's current pose, supplied by wheel odometry or a fixed assumption of forward-facing. Here $t \approx 0$ (ignore) and $R \approx I$, which is why calibration alone suffices.
4. **Metric.** Report the **reprojection RMS in pixels** from the checkerboard calibration (slide p.36), and additionally validate on **held-out distances** — place an object at surveyed 2, 4, 6, 10 m and measure error against true distance. Slide p.36's threshold is *"well under ~0.5 px."* Report both, because a good reprojection RMS can still coexist with a depth bias if the principal point is wrong.
5. **Failure mode.** **A ramp changes the ground plane, and the pinhole model's flat-world assumption is only implicit here — but the deeper break is object scale.** More precisely, the model assumes the object's real size $H$ is known and constant. On a ramp, (i) apparent height changes because of the new viewing angle relative to gravity (foreshortening), and (ii) more critically, if the safety logic infers depth from a *vertical* extent while the camera pitches with the ramp, the $Z$ it computes is the depth **along the optical axis**, not along the ground. On a 10% grade at 3 m that is a ~0.3 m systematic error, enough to defeat a 3 m threshold.

**Sub-questions.**

**E-a — why a floor homography fails.**
A homography maps the **floor plane** to the image and back exactly, so it would be excellent for *"where on the floor is this point"* — and useless for *"how tall is that pallet."* Slide p.26's mechanism: $H$ is derived by imposing $Z = 0$, so the third column of $R$ drops out; anything off that plane has a **non-zero** $Z$ and that dropped term reappears as error, producing *"height smeared along the ground."* A 1 m pallet would project to a smeared footprint, so **every pixel of it looks like floor**. The team would get a geometrically perfect map of the ground plane and no information at all about obstacles standing on it — which is the one thing a forklift must not miss. Concretely, the homography cannot answer *"is anything within 3 m"* because obstacles are exactly the off-plane content it discards.

**E-b — what the second camera changes, with numbers.**
Without it, each pixel searches **$720\times1280 = 921{,}600$** positions. With epipolar geometry, the search collapses onto the line $l'$, giving at most **$1{,}280$** — a **720× reduction** (slide p.47). Two further consequences beyond raw cost: (i) it is *"far fewer look-alike mistakes"* — repeated lane markings produce many visually similar candidates, and restricting to the line eliminates almost all of them; (ii) the epipole $e'$ depends on relative pose, so in the **rectified** case *"lines become rows, so the match is just a horizontal shift: disparity"* (slide p.48) — turning a general 2-D search into a 1-D one along $v = \text{const}$. Note the geometry alone does **not** give depth: slide p.48's chain is *epipolar geometry → correspondence → depth*, and only the last step needs the two-view configuration.

**E-c — correct order, and the symptom of each wrong ordering.**

**Correct:**
```
world X  ──R──▶  X_c  ──+t──▶  (X_c,Y_c,Z_c)  ──K──▶  (u·Z_c, v·Z_c, Z_c)  ──÷Z_c──▶  (u,v)
```
i.e. **rotate → translate → intrinsics → divide last.**

| Wrong order | Symptom |
|---|---|
| Divide by world $Z$ first | Points no longer land on a consistent image plane; large- and small-$Z$ regions warp differently. Looks like a **barrel-distorted** or torn image. |
| Apply $K$ before $R$ | $f_x$ and $c_x$ scale/rotate *in world axes*; the principal point ends up rotated with the camera. Symptom: **the image rotates about the image centre** and $f_x \ne f_y$ produces a visible shear. |
| Apply $t$ before $R$ | Translation gets rotated by $R$. Symptom: **offset in a direction that changes as the camera turns** — in driving, a forward-facing constant $t$ appears to swing sideways with heading. |
| Omit the final division | Every point lands on the **last row** of the image, since $u$ becomes $uZ_c$. |
| Use $R$ where $R^T$ is required | The object appears mirrored or yawed the wrong way; $+90°$ becomes $-90°$. |

### Section F — Solutions

**F1.** *Formula:* $u = f_x\frac{X}{Z} + c_x$, $v = f_y\frac{Y}{Z} + c_y$.

*Point 1, $(0.5, 0, 12)$:*
$$u = 1200\cdot\frac{0.5}{12} + 954 = \frac{600}{12} + 954 = 50 + 954 = \mathbf{1004.0}$$
$$v = 1180\cdot\frac{0}{12} + 542 = \mathbf{542.0}$$

*Point 2, $(-0.9, 0.3, 7.5)$:*
$$u = 1200\cdot\frac{-0.9}{7.5} + 954 = \frac{-1080}{7.5} + 954 = -144 + 954 = \mathbf{810.0}$$
$$v = 1180\cdot\frac{0.3}{7.5} + 542 = \frac{354}{7.5} + 542 = 47.2 + 542 = \mathbf{589.2}$$

*With $c_x$ = image centre.* For a 1920×1080 sensor the centre is $c_x = 960$, so point 1 gives $u = 50 + 960 = \mathbf{1010.0}$ — a **6-px** shift to the right.

*Interpret:* $c_x$ is the **origin of the pixel grid**, not an object position, so changing it translates the whole image rigidly without changing scale. A 6-px error is invisible on a 1920-wide image but is a systematic 6-px bias in **every** measurement derived from it — including triangulation, where it directly biases depth. Note $f_x \ne f_y$ (1200 vs 1180) means pixels are not exactly square, so $u$ and $v$ must be scaled by *different* constants.

**F2.** *Formula:* $Z = \dfrac{fH}{h}$, $H = 1.5$ m, $h = 60$ px.

$$f = 1000 \Rightarrow Z = \frac{1000 \times 1.5}{60} = \frac{1500}{60} = \mathbf{25.0\ m}$$
$$f = 800 \Rightarrow Z = \frac{800 \times 1.5}{60} = \frac{1200}{60} = \mathbf{20.0\ m}$$

*Both verified.*

*Time gap at 50 km/h:* $50\div3.6 = 13.89$ m/s, so
$$\Delta t = \frac{25 - 20}{13.89} = \frac{5}{13.89} = \mathbf{0.36\ s}$$

*Consequence (slide p.28):* *"the 5 m error, at 50 km/h — **the difference between braking in time and not**."* The mechanism is that $Z$ is **linear in $f$**, so a 20% focal-length error becomes a 20% depth error with **no attenuation**. At 50 km/h a 0.36 s shortfall is roughly **6–8 m of additional travel** under emergency braking, and at 3 m stopping distance the whole safety margin evaporates. Note this is not a subtle accuracy issue: it is the difference between a functional and a non-functional safety function.

**F3.** *Why 4 points (slide p.23):*

| | Count |
|---|---|
| Unknowns in $H$ | $9 - 1$ (scale) $= 8$ |
| Equations per pair | $2$ |
| **Needed** | $8 \div 2 = \mathbf{4}$ |

*Solving.* Each pair $(x,y)\to(u,v)$ contributes:
$$h_{11}x + h_{12}y + h_{13} - u\,h_{31}x - u\,h_{32}y = u$$
$$h_{21}x + h_{22}y + h_{23} - v\,h_{31}x - v\,h_{32}y = v$$
with $h_{33} = 1$. Four pairs → an 8×8 system.

*Result (verified by execution):*
$$H = \begin{bmatrix} 1.5 & 0 & 0 \\ 0 & 2.666667 & 0 \\ 0 & 0 & 1\end{bmatrix}$$

*Verification* — apply to all four corners:

| src | expected | computed |
|---|---|---|
| $(0,0)$ | $(0,0)$ | $(0.0, 0.0)$ ✔ |
| $(400,0)$ | $(600,0)$ | $(600.0, 0.0)$ ✔ |
| $(400,300)$ | $(600,800)$ | $(600.0, 800.0)$ ✔ |
| $(0,300)$ | $(0,800)$ | $(0.0, 800.0)$ ✔ |

*Interpret:* the exact solution is a pure **anisotropic scale** — 1.5× horizontally, 2.667× vertically — with no perspective terms. That is expected: mapping a rectangle to a rectangle needs only scale and translation, so $H$ is diagonal. A real tilted-document photo yields all nine entries non-zero, because perspective introduces the off-axis terms.

*If the four points are collinear:* slide p.23's condition is *"no three of the four points on one line."* Collinear points span only 1 dimension, so the 8×8 matrix becomes **singular** and no unique $H$ exists — **verified: my solver returns `None`**. The correct behaviour is to refuse, not to return a garbage matrix.

**F4.** *Formula:* $\text{feature size} = \left\lceil \dfrac{H}{\text{OS}} \right\rceil$; then $u = f_x\frac{X_c}{Z_c} + c_x$.

| OS | Height | Width |
|---|---|---|
| 32 | $1920/32 = \mathbf{60}$ | $1080/32 = \mathbf{33.75} \to \mathbf{34}$ |
| 16 | $1920/16 = \mathbf{120}$ | $1080/16 = \mathbf{67.5} \to \mathbf{68}$ |

*Projection.* $R = I$, $t = 0$, so $X_c = X_{world} = (2.0, -1.0, 20.0)$ — unchanged.

$$u = 1000\cdot\frac{2.0}{20.0} + 960 = 100 + 960 = \mathbf{1060.0}$$
$$v = 1000\cdot\frac{-1.0}{20.0} + 540 = -50 + 540 = \mathbf{490.0}$$

*Interpret:* the 32-stride map (60×34) discards 97% of the pixels — a 1 m object at 20 m projects to $1000/20 = 50$ px before intrinsics, but only ~1.6 cells survive at OS 32. At OS 16 it is ~3 cells. This is exactly the trade slide p.10 makes: the encoder's job is context and semantics at cheap resolution, and the **decoder** must put resolution back. Note also that $v = 490 < 540$ because $Y = -1.0$ is *above* the optical axis under the Y-down convention — a sign check worth doing.

**F5.** *Unknown counts:*

| | Parameters | DOF |
|---|---|---|
| Intrinsic $K$ | $f_x, f_y, c_x, c_y$ | **4** |
| Extrinsic $R$ | roll, pitch, yaw | **3** |
| Extrinsic $t$ | $T_x, T_y, T_z$ | **3** |
| **Total** | | **10** |

Plus distortion coefficients, which the instructor notes are usually skipped for a pinhole model — *"the distortion in pinhole camera are generally less."*

*Correspondences from one view.* An 8×5 corner grid has $8\times5 = \mathbf{40}$ corners, so **one view supplies 40 point pairs** — and each pair gives 2 equations, so **80 equations** against 10 unknowns (9 intrinsic+extrinsic parameters per view, less the shared $K$).

*Why more views are needed.* Three reasons, all from the material:
1. **Each view has its own $R$, $t$.** A single view gives 80 equations but the unknowns include 6 extrinsics *for that view*, so a single view is heavily over-determined in a way that cannot separate the effects — the system is nearly degenerate for the planar case.
2. **Degenerate geometry.** Slide p.30's fourth checkerboard property — *"Flat — all corners share $Z = 0$"* — is what lets you solve for $K$ **first* via a homography, but it also means one view cannot constrain depth at all. You need views where the board is tilted *relative to the optical axis* to break that degeneracy.
3. **Distortion.** Lens distortion needs many views at different angles and radii to fit, or it will overfit to the centre of the image.

The instructor's ordering (transcript): detect → order → refine, across many images, with *"the whole of the checkerboard visible"* in every one.

**F6.** *Exhaustive search (slide p.47):*
$$720 \times 1280 = \mathbf{921{,}600} \text{ candidates per pixel}$$

*Along the epipolar line:*
$$1 \times 1280 = \mathbf{1{,}280} \text{ candidates}$$

*Reduction:*
$$\frac{921{,}600}{1{,}280} = \mathbf{720\times} \quad\text{(verified)}$$

*What supplies the second number:* the **epipolar constraint** — slide p.48: *"Geometry says where to look. Appearance says what matches."* Given pixel $x$, its unknown depth corresponds to candidates $P_1, P_2, P_3\dots$ along the ray through $x$, and **all of them project onto the same line $l'$** because they share the epipolar plane through $O$, $O'$. So the search collapses from a 2-D area to a 1-D line. The line's *length* is the image height (1280 px), which is where the second number comes from.

*Note on the bound:* 1,280 is an **upper bound** — the actual segment is shorter whenever the epipole lies inside the image, since the line is clipped by the image border. In the special case of a forward-facing camera the epipole is at the image centre (slide p.43's *"focus of expansion"*), so the epipolar line is a full-height row and the bound is achieved.

**F7.** *Verdict.* Slide p.36's rule of thumb: *"RMS error well under ~0.5 px is good."* At **0.42 px** the fit is **accepted** — but note the threshold says *well under* 0.5, so this is near the margin rather than comfortably inside it, and should be re-checked on held-out images.

*Three causes if a second dataset gives 1.8 px (slide p.36's own list):*
1. **Blurry photos** — corner detection returns blobs rather than points; the classic cause of a high RMS on one dataset but not another.
2. **Wrong corners** — mis-detected or mis-**ordered** corners. The ordering failure the instructor flags (*"in a partial image the first index might correspond to some middle checkerboard point"*) produces exactly this: large, systematic, and dataset-specific.
3. **A bad distortion model** — the second dataset may cover a wider field of view, where unmodelled radial distortion dominates; the first dataset's corners all sat near the optical centre where distortion is near zero.

*Arithmetic.* A single correspondence off by $(0.6, -0.8)$ px:
$$\|e\| = \sqrt{0.6^2 + (-0.8)^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = \mathbf{1.0\ px}$$

*How many 1-px errors would it take to explain 1.8?* **None — and that is the diagnostic.* RMS obeys
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_i e_i^2} \;\le\; \max_i |e_i|,$$
so if every error were exactly 1.0 px the RMS would be exactly **1.0**, never 1.8. Reaching an RMS of 1.8 therefore *requires* at least one error of 1.8 px or more, i.e. corners that are wrong by several pixels rather than merely imprecise.

**That rules out cause (1) and confirms causes (2) and (3).**
- **Blurry photos** produce small, noisy localisation errors — a blurred corner lands within roughly a pixel. Those raise the RMS toward ~1.0 and plateau there. They cannot produce 1.8.
- **Mis-ordered corners** produce *systematic* errors: a corner indexed as $k$ is matched against the detector's $k'$-th point, a displacement of a full grid cell — tens of pixels. A handful of such pairings drives the RMS straight past 1.8.
- **A bad distortion model** produces errors that **grow with image radius**. If the second dataset covers a wider field of view than the first, corners far from the centre can be off by many pixels while the first dataset's corners, all near the optical centre, were essentially unaffected.

**So the first thing to check is the second dataset's corner ordering and field of view** — not the sharpness. My earlier phrasing ("roughly 18 such errors") was wrong: the RMS bound rules that out entirely.

**F8.** *Formula:* $u = f_x\frac{X_c}{Z_c} + c_x$, $v = f_y\frac{Y_c}{Z_c} + c_y$; $X_c = RX + t$.

*Case 1, $R = I$, $t = 0$.* $X_c = (-0.5, 0.25, 8)$:
$$u = 900\cdot\frac{-0.5}{8} + 640 = -\frac{450}{8} + 640 = -56.25 + 640 = \mathbf{583.75}$$
$$v = 900\cdot\frac{0.25}{8} + 360 = \frac{225}{8} + 360 = 28.125 + 360 = \mathbf{388.125}$$

Both verified.

*Case 2, $R$ = rotation by $+90°$ about $Z$:*
$$R = \begin{bmatrix}0 & -1 & 0\\ 1 & 0 & 0\\ 0 & 0 & 1\end{bmatrix}$$
$$X_c = (0\cdot(-0.5) - 1\cdot(0.25) + 0,\; 1\cdot(-0.5) + 0 + 0,\; 0+0+1\cdot 8) = (-0.25,\; -0.5,\; 8)$$
$$u = 900\cdot\frac{-0.25}{8} + 640 = -28.125 + 640 = \mathbf{611.875}$$
$$v = 900\cdot\frac{-0.5}{8} + 360 = -56.25 + 360 = \mathbf{303.75}$$

*Which stage was affected.* **The rigid-motion stage.** $R$ maps world axes into camera axes; it does not touch the intrinsics and it does not touch the division by $Z$. Note the evidence in the numbers: $Z_c = 8$ is **unchanged**, because a rotation about $Z$ leaves that component alone — only $X_c, Y_c$ moved. In general $R$ would change all three and hence the depth scale, but the key point is that $f_x, f_y, c_x, c_y$ were never involved.

*Interpret:* $(X,Y) = (-0.5, 0.25) \to (X_c,Y_c) = (-0.25, -0.5)$ is a clean $+90°$ rotation: $(x,y)\mapsto(-y,x)$. The pixel moved right and **up**, from $(583.75, 388.13)$ to $(611.88, 303.75)$ — up because $v$ grows downward, so a decrease in $Y_c$ means moving up in the image. Depth is invariant under pure rotation, which is exactly why the division stage is untouched; translation, by contrast, *would* change $Z_c$ and therefore change the apparent size.

---

## J. Sources

### Course sources

| Source | File | Detail |
|---|---|---|
| Presentation | `Session_11_12.pdf` pages 1–48 | *"From Pixels to Points — 3D Vision"*; text layer |
| Transcript | `Session_11_—_Camera_Geometry_….txt` | 26/9/2026, 1:45 PM, 120 min; 11k words |
| **Professor repository** | **None in this workspace** | — |

### Key slides

| Slides | Topic |
|---|---|
| p.2–6 | Why 3D vision; 2D vs 3D perception; **the pipeline and roadmap** |
| p.8–11 | Eye as camera; **pinhole**; similar triangles; **size ∝ 1/Z** |
| p.12–16 | Three frames; camera frame; **$u=f_xX/Z+c_x$**; intrinsics; extrinsics |
| p.17 | **$\mathbf{x}\simeq K[R\mid t]\mathbf{X}$** and the meaning of $\simeq$ |
| p.19–26 | Homography; **derived from the camera model with $Z=0$**; why 4 points; limitations |
| p.28–36 | Why calibrate (**the 0.36 s example**); problem definition; **checkerboard's four reasons**; corner detection/ordering/refinement; reprojection error and the 0.5 px rule |
| p.38–48 | Epipolar plane, **line**, epipole; **F (7 DOF) vs E (5 DOF)**; **720× search reduction** |

### Supplementary

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed.) — the standard reference for all of Modules 1–6; freely available. https://www.robots.ox.ac.uk/~vgg/hzbook/
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — pinhole model, projection, calibration chapters. https://www.cs.cmu.edu/~./szeliski/Book/
- Zhang, "A Flexible New Technique for Camera Calibration" (PAMI 2000) — the standard calibration formulation. https://www.cs.cmu.edu/~./zhangxx/papers/Zhang98-pami.pdf
- OpenCV camera calibration and `findHomography` / `findFundamentalMat` docs. https://docs.opencv.org/4.x/

### Unresolved

1. **Slide p.34 ("Camera calibration steps") and p.35 ("the optimization idea") carry only a page number — their diagrams did not extract.** The step-by-step order in §B10 is reconstructed from the instructor's spoken derivation in the transcript (homography per view → solve for $K$, $R_1$, $R_2$, $t$ using $R^TR=I$ → distortion → reprojection error). The spoken account is unambiguous; the slide diagrams themselves could not be read.
2. **Slides p.23/24 and p.32/33 are duplicates** (the deck shows the same content on two pages). No content is lost.
3. **The deck's own roadmap (p.6) lists seven modules but only six are numbered in the deck**, with a Module 7 ("Putting it together") announced and never delivered. The equivalent material appears as the *Stereo 3D Reconstruction* pages at the end, which belong to Session 12.

---

*Questions A–F are original, written against the Session 11 sources. Every numerical answer and all three code solutions were executed and verified; two errors were caught and fixed — a test harness that passed the intrinsics matrix where the rotation matrix belonged, and a "degenerate input" guard in the homography solver that rejected the legitimate (0,0) corner and made every real call fail.*