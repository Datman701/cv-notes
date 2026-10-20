# Coding — 10 easy options that cover the likely Section D

**The bet.** Section D gives you **3 options and you write 1**. You do not need breadth — you need three templates deep, so at least one option always matches. The official sample is `argmax` (6 lines). Everything below is 6–12 lines and uses only NumPy basics.

**Every answer follows the same skeleton:**

```python
def solve(...):
    # 1. validate / empty case
    # 2. compute output shape
    # 3. loop or vectorise
    # 4. return the type the spec asks for
```

**Before you write, read the spec for:** return type (`list` vs `np.array`), rounding, and what empty input should return. When in doubt: return `[]` for empty and round only at the very end. **If the spec says "list of lists", append `.tolist()` to the NumPy result.**

## Template map

| If the option asks for… | Use |
|---|---|
| a filter / blur / convolution | Q2, Q3 |
| pooling / downsampling | Q4 |
| boxes, overlap, detection | Q5, Q6 |
| segmentation masks | Q5, Q7 |
| attention / transformers | Q8 |
| similarity / retrieval / CLIP | Q9 |
| depth / camera / stereo | Q10 |

**Minimum viable prep:** Q1–Q7. They are the shortest and cover the most likely easy option. Q8–Q10 add one topic each.

---

## Q1 — Argmax with ties

**Why likely:** this is literally the sample question in `session16.pdf`.

**Spec.** Return the index of the highest score. Empty → `-1`. Ties → the **first** index.

**Scaffold**

```python
def argmax_index(scores):
    if not scores:
        return ___                 # BLANK 1
    best = 0
    for i in range(1, len(scores)):
        if scores[i] ___ scores[best]:   # BLANK 2
            best = i
    return best
```

**Blanks:** 1 → `-1`; 2 → `>` (strictly greater keeps the first on ties).

**Solution**

```python
def argmax_index(scores):
    if not scores:
        return -1
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best
```

**Tests**

```
argmax_index([0.2, 0.8, 0.8, 0.1])  ->  1
argmax_index([])                    ->  -1
argmax_index([0.5, 0.5])            ->  0
argmax_index([0.4])                 ->  0
```

**Trap:** `np.argmax` returns the first max too, but the spec may forbid NumPy — write the loop. If `scores` is a NumPy array, `if not scores` is ambiguous; use `if len(scores) == 0`.

---

## Q2 — 2D convolution (valid)

**Why likely:** Class 01 Option D1; convolution is the most-mentioned idea in the transcripts (127 mentions).

**Spec.** `image` and `kernel` are 2D lists/arrays. Return the **valid** response (no padding, stride 1). Kernel larger than the image → `[]`.

**Scaffold**

```python
import numpy as np

def conv2d_valid(image, kernel):
    x = np.asarray(image, dtype=float)
    w = np.asarray(kernel, dtype=float)
    if x.ndim != 2 or w.ndim != 2 or x.size == 0 or w.size == 0:
        return []
    if w.shape[0] > x.shape[0] or w.shape[1] > x.shape[1]:
        return []
    H = ___                                  # BLANK 1
    W = ___                                  # BLANK 2
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = ___                  # BLANK 3
    return out
```

**Blanks**

1. `x.shape[0] - w.shape[0] + 1`
2. `x.shape[1] - w.shape[1] + 1`
3. `np.sum(x[i:i+w.shape[0], j:j+w.shape[1]] * w)`

**Solution**

```python
import numpy as np

def conv2d_valid(image, kernel):
    x = np.asarray(image, dtype=float)
    w = np.asarray(kernel, dtype=float)
    if x.ndim != 2 or w.ndim != 2 or x.size == 0 or w.size == 0:
        return []
    if w.shape[0] > x.shape[0] or w.shape[1] > x.shape[1]:
        return []
    H = x.shape[0] - w.shape[0] + 1
    W = x.shape[1] - w.shape[1] + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(x[i:i+w.shape[0], j:j+w.shape[1]] * w)
    return out
```

**Tests**

```
img = [[1,2,3],[4,5,6],[7,8,9]]
k   = [[1,2,0],[0,0,0],[0,0,-1]]

conv2d_valid(img, k)       ->  array([[-4.]])
conv2d_valid(img, [[5]])   ->  array([[ 5., 10., 15.],
                                      [20., 25., 30.],
                                      [35., 40., 45.]])
conv2d_valid(img, [[1]*4]*4) ->  []
```

**Trap:** this computes **cross-correlation** (no kernel flip), which is what `nn.Conv2d` actually does. If the question explicitly says *convolution*, use `w[::-1, ::-1]` inside the sum. The output shape does not change.

**If they add stride/padding:** pad first, then use the same loops:

```python
x = np.pad(x, ((p, p), (p, p)))
H = (x.shape[0] - w.shape[0]) // s + 1
W = (x.shape[1] - w.shape[1]) // s + 1
out[i, j] = np.sum(x[i*s:i*s+w.shape[0], j*s:j*s+w.shape[1]] * w)
```

---

## Q3 — Moving average filter

**Why likely:** Class 01 Option D2; it is a convolution with a box kernel, and the spec is tiny.

**Spec.** `k` is an odd positive integer. Return the valid `k×k` mean response as floats. Even `k` → `ValueError`. Empty or `k` too large → `[]`.

**Scaffold**

```python
import numpy as np

def moving_average(image, k):
    x = np.asarray(image, dtype=float)
    if ___:                                  # BLANK 1
        raise ValueError("k must be odd")
    if x.ndim != 2 or x.size == 0 or k > min(x.shape):
        return []
    H = x.shape[0] - k + 1
    W = x.shape[1] - k + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = ___                  # BLANK 2
    return out
```

**Blanks:** 1 → `k % 2 == 0`; 2 → `x[i:i+k, j:j+k].mean()`.

**Solution**

```python
import numpy as np

def moving_average(image, k):
    x = np.asarray(image, dtype=float)
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if x.ndim != 2 or x.size == 0 or k > min(x.shape):
        return []
    H = x.shape[0] - k + 1
    W = x.shape[1] - k + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = x[i:i+k, j:j+k].mean()
    return out
```

**Tests**

```
moving_average([[0,0,0],[0,90,0],[0,0,0]], 3)  ->  array([[10.]])
moving_average([[0,0,0],[90,90,90],[0,0,0]], 3) -> array([[30.]])
moving_average([[1,2,3],[4,5,6],[7,8,9]], 2)    ->  raises ValueError
```

**Trap:** `.mean()` returns a float automatically. If you use `sum(...) / (k*k)` in Python 2 style, force `/ float(...)` — not needed in Python 3, but the spec says "never truncated integers".

---

## Q4 — Max pooling / average pooling

**Why likely:** pooling has 52 transcript mentions and is the natural companion to convolution; the Section F sample is pooling output size.

**Spec.** 2D input, square window `k`, `stride` defaults to `k`, no padding. `mode="max"` or `"avg"`. Output size `floor((N-k)/s)+1` per axis.

**Scaffold**

```python
import numpy as np

def pool2d(x, k, stride=None, mode="max"):
    x = np.asarray(x, dtype=float)
    s = ___ if stride is None else stride     # BLANK 1
    if x.ndim != 2 or x.size == 0 or k <= 0 or s <= 0 or k > min(x.shape):
        return []
    H = ___                                   # BLANK 2
    W = ___                                   # BLANK 3
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            win = ___                         # BLANK 4
            out[i, j] = ___ if mode == "max" else ___   # BLANK 5
    return out
```

**Blanks**

1. `k`
2. `(x.shape[0] - k) // s + 1`
3. `(x.shape[1] - k) // s + 1`
4. `x[i*s:i*s+k, j*s:j*s+k]`
5. `win.max()` … `win.mean()`

**Solution**

```python
import numpy as np

def pool2d(x, k, stride=None, mode="max"):
    x = np.asarray(x, dtype=float)
    s = k if stride is None else stride
    if x.ndim != 2 or x.size == 0 or k <= 0 or s <= 0 or k > min(x.shape):
        return []
    H = (x.shape[0] - k) // s + 1
    W = (x.shape[1] - k) // s + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            win = x[i*s:i*s+k, j*s:j*s+k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out
```

**Tests**

```
a = np.arange(16).reshape(4, 4)

pool2d(a, 2)                ->  array([[ 5.,  7.],
                                       [13., 15.]])
pool2d(a, 2, mode="avg")    ->  array([[ 2.5,  4.5],
                                       [10.5, 12.5]])
pool2d(np.zeros((28,28)), 2).shape  ->  (14, 14)
```

**Trap:** `stride` must be applied **inside** the slice (`i*s`), not just used for the output size.

---

## Q5 — IoU and Dice

**Why likely:** Class 08 D3 and Class 14 D1; used in detection and segmentation, and Dice ≥ IoU is a favourite 2-mark question.

**Spec.** `iou(a,b)` takes `[x1,y1,x2,y2]` boxes, returns `0.0` when they do not overlap. `dice(a,b)` takes boolean masks, returns `0.0` when both are empty.

**Scaffold**

```python
import numpy as np

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = ___, ___                       # BLANK 1
    ix2, iy2 = ___, ___                       # BLANK 2
    inter = ___                               # BLANK 3
    area_a = ___                              # BLANK 4
    area_b = ___                              # BLANK 5
    union = area_a + area_b - inter
    return 0.0 if union == 0 else ___

def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = ___                               # BLANK 6
    denom = ___                               # BLANK 7
    return 0.0 if denom == 0 else 2.0 * inter / denom
```

**Blanks**

1. `max(ax1, bx1), max(ay1, by1)`
2. `min(ax2, bx2), min(ay2, by2)`
3. `max(0.0, ix2-ix1) * max(0.0, iy2-iy1)`
4. `max(0.0, ax2-ax1) * max(0.0, ay2-ay1)`
5. `max(0.0, bx2-bx1) * max(0.0, by2-by1)`
6. `np.logical_and(a, b).sum()`
7. `a.sum() + b.sum()`

**Solution**

```python
import numpy as np

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return 0.0 if union == 0 else inter / union

def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 0.0 if denom == 0 else 2.0 * inter / denom
```

**Tests**

```
iou([0,0,10,10], [5,5,15,15])   ->  0.14285714285714285
iou([0,0,10,10], [2,2,8,8])     ->  0.36
iou([0,0,10,10], [20,20,30,30]) ->  0.0

dice([[1,1,0],[0,1,0]], [[1,0,0],[0,1,1]])  ->  0.6666666666666666
dice([[0,0],[0,0]], [[0,0],[0,0]])          ->  0.0
```

**Trap:** clamp the intersection width/height at 0. Without `max(0, ...)`, non-overlapping boxes give a positive (impossible) intersection.

**Dice for boxes:** replace the mask sums with areas: `2*inter/(area_a+area_b)`.

---

## Q6 — Non-maximum suppression

**Why likely:** NMS has 11 transcript mentions but is the standard detection coding question; detection spans Classes 6–8.

**Spec.** `boxes` and `scores` are parallel lists. Return kept **indices**, highest score first. Ties → lower original index first. Suppress any box with `IoU > threshold` against a kept box.

**Scaffold**

```python
def nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: (___, i))   # BLANK 1
    keep = []
    while order:
        i = ___                                                  # BLANK 2
        keep.append(i)
        order = [j for j in order if ___]                        # BLANK 3
    return keep
```

**Blanks**

1. `-scores[i]` (descending score, original index as tie-break)
2. `order.pop(0)`
3. `iou(boxes[i], boxes[j]) <= thr`

**Solution** (assumes `iou` from Q5)

```python
def nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep
```

**Tests**

```
nms([[0,0,10,10],[1,1,11,11],[20,20,30,30]], [0.9,0.8,0.7], 0.5)  ->  [0, 2]
nms([[0,0,10,10],[1,1,11,11]], [0.8,0.8], 0.5)                    ->  [0]
nms([], [], 0.5)                                                   ->  []
```

**Trap:** sorting by score alone is not stable across implementations. The `( -score, index )` key makes ties deterministic.

**Class-wise NMS:** filter indices by class first, run NMS per class, concatenate.

---

## Q7 — Segmentation metrics

**Why likely:** Class 10 Option D2; segmentation has 45 transcript mentions and Dice is explicitly contrasted with IoU in Class 14.

**Spec.** `conf = {class: (TP, FP, FN)}`. Return per-class IoU, per-class Dice, mIoU, and pixel accuracy. Empty dict → `{}`.

**Scaffold**

```python
def seg_metrics(conf):
    if not conf:
        return {}
    ious, dices = {}, {}
    tp = fn = 0
    for c, (t, f, n) in conf.items():
        denom = ___                                      # BLANK 1
        ious[c] = ___ if denom else 0.0                  # BLANK 2
        d2 = ___                                         # BLANK 3
        dices[c] = ___ if d2 else 0.0                    # BLANK 4
        tp += t
        fn += n
    return {"IoU": ious, "Dice": dices,
            "mIoU": ___,                                 # BLANK 5
            "pixel_acc": ___}                            # BLANK 6
```

**Blanks**

1. `t + f + n`
2. `t / denom`
3. `2*t + f + n`
4. `2*t / d2`
5. `sum(ious.values()) / len(ious)`
6. `tp / (tp + fn) if tp + fn else 0.0`

**Solution**

```python
def seg_metrics(conf):
    if not conf:
        return {}
    ious, dices = {}, {}
    tp = fn = 0
    for c, (t, f, n) in conf.items():
        denom = t + f + n
        ious[c] = t / denom if denom else 0.0
        d2 = 2*t + f + n
        dices[c] = 2*t / d2 if d2 else 0.0
        tp += t
        fn += n
    return {"IoU": ious, "Dice": dices,
            "mIoU": sum(ious.values()) / len(ious),
            "pixel_acc": tp / (tp + fn) if tp + fn else 0.0}
```

**Tests**

```
seg_metrics({0: (8,1,1), 1: (6,1,1)})
-> {'IoU': {0: 0.8, 1: 0.75},
    'Dice': {0: 0.8888888888888888, 1: 0.8571428571428571},
    'mIoU': 0.775,
    'pixel_acc': 0.875}

seg_metrics({})  ->  {}
```

**Trap:** pixel accuracy is $14/16$, not $14/18$. The denominator counts **actual** pixels ($TP+FN$); FP belongs to another class.

**If the spec asks for 6 decimal places:** wrap the values at the end, not inside the loop:

```python
res = seg_metrics(conf)
return {k: (round(v, 6) if isinstance(v, float) else
            {c: round(x, 6) for c, x in v.items()}) for k, v in res.items()}
```

---

## Q8 — Scaled dot-product attention

**Why likely:** Class 08 Option D1; attention has 68 mentions. This is the one "medium" problem worth learning because it is heavily emphasised.

**Spec.** `Q` is `n_q × d`, `K` is `n_k × d`, `V` is `n_k × d_v`. Return `n_q × d_v`, rounded to 6 dp. Empty or mismatched → `[]`. Softmax row-wise and numerically stable.

**Scaffold**

```python
import numpy as np

def attention(Q, K, V):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.size == 0 or K.size == 0 or V.size == 0:
        return []
    if ___:                                         # BLANK 1
        return []
    scores = ___ / np.sqrt(___)                      # BLANK 2
    scores = scores - ___                            # BLANK 3
    w = ___                                          # BLANK 4
    w = ___                                          # BLANK 5
    return np.round(___, 6)                          # BLANK 6
```

**Blanks**

1. `K.shape[0] != V.shape[0] or Q.shape[1] != K.shape[1]`
2. `Q @ K.T` … `Q.shape[1]` (this is $d_k$)
3. `scores.max(axis=1, keepdims=True)`
4. `np.exp(scores)`
5. `w / w.sum(axis=1, keepdims=True)`
6. `w @ V`

**Solution**

```python
import numpy as np

def attention(Q, K, V):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.size == 0 or K.size == 0 or V.size == 0:
        return []
    if K.shape[0] != V.shape[0] or Q.shape[1] != K.shape[1]:
        return []
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)
    return np.round(w @ V, 6)
```

**Tests**

```
Q = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
K = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
V = [[1,0],[0,1],[0,0]]

attention(Q, K, V) -> array([[0.451863, 0.274069],
                             [0.274069, 0.451863],
                             [0.274069, 0.274069]])

attention([[1,0]], [[1,0],[0,1]], [[5,0],[0,5]])
-> array([[3.348808, 1.651192]])
```

**Trap 1:** `keepdims=True` is mandatory. Without it the max has shape `(n_q,)` and broadcasting subtracts the wrong element from each row.

**Trap 2:** `Q.shape[1]` is $d_k$, not the number of queries. They are equal only in self-attention with square inputs.

---

## Q9 — Cosine similarity and top-k retrieval

**Why likely:** Class 13 Option D1 and Class 05; CLIP/retrieval is the natural "foundation model" coding option.

**Spec.** `cosine(a,b)` returns a float; zero vector → `0.0`. `top_k(query, gallery, k)` returns `[(index, similarity), …]`, highest first, at most `k` items; empty → `[]`.

**Scaffold**

```python
import numpy as np

def cosine(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = ___, ___                              # BLANK 1
    if na == 0 or nb == 0:
        return 0.0
    return float(___)                              # BLANK 2

def top_k(query, gallery, k):
    q = np.asarray(query, dtype=float).ravel()
    G = np.asarray(gallery, dtype=float)
    if G.ndim == 1:
        G = G[None, :]
    if q.size == 0 or G.size == 0 or k <= 0:
        return []
    qn = q / np.linalg.norm(q) if np.linalg.norm(q) else q
    Gn = ___                                       # BLANK 3
    sims = ___                                     # BLANK 4
    order = np.argsort(___, kind="stable")[:k]     # BLANK 5
    return [(int(i), float(sims[i])) for i in order]
```

**Blanks**

1. `np.linalg.norm(a), np.linalg.norm(b)`
2. `a @ b / (na * nb)`
3. `G / np.linalg.norm(G, axis=1, keepdims=True)`
4. `Gn @ qn`
5. `-sims`

**Solution**

```python
import numpy as np

def cosine(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

def top_k(query, gallery, k):
    q = np.asarray(query, dtype=float).ravel()
    G = np.asarray(gallery, dtype=float)
    if G.ndim == 1:
        G = G[None, :]
    if q.size == 0 or G.size == 0 or k <= 0:
        return []
    qn = q / np.linalg.norm(q) if np.linalg.norm(q) else q
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    sims = Gn @ qn
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in order]
```

**Tests**

```
cosine([1,2,3], [2,4,6])   ->  1.0
cosine([1,0], [0,1])       ->  0.0
cosine([0,0], [1,0])       ->  0.0

top_k([1,0], [[1,0],[0,1],[0.6,0.8]], 2)
->  [(0, 1.0), (2, 0.6)]
```

**Trap:** normalise **rows** of the gallery (`axis=1, keepdims=True`). Normalising the whole matrix gives the wrong denominator.

---

## Q10 — Disparity to depth, and projection

**Why likely:** Class 12 Option D1 and Class 11 Option D1; `Z=fB/d` is named in `MISSION.md` as a Section F target.

**Spec.** `disparity_to_depth(d, f, B)` returns `f*B/d` where `d>0`, else `inf`. `depth_error(Z, Δd, f, B)` returns `Z²Δd/(fB)`. `project_point` returns pixel `(u,v)`.

**Scaffold**

```python
import numpy as np

def disparity_to_depth(d, f, B):
    d = np.asarray(d, dtype=float)
    out = ___                                        # BLANK 1
    mask = ___                                       # BLANK 2
    out[mask] = ___                                  # BLANK 3
    return out

def depth_error(Z, delta_d, f, B):
    return ___                                       # BLANK 4

def project_point(X, Y, Z, fx, fy, cx, cy):
    return ___, ___                                  # BLANK 5
```

**Blanks**

1. `np.full(d.shape, np.inf, dtype=float)`
2. `d > 0`
3. `f * B / d[mask]`
4. `Z**2 * delta_d / (f * B)`
5. `fx*X/Z + cx` … `fy*Y/Z + cy`

**Solution**

```python
import numpy as np

def disparity_to_depth(d, f, B):
    d = np.asarray(d, dtype=float)
    out = np.full(d.shape, np.inf, dtype=float)
    mask = d > 0
    out[mask] = f * B / d[mask]
    return out

def depth_error(Z, delta_d, f, B):
    return Z**2 * delta_d / (f * B)

def project_point(X, Y, Z, fx, fy, cx, cy):
    return fx*X/Z + cx, fy*Y/Z + cy
```

**Tests**

```
disparity_to_depth(24, 720, 0.54)   ->  16.2
disparity_to_depth(23, 720, 0.54)   ->  16.904347826086956
disparity_to_depth(0, 720, 0.54)    ->  inf

depth_error(16.2, 1, 720, 0.54)     ->  0.675

project_point(-0.30, -0.15, 2.0, 800, 800, 640, 360)
->  (520.0, 300.0)
```

**Trap:** guard `d=0` **before** dividing. `np.where(d>0, f*B/d, inf)` still evaluates the division and emits a warning; the masked assignment avoids it.

**Units:** `f` in pixels, `B` in metres, `d` in pixels → `Z` in metres. Always state this in Section F.

---

## Backups — one-liners worth 10 minutes each

These are short enough to memorise the night before.

```python
# Focal loss: p is the full probability vector, y is the true class index
def focal_loss(p, y, alpha=0.25, gamma=2.0):
    p = np.asarray(p, dtype=float)
    pt = p[y]
    return float(-alpha * (1 - pt)**gamma * np.log(pt))

# EMA teacher update
def ema_update(old, new, decay=0.996):
    return decay * old + (1 - decay) * new

# Forward diffusion, jump to any t
def q_sample(x0, abar, eps):
    return np.sqrt(abar) * x0 + np.sqrt(1 - abar) * eps

# Classifier-free guidance
def cfg(cond, uncond, s):
    return uncond + s * (cond - uncond)

# Nearest-neighbour upsampling
def nearest_upsample(x, out_h, out_w):
    x = np.asarray(x)
    H, W = x.shape
    rows = np.arange(out_h) * H // out_h
    cols = np.arange(out_w) * W // out_w
    return x[rows][:, cols]

# Output-size calculator (free marks)
def conv_output_size(n, k, s=1, p=0, d=1):
    k_eff = k + (k - 1) * (d - 1)
    return (n + 2*p - k_eff) // s + 1
```

---

## The 90-minute drill

| Time | Do |
|---|---|
| 0–25 min | Q1–Q4: write each from memory, run the two tests |
| 25–50 min | Q5–Q7: IoU/Dice, NMS, seg metrics |
| 50–70 min | Q8–Q10: attention, cosine/top-k, depth |
| 70–90 min | Close the notes. Write Q2, Q5, Q7 cold. If one fails, rewrite it once. |

**In the exam:** spend the first minute choosing the option that matches a template, not the one that looks most impressive. Then write the empty case, the shape, the loop, the return. Test with the smallest input you can.

**Partial credit:** a correct shape calculation and a correct loop with one wrong edge case usually earns more than an empty page.
