# Python & NumPy help — the 60-second lookup

**The one rule:** never spend more than **60 seconds** looking something up. If the docstring has not answered by then, write the fallback (last section) and move on. A working 8-line loop beats a perfect function you never finish.

**Check the version first.** `np.__version__` tells you which API you have. In **NumPy 2.x**, `np.lookfor` and `np.source` were removed — the recipes below use only what still exists.

---

## 1. The lookup ladder

Run these in order. Stop as soon as you have the answer.

| Step | Command | What it gives you |
|---|---|---|
| 1 | `help(fn)` | Full docstring, paged |
| 2 | `print(fn.__doc__)` | Same text, no pager — best inside a script |
| 3 | `inspect.signature(fn)` | Argument names, order, defaults |
| 4 | `dir(obj)` | Every attribute/method name |
| 5 | `[n for n in dir(np) if 'pad' in n.lower()]` | Find a name you half-remember |
| 6 | `np.info(fn)` | NumPy's own short reference |
| 7 | `inspect.getsource(fn)` | Source, when available |

```python
import numpy as np
import inspect

help(np.pad)                      # 1. paged; press q to quit
print(np.pad.__doc__[:400])       # 2. no pager
print(inspect.signature(np.pad))  # 3. (array, pad_width, mode='constant', **kwargs)
print([n for n in dir(np) if 'pad' in n.lower()])   # 4-5. ['pad']
np.info(np.pad)                   # 6. short reference
```

**Pager warning.** `help()` opens a pager in the interactive REPL. Press **`q`** to get out. Inside a script or a notebook cell, use `print(fn.__doc__)` instead — it cannot trap you.

**If `__doc__` is `None`** (some C extensions), fall back to `help(fn)` or `np.info(fn)`.

---

## 2. Read a NumPy docstring in 20 seconds

Do not read top to bottom. Read these four sections only:

1. **Signature** — argument order and defaults.
2. **Parameters** — the **shape** and **dtype** of each argument.
3. **Returns** — shape, dtype, and whether it is a copy or a view.
4. **Examples** — one example tells you more than the prose.

Skip `Notes`, `References`, `See Also`, and the long prose paragraph.

Ask four questions:

- Which axis does it operate on? (`axis=`)
- Does it keep the reduced dimension? (`keepdims=`)
- Does it return a copy or a view? (`np.pad` copies; `sliding_window_view` returns a **read-only** view)
- Does it modify the input in place? (`out=`)

Worked example — `np.pad`:

```python
print(inspect.signature(np.pad))
# (array, pad_width, mode='constant', **kwargs)

print([l for l in np.pad.__doc__.splitlines() if l.strip()][:6])
# ['Pad an array.', 'Parameters', '----------',
#  'array : array_like of rank N', '    The array to pad.', ...]
```

That is enough to write:

```python
x = np.pad(x, ((p, p), (p, p)))              # same padding on H and W
x = np.pad(x, ((1, 0), (0, 1)), mode='edge') # asymmetric, replicate edges
```

Worked example — `sliding_window_view`:

```python
print(inspect.signature(np.lib.stride_tricks.sliding_window_view))
# (x, window_shape, axis=None, *, subok=False, writeable=False)
```

`axis=None` slides over **all** axes. For a 2D convolution you want both axes; for a 1D moving average over the last axis only, pass `axis=-1`.

```python
a = np.arange(6)
print(np.lib.stride_tricks.sliding_window_view(a, 3))
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]
#  [3 4 5]]
```

The result is a **read-only view** (`writeable=False`). Pass `writeable=True` if you need to assign into it, or `.copy()` to get a normal array.

---

## 3. Forgotten-name recipes

**You know the concept, not the name.** Filter `dir()`:

```python
[n for n in dir(np) if 'pad' in n.lower()]        # ['pad']
[n for n in dir(np.linalg) if 'norm' in n.lower()] # ['matrix_norm', 'norm', 'vector_norm']
```

**You know neither.** Describe the operation in the docs, not with `np.lookfor`:

```python
# NumPy 2.x: np.lookfor was removed.
# Use the online docs, or dir() + __doc__:
for name in dir(np):
    obj = getattr(np, name)
    doc = getattr(obj, '__doc__', '') or ''
    if 'moving average' in doc.lower():
        print(name)
```

**You know the module, not the function.** `dir(np.linalg)`, `dir(np.random)`, `dir(np.fft)`.

**You know the Python type, not the method.** `dir(str)`, `dir(list)`, `dir(dict)`.

**Version check before using a half-remembered function:**

```python
print(np.__version__)          # 2.5.4 here
print(hasattr(np, 'lookfor'))  # False in NumPy 2.x
```

---

## 4. The 3-line probe

When the docstring is ambiguous, test the behaviour on a tiny array. This is faster than reading three paragraphs.

```python
a = np.arange(6).reshape(2, 3)
print(a.shape)                              # (2, 3)
print(a.sum(axis=0).shape)                  # (3,)   -> columns
print(a.sum(axis=1, keepdims=True).shape)   # (2, 1) -> rows, keeps dim
print((np.ones((2,1)) + np.ones((1,3))).shape)  # (2, 3) -> broadcasting
print(np.array([1,2]) / 2, np.array([1,2]) // 2)
# [0.5 1. ] [0 1]  -> / is float, // is floor
```

**What to probe:** `axis` direction, `keepdims` shape, broadcasting compatibility, integer vs float division, and whether the result is a view.

---

## 5. Top 30 NumPy calls for this exam

| Call | Signature fragment | Use | Gotcha |
|---|---|---|---|
| `np.pad` | `(array, pad_width, mode='constant')` | zero/edge padding | `pad_width=((top,bottom),(left,right))` |
| `sliding_window_view` | `(x, window_shape, axis=None)` | windows for conv/pool | **read-only** view |
| `np.einsum` | `(subscripts, *operands)` | arbitrary contractions | `'ij,jk->ik'` = matmul |
| `np.tensordot` | `(a, b, axes=2)` | tensor contraction | `axes` can be int or pair |
| `np.convolve` | `(a, v, mode='full')` | **1-D only** | 2-D needs loops |
| `np.correlate` | `(a, v, mode='valid')` | **1-D only** | same |
| `np.clip` | `(a, a_min, a_max)` | clamp values | `np.clip(x, 0, 1)` |
| `np.where` | `(condition, x=None, y=None)` | select / indices | 1-arg returns indices |
| `np.argmax` | `(a, axis=None, keepdims=False)` | index of max | first max on ties |
| `np.maximum` | `(x1, x2)` | elementwise max | `np.maximum(0, x)` = ReLU |
| `np.exp` / `np.log` | `(x)` | natural exp/log | not base-10 |
| `np.sum` | `(a, axis=None, keepdims=False)` | sum | empty sum = 0 |
| `np.mean` | `(a, axis=None, keepdims=False)` | mean | empty mean = `nan` |
| `np.var` / `np.std` | `(a, axis=None, ddof=0)` | spread | default is **population** |
| `np.linalg.norm` | `(x, ord=None, axis=None)` | vector/matrix norm | `axis=1` for row norms |
| `np.linalg.svd` | `(a, full_matrices=True)` | SVD | returns `U, s, Vh` |
| `np.linalg.solve` | `(a, b)` | solve `Ax=b` | prefer over `inv` |
| `np.linalg.inv` | `(a)` | inverse | slower, less stable |
| `np.linalg.det` | `(a)` | determinant | — |
| `np.meshgrid` | `(*xi, indexing='xy')` | coordinate grids | default is `'xy'` |
| `np.stack` | `(arrays, axis=0)` | new axis | shapes must match |
| `np.concatenate` | `(arrays, axis=0)` | join along existing axis | shapes must match elsewhere |
| `np.transpose` | `(a, axes=None)` | reorder axes | `a.T` reverses |
| `np.reshape` | `(a, shape, order='C')` | change shape | `-1` infers one dim |
| `np.broadcast_to` | `(array, shape)` | expand view | result is read-only |
| `np.allclose` | `(a, b, rtol=1e-05, atol=1e-08)` | tolerant compare | use in tests |
| `np.argsort` | `(a, axis=-1, kind=None)` | sort indices | `kind='stable'`; `-a` for desc |
| `np.bincount` | `(x, minlength=0)` | histogram of ints | non-negative ints only |
| `np.cumsum` | `(a, axis=None)` | running sum | histogram CDF |
| `np.unique` | `(ar, return_counts=False)` | distinct values | sorts by default |

**The five you will actually use in Section D:** `np.asarray`, `np.zeros`, `np.pad`, `np.sum`, `np.argmax`.

---

## 6. Python built-ins you will blank on

| Call | Help line | Signature |
|---|---|---|
| `sorted` | `help(sorted)` | `(iterable, key=None, reverse=False)` |
| `enumerate` | `help(enumerate)` | `(iterable, start=0)` |
| `zip` | `help(zip)` | `(*iterables, strict=False)` |
| `any` / `all` | `help(any)` | `(iterable)` |
| `sum` | `help(sum)` | `(iterable, start=0)` |
| `max` / `min` | `help(max)` | `(iterable, key=None)` |
| `round` | `help(round)` | `(number, ndigits=None)` |
| `range` | `help(range)` | `(start, stop, step)` |
| `dict.items` | `help(dict.items)` | `()` |
| `math.isclose` | `help(math.isclose)` | `(a, b, rel_tol=1e-09)` |
| `functools.reduce` | `help(functools.reduce)` | `(function, iterable)` |
| `collections.Counter` | `help(collections.Counter)` | `(iterable)` |
| `heapq.nlargest` | `help(heapq.nlargest)` | `(n, iterable, key=None)` |

**Sorting a detection list by score, ties by index** — the pattern you need for NMS:

```python
order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
```

Read that as: for each index `i`, sort by the tuple `(-score, i)`. Negative score makes it descending; `i` breaks ties deterministically.

---

## 7. "Forgot X → do Y instead"

| If you forget… | Write this instead |
|---|---|
| `np.pad` | `out = np.zeros((H+2*p, W+2*p)); out[p:p+H, p:p+W] = x` |
| `sliding_window_view` | nested `for` loops over `i`, `j` (the Q2 template) |
| `np.einsum` | `A @ B`, then `np.sum(..., axis=...)` |
| `np.clip` | `np.minimum(np.maximum(x, lo), hi)` |
| `np.allclose` | `np.max(np.abs(a-b)) < 1e-6` |
| `np.argsort` | `sorted(range(len(a)), key=lambda i: a[i])` |
| `np.stack` | `np.array(list_of_arrays)` |
| `np.bincount` | a `dict` counter loop |
| `np.cumsum` | a loop that adds into a running total |
| `np.linalg.norm` | `np.sqrt(np.sum(a*a))` |
| `np.linalg.solve` | `np.linalg.inv(A) @ b` (slower, but works) |
| `np.linalg.svd` | `U, s, Vh = np.linalg.svd(A, full_matrices=False)` — just memorise this one |

**The principle:** every NumPy function has a 3–5 line loop equivalent. In an exam, the loop is acceptable; the blank page is not.

---

## 8. Six lookup drills

Do these once with the docs open. They take 15 minutes and cover the whole ladder.

1. Find the `np.pad` mode that **replicates the edge pixel** instead of adding zeros.
2. Find how to make `sliding_window_view` slide over the **last axis only**.
3. Find the default `ddof` of `np.var`, and the shape change when you pass `axis=0`.
4. Find the return order and shapes of `np.linalg.svd(a, full_matrices=False)`.
5. Find the signature of `sorted` and write the NMS sort key from memory.
6. Find what `np.where(condition)` returns with **one** argument (not three).

Answers:

1. `mode='edge'`.
2. `axis=-1`.
3. `ddof=0`; `axis=0` reduces rows, leaving one value per column, shape `(W,)`.
4. `U` is `(M,K)`, `s` is `(K,)`, `Vh` is `(K,N)` with `K=min(M,N)`.
5. `(iterable, key=None, reverse=False)`; key = `lambda i: (-scores[i], i)`.
6. The indices where the condition is true, as a tuple of arrays.

---

## 9. Exam rules

1. **Time-box every lookup to 60 seconds.** Set the intention before you start: "I am looking for the argument name, nothing else."
2. **Read only four sections:** Signature → Parameters → Returns → Examples.
3. **`help()` pages; `__doc__` does not.** Use `print(fn.__doc__)` in scripts and notebooks.
4. **`np.lookfor` does not exist in NumPy 2.x.** Use `dir()` + a name filter.
5. **`np.info` is shorter than `help`.** Start there for NumPy functions.
6. **If the name is gone, describe the shape you need** and write the loop. The marker awards the logic.
7. **Never debug by printing a huge array.** Print `.shape`, `.dtype`, and one row.
8. **`sliding_window_view` is read-only by default.** Use `.copy()` before modifying; ordinary slices can write through.
9. **Round once, at the end.** Not inside a loop.
10. **Write the empty case first.** It is the most common hidden test.

```python
# The 10-second sanity block to put at the end of any answer
out = np.asarray(out)          # only if the spec lets you return an array
print(out.shape, out.dtype, out)
```

If that prints the right shape and dtype, you have probably passed the hidden tests.
