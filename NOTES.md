# NOTES — teaching preferences and working state

## How to teach this user

**Explicit feedback, session 4 (after Class 09):**
> "the content till now has been verbose and unintuitive … from this class
> onwards, make it intuitive and easy to understand, the questions are fine"

Concretely:

- **Compress the teaching sections hard.** Target roughly half the length of the
  Class 01/08/09 guides for sections B, C, D, E, F, G.
- **Lead with the intuition and the visual, then the formula.** One short
  paragraph of "why this exists" beats four of formal definition.
- **Keep the questions and worked solutions at current depth.** They were
  explicitly approved. Do not thin out the A–F banks.
- Fewer, smaller ASCII diagrams — not more. A diagram that carries one idea beats
  a set of seven that carry one idea each.
- Comparison tables earn their place only when a decision depends on them.

## Course facts established so far

- **Exam format** (`session16.pdf`, 9 slides): 6 sections, 100 marks, 120 min.
  A 10×1 · B 10×2 · C 10×2 (≤100 words) · D 20 (pick 1 of 3) · E 15 · F 15.
- **Grading** (confirmed by the user, overrides slide p.5):
  Quiz 25% · End Term 35% · Projects best-2-of-3 40%, where
  `Project Score = test-case score × project-understanding section in the exam`.
- **Assignments**: 3 Kaggle competitions, best 2 count. Briefs pulled via the
  user's `kaggle` CLI into `assignments/`.
- **Submission must be on the Kaggle leaderboard**, not the Google Form — a
  Form-only submission is not visible to the instructor and scores zero.

## Source-quality ledger (learn this before trusting a source)

| Session | Transcript | Slides | Confidence |
|---|---|---|---|
| 01 | full | text layer | high |
| 03, 05, 08, 09 | full | OCR (image-only) | high — OCR verified good |
| 06 | full | OCR, partial text | medium |
| 13 | full | OCR, **rotated 90°** | medium |
| **10** | **EMPTY (59 words)** | text layer | medium — no audio at all |
| 07 | full | **none exist** | low — transcript only |
| 16 | none | 9 slides | exam spec only |

**Always run the OCR/vision checks** rather than trusting a garbled line. Slide
39 (Class 08) renders "50%, 5%, 95%" — a slide-export artefact, confirmed at
190 DPI, not OCR noise.

**Rule adopted:** the Session 9 midterm quiz is excluded from difficulty/style
calibration and is never imitated.

## Build order (student prioritised 8 → 9 → 10 onward)

- Class 08 — detection transformers ✅
- Class 09 — video, project mechanics ✅
- Class 10 — segmentation ✅ (format compressed per feedback above)
- Class 11 — camera geometry ✅ (33/33 audit)
- Class 12 — 3D vision + motion ✅ (37/37 audit; example-first §B per user request)
- Class 13 — vision foundation models ✅ (22/22 audit; CLIP/CoCa/LLaVA/Flamingo)
- Class 14 — vision++ with foundational models ✅ (38/38 audit; DINO/Grounding DINO/SAM/Depth Anything)
- Class 15 — generative computer vision ✅ (32/32 audit; VAE/ELBO/DDPM/Imagen)
- Class 16 — exam strategy

## Exam-prep docs (built from `session16.pdf`)

- `study-notes/exam-prep/formula-sheet.md` — 28 one-line formulas, ★ priority
  marks, one micro-example each, explicit "not worth memorising" footer.
- `study-notes/exam-prep/coding-questions.md` — the 10 easiest high-probability
  Section-D options, each as a fill-in-the-blank scaffold + full NumPy solution
  + 2 tests + one trap; 6 one-liner backups; 90-minute drill.
- `study-notes/exam-prep/python-numpy-help.md` — 60-second lookup ladder
  (`help` / `__doc__` / `inspect.signature` / `dir` / `np.info`), 20-second
  docstring reading, 3-line probe, top-30 NumPy table, fallbacks.

**Environment change:** `numpy 2.5.4` is now installed for verification
(`python3 -m pip install --user numpy`). Note for the exam: **NumPy 2.x removed
`np.lookfor` and `np.source`**; the help doc documents the replacements.

**Audit:** all 10 coding solutions + 6 backups executed against every quoted
test (0 failures); all 28 formula examples recomputed from first principles;
every help command checked on NumPy 2.5.4.

## Source-quality ledger update (Classes 14–15)

- **14** — `session_14__2_.pdf` is 51 slides, **image-only but upright** (Page
  rot 0; no rotation needed, unlike Class 13). Transcript full. Confidence high.
- **15** — `session_15_annotated.pdf` (40 slides) **has a text layer** and ships
  embedded "EXPLANATION" sections, i.e. it is a *prepared/annotated* deck, not a
  raw export. Transcript full. Confidence high; formula images still lack text.

## Verification discipline that has repeatedly paid off

Execute every numeric answer and every code solution. Bugs found this way:
cross-correlation/convolution sign slip (C1), a `moving_average` guard-order bug,
a D1 `V`-row `IndexError` in attention, an inverted `sqrt(d_k)` simulation
variable, a 3×3×3×3 = **81** mis-multiplied as 27, and a pixel-accuracy
denominator that double-counted misclassified pixels.

**Pattern:** most bugs are in the *expected values*, not the code. Write the
audit before trusting a number. Latest examples (Class 12): the B5 baseline
table and F3 solution had the narrow-baseline depth errors inverted
(0.10 m → 6.9 m instead of the correct 34.7 m), and a case-study stem asserted
"disparity below one pixel at 60 m" when $fB/Z = 6.3$ px. Always re-derive
every quoted number from first principles.