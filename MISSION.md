# Mission: Pass the SST Computer Vision Final Exam

## Why
The Computer Vision final is **35% of the grade** and sits inside a 6-day window
that is already split with a separate C++ exam for a DevOps role, plus an FDE
assignment due the same week. Half the available study time — roughly **3 days** —
is all there is for CV. Passing requires exam-mechanics fluency under time
pressure, not a general CV education.

## Success looks like
- Sit the 120-minute, 100-mark paper in six sections (A MCQ · B multi-select ·
  C descriptive · D coding · E case study · F numericals) and finish it.
- Clear Sections A–C by recognition and D by writing tested Python.
- Get the Section F arithmetic right first time: conv/upsampling shapes, output
  stride, IoU/Dice/mIoU/PQ, receptive fields, camera depth `Z = f·B/d`.
- Hold the 40% project component: best 2 of 3 Kaggle assignments, submitted on
  the leaderboard (not the Google Form), with a written project-understanding
  section in the exam itself.

## Constraints
- ~3 days of CV study time inside 6.
- Classes 1–7 already studied (mid-term prep) — **do not re-teach anchors,
  R-CNN/Faster R-CNN, or one-stage detector mechanics.**
- No professor repository, notebooks, or Kaggle starter code exist locally.
  Class guides' Section D is written from scratch in stdlib Python.
- **numpy 2.5.4 is installed** (added for exam-prep verification). torch / cv2
  are still absent. The exam-prep coding doc uses **NumPy**, because the real
  Section D runs in Python + NumPy; the class guides remain pure-Python.

## Out of scope
- Classes 1–7 in depth (already covered).
- Building working Kaggle solutions — question banks only.
- Anything not traceable to the course slides/transcripts or a cited external
  source.