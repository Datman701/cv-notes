# Compression must not drop examinable content

Session 10 was written at roughly half the length of Classes 01/08/09, after
the user said the earlier guides were "verbose and unintuitive". The compression
was correct in principle but **over-shot**: the eight architectures on slides
p.47–58 (FCN, SegNet, U-Net, DeepLab v1/v2, ASPP, PSPNet, Mask R-CNN, SETR)
survived only as one table row each, with no explanation of how any of them
works. The user noticed immediately and asked for them.

**Evidence:** the user listed eight architectures by name against a deck I had
summarised. Nothing else in that message was a complaint — the questions had
been approved in the previous turn, so the question bank was not the problem.

**Why this matters beyond Class 10.** Slide decks have a *coverage* dimension
and a *depth* dimension. Compressing should reduce depth — fewer examples, less
formality, fewer diagrams per idea — and never coverage. A guide that names an
architecture in a table but cannot answer "how does it work, and why was it
built that way?" has not taught it, regardless of how well it reads.

**Implications for later sessions.**
- Before writing, list the deck's section headings and check each one survives
  into the guide. Architecture-heavy decks (Class 10 p.47–58, Class 12's
  module list) need a dedicated section, not a summary table.
- Compression targets: prose, redundant restatement, and diagram count.
  Non-negotiable: every distinct model/method the slides introduce.
- When in doubt, cut *why it matters* and *worked detail* before cutting
  *what the thing is*.

Related: [[NOTES.md]] records the format preference itself.