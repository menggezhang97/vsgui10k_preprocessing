# Baselines

We currently use three baselines on VSGUI10K, moving from raw image space to a coarse structured UI space:

- **CNN image baseline**
- **Natural patch transformer baseline**
- **Simple UI-sequence transformer baseline**

---

## 1. CNN Image Baseline

This is the simplest image-space baseline.

It takes:

- the full GUI image,
- recent fixation history `(x, y, dur)`,
- and the target cue,

and predicts the **next fixation location** `(x, y)`.

The image is encoded with a ResNet18 backbone, the history is encoded with a small MLP, and the cue is embedded separately before fusion.

### Result

Full test evaluation:

- normalized distance mean: **0.2281**
- normalized distance median: **0.1902**
- pixel distance mean: **154.04**
- pixel distance median: **127.74**
- target hit rate:
  - r=1.0: **0.0314**
  - r=1.5: **0.0587**
  - r=2.0: **0.0850**
  - r=3.0: **0.1337**

This serves as the basic natural-image reference.

---

## 2. Natural Patch Transformer Baseline

This baseline replaces the global CNN image feature with **patch-level visual tokens**.

It takes:

- image patch tokens,
- fixation history `(x, y, dur)`,
- and the cue,

and predicts the **next fixation location** `(x, y)`.

The goal is to test whether tokenized visual representations are stronger than a single global image feature.

### Result

Best pilot checkpoint, evaluated on the full test split:

- normalized distance mean: **0.2251**
- normalized distance median: **0.1813**
- pixel distance mean: **151.51**
- pixel distance median: **120.97**
- target hit rate:
  - r=1.0: **0.0366**
  - r=1.5: **0.0646**
  - r=2.0: **0.0951**
  - r=3.0: **0.1449**

This baseline is slightly better than the CNN baseline on all main fixation-space metrics.

### Training note

We also tested longer training. Increasing the number of epochs beyond 2 did **not** help. Validation distance was best at epoch 2 and then became worse, which suggests that this model reaches its best generalization point quite early under the current setup.

---

## 3. Simple UI-Sequence Transformer Baseline

This is our first bridge baseline toward structured UI modelling.

It does **not** use the raw image directly. Instead, each fixation is mapped to a discrete UI token using the existing segmentation-based UI mapping, and the model predicts the **next UI token** from:

- the cue,
- the history of UI tokens,
- and the history of fixation `(x, y, dur)` features.

So this is a **next UI token classification** model, not a fixation regression model.

### Important limitation

The current UI tokens come from the dataset’s automatic segmentation / UI mapping.

We checked this mapping visually, and it is quite coarse. The extracted UI elements are often rough and not finely aligned. So this baseline should be treated as a **coarse structured approximation**, not a clean or high-fidelity UI representation.

The current token definition is also simple and screen-specific:

- `img_name::ui_class::ui_idx`

### Results

We tested this baseline under several training budgets.

**Aligned pilot (12k / 2k / 2k, 2 epochs)**

- test_top1: **0.0940**
- test_top5: **0.1205**

**12k / 2k / 2k, 10 epochs**

- best val_top1: **0.2305**
- best val_top5: **0.3185**

**30k / 4k / 4k, 5 epochs**

- best val_top1: **0.2858**
- best val_top5: **0.3960**

**Full scan, 5 epochs**

- best val_top1: **0.3850**
- best val_top5: **0.6171**

### Interpretation

This baseline is clearly learnable, and it improves a lot with more data and more training. This means that even the current coarse UI mapping still contains meaningful structural signal.

At the same time, it is still very limited:

- it only uses the fixation-hit UI sequence,
- it has no full UI element set as memory,
- and it has no explicit interaction between fixation history and the full UI structure.

So this model is useful as a **bridge baseline**, but it is not the final structured UI model.

---

# Summary

At this stage:

- the **CNN baseline** provides a stable image-space reference,
- the **natural patch transformer** is a stronger intermediate baseline and performs slightly better than the CNN baseline,
- the **simple UI-sequence baseline** is much cheaper to train and does learn meaningful signal, even with coarse automatic UI mapping, but its current formulation is still limited.

---

# Next Step

The next model should move beyond the current history-only UI sequence formulation and explicitly model:

- the **full UI token set**,
- the **fixation history**,
- the **cue-conditioned query**,
- and the interaction between gaze history and UI structure through **cross-attention / structured UI memory**.
