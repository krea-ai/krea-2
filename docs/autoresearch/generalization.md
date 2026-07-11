# Reducing DRaFT face and expression overfitting

## Failure mode

The current DRaFT stage improves held-out antelopev2 identity similarity, but
some neutral prompts reproduce the pose or expression of one reference image.
This is reward-driven prototype collapse rather than only conventional SFT
overfitting: identity improves while seed diversity, expression control, and
distance from the nearest training crop regress.

The current face reward makes this shortcut attractive. For normalized
generated embedding `z`, normalized reference embeddings `r_i`, and their
normalized centroid `mu`, it uses:

```text
identity(z) = 0.75 * cosine(z, mu)
            + 0.25 * smooth_max_i cosine(z, r_i)
```

The second term explicitly rewards moving toward one training view. A
discriminative recognition embedding is also not guaranteed to be perfectly
invariant to expression, pose, crop quality, or closed eyes when optimized
through a generator.

## Recommended order

### 1. Fix and saturate the identity reward

Use the normalized identity centroid only at first
(`nearest_reference_weight=0`). Do not keep pushing cosine similarity toward
one indefinitely. Replace the linear score with a soft hinge that has little
gradient after reaching a character-specific target:

```text
L_identity = softplus((target_similarity - cosine(z, mu)) / temperature)
```

Choose `target_similarity` from held-out SFT-500 generations rather than the
training references. This changes identity similarity from the sole objective
into a constraint: once identity is adequate, prompt alignment, diversity, and
the SFT prior determine the result.

If individual-reference information is later useful, add an anti-copy term
instead of a nearest-reference bonus. With
`p_i = softmax(cosine(z, r_i) / temperature)`, penalize excessive reference
selection using `KL(p || uniform)`. Calibrate its allowed margin from
leave-one-reference-out embeddings so naturally unusual views are not forced
to match every reference equally. This term is undefined for a one-image
dataset, where prior preservation is essential.

Build the centroid with one equal vote per source image. Downweight or exclude
blurred, heavily occluded, extreme-profile, and closed-eye crops as identity
anchors while retaining them as SFT images with accurate captions.

### 2. Anchor DRaFT to the frozen SFT-500 model

Keep a frozen copy of the initial SFT LoRA and penalize deviation of the
trainable model's velocity prediction on the same `(x_t, prompt, t)`:

```text
L = L_identity
  + lambda_anchor * mean(||v_draft - stopgrad(v_sft500)||^2)
```

Evaluate the anchor at the differentiable final step already used by DRaFT-K,
so it needs another LoRA evaluation rather than another full denoising
trajectory. A small fraction of steps should use generic person prompts and
the anchor only. This is the DRaFT analogue of class-prior preservation: it
keeps neutral face, pose, expression, and prompt behavior close to the model
that existed before reward optimization.

Regularizing effective LoRA deltas is a cheaper fallback, but raw `A/B` L2 is
factorization-dependent. Prediction anchoring is the better primary method.

### 3. Restrict which SFT LoRA tensors DRaFT may update

The current trainer updates every INT8 linear inside each DiT block:
attention `wq`, `wk`, `wv`, attention `gate`, `wo`, and MLP `gate`, `up`, and
`down`. For DRaFT, load the complete SFT adapter but freeze all except the
selected modules. Do not remove the frozen SFT tensors from the adapter.

Test these in order:

1. `wv + wo` only, rank 32;
2. `wq + wk + wv + wo`, rank 32;
3. QKVO, rank 16;
4. all block linears, current control.

QKVO should reduce capacity substantially by freezing attention gating and
all MLPs. However, Krea 2 uses single-stream attention over combined text and
image tokens, not classic cross-attention. Changing Q/K changes token routing
and can weaken prompt or expression control. V/O-only is therefore a serious
candidate, not merely an additional low-capacity baseline. Custom Diffusion's
finding that a small attention subset is enough for subject customization
supports this direction, but its cross-attention K/V result does not transfer
verbatim to this architecture.

### 4. Optimize expected reward over face augmentations

Apply stochastic, differentiable transforms after landmark alignment and
before the recognition network, then average two identity losses. Recommended
weak transforms are:

- horizontal flip with some clean, unflipped probability;
- scale/translation and at most about five degrees of rotation;
- brightness, contrast, gamma, mild color temperature, blur, and sensor noise;
- small random occlusion of the eye or mouth region using aligned landmarks.

Use the same augmentation family offline when constructing each reference
embedding, but average augmentations within each source image before averaging
images. This prevents one image from receiving more prototype weight merely
because it produced more augmented crops.

Eye/mouth dropout is more relevant to the reported failure than generic color
jitter: it makes a closed-eye configuration or particular smile unreliable as
an identity shortcut. Keep transforms weak enough that antelopev2 embeddings
remain in-distribution. Do not transform the full image before the
non-differentiable detector; augment the aligned 112x112 crop.

### 5. Separate expression from identity and reward diversity

Caption every reference expression, gaze, and coarse head pose. DRaFT prompt
generation should balance neutral, smiling, serious, eyes-open, eyes-closed,
left/right profile, and looking-away examples. Randomly drop the expression
clause on some training prompts so an unspecified expression learns a
distribution rather than one canonical training face.

For neutral or expression-unspecified prompts, periodically generate two
independent full trajectories rather than only the correlated LV perturbation.
Use:

```text
L_pair = mean(L_identity(z1), L_identity(z2))
       + lambda_anchor * L_anchor
       + lambda_diversity * relu(margin - distance(a1, a2))
```

Here `a1/a2` are frozen expression/pose attributes (action units, landmarks,
or a 3D face model), not ArcFace identity embeddings. The pair must retain
similar identity while differing in expression or pose. Run this more
expensive step intermittently, for example one update in four. For prompts
that specify an expression, replace diversity with expression alignment.

Face2Diffusion independently identifies expression leakage as an
identity/editability problem and combines expression guidance with
class-guided denoising regularization. PhotoMaker's multi-image stacked
identity representation likewise supports representing identity from the
whole reference set rather than selecting one view.

## Evaluation gate

Mean face similarity alone cannot select this change. Keep the existing ten
held-out prompts and add the following metrics:

- neutral-prompt seed grid: at least eight seeds for each character;
- nearest-reference pixel/LPIPS or DINO distance on aligned face crops;
- nearest-reference assignment entropy and `max_reference - centroid` gap;
- expression/pose variance across seeds for expression-unspecified prompts;
- explicit-expression accuracy and text-image alignment;
- face detection and centroid identity similarity.

Reject a candidate if identity rises while nearest-reference distance falls
sharply or neutral-prompt expression variance collapses. Include deliberately
bad reference expressions (closed eyes, blur, extreme profile) in the audit.

## Six-character LV-60 control

The finalized low-LR LV-60 control was rerun with ten held-out prompts per
character, seed 42, 12 training steps, and 20 evaluation steps. Similarity
counts a missed face as zero.

| Character | LV-60 similarity / detection | SFT-1000 similarity / detection | LV / SFT time |
| --- | ---: | ---: | ---: |
| Finn Wolfhard | 0.2430 / 70% | 0.1281 / 70% | 594.2 / 549.5 s |
| John Francis Flynn | 0.3193 / 80% | 0.2107 / 90% | 576.7 / 549.0 s |
| Julia Jacklin | 0.6075 / 100% | 0.2412 / 100% | 560.8 / 547.3 s |
| One photo | 0.5126 / 90% | 0.2154 / 90% | 584.4 / 562.7 s |
| Saya Gray | 0.5530 / 100% | 0.2508 / 100% | 577.6 / 544.0 s |
| Tommy Guerrero | 0.5848 / 100% | 0.3421 / 100% | 617.7 / 565.3 s |
| **Mean** | **0.4700 / 90.0%** | **0.2314 / 91.7%** | **585.2 / 553.0 s** |

LV-60 more than doubles mean identity similarity (`+103%`) for 5.8% more
optimization time on average. The slightly lower detection rate and observed
reference-expression copying show why automatic identity similarity cannot be
the only promotion criterion for the next phase.

## Minimal ablation matrix

Reuse the exact SFT-500 states and run only 60-step DRaFT branches on Julia
Jacklin and Tommy Guerrero:

| ID | Trainable modules | Identity reward | Anchor | Crop EOT |
| --- | --- | --- | --- | --- |
| G0 | all linears | current 75/25 | no | no |
| G1 | all linears | saturated centroid | no | no |
| G2 | QKVO | saturated centroid | no | no |
| G3 | V/O | saturated centroid | no | no |
| G4 | best of G2/G3 | saturated centroid | yes | no |
| G5 | best of G2/G3 | saturated centroid | yes | two views |
| G6 | G5 | saturated centroid | yes | two views + intermittent expression diversity |

G1 isolates the reward bug before capacity changes. G2/G3 test the proposed
module restriction. G4 is expected to provide the largest generalization gain
after G1; G5/G6 target expression-specific copying.

## Literature basis

- [DreamBooth](https://arxiv.org/abs/2208.12242) introduces class-specific
  prior preservation to retain diverse pose, view, scene, and lighting
  behavior during few-shot personalization.
- [Custom Diffusion](https://arxiv.org/abs/2212.04488) demonstrates that a
  small attention parameter subset can represent a new concept and uses
  regularization images to limit overfitting.
- [Face2Diffusion](https://arxiv.org/abs/2403.05094) explicitly removes
  identity-irrelevant information, separates expression guidance, and uses
  class-guided denoising regularization.
- [PhotoMaker](https://arxiv.org/abs/2312.04461) constructs a unified identity
  representation from a stack of reference images to retain identity while
  supporting flexible attributes.
- [ArcFace](https://arxiv.org/abs/1801.07698) explains the normalized angular
  identity geometry used by antelopev2, but does not make the embedding a
  complete perceptual or expression-invariant generator objective.
