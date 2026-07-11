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

### 1. Fix and saturate the identity reward — implemented

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

### 2. Anchor DRaFT to the frozen SFT-500 model — not implemented

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

This teacher-forward design was deliberately rejected for the current profile.
It adds another DiT evaluation and directly constrains outputs, but its compute
cost works against the shortest-training-time objective.

#### Lower-cost anchoring candidates

The preferred future experiment is **effective-delta L2-SP**. Preserve the
SFT-500 effective LoRA update, not its arbitrary factorization:

```text
L_SP = lambda_SP * sum_layers ||B A - B_SFT A_SFT||_F^2
```

The Frobenius term can be evaluated through rank-by-rank Gram matrices without
materializing the large dense update. This is invariant to equivalent LoRA
factorizations, adds no DiT forward, and anchors exactly to SFT rather than to
zero. Ordinary AdamW is not a substitute: weight decay pulls `A/B` toward zero
and can erase the useful SFT adapter. [L2-SP](https://proceedings.mlr.press/v80/li18a/li18a.pdf)
is the closest established starting-point regularizer.

Other candidates, in priority order:

1. **Interleaved cached-SFT replay:** every Nth update uses the saved SFT
   latent/text cache and flow objective. It needs no teacher model and protects
   actual denoising behavior, but adds a full DiT forward/backward and requires
   bringing the SFT cache into the DRaFT process.
2. **Fisher-weighted effective-delta penalty:** estimate QKVO importance from a
   small SFT calibration pass, then penalize movement in important directions.
   This follows [Elastic Weight Consolidation](https://doi.org/10.1073/pnas.1611835114),
   but the diagonal Fisher is basis-dependent unless applied carefully to the
   effective update.
3. **SFT-gradient subspace projection:** save a low-rank sketch of SFT gradient
   directions and project DRaFT gradients away from them. This is related to
   [Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104), avoids a
   teacher forward, but introduces optimizer complexity and may remove useful
   identity directions.
4. **Self-regularized/orthogonal LoRA updates:** constrain the DRaFT increment
   relative to the SFT LoRA subspace, following the motivation of
   [Continual Diffusion C-LoRA](https://arxiv.org/abs/2304.06027). This is more
   structural than L2-SP and should follow, not precede, the simpler baseline.

Effective-delta L2-SP is the recommended first anchor because it is cheap,
SFT-relative, factorization-invariant, and compatible with QKVO-only updates.
It remains an investigation item and is not active in the trainer.

### 3. Restrict which SFT LoRA tensors DRaFT may update — implemented

SFT updates every INT8 linear inside each DiT block: attention `wq`, `wk`,
`wv`, attention `gate`, `wo`, and MLP `gate`, `up`, and `down`. DRaFT loads the
complete SFT adapter but freezes all except the selected modules.

DRaFT now updates Q/K/V/O only, at rank 32. Attention gating and all MLP LoRA
tensors retain their exact SFT-500 values. The full adapter is still loaded and
saved, preserving compatibility with existing inference and ComfyUI conversion.
Custom Diffusion's finding that a small attention subset is enough for subject
customization supports this capacity restriction, though its cross-attention
K/V result does not transfer verbatim to Krea's single-stream architecture.

### 4. Optimize expected reward over face augmentations — implemented

The implementation applies stochastic, differentiable transforms after
landmark alignment and before the recognition network, then averages two
normalized embeddings. Active weak transforms are:

- horizontal flip with some clean, unflipped probability;
- scale/translation and at most about five degrees of rotation;
- brightness, contrast, blur, and sensor noise;
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

### 5. Separate expression from identity and reward diversity — implemented

DRaFT prompt generation now balances neutral, smiling, serious, eyes-open,
eyes-closed, looking-aside, surprised, laughing, and thoughtful/lowered-gaze
examples. It omits the expression clause on some training prompts so an
unspecified expression learns a distribution rather than one canonical face.

For neutral or expression-unspecified prompts, periodically generate two
independent full trajectories rather than only the correlated LV perturbation.
Use:

```text
L_pair = mean(L_identity(z1), L_identity(z2))
       + lambda_diversity * relu(margin - distance(a1, a2))
```

Here `a1/a2` are differentiable low-frequency descriptors of the aligned eye
and mouth regions, not ArcFace identity embeddings. The pair retains similar
identity while differing in expression-related image structure. The extra
trajectory runs one update in four. Prompts that specify an expression receive
independent identity supervision but no diversity penalty.

The active schedule leaves 25% of prompts expression-unspecified and balances
the other 75% over neutral, smiling, serious, laughing, looking aside,
surprised, thoughtful/lowered-gaze, and closed-eye cases. Every fourth DRaFT
step replaces the LV auxiliary sample with an independent trajectory, keeping
the number of differentiable image graphs at two. The diversity descriptor is
a low-frequency aligned eye/mouth representation; explicit-expression prompts
receive independent identity supervision without a diversity penalty.

A separate deterministic viewpoint schedule leaves 25% of prompts open and
balances the remainder over frontal, left/right three-quarter, left/right
profile, high, low, and oblique views. Generated EOT always contains a mirrored
aligned crop, rather than relying on a random flip, so both yaw directions
receive identity gradients even when the reference folder is one-sided.

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

## Implemented profile and four-character comparison

The selected profile is fixed rather than exposed as a module-target ablation:
QKVO rank 32, 60 DRaFT updates at 1e-4, saturated centroid target 0.45,
anti-copy entropy weight 0.02, two generated EOT views, four reference EOT
views, and an independent pair every fourth update. Reward memory is bounded
to the largest generated face and one secondary duplicate; this fixed an OOM
caused by spurious multi-face detections without letting a small background
face win the identity maximum.

Four comparisons were retrained from scratch. Each uses 64 DRaFT prompts, ten
held-out prompts, seed 42, 12 training and 20 evaluation steps. Similarity
counts a missed detection as zero.

| Character | QKVO DRaFT-60 | SFT-1000 | Detection (both) | Training seconds (hybrid / SFT) |
| --- | ---: | ---: | ---: | ---: |
| One photo | 0.6267 | 0.3721 | 100% | 636.5 / 502.8 |
| Julia Jacklin | 0.6223 | 0.3442 | 100% | 660.4 / 575.9 |
| Tommy Guerrero | 0.4187 | 0.2195 | 90% | 684.7 / 560.1 |
| John Francis Flynn | 0.4924 | 0.3542 | 90% | 574.3 / 563.3 |
| **Mean** | **0.5400** | **0.3225** | **95%** | **639.0 / 550.5** |

The protected hybrid improves mean identity similarity by 67.4% at identical
mean detection. It uses 16.1% more optimization time in aggregate, and every
hybrid run remains below 685 seconds. For the multi-reference characters, the
final mean reference-assignment maximum probability is 0.229–0.307; the
single-photo value is necessarily 1.0 and is not a collapse diagnostic.

The first Julia pose-aware learning curve also ran to step 100 before the
production length was reset to 60. Its extreme profile/oblique held-out scores
rose from 0.320 in the original protected run to 0.594–0.615, while mean
similarity reached 0.6569. This established that viewpoint coverage, rather
than only more updates, fixes the corner-angle failure; 60 steps was retained
for the requested faster default.

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
