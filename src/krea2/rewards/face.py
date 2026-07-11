"""Differentiable antelopev2 face-similarity reward for DRaFT-K training.

Detection runs through ONNXRuntime/SCRFD on detached images. Recognition is a
small PyTorch executor for the antelopev2 recognition ONNX graph, so gradients
flow from the cosine-distance reward through the aligned crop to generated
images.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from insightface.model_zoo.scrfd import SCRFD

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

EXPRESSION_PROMPT_RE = re.compile(
    r"\b(?:expression|smil(?:e|es|ing)|grin(?:s|ning)?|laugh(?:s|ing)?|"
    r"surpris(?:ed|e)|frown(?:s|ing)?|"
    r"eyes?\s+(?:open|closed)|closed[- ]eyes?|gaze|looking\s+(?:away|aside)|"
    r"raised eyebrows?)\b",
    re.I,
)


@dataclass(frozen=True)
class OnnxNode:
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any]


def _providers(providers: list[str] | tuple[str, ...] | None) -> list[str]:
    requested = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    available = set(ort.get_available_providers())
    selected = [provider for provider in requested if provider in available]
    return selected or ["CPUExecutionProvider"]


def _det_size(value) -> tuple[int, int] | list[tuple[int, int]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple, np.ndarray))
    ):
        return [tuple(map(int, item)) for item in value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("det_size must be a width,height pair")
    return tuple(map(int, value))


def _reference_paths(reference_images) -> list[Path]:
    if reference_images is None:
        raise ValueError("reference_images is required")
    if isinstance(reference_images, (str, Path)):
        paths = [Path(reference_images)]
    else:
        paths = [Path(item) for item in reference_images]

    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(
                sorted(
                    item
                    for item in path.iterdir()
                    if item.is_file()
                    and (
                        item.suffix.lower() in IMAGE_EXTS
                        or (not item.suffix and cv2.haveImageReader(str(item)))
                    )
                )
            )
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise ValueError("reference_images did not resolve to any image files")
    return out


def _attrs(node) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


class OnnxRecognitionTorch(nn.Module):
    """PyTorch executor for antelopev2/recognition/model.onnx.

    The recognition graph uses only Conv, BatchNormalization, PRelu, Add,
    Flatten and Gemm, so a direct eager executor is enough and keeps the module
    differentiable with respect to its input crop.
    """

    def __init__(self, onnx_path: str | Path):
        super().__init__()
        graph = onnx.load(str(onnx_path)).graph
        self.input_name = graph.input[0].name
        self.output_name = graph.output[0].name
        self.constant_names: dict[str, str] = {}
        for idx, initializer in enumerate(graph.initializer):
            array = np.array(onnx.numpy_helper.to_array(initializer), copy=True)
            tensor = torch.from_numpy(array)
            if tensor.is_floating_point():
                tensor = tensor.float()
            buffer_name = f"const_{idx}"
            self.register_buffer(buffer_name, tensor)
            self.constant_names[initializer.name] = buffer_name
        self.nodes = [
            OnnxNode(
                op_type=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attrs=_attrs(node),
            )
            for node in graph.node
        ]

    def _value(self, name: str, values: dict[str, torch.Tensor]) -> torch.Tensor:
        if name in values:
            return values[name]
        return getattr(self, self.constant_names[name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: dict[str, torch.Tensor] = {self.input_name: x.float()}
        for node in self.nodes:
            inputs = [self._value(name, values) for name in node.inputs if name]
            if node.op_type == "Conv":
                y = self._conv(node, inputs)
            elif node.op_type == "BatchNormalization":
                y = F.batch_norm(
                    inputs[0],
                    running_mean=inputs[3],
                    running_var=inputs[4],
                    weight=inputs[1],
                    bias=inputs[2],
                    training=False,
                    eps=float(node.attrs.get("epsilon", 1e-5)),
                )
            elif node.op_type == "PRelu":
                slope = inputs[1].reshape(-1)
                if slope.numel() not in (1, inputs[0].shape[1]):
                    slope = slope.reshape(1)
                y = F.prelu(inputs[0], slope)
            elif node.op_type == "Add":
                y = inputs[0] + inputs[1]
            elif node.op_type == "Flatten":
                axis = int(node.attrs.get("axis", 1))
                if axis < 0:
                    axis += inputs[0].dim()
                y = torch.flatten(inputs[0], start_dim=axis)
            elif node.op_type == "Gemm":
                y = self._gemm(node, inputs)
            else:
                raise NotImplementedError(f"unsupported ONNX op: {node.op_type}")
            values[node.outputs[0]] = y
        return values[self.output_name]

    @staticmethod
    def _conv(node: OnnxNode, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, weight = inputs[:2]
        bias = inputs[2] if len(inputs) > 2 else None
        pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
        if pads[0] == pads[2] and pads[1] == pads[3]:
            padding = (pads[0], pads[1])
        else:
            x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
            padding = (0, 0)
        return F.conv2d(
            x,
            weight,
            bias,
            stride=tuple(node.attrs.get("strides", [1, 1])),
            padding=padding,
            dilation=tuple(node.attrs.get("dilations", [1, 1])),
            groups=int(node.attrs.get("group", 1)),
        )

    @staticmethod
    def _gemm(node: OnnxNode, inputs: list[torch.Tensor]) -> torch.Tensor:
        a, b = inputs[:2]
        c = inputs[2] if len(inputs) > 2 else None
        if int(node.attrs.get("transA", 0)):
            a = a.t()
        if int(node.attrs.get("transB", 0)):
            b = b.t()
        y = float(node.attrs.get("alpha", 1.0)) * (a @ b)
        if c is not None:
            y = y + float(node.attrs.get("beta", 1.0)) * c
        return y


def estimate_arcface_matrix(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        raise ValueError(f"expected 5x2 landmarks, got {kps.shape}")
    dst = ARCFACE_DST * (float(image_size) / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("failed to estimate ArcFace alignment")
    return matrix.astype(np.float32)


def bgr_to_rgb_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = image.detach().float().clamp(-1, 1).cpu()
    rgb = ((rgb + 1.0) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[:, :, ::-1])


def warp_affine_tensor(
    image: torch.Tensor,
    matrix: np.ndarray,
    *,
    output_size: int = 112,
) -> torch.Tensor:
    """Warp RGB CHW tensor using a source->destination pixel affine matrix."""
    device = image.device
    dtype = torch.float32
    matrix_t = torch.as_tensor(matrix, device=device, dtype=dtype)
    full = torch.eye(3, device=device, dtype=dtype)
    full[:2, :] = matrix_t
    inv = torch.linalg.inv(full)[:2, :]

    y, x = torch.meshgrid(
        torch.arange(output_size, device=device, dtype=dtype),
        torch.arange(output_size, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    dst = torch.stack((x, y, ones), dim=-1).reshape(-1, 3).t()
    src = (inv @ dst).t().reshape(output_size, output_size, 2)

    h, w = image.shape[-2:]
    grid_x = (src[..., 0] + 0.5) * (2.0 / float(w)) - 1.0
    grid_y = (src[..., 1] + 0.5) * (2.0 / float(h)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def bbox_matrix(bbox: np.ndarray, image_size: int = 112) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    scale_x = (image_size - 1) / width
    scale_y = (image_size - 1) / height
    return np.array(
        [[scale_x, 0.0, -x1 * scale_x], [0.0, scale_y, -y1 * scale_y]],
        dtype=np.float32,
    )


class FaceSimilarityReward(nn.Module):
    def __init__(
        self,
        reference_images=None,
        model_dir: str | Path = "../antelopev2",
        providers: list[str] | tuple[str, ...] | None = None,
        ctx_id: int = 0,
        det_size=(640, 640),
        det_thresh: float = 0.5,
        crop_mode: str = "aligned",
        no_face_reward: float | None = None,
        no_face_penalty: float = 0.5,
        reference_face_policy: str = "largest",
        nearest_reference_weight: float = 0.0,
        nearest_temperature: float = 0.07,
        target_similarity: float = 0.45,
        saturation_temperature: float = 0.05,
        reference_entropy_weight: float = 0.02,
        reference_entropy_temperature: float = 0.10,
        eot_views: int = 2,
        reference_eot_views: int = 4,
        expression_diversity_weight: float = 0.05,
        expression_diversity_margin: float = 0.15,
        duplicate_identity_weight: float = 0.25,
        duplicate_identity_threshold: float = 0.35,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.model_dir = Path(model_dir).expanduser()
        self.det_size = _det_size(det_size)
        self.det_thresh = float(det_thresh)
        self.crop_mode = crop_mode
        if no_face_reward is not None:
            warnings.warn(
                "no_face_reward is deprecated and ignored; missed detections use "
                "the differentiable fallback reward plus no_face_penalty",
                DeprecationWarning,
                stacklevel=2,
            )
        self.no_face_penalty = float(no_face_penalty)
        self.reference_face_policy = reference_face_policy
        self.nearest_reference_weight = float(nearest_reference_weight)
        self.nearest_temperature = float(nearest_temperature)
        self.target_similarity = float(target_similarity)
        self.saturation_temperature = float(saturation_temperature)
        self.reference_entropy_weight = float(reference_entropy_weight)
        self.reference_entropy_temperature = float(reference_entropy_temperature)
        self.eot_views = int(eot_views)
        self.reference_eot_views = int(reference_eot_views)
        self.expression_diversity_weight = float(expression_diversity_weight)
        self.expression_diversity_margin = float(expression_diversity_margin)
        self.duplicate_identity_weight = float(duplicate_identity_weight)
        self.duplicate_identity_threshold = float(duplicate_identity_threshold)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.crop_mode not in {"aligned", "bbox"}:
            raise ValueError("crop_mode must be 'aligned' or 'bbox'")
        if self.reference_face_policy not in {"largest", "highest", "single", "all"}:
            raise ValueError(
                "reference_face_policy must be largest, highest, single, or all"
            )
        if not 0.0 <= self.nearest_reference_weight <= 1.0:
            raise ValueError("nearest_reference_weight must be in [0, 1]")
        if self.nearest_temperature <= 0.0:
            raise ValueError("nearest_temperature must be positive")
        if self.nearest_reference_weight:
            warnings.warn(
                "nearest_reference_weight is deprecated; the training reward uses "
                "a saturated centroid and anti-copy entropy instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if not -1.0 <= self.target_similarity <= 1.0:
            raise ValueError("target_similarity must be in [-1, 1]")
        if self.saturation_temperature <= 0.0:
            raise ValueError("saturation_temperature must be positive")
        if self.reference_entropy_weight < 0.0:
            raise ValueError("reference_entropy_weight must be non-negative")
        if self.reference_entropy_temperature <= 0.0:
            raise ValueError("reference_entropy_temperature must be positive")
        if self.eot_views <= 0 or self.reference_eot_views <= 0:
            raise ValueError("EOT view counts must be positive")
        if self.expression_diversity_weight < 0.0:
            raise ValueError("expression_diversity_weight must be non-negative")
        if self.expression_diversity_margin < 0.0:
            raise ValueError("expression_diversity_margin must be non-negative")

        detection_path = self.model_dir / "detection" / "model.onnx"
        recognition_path = self.model_dir / "recognition" / "model.onnx"
        if not detection_path.exists():
            raise FileNotFoundError(detection_path)
        if not recognition_path.exists():
            raise FileNotFoundError(recognition_path)

        selected_providers = _providers(providers)
        session = ort.InferenceSession(
            str(detection_path), providers=selected_providers
        )
        self.detector = SCRFD(model_file=str(detection_path), session=session)
        self.detector.prepare(
            ctx_id, input_size=self.det_size, det_thresh=self.det_thresh
        )
        self.valid_reference_images: list[str] = []
        self.skipped_reference_images: list[tuple[str, int]] = []
        reference_crops = self._reference_crops(_reference_paths(reference_images))
        if not reference_crops:
            raise RuntimeError(
                "no valid reference faces found; check detection threshold and images"
            )

        self.recognition = OnnxRecognitionTorch(recognition_path).eval().to(self.device)
        for param in self.recognition.parameters():
            param.requires_grad_(False)
        embeddings = self._encode_reference_crops(reference_crops)
        self.register_buffer("reference_embeddings", embeddings)
        prototype = F.normalize(embeddings.float().mean(dim=0, keepdim=True), dim=-1)
        self.register_buffer("reference_prototype", prototype)

    @torch.no_grad()
    def detect_faces(
        self, image_bgr: np.ndarray
    ) -> list[dict[str, np.ndarray | float]]:
        bboxes, kpss = self.detector.detect(
            image_bgr,
            input_size=self.det_size,
            max_num=0,
            metric="default",
        )
        if bboxes.shape[0] == 0:
            return []
        faces = []
        for idx in range(bboxes.shape[0]):
            faces.append(
                {
                    "bbox": bboxes[idx, :4].astype(np.float32),
                    "score": float(bboxes[idx, 4]),
                    "kps": None if kpss is None else kpss[idx].astype(np.float32),
                }
            )
        faces.sort(key=lambda item: float(item["score"]), reverse=True)
        return faces

    def encode_faces(self, crops: torch.Tensor) -> torch.Tensor:
        embeddings = self.recognition(crops.to(self.device).float())
        return F.normalize(embeddings.float(), dim=-1)

    @staticmethod
    def _random_values(
        count: int,
        crops: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.rand(
            count,
            device=crops.device,
            dtype=torch.float32,
            generator=generator,
        )

    def _augment_aligned_view(
        self,
        crops: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        force_flip: bool | None = None,
    ) -> torch.Tensor:
        """Apply weak differentiable identity-preserving crop transforms."""
        batch = crops.shape[0]
        values = self._random_values(batch * 9, crops, generator=generator).reshape(
            batch, 9
        )
        angle = (values[:, 0] * 2.0 - 1.0) * math.radians(5.0)
        scale = 0.94 + values[:, 1] * 0.12
        tx = (values[:, 2] * 2.0 - 1.0) * 0.05
        ty = (values[:, 3] * 2.0 - 1.0) * 0.05
        cos = torch.cos(angle) / scale
        sin = torch.sin(angle) / scale
        theta = torch.zeros(batch, 2, 3, device=crops.device, dtype=torch.float32)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, crops.shape, align_corners=False)
        view = F.grid_sample(
            crops.float(),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

        if force_flip is None:
            flip = values[:, 4].view(batch, 1, 1, 1) < 0.5
        else:
            flip = torch.full(
                (batch, 1, 1, 1),
                force_flip,
                device=crops.device,
                dtype=torch.bool,
            )
        view = torch.where(flip, torch.flip(view, dims=(-1,)), view)
        brightness = 0.90 + values[:, 5].view(batch, 1, 1, 1) * 0.20
        contrast = 0.90 + values[:, 6].view(batch, 1, 1, 1) * 0.20
        spatial_mean = view.mean(dim=(-2, -1), keepdim=True)
        view = (view - spatial_mean) * contrast + spatial_mean
        view = view * brightness
        blurred = F.avg_pool2d(view, kernel_size=3, stride=1, padding=1)
        blur = values[:, 7].view(batch, 1, 1, 1) < 0.30
        view = torch.where(blur, blurred, view)
        noise_scale = values[:, 8].view(batch, 1, 1, 1) * 0.015
        noise = torch.randn(
            view.shape,
            device=view.device,
            dtype=view.dtype,
            generator=generator,
        )
        view = view + noise * noise_scale

        # Canonical ArcFace alignment places eyes near y=52 and mouth near y=92.
        # Occluding either region makes a particular blink or smile unreliable as
        # the sole identity shortcut while retaining most of the aligned face.
        occlusion = self._random_values(batch * 2, crops, generator=generator).reshape(
            batch, 2
        )
        eye_mask = torch.ones(1, 1, 112, 112, device=view.device, dtype=view.dtype)
        eye_mask[:, :, 42:61, 18:94] = 0.0
        mouth_mask = torch.ones_like(eye_mask)
        mouth_mask[:, :, 77:105, 27:86] = 0.0
        region_mask = torch.where(
            (occlusion[:, 1] < 0.5).view(batch, 1, 1, 1),
            eye_mask,
            mouth_mask,
        )
        mask = torch.where(
            (occlusion[:, 0] < 0.35).view(batch, 1, 1, 1),
            region_mask,
            torch.ones_like(region_mask),
        )
        fill = view.mean(dim=(-2, -1), keepdim=True)
        view = view * mask + fill * (1.0 - mask)
        return view.clamp(-1.0, 1.0)

    def _eot_crops(
        self,
        crops: torch.Tensor,
        views: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        variants = [crops.float()]
        variants.extend(
            self._augment_aligned_view(
                crops,
                generator=generator,
                force_flip=view_index % 2 == 1,
            )
            for view_index in range(1, views)
        )
        return torch.cat(variants, dim=0)

    def encode_reward_faces(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode clean or EOT-averaged crops depending on autograd context."""
        views = self.eot_views if torch.is_grad_enabled() and crops.requires_grad else 1
        embeddings = self.encode_faces(self._eot_crops(crops, views))
        if views == 1:
            return embeddings
        embeddings = embeddings.reshape(views, crops.shape[0], -1).mean(dim=0)
        return F.normalize(embeddings, dim=-1)

    def crop_tensor(self, image: torch.Tensor, face: dict[str, Any]) -> torch.Tensor:
        if self.crop_mode == "aligned" and face.get("kps") is not None:
            matrix = estimate_arcface_matrix(face["kps"])
        else:
            matrix = bbox_matrix(face["bbox"])
        return warp_affine_tensor(image, matrix, output_size=112)

    def _reference_crops(self, paths: list[Path]) -> list[torch.Tensor]:
        crops = []
        with torch.no_grad():
            for path in paths:
                image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    self.skipped_reference_images.append((str(path), -1))
                    continue
                faces = self.detect_faces(image_bgr)
                if not faces:
                    self.skipped_reference_images.append((str(path), len(faces)))
                    continue
                if self.reference_face_policy == "single" and len(faces) != 1:
                    self.skipped_reference_images.append((str(path), len(faces)))
                    continue
                if self.reference_face_policy == "largest":
                    faces = [
                        max(
                            faces,
                            key=lambda face: float(
                                max(0.0, face["bbox"][2] - face["bbox"][0])
                                * max(0.0, face["bbox"][3] - face["bbox"][1])
                            ),
                        )
                    ]
                elif self.reference_face_policy == "highest":
                    faces = faces[:1]
                image = bgr_to_rgb_tensor(image_bgr).to(self.device)
                crops.extend(self.crop_tensor(image, face) for face in faces)
                self.valid_reference_images.append(str(path))
        return crops

    def _encode_reference_crops(self, crops: list[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            generator = torch.Generator(device=self.device).manual_seed(17_042)
            embeddings = []
            for crop in crops:
                views = self._eot_crops(
                    crop,
                    self.reference_eot_views,
                    generator=generator,
                )
                embedding = self.encode_faces(views).mean(dim=0, keepdim=True)
                embeddings.append(F.normalize(embedding, dim=-1))
            return torch.cat(embeddings, dim=0).detach()

    def identity_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Raw centroid cosine used for deterministic evaluation metrics."""
        embeddings = F.normalize(embeddings.float(), dim=-1)
        prototype = self.reference_prototype.to(embeddings)
        return (embeddings @ prototype.t()).squeeze(-1)

    def identity_reward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Saturated centroid reward with a small anti-copy entropy penalty."""
        embeddings = F.normalize(embeddings.float(), dim=-1)
        scores = self.identity_scores(embeddings)
        temperature = self.saturation_temperature
        reward = -temperature * F.softplus(
            (self.target_similarity - scores) / temperature
        )
        references = self.reference_embeddings.to(embeddings)
        if references.shape[0] > 1 and self.reference_entropy_weight > 0.0:
            logits = (embeddings @ references.t()) / self.reference_entropy_temperature
            probabilities = logits.softmax(dim=-1)
            log_probabilities = logits.log_softmax(dim=-1)
            kl_uniform = (
                probabilities * (log_probabilities + math.log(references.shape[0]))
            ).sum(dim=-1)
            reward = reward - self.reference_entropy_weight * kl_uniform
        return reward

    def reference_assignment_stats(
        self, embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Measure whether an embedding collapses toward one reference view."""
        embeddings = F.normalize(embeddings.float(), dim=-1)
        references = self.reference_embeddings.to(embeddings)
        reference_scores = embeddings @ references.t()
        probabilities = (reference_scores / self.reference_entropy_temperature).softmax(
            dim=-1
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        nearest = reference_scores.max(dim=-1).values
        centroid = self.identity_scores(embeddings)
        return {
            "nearest_reference_similarity": nearest,
            "nearest_centroid_gap": nearest - centroid,
            "reference_assignment_entropy": entropy,
            "reference_assignment_max_probability": probabilities.max(dim=-1).values,
        }

    @staticmethod
    def _fallback_crops(image: torch.Tensor) -> torch.Tensor:
        """Differentiable center/upper-body hypotheses for missed detections."""
        _, height, width = image.shape
        boxes = (
            (0.20, 0.00, 0.80, 0.62),
            (0.12, 0.00, 0.88, 0.78),
            (0.25, 0.08, 0.75, 0.70),
        )
        crops = []
        for x0, y0, x1, y1 in boxes:
            left, right = int(x0 * width), max(int(x1 * width), int(x0 * width) + 1)
            top, bottom = int(y0 * height), max(int(y1 * height), int(y0 * height) + 1)
            crop = image[:, top:bottom, left:right].unsqueeze(0).float()
            crops.append(
                F.interpolate(
                    crop,
                    size=(112, 112),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            )
        return torch.cat(crops, dim=0)

    def _smooth_max(self, values: torch.Tensor) -> torch.Tensor:
        temperature = self.nearest_temperature
        return temperature * (
            torch.logsumexp(values / temperature, dim=0) - math.log(values.numel())
        )

    def _fallback_reward(self, image: torch.Tensor) -> torch.Tensor:
        crops = self._fallback_crops(image)
        embeddings = self.encode_reward_faces(crops)
        return self._smooth_max(self.identity_reward(embeddings)) - self.no_face_penalty

    def _primary_crop(self, image: torch.Tensor) -> torch.Tensor | None:
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError("pairwise face rewards expect one image at a time")
            image = image[0]
        faces = self.detect_faces(tensor_to_bgr(image))
        faces.sort(key=self._face_area, reverse=True)
        for face in faces:
            try:
                return self.crop_tensor(image, face)
            except Exception:  # noqa: BLE001 - ignore only invalid face geometry.
                continue
        return None

    @staticmethod
    def _face_area(face: dict[str, Any]) -> float:
        x1, y1, x2, y2 = np.asarray(face["bbox"], dtype=np.float32)
        return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)

    @staticmethod
    def expression_descriptor(crop: torch.Tensor) -> torch.Tensor:
        """Low-frequency aligned eye/mouth descriptor with direct image gradients."""
        gray = crop[:, 0:1] * 0.299 + crop[:, 1:2] * 0.587 + crop[:, 2:3] * 0.114
        gray = F.avg_pool2d(gray, kernel_size=5, stride=4, padding=2)
        eyes = gray[:, :, 9:17, 3:25]
        mouth = gray[:, :, 17:27, 6:23]
        descriptor = torch.cat((eyes.flatten(1), mouth.flatten(1)), dim=1)
        return F.layer_norm(descriptor, (descriptor.shape[-1],))

    def pairwise_reward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        prompt: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encourage expression diversity only when the prompt leaves it open."""
        del kwargs
        zero = (first.sum() + second.sum()) * 0.0
        if self.expression_diversity_weight <= 0.0 or (
            prompt and EXPRESSION_PROMPT_RE.search(prompt)
        ):
            return zero
        first_crop = self._primary_crop(first)
        second_crop = self._primary_crop(second)
        if first_crop is None or second_crop is None:
            return zero
        first_descriptor = self.expression_descriptor(first_crop)
        second_descriptor = self.expression_descriptor(second_crop)
        distance = F.smooth_l1_loss(
            first_descriptor,
            second_descriptor,
            reduction="mean",
            beta=0.1,
        )
        return -self.expression_diversity_weight * F.relu(
            self.expression_diversity_margin - distance
        )

    def forward(self, image: torch.Tensor, prompt: str | None = None, **kwargs):
        del prompt, kwargs
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(
                f"expected image [B,3,H,W] or [3,H,W], got {tuple(image.shape)}"
            )

        rewards = []
        for idx, img in enumerate(image):
            faces = self.detect_faces(tensor_to_bgr(img))
            if not faces:
                rewards.append(self._fallback_reward(img))
                continue
            # A character prompt has one primary subject. Bounding the reward to
            # the largest face plus one secondary keeps recognition activations
            # independent of spurious detector count and prevents a tiny
            # background face from winning the identity max.
            faces.sort(key=self._face_area, reverse=True)
            crops = []
            for face in faces[:2]:
                try:
                    crops.append(self.crop_tensor(img, face))
                except Exception:  # noqa: BLE001 - ignore only the invalid geometry.
                    continue
            if not crops:
                rewards.append(self._fallback_reward(img))
                continue
            primary_embedding = self.encode_reward_faces(crops[0])
            value = self.identity_reward(primary_embedding).squeeze(0)
            if len(crops) > 1 and self.duplicate_identity_weight > 0.0:
                secondary_embedding = self.encode_faces(crops[1])
                secondary_score = self.identity_scores(secondary_embedding).squeeze(0)
                duplicate = F.relu(secondary_score - self.duplicate_identity_threshold)
                value = value - self.duplicate_identity_weight * duplicate
            rewards.append(value.to(image.device))

        return torch.stack(rewards)
