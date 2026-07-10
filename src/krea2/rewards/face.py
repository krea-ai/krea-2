"""Differentiable antelopev2 face-similarity reward for DRaFT-K training.

Detection runs through ONNXRuntime/SCRFD on detached images. Recognition is a
small PyTorch executor for the antelopev2 recognition ONNX graph, so gradients
flow from the cosine-distance reward through the aligned crop to generated
images.
"""

from __future__ import annotations

import math
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
        nearest_reference_weight: float = 0.25,
        nearest_temperature: float = 0.07,
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
            return self.encode_faces(torch.cat(crops, dim=0)).detach()

    def identity_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Robust cosine identity score using a prototype and nearby views."""
        embeddings = F.normalize(embeddings.float(), dim=-1)
        prototype = self.reference_prototype.to(embeddings)
        references = self.reference_embeddings.to(embeddings)
        prototype_score = embeddings @ prototype.t()
        reference_scores = embeddings @ references.t()
        temperature = self.nearest_temperature
        smooth_nearest = temperature * (
            torch.logsumexp(reference_scores / temperature, dim=-1)
            - math.log(reference_scores.shape[-1])
        )
        weight = self.nearest_reference_weight
        return (1.0 - weight) * prototype_score.squeeze(-1) + weight * smooth_nearest

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
        embeddings = self.encode_faces(self._fallback_crops(image))
        return (
            self._smooth_max(self.identity_scores(embeddings))
            - 1.0
            - self.no_face_penalty
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
            crops = []
            for face in faces:
                try:
                    crops.append(self.crop_tensor(img, face))
                except Exception:  # noqa: BLE001 - ignore only the invalid geometry.
                    continue
            if not crops:
                rewards.append(self._fallback_reward(img))
                continue
            embeddings = self.encode_faces(torch.cat(crops, dim=0))
            scores = self.identity_scores(embeddings)
            ordered = torch.sort(scores, descending=True).values
            value = ordered[0] - 1.0
            if ordered.numel() > 1 and self.duplicate_identity_weight > 0.0:
                duplicate = F.relu(
                    ordered[1:] - self.duplicate_identity_threshold
                ).mean()
                value = value - self.duplicate_identity_weight * duplicate
            rewards.append(value.to(image.device))

        return torch.stack(rewards)
