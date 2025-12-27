from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# Video Reader (NO cv2)
# =========================
try:
    import imageio.v3 as iio  # new
    _HAS_IMAGEIO = True
except Exception:
    try:
        import imageio as iio  # old fallback
        _HAS_IMAGEIO = True
    except Exception:
        _HAS_IMAGEIO = False


def _read_video_rgb(video_path: str) -> List[np.ndarray]:
    if not _HAS_IMAGEIO:
        raise RuntimeError("Install: pip install imageio imageio-ffmpeg")

    frames: List[np.ndarray] = []

    # Try v3 fast path
    try:
        arr = iio.imread(video_path)
        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            for fr in arr:
                frames.append(fr[..., :3].astype(np.uint8))
        else:
            raise RuntimeError("v3 returned unexpected shape")
    except Exception:
        reader = iio.get_reader(video_path)
        for fr in reader:
            fr = np.asarray(fr)
            frames.append(fr[..., :3].astype(np.uint8))
        reader.close()

    if not frames:
        raise RuntimeError("No frames extracted from video.")
    return frames


# =========================
# Motion helpers (key improvement)
# =========================
def _motion_scores(frames: List[np.ndarray], step: int = 2) -> np.ndarray:
    """
    Computes simple motion score per frame transition.
    step=2 means compare i and i-2 to reduce noise.
    """
    n = len(frames)
    if n < 3:
        return np.zeros(max(1, n - 1), dtype=np.float32)

    scores = np.zeros(n, dtype=np.float32)
    prev = frames[0].astype(np.int16)

    for i in range(step, n, step):
        cur = frames[i].astype(np.int16)
        scores[i] = float(np.abs(cur - prev).mean())
        prev = cur

    # smooth a bit
    if n >= 5:
        k = 5
        pad = k // 2
        x = np.pad(scores, (pad, pad), mode="edge")
        scores = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="valid")

    return scores


def _pick_motion_window(frames: List[np.ndarray], T: int, window_factor: float = 2.5) -> List[np.ndarray]:
    """
    Pick a window around the most moving segment.
    window_len ~= T*window_factor (>=T)
    """
    n = len(frames)
    if n <= T:
        return frames

    scores = _motion_scores(frames)
    window_len = min(n, max(T, int(T * window_factor)))

    # sliding sum
    best_s = 0
    best_val = -1.0
    # scores defined per-frame, so use sum over window
    for s in range(0, n - window_len + 1):
        val = float(scores[s : s + window_len].sum())
        if val > best_val:
            best_val = val
            best_s = s

    return frames[best_s : best_s + window_len]


def _uniform_idxs(n: int, T: int) -> List[int]:
    if n <= 1:
        return [0] * T
    return np.linspace(0, n - 1, T).astype(int).tolist()


# =========================
# Model A: SimpleTemporalResNet
# =========================
class SimpleTemporalResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 512
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        f = self.backbone(x).view(b, t, self.feat_dim)
        f = f.mean(dim=1)
        f = self.dropout(f)
        return self.classifier(f)


# =========================
# Model B: TemporalTransformer
# =========================
class TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        max_len: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        rn = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(rn.children())[:-1])  # [B,512,1,1]
        self.feat_dim = 512
        assert self.feat_dim == d_model, "d_model must be 512 for ResNet18 backbone."

        self.temporal = nn.Module()
        self.temporal.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal.pos = nn.Module()
        self.temporal.pos.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.temporal.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        f = self.backbone(x).flatten(1).view(b, t, 512)

        cls = self.temporal.cls.expand(b, -1, -1)
        seq = torch.cat([cls, f], dim=1)

        pe = self.temporal.pos.pe[:, : seq.size(1), :]
        seq = seq + pe

        seq = self.temporal.enc(seq)
        out = seq[:, 0]
        return self.head(out)


def _infer_transformer_config(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    pe = sd["temporal.pos.pe"]  # [1, max_len, d_model]
    max_len = int(pe.shape[1])
    d_model = int(pe.shape[2])

    layer_ids = set()
    for k in sd.keys():
        if k.startswith("temporal.enc.layers."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                layer_ids.add(int(parts[3]))
    num_layers = (max(layer_ids) + 1) if layer_ids else 2

    ff_key = "temporal.enc.layers.0.linear1.weight"
    dim_feedforward = int(sd[ff_key].shape[0]) if ff_key in sd else 2048

    return {"max_len": max_len, "d_model": d_model, "num_layers": num_layers, "dim_feedforward": dim_feedforward}


def _remap_backbone_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # if checkpoint uses backbone.backbone.* -> backbone.*
    keys = list(sd.keys())
    if any(k.startswith("backbone.backbone.") for k in keys):
        fixed = {}
        for k, v in sd.items():
            if k.startswith("backbone.backbone."):
                fixed[k.replace("backbone.backbone.", "backbone.", 1)] = v
            else:
                fixed[k] = v
        return fixed
    return sd


# =========================
# Recognizer (IMPROVED)
# =========================
class ASLSignRecognizer:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        topk: int = 5,
        num_clips: int = 12,              # default improved
        use_flip_tta: bool = True,
        temperature: float = 1.0,
        deterministic: bool = True,       # default improved
        motion_window_factor: float = 2.5 # default improved
    ):
        self.model_path = str(model_path).strip()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(self.model_path, map_location="cpu")
        sd = ckpt.get("model_state_dict")
        if sd is None:
            raise ValueError("Checkpoint missing 'model_state_dict'.")

        self.label_to_idx = ckpt.get("label_to_idx")
        if self.label_to_idx is None:
            raise ValueError("Checkpoint missing 'label_to_idx'.")

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        sd = _remap_backbone_backbone(sd)

        self.topk = int(topk)
        self.num_clips = int(num_clips)
        self.use_flip_tta = bool(use_flip_tta)
        self.temperature = float(max(1e-6, temperature))
        self.deterministic = bool(deterministic)
        self.motion_window_factor = float(motion_window_factor)

        self.img_size = int(ckpt.get("img_size", 112))

        # Decide model type by keys
        if "temporal.pos.pe" in sd:
            cfg = _infer_transformer_config(sd)
            d_model = cfg["d_model"]
            if d_model % 8 == 0:
                nhead = 8
            elif d_model % 4 == 0:
                nhead = 4
            else:
                nhead = 2

            self.model = TemporalTransformer(
                num_classes=self.num_classes,
                max_len=cfg["max_len"],
                d_model=d_model,
                nhead=nhead,
                num_layers=cfg["num_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=0.1,
                norm_first=True,
                pretrained_backbone=False,
            )
            self.frames_per_clip = int(cfg["max_len"] - 1)
            model_type = "transformer"
        else:
            self.model = SimpleTemporalResNet(num_classes=self.num_classes, pretrained=False)
            self.frames_per_clip = int(ckpt.get("frames_per_clip", 11))
            model_type = "resnet_avg"

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # preprocessing (val-like)
        self.tf = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(
            f"[INFO] Loaded {Path(self.model_path).name} | type={model_type} classes={self.num_classes} "
            f"frames_per_clip={self.frames_per_clip} img_size={self.img_size} device={self.device} "
            f"num_clips={self.num_clips} flip_tta={self.use_flip_tta} temp={self.temperature:.2f} "
            f"deterministic={self.deterministic} motion_w={self.motion_window_factor}"
        )

    def _frames_to_tensor(self, frames: List[np.ndarray], flip: bool) -> torch.Tensor:
        clip = []
        for fr in frames:
            if flip:
                fr = np.ascontiguousarray(fr[:, ::-1, :])
            img = Image.fromarray(fr)
            clip.append(self.tf(img))
        return torch.stack(clip, dim=0).unsqueeze(0)  # [1,T,C,H,W]

    @torch.inference_mode()
    def predict_video(self, video_path: str) -> Tuple[str, float, List[Dict[str, float]]]:
        frames = _read_video_rgb(video_path)
        T = self.frames_per_clip
        n = len(frames)

        # 1) focus on most moving region
        focus = _pick_motion_window(frames, T, window_factor=self.motion_window_factor)
        nf = len(focus)

        # 2) choose clip starts deterministically inside focus window
        if nf <= T:
            starts = [0]
        else:
            if self.num_clips <= 1:
                starts = [0 if self.deterministic else random.randint(0, nf - T)]
            else:
                if self.deterministic:
                    starts = np.linspace(0, nf - T, self.num_clips).astype(int).tolist()
                else:
                    starts = [random.randint(0, nf - T) for _ in range(self.num_clips)]

        # 3) accumulate logits AND do voting
        logits_sum = None
        denom = 0

        vote_counts: Dict[int, int] = {}
        vote_scores: Dict[int, float] = {}

        def _acc_vote(logits: torch.Tensor):
            # logits: [1,C]
            pred = int(torch.argmax(logits, dim=1).item())
            score = float(torch.max(torch.softmax(logits, dim=1)).item())
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
            vote_scores[pred] = vote_scores.get(pred, 0.0) + score

        for s in starts:
            if nf > T:
                sub = focus[s : s + T]
            else:
                sub = focus

            idxs = _uniform_idxs(len(sub), T)
            clip_frames = [sub[i] for i in idxs]

            x = self._frames_to_tensor(clip_frames, flip=False).to(self.device)
            logits = self.model(x)
            _acc_vote(logits)

            logits_sum = logits if logits_sum is None else (logits_sum + logits)
            denom += 1

            if self.use_flip_tta:
                xf = self._frames_to_tensor(clip_frames, flip=True).to(self.device)
                logits_f = self.model(xf)
                _acc_vote(logits_f)

                logits_sum = logits_sum + logits_f
                denom += 1

        logits_avg = logits_sum / max(1, denom)
        logits_avg = logits_avg / self.temperature

        probs = torch.softmax(logits_avg, dim=1)[0]  # [C]

        # 4) final decision: voting first, fallback to avg logits
        if vote_counts:
            # max count then max accumulated score
            best = sorted(vote_counts.keys(), key=lambda k: (vote_counts[k], vote_scores.get(k, 0.0)), reverse=True)[0]
            label = self.idx_to_label[best]
            confidence = float(probs[best].item())
        else:
            best = int(torch.argmax(probs).item())
            label = self.idx_to_label[best]
            confidence = float(probs[best].item())

        # 5) top-k output
        k = min(self.topk, probs.numel())
        vals, idxs = torch.topk(probs, k=k)

        topk = []
        for v, i in zip(vals, idxs):
            ii = int(i.item())
            topk.append({"label": self.idx_to_label[ii], "confidence": float(v.item())})

        return label, confidence, topk
