import os, time, math, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "preprocessing"
SAVE_DIR = PROJECT_ROOT / "models"
SAVE_NAME = "asl_best_gpu.pth"

FRAMES_PER_CLIP = 8
IMG_SIZE = 112

BATCH_SIZE = 8
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
FREEZE_EPOCHS = 2
LABEL_SMOOTHING = 0.1

WARMUP_EPOCHS = 2
EARLY_STOP_PATIENCE = 6
LOG_EVERY = 20


# ========= HELPERS =========
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def vram_str():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        return f" | VRAM alloc={a:.0f}MB reserved={r:.0f}MB"
    return ""


def find_and_import_class(project_root: Path, class_name: str):
    import importlib.util

    for py in project_root.rglob("*.py"):
        p = str(py).lower()
        if "/.venv/" in p or "__pycache__" in p:
            continue
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if f"class {class_name}" not in txt:
            continue

        spec = importlib.util.spec_from_file_location(f"_auto_{py.stem}", py)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except Exception:
                pass

    raise ImportError(f"Class {class_name} not found")


def make_lr_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        p = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def top5_accuracy(logits, y):
    k = min(5, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    return topk.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()


# ========= MAIN =========
def main():
    seed_everything(42)

    FramesFolderDataset = find_and_import_class(PROJECT_ROOT, "FramesFolderDataset")
    ResNetTemporalAttn = find_and_import_class(PROJECT_ROOT, "ResNetTemporalAttn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FramesFolderDataset(str(DATA_ROOT), "train", FRAMES_PER_CLIP, train_tf)

    try:
        val_ds = FramesFolderDataset(
            str(DATA_ROOT), "val", FRAMES_PER_CLIP, val_tf,
            label_to_idx=train_ds.label_to_idx
        )
    except TypeError:
        val_ds = FramesFolderDataset(str(DATA_ROOT), "val", FRAMES_PER_CLIP, val_tf)
        if hasattr(val_ds, "label_to_idx"):
            val_ds.label_to_idx = train_ds.label_to_idx

    num_classes = len(train_ds.label_to_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=(NUM_WORKERS > 0)
    )

    model = ResNetTemporalAttn(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_lr_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    from torch.amp import GradScaler, autocast
    scaler = GradScaler(enabled=(device.type == "cuda"))

    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / SAVE_NAME

    best_val = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, EPOCHS + 1):

        if hasattr(model, "backbone") and epoch == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True

        model.train()
        run_loss = run_correct = run_total = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * x.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += y.size(0)

        train_acc = run_correct / run_total

        model.eval()
        vloss = vcorrect = vtotal = 0
        vtop5_sum = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)

                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    out = model(x)
                    loss = criterion(out, y)

                bs = y.size(0)
                vloss += loss.item() * bs
                vcorrect += (out.argmax(1) == y).sum().item()
                vtop5_sum += top5_accuracy(out, y) * bs
                vtotal += bs

        val_acc = vcorrect / vtotal
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": train_ds.label_to_idx,
                "epoch": epoch,
                "val_acc": best_val,
            }, save_path)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    print(f"Best val_acc={best_val:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
