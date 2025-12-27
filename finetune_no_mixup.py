import os, time, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from train_gpu_ready import ClipFramesDataset, DATA_ROOT, IMG_SIZE

CKPT_IN  = Path("models/asl_best_gpu.pth")
CKPT_OUT = Path("models/asl_finetuned_nomix_best.pth")

FRAMES_PER_CLIP = 8          # لازم نفس اللي اتدرّبت عليه
BATCH_SIZE = 16              # لو OOM خليها 8
EPOCHS = 12
LR = 7e-5                    # أقل من قبل
WEIGHT_DECAY = 2e-4          # أعلى شوية ضد overfit
NUM_WORKERS = 0              # Windows safe

LABEL_SMOOTHING = 0.05       # قللها شوية
EARLY_STOP_PATIENCE = 4
LOG_EVERY = 20


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class SimpleTemporalResNetDrop(nn.Module):
    """ResNet18 backbone + temporal mean + dropout classifier"""
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.35):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 512
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        f = self.backbone(x)               # [B*T, 512, 1, 1]
        f = f.view(b, t, self.feat_dim)    # [B, T, 512]
        f = f.mean(dim=1)                  # [B, 512]
        f = self.drop(f)
        return self.classifier(f)


def eval_val(model, loader, device, criterion):
    model.eval()
    vloss, vcorrect, vtotal = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            out = model(x)
            loss = criterion(out, y)
            bs = y.size(0)
            vloss += loss.item() * bs
            vcorrect += (out.argmax(1) == y).sum().item()
            vtotal += bs
    return vloss / max(1, vtotal), vcorrect / max(1, vtotal)


def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if device.type == "cuda":
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    # ✅ Aug قوي (بدون MixUp)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.08),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.18)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=True)
    label_to_idx = ckpt["label_to_idx"]
    num_classes = len(label_to_idx)

    model = SimpleTemporalResNetDrop(num_classes=num_classes, pretrained=False, dropout=0.35).to(device)

    # حمل أوزان الموديل الأساسي (نفس الأسماء تقريبًا)
    # هنحمّل backbone + classifier لو مطابقين
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[LOAD] missing:", len(missing), "unexpected:", len(unexpected))

    # افتح الـ backbone للتعلم (fine-tune)
    for p in model.backbone.parameters():
        p.requires_grad = True

    train_ds = ClipFramesDataset(str(DATA_ROOT), "train", FRAMES_PER_CLIP, train_tf)
    val_ds   = ClipFramesDataset(str(DATA_ROOT), "val", FRAMES_PER_CLIP, val_tf, label_to_idx=label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    from torch.amp import GradScaler, autocast
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = -1.0
    best_epoch = 0
    patience = 0
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                    desc=f"FT NOMIX {epoch:02d}/{EPOCHS}", ncols=120)

        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)

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

            if step % LOG_EVERY == 0 or step == len(train_loader):
                pbar.set_postfix(loss=f"{(run_loss/max(1,run_total)):.4f}",
                                 acc=f"{(run_correct/max(1,run_total)):.3f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        val_loss, val_acc = eval_val(model, val_loader, device, criterion)
        scheduler.step()

        print(f"[VAL] epoch {epoch:02d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "arch": "ResNet18 temporal + dropout",
                "frames_per_clip": FRAMES_PER_CLIP,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_acc": best_val,
            }, CKPT_OUT)
            print(f"[SAVE] ✅ best -> {CKPT_OUT} (val_acc={best_val:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[EARLY STOP] best_epoch={best_epoch} best_val={best_val:.4f}")
                break

    print("[DONE] best_val:", best_val, "best_epoch:", best_epoch)


if __name__ == "__main__":
    main()
