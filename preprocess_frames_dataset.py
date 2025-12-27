from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
from torchvision import transforms
from tqdm import tqdm


# ========= CONFIG =========
INPUT_ROOT = Path("data/preprocessing")
OUTPUT_ROOT = Path("data/processed")
IMG_SIZE = 112
MAX_FRAMES = 16
MOTION_THRESHOLD = 15


# ========= TRANSFORMS =========
resize_tf = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


# ========= UTILS =========
def image_diff(img1, img2):
    diff = ImageChops.difference(img1, img2)
    return np.mean(np.array(diff))


def select_motion_frames(frames, max_frames):
    scores = []
    for i in range(1, len(frames)):
        scores.append((image_diff(frames[i - 1], frames[i]), i))

    scores.sort(reverse=True)
    selected = sorted([i for _, i in scores[:max_frames]])

    if len(selected) < max_frames:
        selected = list(range(min(len(frames), max_frames)))

    return selected


# ========= PROCESS =========
def process_clip(clip_dir, out_dir):
    frames = sorted(clip_dir.glob("*.jpg"))
    if len(frames) < 2:
        return

    images = [Image.open(f).convert("RGB") for f in frames]
    idxs = select_motion_frames(images, MAX_FRAMES)

    out_dir.mkdir(parents=True, exist_ok=True)

    for j, i in enumerate(idxs):
        img = resize_tf(images[i])
        img = transforms.ToPILImage()(img)
        img.save(out_dir / f"frame_{j:03d}.jpg")


def main():
    for split in ["train", "val", "test"]:
        in_split = INPUT_ROOT / split / "frames"
        out_split = OUTPUT_ROOT / split / "frames"

        for label_dir in tqdm(list(in_split.iterdir()), desc=split):
            if not label_dir.is_dir():
                continue

            for clip_dir in label_dir.iterdir():
                if not clip_dir.is_dir():
                    continue

                process_clip(
                    clip_dir,
                    out_split / label_dir.name / clip_dir.name
                )


if __name__ == "__main__":
    main()
