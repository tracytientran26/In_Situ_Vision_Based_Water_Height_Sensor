from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

IMAGE_DIR = Path(r"C:\Users\tranp\OneDrive\Desktop\ARTS lab\In_Situ_Vision_Based_Water_Height_Sensor\Images\training images")
USGS_CSV = Path(r"C:\Users\tranp\OneDrive\Desktop\ARTS lab\In_Situ_Vision_Based_Water_Height_Sensor\Code\USGS data\usgs_water_height_dataset.csv")

# Change only these names if your exported USGS CSV uses different headers.
USGS_TIME_COLUMN = "dateTime"
USGS_LEVEL_COLUMN = "gage_height_ft"

@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 720
    batch_size: int = 8
    max_epochs: int = 100
    learning_rate: float = 1e-3
    validation_fraction: float = 0.20
    early_stopping_patience: int = 28
    huber_delta_ft: float = 0.30
    random_seed: int = 42
    max_time_difference_minutes: float = 10.0


def parse_image_timestamp(image_path: Path) -> pd.Timestamp:
    """Read YYYYMMDD_HHMMSS from an image name such as 20260826_123000.jpg."""
    try:
        return pd.to_datetime(image_path.stem[:15], format="%Y%m%d_%H%M%S", utc=True)
    except ValueError as exc:
        raise ValueError(
            f"{image_path.name} must begin with YYYYMMDD_HHMMSS, e.g. 20260826_123000.jpg"
        ) from exc


def build_samples(image_dir: Path, usgs_csv: Path, cfg: TrainConfig) -> pd.DataFrame:
    """Pair each image with the nearest USGS water level within the tolerance."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {image_dir}")
    if not usgs_csv.is_file():
        raise FileNotFoundError(f"USGS_CSV does not exist: {usgs_csv}")

    usgs = pd.read_csv(usgs_csv)
    missing = {USGS_TIME_COLUMN, USGS_LEVEL_COLUMN} - set(usgs.columns)
    if missing:
        raise ValueError(f"USGS CSV is missing columns {missing}. Available columns: {list(usgs.columns)}")
    usgs = usgs[[USGS_TIME_COLUMN, USGS_LEVEL_COLUMN]].copy()

    usgs.columns = ["timestamp", "water_level_ft"]
    usgs["timestamp"] = pd.to_datetime(usgs["timestamp"], utc=True, errors="coerce")
    usgs["water_level_ft"] = pd.to_numeric(usgs["water_level_ft"], errors="coerce")
    usgs = usgs.dropna().sort_values("timestamp")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions)
    if not images:
        raise ValueError(f"No image files were found in {image_dir}")
    photo_rows = []
    for image in images:
        photo_rows.append({"image_path": str(image), "timestamp": parse_image_timestamp(image)})
    photos = pd.DataFrame(photo_rows).sort_values("timestamp")

    samples = pd.merge_asof(
        photos, usgs, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(minutes=cfg.max_time_difference_minutes),
    ).dropna(subset=["water_level_ft"])
    if len(samples) < 5:
        raise ValueError(f"Only {len(samples)} image/USGS pairs were created. Check timestamps and matching tolerance.")
    print(f"Paired {len(samples)} images with USGS targets; target range: "
          f"{samples.water_level_ft.min():.3f} to {samples.water_level_ft.max():.3f} ft")
    return samples.reset_index(drop=True)


class WaterLevelDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, image_size: int, target_mean: float, target_std: float):
        self.samples, self.image_size = samples.reset_index(drop=True), image_size
        self.target_mean, self.target_std = target_mean, target_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        rgb = Image.open(row.image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb_array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # CNN input shape is [3, H, W]: red, green, and blue values.
        x = torch.from_numpy(rgb_array)
        y = (float(row.water_level_ft) - self.target_mean) / self.target_std
        return x, torch.tensor(y, dtype=torch.float32), str(row.image_path)


class WaterLevelCNN(nn.Module):
    """Two convolution + two max-pooling layers; final layer has no activation (linear)."""
    def __init__(self, image_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flattened = 32 * (image_size // 4) * (image_size // 4)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(flattened, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(1)


def evaluate(model, loader, loss_fn, device) -> float:
    model.eval(); total_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            total_loss += loss_fn(model(x.to(device)), y.to(device)).item() * len(y)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--usgs-csv", type=Path, default=USGS_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("water_level_cnn_output"))
    args = parser.parse_args()
    cfg = TrainConfig()
    random.seed(cfg.random_seed); np.random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    samples = build_samples(args.image_dir, args.usgs_csv, cfg)
    train_count = int((1 - cfg.validation_fraction) * len(samples))
    indices = torch.randperm(len(samples), generator=torch.Generator().manual_seed(cfg.random_seed)).tolist()
    train_rows, val_rows = samples.iloc[indices[:train_count]], samples.iloc[indices[train_count:]]
    target_mean, target_std = train_rows.water_level_ft.mean(), train_rows.water_level_ft.std()
    target_std = max(float(target_std), 1e-6)
    dataset = WaterLevelDataset(samples, cfg.image_size, target_mean, target_std)
    train_loader = DataLoader(Subset(dataset, indices[:train_count]), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[train_count:]), batch_size=cfg.batch_size, shuffle=False)

    model = WaterLevelCNN(cfg.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta_ft / target_std)
    best_loss, stale_epochs, best_state = float("inf"), 0, None
    train_loss_history, val_loss_history = [], []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y, _ in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step()
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d}/{cfg.max_epochs}: validation Huber loss = {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs."); break

    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training and validation Huber loss.
    plt.figure(figsize=(8, 5))
    epochs_ran = range(1, len(train_loss_history) + 1)
    plt.plot(epochs_ran, train_loss_history, label="Training Huber loss")
    plt.plot(epochs_ran, val_loss_history, label="Validation Huber loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("Rocky Creek Branch CNN: Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_validation_loss.png", dpi=200)
    plt.close()
    torch.save({"model_state_dict": model.state_dict(), "target_mean_ft": target_mean,
                "target_std_ft": target_std, "config": asdict(cfg)}, args.output_dir / "best_water_level_cnn.pt")
    with open(args.output_dir / "training_configuration.json", "w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    model.eval(); prediction_rows = []
    with torch.no_grad():
        for x, y, paths in val_loader:
            predicted_ft = model(x.to(device)).cpu().numpy() * target_std + target_mean
            actual_ft = y.numpy() * target_std + target_mean
            prediction_rows.extend({"image_path": path, "actual_usgs_level_ft": actual, "predicted_level_ft": predicted,
                                    "absolute_error_ft": abs(actual - predicted)} for path, actual, predicted in zip(paths, actual_ft, predicted_ft))
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    # Plot the validation time series: USGS vs CNN.
    plot_predictions = predictions.copy()
    plot_predictions["timestamp"] = pd.to_datetime(
        plot_predictions["image_path"].map(lambda p: Path(p).stem[:15]),
        format="%Y%m%d_%H%M%S", errors="coerce", utc=True
    )
    plot_predictions = plot_predictions.sort_values("timestamp")

    plt.figure(figsize=(10, 5))
    plt.plot(plot_predictions["timestamp"], plot_predictions["actual_usgs_level_ft"], label="USGS actual water level")
    plt.plot(plot_predictions["timestamp"], plot_predictions["predicted_level_ft"], label="CNN predicted water level")
    plt.xlabel("Time")
    plt.ylabel("Water level (ft)")
    plt.title("Rocky Creek Branch: CNN Validation vs USGS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir / "validation_usgs_vs_cnn.png", dpi=200)
    plt.close()

    # Plot predicted vs actual values.
    plt.figure(figsize=(6, 6))
    x_actual = plot_predictions["actual_usgs_level_ft"]
    y_pred = plot_predictions["predicted_level_ft"]
    plt.scatter(x_actual, y_pred, alpha=0.75)
    low = min(x_actual.min(), y_pred.min())
    high = max(x_actual.max(), y_pred.max())
    plt.plot([low, high], [low, high], linestyle="--", label="Ideal 1:1")
    plt.xlabel("USGS actual water level (ft)")
    plt.ylabel("CNN predicted water level (ft)")
    plt.title("Validation: CNN Predicted vs USGS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "validation_predicted_vs_usgs_scatter.png", dpi=200)
    plt.close()

    print(f"Saved model and validation predictions in: {args.output_dir.resolve()}")
    print(f"Validation MAE: {predictions.absolute_error_ft.mean():.3f} ft")
    print("Saved plots: training_validation_loss.png, validation_usgs_vs_cnn.png, validation_predicted_vs_usgs_scatter.png")


if __name__ == "__main__":
    main()
