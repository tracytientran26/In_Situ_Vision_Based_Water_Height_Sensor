from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple
import json
from pathlib import Path
from PIL import Image
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

# import dataset address
IMG_PATH = Path(r"C:\Users\tranp\OneDrive\Desktop\ARTS lab\In_Situ_Vision_Based_Water_Height_Sensor\Images\training images")
USGS_PATH = Path(r"C:\Users\tranp\OneDrive\Desktop\ARTS lab\In_Situ_Vision_Based_Water_Height_Sensor\Code\USGS data\usgs_water_height_dataset.csv")
DATE_TIME_USGS = "dateTime"
DATA_USGS = "gage_height_ft"

# verify the address:
def verification(image_path, usgs_path):
    if not IMG_PATH.is_dir():
        raise FileNotFoundError(f"Folder not found: {IMG_PATH}")
    if not USGS_PATH.is_dir():
        raise FileNotFoundError(f"File not found: {USGS_PATH}")

# CNN architecture
@dataclass
class train_parameter:
    image_size: int = 720
    epoch: int = 100
    batch_size: int = 8
    early_stopping: int = 14
    learning_rate: float = 1e-3
    validation_rate: float = 0.2
    random_seed: int = 42
    time_displacement: int = 10
    huber_loss_ft: int = 0.15

def address_to_time(image_path: Path) -> pd.Timestamps:
    try:
        return pd.to_datetime(image_path.stem[:15], format = "%Y%m%D_%H%M%S", utc = True)
    except ValueError as exc:
        raise ValueError(f"Change {image_path}.name to YYYYMMDD_HHMMSS.") from exc

def pair_images_USGS(image_path: Path, usgs_path: Path, cfg: train_parameter) -> pd.DataFrame:
    usgs = pd.read(usgs_path)
    missing = {DATE_TIME_USGS, DATA_USGS} - set(usgs.columns)
    if missing:
        raise ValueError(f"Missing data in USGS.")
    usgs = usgs[[DATE_TIME_USGS, DATA_USGS]].copy()

    usgs.columns = ["timestamp", "water_level_ft"]
    usgs["timestamp"] = pd.to_datetime(usgs["timestamp"], utc = True, errors = "coerce")
    usgs["water_level_ft"] = pd.to_numeric(usgs["water_level_ft"], utc = True, errors = "coerce")
    usgs = usgs.dropna().sort_values("timestamp")

    image_extension = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_path.dir.iterdir() if p.suffix.lower() in image_extension)
    if not images: 
        raise ValueError(f"No image found in {image_path}")
    photo_rows = []
    for image in images:
        photo_rows.append({"image_path": str(image), "timestamp": address_to_time(image)})
    photos = pd.DataFrame(photo_rows).sort_values("timestamp")

    samples = pd.merge_asof(
        photos, usgs, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(minutes= train_parameter.max_time_difference_minutes),
    ).dropna(subset=["water_level_ft"])
    if len(samples) < 5:
        raise ValueError(f"Only {len(samples)} image/USGS pairs were created. Check timestamps and matching tolerance.")
    print(f"Paired {len(samples)} images with USGS targets; target range: "
          f"{samples.water_level_ft.min():.3f} to {samples.water_level_ft.max():.3f} ft")
    return samples.reset_index(drop=True)

class water_level_dataset(Dataset):
    def __init__(self, samples: pd.DataFrame, image_size: int, target_mean: float, target_std: float):
        self.sample, self.image_size = samples.reset_index(drop = True), image_size
        self.target_mean, self.target_std = target_mean, target_std

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        rgb = Image.open(row.image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb_array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # CNN input shape is [3, H, W]: red, green, and blue values.
        x = torch.from_numpy(rgb_array)
        y = (float(row.water_level_ft) - self.target_mean) / self.target_std
        return x, torch.tensor(y, dtype=torch.float32), str(row.image_path)

class CNN_module(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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
    # initiate parser
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--image-path", type = Path, default = IMG_PATH)      
    parser.add_argument("--usgs-path", type = Path, default = USGS_PATH)
    parser.add_argument("--output-dir", type = Path, default = Path("water_level_cnn_output"))
    args = parser.parse_args()
    # initiate random seed
    cfg = train_parameter()
    random.seed(cfg.random_seed); np.random.seed(train_parameter.random_seed); torch.manual_seed(train_parameter.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # build table dataframe
    samples = pair_images_USGS(args.image_path, args.usgs_path, train_parameter)
    train_count = int((1 -cfg.validation_fraction) * len(samples))
    indices = torch.randperm(len(samples), generator = torch.Generator().manual_seed(cfg.random_seed)).tolist()
    train_rows, val_rows = samples.iloc[indices[:train_count]], samples.iloc[indices[train_count:]]
    target_mean, target_std = train_rows.water_level_ft.mean(), train_rows.water_level_ft.std()
    target_std = max(float(target_std), 1e-6)
    dataset = water_level_dataset(samples, cfg.image_size, target_mean, target_std)
    train_loader = DataLoader(Subset(dataset, indices[:train_count]), batch_size = cfg.batch_size, shuffle = True)
    val_loader = DataLoader(Subset(dataset, indices[train_count:]), batch_size = cfg.batch_size, shuffle = False)

    model = water_level_dataset(train_parameter.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameter(), lr = train_parameter.learning_rate)
    loss_fn = nn.HuberLoss(delta = train_parameter.huber_delta ft / target_std)
    best_loss, stale_epochs, best_state = float("inf"), 0, None
    for epoch in range(1, train_parameter.max_epochs + 1):
        model.train()
        for x, y, _ in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step()
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d}/{train_parameter.max_epochs}: validation Huber loss = {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: 
            stale_epochs += 1
            if stale_epochs >= train_parameter.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs."); break

    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents = True, exist_ok = True)
    torch.save({"model_state_dict": model.state_dict(), "target_mean_ft": target_mean,
                "target_std_ft": target_std, 
                "config": asdict(train_parameter)}, args.output_dir / "best_water_level_cnn.pt")
    with open(args.output_dir / "training_configuration.json", "w", encoding = "utf-8") as file:
        json.dump(asdict(train_parameter), file, indent = 2)

    model.eval(); prediction_rows = []
    with torch.no_grad():
        for x, y, paths in val_loader:
            predicted_ft = model(x.to(device)).cpu().numpy() * target_std + target_mean
            actual_ft = y.numpy() * target_std + target_mean
            prediction_rows.extend({"image path": path, "actual_usgs_ft": actual, "precdicted_levle_ft": predicted,
                                "absolute_error_ft": abs(actual - predicted)} for path, actual, predicted in zip(paths, actual_ft, predicted_ft))
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    print(f"Saved model and validation predictions in: {args.output_dir.resolve()}")
    print(f"Validation MAE: {predictions.absolute_error_ft.mean():.3f} ft")

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
    
    
if __name__ == "__main__":
    main()

    

        



