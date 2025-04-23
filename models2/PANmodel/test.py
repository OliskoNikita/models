import numpy as np
import segmentation_models_pytorch as smp
import torch
import os
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Параметры конфигурации
dataset_root = "C:/dataset/dataset_after_slicing/"
test_dir = os.path.join(dataset_root, "test")
local_batch_size = 4
Epoch = 50
backbone = "mobilenet_v2"
model_serialization = "PAN_mobilenet_v2"

class MyDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.split = split
        self.dataset = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        df_row = self.dataset.iloc[index]

        # Load RGB image
        rgb_image = cv2.imread(df_row["rgb_image_path"], cv2.IMREAD_COLOR)
        if rgb_image is None:
            raise ValueError(f"Image not found: {df_row['rgb_image_path']}")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) / 255.0

        if self.split == "test":
            image_tensor = rgb_image.transpose((2, 0, 1)).astype("float32")
            return {"image": image_tensor}

        # Load mask
        mask_path = df_row["mask_path"]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found: {mask_path}")
        mask = mask / 255.0

        if self.split == "train" and self.transform:
            augmented = self.transform(image=rgb_image, mask=mask)
            rgb_image = augmented["image"]
            mask = augmented["mask"]

        return {
            "image": rgb_image.transpose((2, 0, 1)).astype("float32"),
            "mask": mask.astype("int64"),
        }

def get_filename(filepath):
    return os.path.split(filepath)[1]


def create_df(main_dir, split="train"):
    rgb_image_paths = sorted(glob.glob(main_dir + '/**/rgb/*.png', recursive=True))
    rgb_image_names = [get_filename(pth) for pth in rgb_image_paths]
    mask_paths = []

    for i in range(len(rgb_image_paths)):
        # Путь к изображению vh
        mask_name = rgb_image_names[i]
        if split != "test":
            if ("_1_" in mask_name):
                mask_path = os.path.join(
                    main_dir, "before_flood", "water_label", mask_name
                )
                mask_paths.append(mask_path)
            if ("_2_" in mask_name):
                mask_path = os.path.join(
                    main_dir, "flood", "water_label", mask_name
                )
                mask_paths.append(mask_path)
        else:
            mask_paths.append(np.NaN)

    paths = {
        "rgb_image_path": rgb_image_paths,
        "mask_path": mask_paths,
    }

    return pd.DataFrame(paths)


def calculate_flood_percentage(mask):
    flood_pixels = np.sum(mask == 1)
    total_pixels = mask.size
    return (flood_pixels / total_pixels) * 100

def visualize_predictions(images, predictions):
    """Визуализация входных изображений и их предсказаний."""
    batch_size = images.shape[0]
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Перестановка осей и перевод на CPU
        prediction = predictions[i].argmax(dim=0).cpu().numpy()  # Перевод предсказания на CPU
        flood_percentage = calculate_flood_percentage(prediction)
        
        # Отображение изображений
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Входное изображение")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(prediction, cmap="gray")
        plt.title(f"Предсказание модели\n{flood_percentage:.2f}% воды")
        plt.axis("off")
        
        plt.show()

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"C:/models2/PANmodel/PAN_mobilenet_v2_{Epoch}.pth")
    
    # Создание модели
    model = smp.PAN(
        encoder_name=backbone, encoder_weights='imagenet', in_channels=3, classes=2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Создание DataLoader-ов
    test_df = create_df(test_dir)
    if test_df.empty:
        raise ValueError("Test DataFrame is empty!")

    test_dataset = MyDataset(test_df, split="test")

    test_loader = DataLoader(
        test_dataset, batch_size=local_batch_size, shuffle=True, num_workers=0
    )
    if len(test_loader) == 0:
        raise ValueError("test DataLoader is empty!")

    # Оценка на валидационном наборе
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Test", unit="batch", leave=True)
        for batch in progress_bar:
            image = batch["image"].to(device)
            pred = model(image)
            visualize_predictions(images = image, predictions = pred)
if __name__ == "__main__":
    test()
