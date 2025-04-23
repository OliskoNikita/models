import numpy as np
import segmentation_models_pytorch as smp
import torch
import os
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Параметры конфигурации
dataset_root = "C:/dataset_etci21/home/juputer/Flood_Comp/"
test_dir = os.path.join(dataset_root, "test_internal")
local_batch_size = 4
Epoch = 25
model_serialization = "segformer"

def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image

class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.split = split
        self.dataset = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):

        df_row = self.dataset.iloc[index]

        # load vv and vh images
        vv_image = cv2.imread(df_row["vv_image_path"], 0)
        vh_image = cv2.imread(df_row["vh_image_path"], 0)
        if vv_image is None or vh_image is None:
            raise ValueError(f"Image not found: {df_row['vv_image_path']} or {df_row['vh_image_path']}")
        vv_image = vv_image / 255.0
        vh_image = vh_image / 255.0

        # convert vv and vh images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)

        if self.split == "test":
            image_tensor = rgb_image.transpose((2, 0, 1)).astype("float32")
            return {"image": image_tensor}
        # load ground truth flood mask
        flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0

        # apply augmentations if specified
        if self.split == "train" and self.transform:
            augmented = self.transform(image=rgb_image, mask=flood_mask)
            rgb_image = augmented["image"]
            flood_mask = augmented["mask"]

        # For validation: no augmentation
        return {
            "image": rgb_image.transpose((2, 0, 1)).astype("float32"),
            "mask": flood_mask.astype("int64"),
        }


def get_filename(filepath):
    return os.path.split(filepath)[1]


def create_df(main_dir, split="train"):
    vv_image_paths = sorted(glob.glob(main_dir + '/**/vv/*.png', recursive=True))
    vv_image_names = [get_filename(pth) for pth in vv_image_paths]
    region_name_dates = ["_".join(n.split("_")[:2]) for n in vv_image_names]
    vh_image_paths, flood_label_paths, water_body_label_paths, region_names = (
        [],
        [],
        [],
        [],
    )

    for i in range(len(vv_image_paths)):
        # Путь к изображению vh
        vh_image_name = vv_image_names[i].replace("vv", "vh")
        vh_image_path = os.path.join(
            main_dir, region_name_dates[i], "tiles", "vh", vh_image_name
        )
        vh_image_paths.append(vh_image_path)

        # Путь к маске наводнения
        if split != "test":
            flood_image_name = vv_image_names[i].replace("_vv", "")
            flood_label_path = os.path.join(
                main_dir, region_name_dates[i], "tiles", "flood_label", flood_image_name
            )
            flood_label_paths.append(flood_label_path)
        else:
            flood_label_paths.append(np.NaN)

        # Путь к маске водоёма
        water_body_label_name = vv_image_names[i].replace("_vv", "")
        water_body_label_path = os.path.join(
            main_dir,
            region_name_dates[i],
            "tiles",
            "water_body_label",
            water_body_label_name,
        )
        water_body_label_paths.append(water_body_label_path)

        # Регион
        region_name = region_name_dates[i].split("_")[0]
        region_names.append(region_name)

    paths = {
        "vv_image_path": vv_image_paths,
        "vh_image_path": vh_image_paths,
        "flood_label_path": flood_label_paths,
        "water_body_label_path": water_body_label_paths,
        "region": region_names,
    }

    return pd.DataFrame(paths)


def filter_df(df):
    remove_indices = []
    for i, image_path in enumerate(df["vv_image_path"].tolist()):
        image = cv2.imread(image_path, 0)
        image_values = list(np.unique(image))

        binary_value_check = (
        np.array_equal(image_values, [0, 255])
        or np.array_equal(image_values, [0])
        or np.array_equal(image_values, [255])
        )   

        if binary_value_check:
            remove_indices.append(i)
    return remove_indices

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
        plt.title(f"Предсказание модели\n{flood_percentage:.2f}% наводнения")
        plt.axis("off")
        
        plt.show()

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"C:/models1/SegFormer/segformer_{Epoch}.pth")
    
    # Создание модели
    model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Создание DataLoader-ов
    test_df = create_df(test_dir)
    if test_df.empty:
        raise ValueError("Test DataFrame is empty!")

    # Удаление недопустимых изображений
    test_df = test_df.drop(test_df.index[filter_df(test_df)])

    test_dataset = ETCIDataset(test_df, split="test")

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
            pred = pred.logits
            pred = F.interpolate(pred, size=[256, 256], mode='bilinear', align_corners=False)
            visualize_predictions(images = image, predictions = pred)


if __name__ == "__main__":
    test()
