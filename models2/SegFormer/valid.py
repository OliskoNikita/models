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
dataset_root = "C:/dataset/dataset_after_slicing/"
validation_dir = os.path.join(dataset_root, "validation")
local_batch_size = 1
Epoch = 25
model_serialization = "segformer"

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

def calculate_iou(pred, target):
    """
    Расчёт Intersection over Union (IoU).
    """
    num_classes=2
    batch_iou_0 = []
    batch_iou_1 = []
    batch_iou_2 = []
    for i in range(pred.shape[0]):  # Проходим по каждому изображению в батче
        pred_i = pred[i].argmax(dim=0).view(-1)
        target_i = target[i].view(-1)
        iou_per_class_0 = []
        iou_per_class_1 = []
        for cls in range(num_classes):
            intersection = ((pred_i == cls) & (target_i == cls)).sum().item()
            union = ((pred_i == cls) | (target_i == cls)).sum().item()
            if union > 0:
                iou_per_class_0.append(intersection / union)
                iou_per_class_1.append(intersection / union)
            else:
                iou_per_class_0.append(0.0)
        batch_iou_0.append(np.mean(iou_per_class_0))
        batch_iou_1.append(iou_per_class_0[0])
        if len(iou_per_class_1) > 1:
            batch_iou_2.append(iou_per_class_0[1])
    if len(batch_iou_2) > 0:
        return [np.mean(batch_iou_0), np.mean(batch_iou_1), batch_iou_2]
    else: 
        return [np.mean(batch_iou_0), np.mean(batch_iou_1), 2.0]

def calculate_pixel_accuracy(pred, target):
    """
    Расчёт Pixel Accuracy.
    """
    batch_accuracy = []
    for i in range(pred.shape[0]):
        pred_i = pred[i].argmax(dim=0)  # Преобразуем в прогнозируемый класс
        correct_i = (pred_i == target[i]).sum().item()
        total_i = target[i].numel()
        batch_accuracy.append(correct_i / total_i)
    return np.mean(batch_accuracy)

def calculate_dice(pred, target):
    batch_dice = []
    for i in range(pred.shape[0]):
        pred_i = pred[i].argmax(dim=0)  # Выбираем предсказанный класс
        intersection_i = (pred_i * target[i]).sum().item()
        batch_dice.append((2. * intersection_i) / (pred_i.sum().item() + target[i].sum().item() + 1e-6))
    return np.mean(batch_dice)

def calculate_precision_recall(pred, target):
    batch_precision = []
    batch_recall = []
    for i in range(pred.shape[0]):
        pred_i = pred[i].argmax(dim=0)
        true_positive_i = ((pred_i == 1) & (target[i] == 1)).sum().item()
        false_positive_i = ((pred_i == 1) & (target[i] == 0)).sum().item()
        false_negative_i = ((pred_i == 0) & (target[i] == 1)).sum().item()

        batch_precision.append(true_positive_i / (true_positive_i + false_positive_i + 1e-6))
        batch_recall.append(true_positive_i / (true_positive_i + false_negative_i + 1e-6))
    return [np.mean(batch_precision), np.mean(batch_recall)]

def calculate_f1_score(pred, target):
    batch_f1_score = []
    for i in range(pred.shape[0]):
        pred_i = pred[i].argmax(dim=0)
        true_positive_i = ((pred_i == 1) & (target[i] == 1)).sum().item()
        false_positive_i = ((pred_i == 1) & (target[i] == 0)).sum().item()
        false_negative_i = ((pred_i == 0) & (target[i] == 1)).sum().item()

        precision_i = true_positive_i / (true_positive_i + false_positive_i + 1e-6)
        recall_i = true_positive_i / (true_positive_i + false_negative_i + 1e-6)
        batch_f1_score.append(2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-6))
    return np.mean(batch_f1_score)


def calculate_balanced_accuracy(pred, target):
    batch_sensitivity = []
    batch_specificity = []
    for i in range(pred.shape[0]):
        pred_i = pred[i].argmax(dim=0)
        true_positive_i = ((pred_i == 1) & (target[i] == 1)).sum().item()
        true_negative_i = ((pred_i == 0) & (target[i] == 0)).sum().item()
        total_positive_i = (target[i] == 1).sum().item()
        total_negative_i = (target[i] == 0).sum().item()

        batch_sensitivity.append(true_positive_i / (total_positive_i + 1e-6))
        batch_specificity.append(true_negative_i / (total_negative_i + 1e-6))
    return (np.mean(batch_sensitivity) + np.mean(batch_specificity)) / 2


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

def visualize_predictions(images, masks, predictions):
    """
    Визуализация входных изображений, их масок и предсказаний модели.
    """
    batch_size = images.shape[0]
    for i in range(batch_size):
        # Преобразование изображений
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Перестановка осей и перевод на CPU
        mask = masks[i].cpu().numpy()  # Перевод маски на CPU
        prediction = predictions[i].argmax(dim=0).cpu().numpy()  # Перевод предсказания на CPU
        flood_percentage = calculate_flood_percentage(prediction)

        # Отображение изображений
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Входное изображение")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Маска")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap="gray")
        plt.title(f"Предсказание модели\n{flood_percentage:.2f}% воды")
        plt.axis("off")

        plt.show()

def valid():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"C:/models2/SegFormer/segformer_{Epoch}.pth")
    
    # Создание модели
    model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Создание DataLoader-ов
    validation_df = create_df(validation_dir)
    if validation_df.empty:
        raise ValueError("Validation DataFrame is empty!")

    validation_dataset = MyDataset(validation_df, split="validation")

    validation_loader = DataLoader(
        validation_dataset, batch_size=local_batch_size, shuffle=False, num_workers=0
    )
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader is empty!")

    # Оценка на валидационном наборе
    model.eval()
    with torch.no_grad():
        MIoU_metric = []
        IoU_metric1 = []
        IoU_metric2 = []
        Dice_metric = []
        Accuracy_metric = []
        Precision_metric = []
        Recall_metric = []
        F1_metric = []
        BalancedAcc_metric = []
        progress_bar = tqdm(validation_loader, desc="Valid", unit="batch", leave=True)
        for batch in progress_bar:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            pred = pred.logits
            pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=False)

            IoU = calculate_iou(pred, mask)
            MIoU_metric.append(IoU[0])
            IoU_metric1.append(IoU[1])
            if IoU[2] != 2.0:
                IoU_metric2 += IoU[2]
            Dice_metric.append(calculate_dice(pred, mask))
            Accuracy_metric.append(calculate_pixel_accuracy(pred, mask))
            precision_recall = calculate_precision_recall(pred, mask)
            Precision_metric.append(precision_recall[0])
            Recall_metric.append(precision_recall[1])
            F1_metric.append(calculate_f1_score(pred, mask))
            BalancedAcc_metric.append(calculate_balanced_accuracy(pred, mask))
            visualize_predictions(images = image, masks = mask, predictions = pred)
        print(
            f"MIoU: {sum(MIoU_metric)/(len(MIoU_metric) +0.00001):.4f} "
            f"IoU_background: {sum(IoU_metric1)/(len(IoU_metric1) +0.00001):.4f} "
            f"IoU_flooding: {sum(IoU_metric2)/(len(IoU_metric2) +0.00001):.4f} "
            f"Dice: {sum(Dice_metric)/(len(Dice_metric) +0.00001):.4f} "
            f"Accuracy: {sum(Accuracy_metric)/(len(Accuracy_metric) +0.00001):.4f} "
            f"Precision: {sum(Precision_metric)/(len(Precision_metric) +0.00001):.4f} "
            f"Recall: {sum(Recall_metric)/(len(Recall_metric) +0.00001):.4f} "
            f"F1-Score: {sum(F1_metric)/(len(F1_metric) +0.00001):.4f} "
            f"Balanced Accuracy: {sum(BalancedAcc_metric)/(len(BalancedAcc_metric) +0.00001):.4f} "
        )

        
if __name__ == "__main__":
    valid()
