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
import albumentations as A

# Параметры конфигурации
dataset_root = "C:/dataset_etci21/home/juputer/Flood_Comp/"
train_dir = os.path.join(dataset_root, "train")
valid_dir = os.path.join(dataset_root, "test")
local_batch_size = 4
learning_rate = 1e-3
num_epochs = 14
initEpoch = 1
model_serialization = "segformer"

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.RandomRotate90(p = 0.5),
    A.VerticalFlip(p = 0.5)
])

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


def train(num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"C:/models1/SegFormer/segformer_{initEpoch}.pth")
    
    # Создание модели
    model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Шедулер
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.31623)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Функция потерь
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")

    # Создание DataLoader-ов
    train_df = create_df(train_dir)
    valid_df = create_df(valid_dir)
    if train_df.empty:
        raise ValueError("Train DataFrame is empty!")
    if valid_df.empty:
        raise ValueError("Validation DataFrame is empty!")

    # Удаление недопустимых изображений
    train_df = train_df.drop(train_df.index[filter_df(train_df)])
    
    valid_df = valid_df.drop(valid_df.index[filter_df(valid_df)])

    train_dataset = ETCIDataset(train_df, split="train", transform = transform)
    validation_dataset = ETCIDataset(valid_df, split="validation")

    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        validation_dataset, batch_size=local_batch_size, shuffle=False, num_workers=0
    )
    if len(train_loader) == 0:
        raise ValueError("Train DataLoader is empty!")
    if len(valid_loader) == 0:
        raise ValueError("Validation DataLoader is empty!")

    ## Начало обучения ##
    for epoch in range(num_epochs):
        # Тренировочный этап
        model.train()
        train_losses = []
        train_MIoU_metric = []
        train_IoU_metric1 = []
        train_IoU_metric2 = []
        train_Dice_metric = []
        train_Accuracy_metric = []
        train_Precision_metric = []
        train_Recall_metric = []
        train_F1_metric = []
        train_BalancedAcc_metric = []
        progress_bar = tqdm(train_loader, desc="Train", unit="batch", leave=True)
        for batch in progress_bar:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            pred = pred.logits
            pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=False)

            # Loss
            loss = criterion_dice(pred, mask)
            
            # Метрики
            train_losses.append(loss.item())
            IoU = calculate_iou(pred, mask)
            train_MIoU_metric.append(IoU[0])
            train_IoU_metric1.append(IoU[1])
            if IoU[2] != 2.0:
                train_IoU_metric2 += IoU[2]
            train_Dice_metric.append(calculate_dice(pred, mask))
            train_Accuracy_metric.append(calculate_pixel_accuracy(pred, mask))
            precision_recall = calculate_precision_recall(pred, mask)
            train_Precision_metric.append(precision_recall[0])
            train_Recall_metric.append(precision_recall[1])
            train_F1_metric.append(calculate_f1_score(pred, mask))
            train_BalancedAcc_metric.append(calculate_balanced_accuracy(pred, mask))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch+initEpoch+1}, "
            f"Current LR: {optimizer.param_groups[0]['lr']:.6f}"
            f"Train Loss: {sum(train_losses)/(len(train_losses) +0.00001):.4f}, "
            f"Train MIoU: {sum(train_MIoU_metric)/(len(train_MIoU_metric) +0.00001):.4f} "
            f"Train IoU_background: {sum(train_IoU_metric1)/(len(train_IoU_metric1) +0.00001):.4f} "
            f"Train IoU_flooding: {sum(train_IoU_metric2)/(len(train_IoU_metric2) +0.00001):.4f} "
            f"Train Dice: {sum(train_Dice_metric)/(len(train_Dice_metric) +0.00001):.4f} "
            f"Train Accuracy: {sum(train_Accuracy_metric)/(len(train_Accuracy_metric) +0.00001):.4f} "
            f"Train Precision: {sum(train_Precision_metric)/(len(train_Precision_metric) +0.00001):.4f} "
            f"Train Recall: {sum(train_Recall_metric)/(len(train_Recall_metric) +0.00001):.4f} "
            f"Train F1-Score: {sum(train_F1_metric)/(len(train_F1_metric) +0.00001):.4f} "
            f"Train Balanced Accuracy: {sum(train_BalancedAcc_metric)/(len(train_BalancedAcc_metric) +0.00001):.4f} "
        )

        scheduler.step()

        # Оценка на валидационном наборе
        model.eval()
        with torch.no_grad():
            valid_losses = []
            valid_MIoU_metric = []
            valid_IoU_metric1 = []
            valid_IoU_metric2 = []
            valid_Dice_metric = []
            valid_Accuracy_metric = []
            valid_Precision_metric = []
            valid_Recall_metric = []
            valid_F1_metric = []
            valid_BalancedAcc_metric = []
            progress_bar = tqdm(valid_loader, desc="Valid", unit="batch", leave=True)
            for batch in progress_bar:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                pred = model(image)
                pred = pred.logits
                pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=False)

                # Loss
                loss = criterion_dice(pred, mask)

                # Метрики
                valid_losses.append(loss.item())
                IoU = calculate_iou(pred, mask)
                valid_MIoU_metric.append(IoU[0])
                valid_IoU_metric1.append(IoU[1])
                if IoU[2] != 2.0:
                    valid_IoU_metric2 += IoU[2]
                valid_Dice_metric.append(calculate_dice(pred, mask))
                valid_Accuracy_metric.append(calculate_pixel_accuracy(pred, mask))
                precision_recall = calculate_precision_recall(pred, mask)
                valid_Precision_metric.append(precision_recall[0])
                valid_Recall_metric.append(precision_recall[1])
                valid_F1_metric.append(calculate_f1_score(pred, mask))
                valid_BalancedAcc_metric.append(calculate_balanced_accuracy(pred, mask))
        print(
            f"Valid Loss: {sum(valid_losses)/(len(valid_losses) +0.00001):.4f}, "
            f"Valid MIoU: {sum(valid_MIoU_metric)/(len(valid_MIoU_metric) +0.00001):.4f} "
            f"Valid IoU_background: {sum(valid_IoU_metric1)/(len(valid_IoU_metric1) +0.00001):.4f} "
            f"Valid IoU_flooding: {sum(valid_IoU_metric2)/(len(valid_IoU_metric2) +0.00001):.4f} "
            f"Valid Dice: {sum(valid_Dice_metric)/(len(valid_Dice_metric) +0.00001):.4f} "
            f"Valid Accuracy: {sum(valid_Accuracy_metric)/(len(valid_Accuracy_metric) +0.00001):.4f} "
            f"Valid Precision: {sum(valid_Precision_metric)/(len(valid_Precision_metric) +0.00001):.4f} "
            f"Valid Recall: {sum(valid_Recall_metric)/(len(valid_Recall_metric) +0.00001):.4f} "
            f"Valid F1-Score: {sum(valid_F1_metric)/(len(valid_F1_metric) +0.00001):.4f} "
            f"Valid Balanced Accuracy: {sum(valid_BalancedAcc_metric)/(len(valid_BalancedAcc_metric) +0.00001):.4f} "
        )
        
        # Сохранение модели
        torch.save({
            'epoch': epoch+1+initEpoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f"{model_serialization}_{epoch+1+initEpoch}.pth")


if __name__ == "__main__":
    train(num_epochs)
