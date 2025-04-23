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
import albumentations as A

# Параметры конфигурации
dataset_root = "C:/dataset/dataset_after_slicing/"
train_dir = os.path.join(dataset_root, "train")
validation_dir = os.path.join(dataset_root, "validation")
local_batch_size = 4
learning_rate = 1e-3
num_epochs = 24
initEpoch = 26
backbone = "mobilenet_v2"
model_serialization = "unet_mobilenet_v2"

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.RandomRotate90(p = 0.5),
    A.VerticalFlip(p = 0.5)
])

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


def train(num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"C:/models2/Unet/unet_mobilenet_v2_{initEpoch}.pth")

    # Создание модели
    model = smp.Unet(
        encoder_name=backbone, encoder_weights='imagenet', in_channels=3, classes=2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Шедулер
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.31623)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Функция потерь
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")

    # Создание DataLoader-ов
    train_df = create_df(train_dir)
    validation_df = create_df(validation_dir)
    if train_df.empty:
        raise ValueError("Train DataFrame is empty!")
    if validation_df.empty:
        raise ValueError("Validation DataFrame is empty!")

    train_dataset = MyDataset(train_df, split="train", transform = transform)
    validation_dataset = MyDataset(validation_df, split="validation", transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=local_batch_size, shuffle=False, num_workers=0
    )
    if len(train_loader) == 0:
        raise ValueError("Train DataLoader is empty!")
    if len(validation_loader) == 0:
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
            
            # Loss
            loss = criterion_dice(pred, mask)


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
            f"Current LR: {optimizer.param_groups[0]['lr']:.6f}, "
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
            progress_bar = tqdm(validation_loader, desc="Valid", unit="batch", leave=True)
            for batch in progress_bar:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                pred = model(image)

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
