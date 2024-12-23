import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

#################################################################################
## Define function loading GTZAN df from file
#################################################################################


def get_gtzan_dataframe(with_val, gtzan_df_path, gtzan_image_dir):
    gtzan_df = pd.read_csv(gtzan_df_path)

    # turn relative paths into absolute paths
    gtzan_df["path"] = gtzan_df["path"].apply(lambda x: gtzan_image_dir / x)

    # SAMPLES: 800/98/100
    if with_val:
        gtzan_train_df = gtzan_df[gtzan_df["set"] == "train"]
        gtzan_val_df = gtzan_df[gtzan_df["set"] == "validation"]
        gtzan_test_df = gtzan_df[gtzan_df["set"] == "test"]
        return gtzan_train_df, gtzan_val_df, gtzan_test_df
    else:  # SAMPLES: 800/198
        gtzan_train_df = gtzan_df[gtzan_df["set"] == "train"]
        gtzan_test_df = gtzan_df[
            (gtzan_df["set"] == "test") | (gtzan_df["set"] == "validation")
        ]
        return gtzan_train_df, gtzan_test_df


#################################################################################
## Define function loading FMA df from file
#################################################################################


def get_fma_dataframe(get_fma_small, with_val, fma_df_path, fma_image_dir):
    fma_df = pd.read_csv(fma_df_path)

    # turn relative paths into absolute paths
    fma_df["path"] = fma_df["path"].apply(lambda x: fma_image_dir / x)

    # otherwise the dataset is FMA-medium
    if get_fma_small:
        fma_df = fma_df[fma_df["size"] == "small"]

    # PERCENTAGES: 80/10/10
    if with_val:
        fma_train_df = fma_df[fma_df["set"] == "training"]
        fma_val_df = fma_df[fma_df["set"] == "validation"]
        fma_test_df = fma_df[fma_df["set"] == "test"]
        return fma_train_df, fma_val_df, fma_test_df
    else:  # PERCENTAGES: 80/20
        fma_train_df = fma_df[fma_df["set"] == "training"]
        fma_test_df = fma_df[
            (fma_df["set"] == "test") | (fma_df["set"] == "validation")
        ]
        return fma_train_df, fma_test_df


#################################################################################
## # Define input data loading functions
#################################################################################


# 0, 1, 2, 3... labels used with nn.CrossEntropyLoss
def numerically_encode_class_label(class_name, available_classes):
    return available_classes.index(class_name)


# currently unused
def one_hot_encode_class_label(class_name, available_classes):
    label = available_classes.index(class_name)
    one_hot = np.zeros(len(available_classes))
    one_hot[label] = 1
    return one_hot


# decode png
def decode_img_data(img_path, image_size, channels=3):
    img = Image.open(img_path).convert(
        "RGB" if channels == 3 else "L"
    )  # convert to RGB if 3 channels, otherwise grayscale
    img = img.resize((image_size, image_size))
    img = np.array(img)
    img = (
        torch.tensor(img).permute(2, 0, 1)
        if channels == 3
        else torch.tensor(img).unsqueeze(0)
    )  # channels first
    return img.float() / 255.0  # normalize to [0, 1]


# return pair: decoded png and class name turned into numerical label
def process_sample(sample, available_classes, image_size, channels=3):
    img_path = sample[0]
    class_label = sample[1]

    img_data = decode_img_data(img_path, image_size, channels)
    numerical_label = numerically_encode_class_label(class_label, available_classes)
    return img_data, numerical_label


class CustomDataset(Dataset):
    def __init__(self, df, available_classes, image_size, channels=3, device="cpu"):
        self.df = df
        self.available_classes = available_classes
        self.image_size = image_size
        self.channels = channels
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        class_label = self.df.iloc[idx]["genre"]
        img_data, numerical_label = process_sample(
            (img_path, class_label),
            self.available_classes,
            self.image_size,
            self.channels,
        )
        return img_data.to(self.device), torch.tensor(
            numerical_label, device=self.device
        )  #! DATA on device


def prepare_dataloader_based_on_df(
    df, available_classes, batch_size=8, channels=3, device="cpu", image_size=128
):
    dataset = CustomDataset(df, available_classes, image_size, channels, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    return dataloader
