#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle bus trucks dataloader.

Created on Wed May 18 21:47:09 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Union

# Third party imports
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure
from numpy import array, newaxis
from numpy.typing import NDArray
from pandas import read_csv
from pytorch_lightning import LightningDataModule
from selectivesearch import selective_search
from sklearn.model_selection import train_test_split
from torch import Tensor, cat, nan_to_num, tensor, vstack, zeros, int64
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.ops import box_iou
from torchvision.transforms import Compose, Normalize, Resize


LABEL2TARGET = {"Etc": 0, "Bus": 1, "Truck": 2}
TARGET2LABEL = {value: key for key, value in LABEL2TARGET.items()}


@dataclass
class OpenImageInfo:
    """Dataclass for open image for R-CNN."""

    image_id: str = ""
    boxes_ratio: list[Tensor] = field(default_factory=list)
    tg_idx: list[int] = field(default_factory=list)


@dataclass
class OpenImageCadidateInfo:
    """Dataclass for cropped image for R-CNN."""

    image_id: str
    boxes_ratio: Tensor
    deltas: Tensor
    tg_idx: Tensor


def bustruck_df2img_info(path_csv: str) -> list[OpenImageInfo]:
    """Convert bus truck csv to list of OpenImageInfo."""
    infos_df = read_csv(path_csv)
    infos_df.sort_values("ImageID")

    ret: dict[str, OpenImageInfo] = defaultdict(OpenImageInfo)

    for info_df in infos_df.itertuples():
        info_ds = ret[info_df.ImageID]
        info_ds.image_id = info_df.ImageID
        info_ds.boxes_ratio.append(
            tensor([info_df.XMin, info_df.YMin, info_df.XMax, info_df.YMax])
        )
        info_ds.tg_idx.append(LABEL2TARGET[info_df.LabelName])

    return list(ret.values())


def extract_candidates_with_ss(
    img_infos: list[OpenImageInfo],
    folder_jpg: str,
    scale: float = 200.0,
    sigma: float = 0.8,
    min_size: int = 100,
    iou_th_for_etc: float = 0.3,
) -> list[OpenImageCadidateInfo]:
    """Extract candidates from image using selective search.

    Args:
        img_infos: list[OpenImageInfo].
        folder_jpg: folder of jpg.
        scale: Free parameter Higher means larger clusters in felzenszwalb segmentation.
            Defaults to 200.0.
        sigma: Width of Gaussian kernel for felzenszwalb segmentation. Defaults to 0.8.
        min_size: Minimum component size for felzenszwalb segmentation. Defaults to 100.
        iou_th_for_etc: If the IoU is less than iou_th_for_etc, it is classified as
            Other. Defaults to 0.3

    Returns:
        list[OpenImageCadidateInfo]: DESCRIPTION.

    """
    ret: list[OpenImageCadidateInfo] = []

    for img_info in img_infos:
        path_img = f"{folder_jpg}/{img_info.image_id}.jpg"

        img = read_image(path_img).permute(1, 2, 0)
        n_y, n_x, _ = img.shape
        ratio2size = tensor([[n_x, n_y, n_x, n_y]])

        tg_coordinates = (vstack(img_info.boxes_ratio) * ratio2size).int()

        # Calc coordinates of cadidates by selective search
        _, ss_infos = selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
        ss_coordinates_l = []
        for ss_info in ss_infos:
            left, top, width, height = ss_info["rect"]
            if width < 2 or height < 2:
                continue
            ss_coordinates_l.append(tensor([left, top, left + width, top + height]))
        ss_coordinates = vstack(ss_coordinates_l)

        # Calc IoU
        ious = nan_to_num(box_iou(ss_coordinates, tg_coordinates))

        # By comparing selective search results with ground truth,
        # the identifier of the target with the largest IoU is assigned
        # the target identifier of candidates.
        ious_best, idxs_best = ious.max(dim=1)
        tg_idx_from_ss = tensor(img_info.tg_idx)[idxs_best.tolist()]
        # If ious < 0.3, the target identifier of candidates is assinged "Etc"
        tg_idx_from_ss[ious_best < iou_th_for_etc] = LABEL2TARGET["Etc"]

        deltas = []
        for ss_coordi, idx_best in zip(ss_coordinates, idxs_best):
            deltas.append(tg_coordinates[idx_best] - ss_coordi)
        deltas = vstack(deltas)

        candidate_info = OpenImageCadidateInfo(
            img_info.image_id,
            ss_coordinates / ratio2size,
            deltas / ratio2size,
            tg_idx_from_ss,
        )
        ret.append(candidate_info)

    return ret


class BusTruckDataset(Dataset):
    """Dataset for Kaggle Bus Truck data.

    You can download dataset at
        https://www.kaggle.com/datasets/sixhky/open-images-bus-trucks/download

    Args:
        infos: List of OpenImageInfo
        folder_jpg: Image folder.

    """

    def __init__(
        self,
        infos: list[OpenImageInfo],
        folder_jpg: str,
    ):
        """."""
        self.infos = infos
        self.folder = folder_jpg

    def __len__(self) -> int:
        """."""
        return len(self.infos)

    def __getitem__(self, idx: int) -> dict:
        """."""
        info = self.infos[idx]
        path_img = f"{self.folder}/{info.image_id}.jpg"

        img = read_image(path_img).float() / 255.0
        _, n_y, n_x = img.shape

        tg_loc_ratio = vstack(info.boxes_ratio)

        return {"img": img, "tg_loc_ratio": tg_loc_ratio, "tg_idx": info.tg_idx}


# %% RCNN DataSet & Module
class BusTruckCroppedDataset(Dataset):
    """Dataset for Kaggle Bus Truck cropped image for rcnn.

    You can download dataset at
        https://www.kaggle.com/datasets/sixhky/open-images-bus-trucks/download

    Args:
        infos: List of OpenImageInfo
        coordinates: List of box coordinates for crop
        folder_jpg: Image folder.

    """

    def __init__(
        self,
        infos: list[OpenImageCadidateInfo],
        folder_jpg: str,
        output_size: tuple[int, int] = (224, 224),
    ):
        """."""
        self.infos = infos
        self.folder = folder_jpg
        self.output_size = output_size
        self.transforms = Compose(
            (
                Resize(output_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        )

    def __len__(self) -> int:
        """."""
        return len(self.infos)

    def __getitem__(self, idx: int) -> dict:
        """."""
        info = self.infos[idx]
        n_crop = len(info.tg_idx)

        path_img = f"{self.folder}/{info.image_id}.jpg"
        img = read_image(path_img).float() / 255.0
        n_ch, n_y, n_x = img.shape
        ratio2size = tensor([n_x, n_y, n_x, n_y])

        img_cropped = zeros((n_crop, n_ch, *self.output_size))

        for idx, box_ratio in enumerate(info.boxes_ratio):
            box = (box_ratio * ratio2size).int()
            _img = img[:, box[1] : box[3], box[0] : box[2]]
            _img = self.transforms(_img)
            img_cropped[idx] = _img

        return {
            "imgs": img_cropped,
            "deltas": info.deltas,
            "tg_idx": info.tg_idx,
            "box_ratio": info.boxes_ratio,
            "image_id": info.image_id,
        }

    def collate_fn(self, batch) -> dict:
        """."""
        keys = batch[0].keys()
        ret = {}
        for key in keys:
            _list = []
            for ele in batch:
                _list.append(ele[key])
            if key != "image_id":
                ret[key] = cat(_list)
            else:
                ret[key] = _list
        return ret


class BusTruckCroppedDataModuleWithSS(LightningDataModule):
    """Data Module for Bus Truck cropped images for rcnn by selective search."""

    def __init__(
        self,
        img_infos: list[OpenImageInfo],
        batch_size: int,
        n_worker: int = 0,
        output_size: tuple[int, int] = (224, 224),
        folder_img: str = "./data/kaggle_bus_trucks/images",
        ss_kwargs: Optional[dict] = None,
    ):
        """."""
        super().__init__()
        self.img_infos = img_infos
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.folder_img = folder_img
        self.output_size = output_size
        if ss_kwargs is None:
            self.ss_kwargs = {
                "scale": 200.0,
                "sigma": 0.8,
                "min_size": 100,
                "iou_th_for_etc": 0.3,
            }
        else:
            self.ss_kwargs = ss_kwargs

        self.train_infos: list[OpenImageCadidateInfo] = []
        self.val_infos: list[OpenImageCadidateInfo] = []
        self.test_infos: list[OpenImageCadidateInfo] = []

    def prepare_data(self):
        """Prepare Data."""
        pass

    def setup(self, stage=None):
        """Read images."""
        print("Apply Selective search to images ----", end=" ")
        candidates_infos = extract_candidates_with_ss(
            self.img_infos, self.folder_img, **self.ss_kwargs
        )

        self.train_infos, leftover_infos = train_test_split(
            candidates_infos, test_size=0.2
        )
        self.val_infos, self.test_infos = train_test_split(
            leftover_infos, test_size=0.5
        )
        print("End")

    def train_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckCroppedDataset(self.train_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckCroppedDataset(self.val_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckCroppedDataset(self.test_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )


# %% Fast RCNN DataSet & Module
class BusTruckDatasetWithRoi(Dataset):
    """Dataset for Kaggle Bus Truck image for fast rcnn.

    You can download dataset at
        https://www.kaggle.com/datasets/sixhky/open-images-bus-trucks/download

    Args:
        infos: List of OpenImageInfo
        coordinates: List of box coordinates for crop
        folder_jpg: Image folder.

    """

    def __init__(
        self,
        infos: list[OpenImageCadidateInfo],
        folder_jpg: str,
        output_size: tuple[int, int] = (224, 224),
    ):
        """."""
        self.infos = infos
        self.folder = folder_jpg
        self.output_size = output_size
        self.transforms = Compose(
            (
                Resize(output_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        )
        self.ratio2size = tensor(
            [
                output_size[0],
                output_size[1],
                output_size[0],
                output_size[1],
            ]
        )

    def __len__(self) -> int:
        """."""
        return len(self.infos)

    def __getitem__(self, idx: int) -> dict:
        """."""
        info = self.infos[idx]

        path_img = f"{self.folder}/{info.image_id}.jpg"
        img = read_image(path_img).float() / 255.0
        n_ch, n_y, n_x = img.shape
        img_resize = self.transforms(img)
        # coordinates in resize image
        box_coordinates_resized = info.boxes_ratio * self.ratio2size

        return {
            "img": img_resize[newaxis, :],
            "deltas": info.deltas,
            "tg_idx": info.tg_idx,
            "box_ratio": info.boxes_ratio,
            "box_coordinates_resized": box_coordinates_resized,
            "image_id": info.image_id,
        }

    def collate_fn(self, batch) -> dict:
        """."""
        keys = batch[0].keys()
        ret = {}
        for key in keys:
            _list = []
            for ele in batch:
                _list.append(ele[key])
            if key == "img":
                ret[key] = cat(_list)
            else:
                ret[key] = _list
        return ret


class BusTruckRoiDataModuleWithSS(LightningDataModule):
    """Data Module for Bus Truck cropped images for rcnn by selective search."""

    def __init__(
        self,
        img_infos: list[OpenImageInfo],
        batch_size: int,
        n_worker: int = 0,
        output_size: tuple[int, int] = (224, 224),
        folder_img: str = "./data/kaggle_bus_trucks/images",
        ss_kwargs: Optional[dict] = None,
    ):
        """."""
        super().__init__()
        self.img_infos = img_infos
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.folder_img = folder_img
        self.output_size = output_size
        if ss_kwargs is None:
            self.ss_kwargs = {
                "scale": 200.0,
                "sigma": 0.8,
                "min_size": 100,
                "iou_th_for_etc": 0.3,
            }
        else:
            self.ss_kwargs = ss_kwargs

        self.train_infos: list[OpenImageCadidateInfo] = []
        self.val_infos: list[OpenImageCadidateInfo] = []
        self.test_infos: list[OpenImageCadidateInfo] = []

    def prepare_data(self):
        """Prepare Data."""
        pass

    def setup(self, stage=None):
        """Read images."""
        print("Apply Selective search to images ----", end=" ")
        candidates_infos = extract_candidates_with_ss(
            self.img_infos, self.folder_img, **self.ss_kwargs
        )

        self.train_infos, leftover_infos = train_test_split(
            candidates_infos, test_size=0.2
        )
        self.val_infos, self.test_infos = train_test_split(
            leftover_infos, test_size=0.5
        )
        print("End")

    def train_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetWithRoi(self.train_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetWithRoi(self.val_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetWithRoi(self.test_infos, self.folder_img, self.output_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )


# %% Faster RCNN DataSet & Module
class BusTruckDatasetFasterRCNN(Dataset):
    """Dataset for Kaggle Bus Truck image for faster rcnn.

    You can download dataset at
        https://www.kaggle.com/datasets/sixhky/open-images-bus-trucks/download

    Args:
        infos: List of OpenImageInfo
        coordinates: List of box coordinates for crop
        folder_jpg: Image folder.

    """

    def __init__(
        self,
        infos: list[OpenImageInfo],
        folder_jpg: str,
        output_size: tuple[int, int] = (224, 224),
    ):
        """."""
        self.infos = infos
        self.folder = folder_jpg
        self.output_size = output_size
        self.transforms = Compose((Resize(output_size),))
        self.ratio2size = tensor(
            [
                output_size[0],
                output_size[1],
                output_size[0],
                output_size[1],
            ]
        )

    def __len__(self) -> int:
        """."""
        return len(self.infos)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        """."""
        info = self.infos[idx]

        path_img = f"{self.folder}/{info.image_id}.jpg"
        img = read_image(path_img).float() / 255.0
        n_ch, n_y, n_x = img.shape
        img_resize = self.transforms(img)
        # coordinates in resize image
        boxes_ratio = vstack(info.boxes_ratio)
        box_coordinates_resized = boxes_ratio * self.ratio2size
        ret_info = {
            "labels": tensor(info.tg_idx, dtype=int64),
            "boxes_ratio": boxes_ratio,
            "boxes": box_coordinates_resized,
            "image_id": info.image_id,
        }

        return img_resize, ret_info

    def collate_fn(self, batch):
        """."""
        return tuple(zip(*batch))


class BusTruckDataModuleFasterRCNN(LightningDataModule):
    """Data Module for Bus Truck cropped images for rcnn by selective search."""

    def __init__(
        self,
        img_infos: list[OpenImageInfo],
        batch_size: int,
        n_worker: int = 0,
        output_size: tuple[int, int] = (224, 224),
        folder_img: str = "./data/kaggle_bus_trucks/images",
    ):
        """."""
        super().__init__()
        self.img_infos = img_infos
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.folder_img = folder_img
        self.output_size = output_size

        self.train_infos: list[OpenImageInfo] = []
        self.val_infos: list[OpenImageInfo] = []
        self.test_infos: list[OpenImageInfo] = []

    def prepare_data(self):
        """Prepare Data."""
        pass

    def setup(self, stage=None):
        """Read images."""
        self.train_infos, leftover_infos = train_test_split(
            self.img_infos, test_size=0.2
        )
        self.val_infos, self.test_infos = train_test_split(
            leftover_infos, test_size=0.5
        )

    def train_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetFasterRCNN(
            self.train_infos, self.folder_img, self.output_size
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetFasterRCNN(
            self.val_infos, self.folder_img, self.output_size
        )
        return DataLoader(
            ds,
            batch_size=1,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """."""
        ds = BusTruckDatasetFasterRCNN(
            self.test_infos, self.folder_img, self.output_size
        )
        return DataLoader(
            ds,
            batch_size=1,
            num_workers=self.n_worker,
            collate_fn=ds.collate_fn,
            drop_last=True,
        )


# %% Test Code
def _draw_box(
    ax: Axes,
    img: NDArray[np.float32],
    box_loc_ratio_all: Tensor,
    labels: Optional[list[str]] = None,
    colors: Union[str, cycle] = "r",
    line_styles: Union[str, cycle] = "-",
):
    """Draw box in img."""
    if isinstance(colors, str):
        colors = cycle([colors])
    if isinstance(line_styles, str):
        line_styles = cycle([line_styles])

    if labels is None:
        labels = [""] * len(box_loc_ratio_all)

    n_y, n_x, _ = img.shape
    ratio2pos = array([[n_x, n_y, n_x, n_y]])

    box_loc_all = box_loc_ratio_all * ratio2pos
    for box_loc, label in zip(box_loc_all, labels):
        color = next(colors)
        ax.add_patch(
            Rectangle(
                box_loc[0:2],
                *(box_loc[2:4] - box_loc[0:2]),
                linewidth=2,
                linestyle=next(line_styles),
                edgecolor=color,
                fill=False,
            )
        )
        if label:
            ax.text(box_loc[0], box_loc[1], label, color=color)


def _draw_results_selective_search(infos: list[OpenImageInfo], folder_img: str):
    """Draw results of selective search."""
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
    line_styles = cycle(["-", "--", "-.", ":"])

    ds_bus = BusTruckDataset(infos, folder_img)

    # Search candindates box using selective search
    candidates_infos = extract_candidates_with_ss(
        infos, folder_img, scale=200.0, sigma=0.8, min_size=10, iou_th_for_etc=0.6
    )

    for idx in range(len(ds_bus)):
        img_tensor, tg_loc_ratio_all, tg_idxs = ds_bus[idx].values()
        img = img_tensor.permute(1, 2, 0).numpy()

        fig = figure(figsize=(12, 5))
        # Plot images with target
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_title("Ground truth target")
        ax1.grid(False)
        _draw_box(ax1, img, tg_loc_ratio_all, None)

        # Plot images with extract_candidates_with_ss
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img)
        c_info = candidates_infos[idx]
        n_y, n_x, _ = img.shape
        for c_ratio, tg_idx in zip(c_info.boxes_ratio, c_info.tg_idx):
            if tg_idx == LABEL2TARGET["Etc"]:
                continue
            _draw_box(
                ax2,
                img,
                vstack([c_ratio]),
                None,
                colors=colors,
                line_styles=line_styles,
            )

        ax2.set_title("Box from selective search")
        ax2.grid(False),
        fig.tight_layout()


def _draw_cropped_img(infos: list[OpenImageInfo], folder_img: str):
    """."""
    candidates_infos = extract_candidates_with_ss(
        infos, folder_img, scale=200.0, sigma=0.8, min_size=10, iou_th_for_etc=0.6
    )

    ds_candidates = BusTruckCroppedDataset(candidates_infos, folder_img)

    for idx1 in range(len(ds_candidates)):
        imgs, _, tg_idxs, *_ = ds_candidates[idx1].values()

        idx_ax = 1
        fig = figure()

        for img, tg_idx in zip(imgs, tg_idxs):
            if tg_idx == LABEL2TARGET["Etc"]:
                continue
            ax1 = fig.add_subplot(2, 2, idx_ax)
            ax1.imshow(img.permute(1, 2, 0).numpy())
            idx_ax += 1
            if idx_ax > 4:
                break


def test_BusTruckCroppedDataModuleWithSS(infos: list[OpenImageInfo], folder_img: str):
    """."""
    dm = BusTruckCroppedDataModuleWithSS(infos, 2)
    dm.setup()
    for batch in dm.train_dataloader():
        for key in batch.keys():
            if key != "image_id":
                print(f"{key}: shape({batch[key].shape})", end="   ")
            else:
                print(f"{key}: {batch[key]}", end="   ")
        print("")


def test_BusTruckRoiDataModuleWithSS(infos: list[OpenImageInfo], folder_img: str):
    """."""
    dm = BusTruckRoiDataModuleWithSS(infos, 2)
    dm.setup()
    for batch in dm.train_dataloader():
        for key in batch.keys():
            if key == "img":
                print(f"{key}: shape({batch[key].shape})", end="   ")
            else:
                print(f"{key}: {batch[key]}", end="   ")
        print("")


def test_BusTruckRoiDataModuleFasterRCNN(infos: list[OpenImageInfo], folder_img: str):
    """."""
    dm = BusTruckRoiDataModuleFasterRCNN(infos, 2)
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)


if __name__ == "__main__":
    path_csv = "./data/kaggle_bus_trucks/df.csv"
    folder_img = "./data/kaggle_bus_trucks/images"

    infos = bustruck_df2img_info(path_csv)[0:50]
    # _draw_results_selective_search(infos, folder_img)
    # _draw_cropped_img(infos, folder_img)
    # test_BusTruckCroppedDataModuleWithSS(infos, folder_img)
    # test_BusTruckRoiDataModuleWithSS(infos, folder_img)
    # test_BusTruckRoiDataModuleFasterRCNN(infos, folder_img)
