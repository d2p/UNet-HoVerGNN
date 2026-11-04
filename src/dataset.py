import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _split_has_tiles(split_dir):
    """True when split_dir holds sample folders that contain image.npy."""
    try:
        names = os.listdir(split_dir)
    except OSError:
        return False
    for name in names:
        sample = os.path.join(split_dir, name)
        if os.path.isdir(sample) and os.path.isfile(os.path.join(sample, "image.npy")):
            return True
    return False


def resolve_dataset_path(data_path, dataset):
    """Directory that contains train/val/test tile folders.

    ``data_path`` may be the parent of the named dataset
    (``../data`` with ``MoNuSeg_37/train``) or the dataset directory
    itself (``/kaggle/input/MoNuSeg_37`` with ``train`` inside it).
    """
    nested = os.path.join(data_path, dataset)
    raw_hit = None
    for root in (nested, data_path):
        train = os.path.join(root, "train")
        if not os.path.isdir(train):
            continue
        if _split_has_tiles(train):
            return root
        if os.path.isdir(os.path.join(train, "img")):
            raw_hit = train
    if raw_hit is not None:
        raise FileNotFoundError(
            f"{raw_hit} has img/labelcol images, not tiled samples. "
            "Run prepare_monuseg.py or prepare_monuseg37.py so each split "
            "contains sample folders with image.npy, mask.npy, h_map.npy, and v_map.npy."
        )
    raise FileNotFoundError(
        f"No tiled train split at {os.path.join(nested, 'train')} "
        f"or {os.path.join(data_path, 'train')}."
    )


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root dataset directory, e.g., "MoNuSAC"
            split (str): One of 'train', 'val', 'test'
            image_transform (callable, optional): Transform applied only to image
        """
        self.split_dir = os.path.join(root_dir, split)
        self.sample_dirs = sorted([
            os.path.join(self.split_dir, d)
            for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        image = np.load(os.path.join(sample_dir, "image.npy"))  # HWC
        mask = np.load(os.path.join(sample_dir, "mask.npy"))    # HW
        h_map = np.load(os.path.join(sample_dir, "h_map.npy"))  # HW
        v_map = np.load(os.path.join(sample_dir, "v_map.npy"))  # HW

        h_map = (h_map.astype(np.float32) / 127.5) - 1.0
        v_map = (v_map.astype(np.float32) / 127.5) - 1.0

        inst_path = os.path.join(sample_dir, "inst.npy")
        if os.path.exists(inst_path):
            inst = np.load(inst_path).astype(np.int64)
        else:
            # Negative ids tell the metric code to rebuild instances from the hover maps.
            inst = np.full(mask.shape, -1, dtype=np.int64)

        if self.transform:
            image = self.transform(image)

        return image, \
                torch.from_numpy(mask).long(), \
                torch.from_numpy(h_map).float(), \
                torch.from_numpy(v_map).float(), \
                torch.from_numpy(inst).long()
    
if __name__ == "__main__":
    from config import Config
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts HWC to CHW and scales to [0, 1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # example: scale to [-1, 1]
    ])

    dataset_path = resolve_dataset_path(Config.DATA_PATH, Config.DATASET)

    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    val_dataset = SegmentationDataset(dataset_path, split="val", transform=transform)
    test_dataset = SegmentationDataset(dataset_path, split="test", transform=transform)

    from torch.utils.data import Dataset, DataLoader

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("First fold:", len(train_dataloader))
    print("Second fold:", len(val_dataloader))
    print("Third fold:", len(test_dataloader))

    image, mask, h_map, v_map, inst = next(iter(train_dataloader))
    print(image.shape)
    print(mask.shape)
    print(h_map.shape)
    print(v_map.shape)
    print(inst.shape)