class Config:
    DATA_PATH = "../data/sample_data"
    OUTPUT_PATH = "../output"
    # MoNuSAC (5), PanNuke (6), CoNSeP_Tiled (8), MoNuSeg / MoNuSeg_37 (2, segmentation only).
    # From src/: python main.py --dataset MoNuSeg_37 --num_classes 2 --data_path ../data
    # Prepare MoNuSeg_37 tiles first: python prepare_monuseg37.py
    DATASET = "MoNuSAC"
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    STAGE_EPOCH = 25
    # Centroid match radius at 40x, as in HoVer-Net (12 px). 6 px would be 20x.
    MATCH_RADIUS = 12
    # Post-processing before validation and test scores. The published method uses it.
    # --no-post-process scores the raw NC argmax with pixel IoU and pixel F1.
    POST_PROCESS = True
    # Score that selects the checkpoint. "dice" is the binary NP Dice.
    # "pq" is the pixel harmonic mean of IoU and F1. Override with --checkpoint.
    CHECKPOINT_METRIC = "dice"
    # Sobel width for the watershed ridges only. The hover loss stays 3x3.
    # Odd integer from 1 to 31. Override with --watershed_sobel.
    WATERSHED_SOBEL = 3
    # Weight init, dropout, and the training-loader shuffle. Override with --seed.
    SEED = 42


def set_seed(seed=None):
    """Fix Python, NumPy, and PyTorch RNGs, including every CUDA device.

    Call this before building a model or a shuffled loader. cuDNN then picks
    its deterministic algorithms. The hash seed applies to child processes.
    """
    import os
    import random

    import numpy as np
    import torch

    if seed is None:
        seed = Config.SEED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(_worker_id):
    """Seed NumPy and Python inside one DataLoader worker.

    PyTorch already seeds the worker from the loader generator. This covers
    libraries that read the other two global RNGs.
    """
    import random

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader_seed(seed=None):
    """Return a generator and worker init so a shuffled loader repeats."""
    import torch

    if seed is None:
        seed = Config.SEED
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator, seed_worker