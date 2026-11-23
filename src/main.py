import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config, loader_seed, set_seed
from dataset import SegmentationDataset
from loss_function import HoverLoss
from postprocess import require_sobel_ksize
from training_loop import train, final_evaluate
from unet_hovergnn import GraphHoverNet
from visualization import plot_validation_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate UNet-HoVerGNN")
    parser.add_argument("--dataset", type=str, default=Config.DATASET,
                        help="Dataset name: MoNuSAC, PanNuke, CoNSeP_Tiled, MoNuSeg, or MoNuSeg_37")
    parser.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES,
                        help="Class count including background. MoNuSeg/MoNuSeg_37: use 2")
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to dataset root directory")
    parser.add_argument("--output_path", type=str, default=Config.OUTPUT_PATH,
                        help="Path to save outputs (logs, checkpoints, metrics)")
    parser.add_argument("--epochs", type=int, default=Config.STAGE_EPOCH,
                        help="Epochs per stage. Pretrain and finetune each run this many")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                        help="Samples per batch for train, validation, and test loaders")
    parser.add_argument("--post_process", action=argparse.BooleanOptionalAction, default=Config.POST_PROCESS,
                        help="Score with post-processing. --no-post-process scores the raw NC argmax")
    parser.add_argument("--checkpoint", choices=("dice", "pq"), default=Config.CHECKPOINT_METRIC,
                        help="Validation score that selects the checkpoint. "
                             "dice is the binary NP Dice (default); pq is the pixel harmonic mean")
    parser.add_argument("--watershed_sobel", type=require_sobel_ksize, default=Config.WATERSHED_SOBEL,
                        help="Sobel width for the watershed ridges. Odd integer from 1 to 31, default 3. "
                             "The hover loss stays 3x3")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Seed for weight init, dropout, and the training shuffle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override Config values if provided
    Config.DATASET = args.dataset
    Config.NUM_CLASSES = args.num_classes
    Config.DATA_PATH = args.data_path
    Config.OUTPUT_PATH = args.output_path
    Config.BATCH_SIZE = args.batch_size
    Config.POST_PROCESS = args.post_process
    Config.CHECKPOINT_METRIC = args.checkpoint
    Config.WATERSHED_SOBEL = args.watershed_sobel
    Config.SEED = args.seed
    set_seed(Config.SEED)
    print(f"Seed: {Config.SEED}")
    print(f"Checkpoint score: {Config.CHECKPOINT_METRIC}")
    print(f"Watershed Sobel: {Config.WATERSHED_SOBEL}")

    # Ensure output path exists
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

    # === Transformations ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET)

    # === Datasets and DataLoaders ===
    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    val_dataset = SegmentationDataset(dataset_path, split="val", transform=transform)
    test_dataset = SegmentationDataset(dataset_path, split="test", transform=transform)

    train_generator, seed_worker = loader_seed(Config.SEED)
    train_dataloader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        generator=train_generator, worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        worker_init_fn=seed_worker,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        worker_init_fn=seed_worker,
    )

    # === Model, Loss, and Training ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES).to(device)
    criterion = HoverLoss()

    # Train model
    train(model, criterion, train_dataloader, val_dataloader, device,
          num_classes=Config.NUM_CLASSES, epochs=args.epochs,
          post_process=Config.POST_PROCESS,
          checkpoint_metric=Config.CHECKPOINT_METRIC,
          sobel_ksize=Config.WATERSHED_SOBEL)

    # Evaluate model (pretraining and finetuning)
    final_evaluate(model, test_dataloader, criterion, device, "pretrain",
                   num_classes=Config.NUM_CLASSES, post_process=Config.POST_PROCESS,
                   sobel_ksize=Config.WATERSHED_SOBEL)
    final_evaluate(model, test_dataloader, criterion, device, "finetune",
                   num_classes=Config.NUM_CLASSES, post_process=Config.POST_PROCESS,
                   sobel_ksize=Config.WATERSHED_SOBEL)

    # Each saved figure shows the raw class map and the post-processed instance map.
    plot_validation_samples(model, test_dataloader, device, Config.NUM_CLASSES,
                            sobel_ksize=Config.WATERSHED_SOBEL)
