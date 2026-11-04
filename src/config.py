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
    PATIENCE = 25
    # Centroid match radius at 40x, as in HoVer-Net (12 px). 6 px would be 20x.
    MATCH_RADIUS = 12
    # Watershed before validation and test scores. The published method uses it.
    # --no-post-process scores the raw NC argmax with pixel IoU and pixel F1.
    POST_PROCESS = True