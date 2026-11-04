import numpy as np
import torch
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def _watershed_instances(np_mask, h_map, v_map, class_map):
    """Marker-controlled watershed on HoVer maps.

    Returns an instance-id map and a type map (majority class inside each instance).
    """
    # Cast the nuclear mask, hover maps, and class map to the dtypes watershed expects.
    np_mask = np.asarray(np_mask).astype(bool)
    h_map = np.asarray(h_map, dtype=np.float64)
    v_map = np.asarray(v_map, dtype=np.float64)
    class_map = np.asarray(class_map).astype(np.int32)

    # Sobel energy is high where the hover maps jump. Invert it so watershed
    # treats those jumps as ridges and floods outward from each marker.
    dx = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(dx ** 2 + dy ** 2)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy = 1.0 - energy

    # One marker per nucleus, at a local peak of the distance transform inside the mask.
    distance = ndi.distance_transform_edt(np_mask)
    coords = peak_local_max(distance, labels=np_mask, footprint=np.ones((3, 3)), exclude_border=False)
    markers = np.zeros(np_mask.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx

    # Grow markers until they meet at the hover ridges. Background stays 0.
    instance_map = watershed(energy, markers=markers, mask=np_mask).astype(np.int32)

    # Label every pixel of an instance with the most common class inside it.
    type_map = np.zeros_like(instance_map, dtype=np.int32)
    for inst_id in range(1, int(instance_map.max()) + 1):
        region = instance_map == inst_id
        class_ids, counts = np.unique(class_map[region], return_counts=True)
        if len(class_ids) > 0:
            type_map[region] = class_ids[np.argmax(counts)]
    return instance_map, type_map


def postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device, return_centroids=False, nr_types=6):
    """Split nuclei with the HoVer watershed.

    Returns:
        instance_map: [B, H, W] long instance ids (0 is background).
        type_map: [B, H, W] long class ids from a majority vote of the NC head.
    """
    del return_centroids
    batch_size = np_logits.shape[0]
    # NP channel 0 is background. NC argmax is the per-pixel class before the vote.
    np_pred = torch.argmax(np_logits, dim=1)
    nc_pred = torch.argmax(nc_logits, dim=1)

    # Watershed each image on CPU. nr_types=0 keeps every instance labelled background.
    instances = []
    types = []
    for i in range(batch_size):
        np_mask = np_pred[i].detach().cpu().numpy() > 0
        h_map = hv_logits[i, 0].detach().cpu().numpy()
        v_map = hv_logits[i, 1].detach().cpu().numpy()
        if nr_types:
            class_map = nc_pred[i].detach().cpu().numpy()
        else:
            class_map = np.zeros(np_mask.shape, dtype=np.int32)
        inst, typ = _watershed_instances(np_mask, h_map, v_map, class_map)
        instances.append(torch.from_numpy(inst).long())
        types.append(torch.from_numpy(typ).long())

    return torch.stack(instances).to(device), torch.stack(types).to(device)


def instances_from_hover(mask, h_map, v_map):
    """Recover a ground-truth instance map from a class mask and hover maps.

    Kaggle samples store class ids, not instance ids. The same watershed used
    at inference is run on the stored ground-truth hover maps. Touching nuclei
    of the same class are separated only as well as those maps allow.
    """
    # Watershed runs in NumPy. Remember the device so the ids can be moved back.
    if torch.is_tensor(mask):
        device = mask.device
        mask_np = mask.detach().cpu().numpy()
        h_np = h_map.detach().cpu().numpy()
        v_np = v_map.detach().cpu().numpy()
    else:
        device = None
        mask_np = np.asarray(mask)
        h_np = np.asarray(h_map)
        v_np = np.asarray(v_map)
    # Seed the same watershed used at inference. The class map is the GT type labels.
    inst, _ = _watershed_instances(mask_np > 0, h_np, v_np, mask_np)
    inst_t = torch.from_numpy(inst).long()
    if device is not None:
        inst_t = inst_t.to(device)
    return inst_t


def resolve_gt_instances(mask, h_map, v_map, inst):
    """Use inst.npy when it is present. A negative id means the sample has none."""
    # MoNuSeg stores real ids. Kaggle samples are filled with -1 and must be rebuilt.
    resolved = inst.clone()
    for b in range(inst.shape[0]):
        if (inst[b] < 0).any():
            resolved[b] = instances_from_hover(mask[b], h_map[b], v_map[b])
    return resolved


def centroids_from_instances(inst_map):
    """One centroid per instance.

    Returns:
        coords: [N, 3] long (y, x, batch)
        inst_ids: [N, 2] long (batch, instance id)
    """
    # An empty map has no nuclei, so both lists stay empty.
    if inst_map.numel() == 0:
        device = inst_map.device
        return (
            torch.zeros((0, 3), dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )
    # Centre of mass, rounded to a pixel. Background id 0 is skipped.
    inst_np = inst_map.detach().cpu().numpy()
    coords = []
    ids = []
    for b in range(inst_np.shape[0]):
        for inst_id in np.unique(inst_np[b]):
            if inst_id == 0:
                continue
            ys, xs = np.where(inst_np[b] == inst_id)
            coords.append([int(round(ys.mean())), int(round(xs.mean())), b])
            ids.append([b, int(inst_id)])
    device = inst_map.device
    if not coords:
        return (
            torch.zeros((0, 3), dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )
    return (
        torch.tensor(coords, dtype=torch.long, device=device),
        torch.tensor(ids, dtype=torch.long, device=device),
    )


def instance_majority_labels(inst_map, class_map, inst_ids):
    """Majority class of class_map inside each instance listed in inst_ids."""
    # One label per listed instance. A missing region is treated as background.
    labels = []
    for row in inst_ids:
        b = int(row[0])
        inst_id = int(row[1])
        region = class_map[b][inst_map[b] == inst_id]
        if region.numel() == 0:
            labels.append(0)
            continue
        vals, counts = torch.unique(region, return_counts=True)
        labels.append(int(vals[torch.argmax(counts)].item()))
    if not labels:
        return torch.zeros((0,), dtype=torch.long, device=inst_map.device)
    return torch.tensor(labels, dtype=torch.long, device=inst_map.device)


def apply_gnn_instance_labels(inst_map, inst_ids, node_logits, fallback_type):
    """Paint each instance with the GNN class of its centroid node."""
    # Start from the NC majority vote and overwrite instances that have a GNN node.
    type_map = fallback_type.clone()
    if node_logits is None or inst_ids is None or inst_ids.numel() == 0:
        return type_map
    classes = torch.argmax(node_logits, dim=1)
    for i in range(inst_ids.shape[0]):
        b = int(inst_ids[i, 0])
        inst_id = int(inst_ids[i, 1])
        if inst_id <= 0:
            continue
        # Paint the whole instance, not only the centroid pixel.
        type_map[b][inst_map[b] == inst_id] = classes[i]
    return type_map

def get_node_labels_from_coords(centroid_coords, type_maps):
    """
    Args:
        centroid_coords: (N, 3) tensor, (y, x, batch_id)
        type_maps: (B, H, W) tensor of type labels

    Returns:
        labels: (N,) long tensor
    """
    # Read the class under each centroid. Clamp so a rounded centre stays in frame.
    y = centroid_coords[:, 0].clamp(0, type_maps.shape[1] - 1)
    x = centroid_coords[:, 1].clamp(0, type_maps.shape[2] - 1)
    b = centroid_coords[:, 2].clamp(0, type_maps.shape[0] - 1)

    return type_maps[b, y, x]

if __name__ == "__main__":
    from config import Config
    from unet_hovergnn import GraphHoverNet
    from loss_function import HoverLoss
    from dataset import SegmentationDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from visualization import visualize_hovernet_output
    import os

    # Scale RGB tiles to [-1, 1], matching the training loader.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts HWC to CHW and scales to [0, 1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # example: scale to [-1, 1]
    ])

    dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET)

    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Pretrain keeps the graph off, so this demo scores the CNN heads only.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    criterion = HoverLoss()
    model.set_stage("pretrain")

    from metrics import HoverNetMetrics

    with torch.no_grad():
        # One training batch, moved to the model device.
        images, masks, h_grads, v_grads, inst = next(iter(train_dataloader))
        images = images.to(device)
        masks = masks.to(device).long()
        h_grads = h_grads.to(device)
        v_grads = v_grads.to(device)
        inst = inst.to(device)

        # Ground-truth nodes: one centroid per nucleus, labelled by its majority class.
        nc_targets = masks
        np_targets = (nc_targets > 0).long()
        gt_inst = resolve_gt_instances(masks, h_grads, v_grads, inst)
        node_coords, node_inst_ids = centroids_from_instances(gt_inst)
        node_labels = instance_majority_labels(gt_inst, masks, node_inst_ids)
        keep = node_labels != 0
        if keep.any():
            node_coords = node_coords[keep]
            node_labels = node_labels[keep]
        else:
            node_coords = None

        np_logits, hv_logits, nc_logits, centroids, gc_logits = model(images, node_coords=node_coords)

        # The graph term is zero while the stage is pretrain or the batch has no nodes.
        if gc_logits is not None and node_labels is not None and node_labels.numel() > 0:
            graph_loss_inputs = (gc_logits, node_labels)
        else:
            graph_loss_inputs = (None, None)

        loss = criterion(
            np_logits, np_targets,
            hv_logits, h_grads, v_grads,
            nc_logits, nc_targets,
            *graph_loss_inputs
        )

        # Instance AJI/PQ from the watershed ids, and per-class F1 from the type map.
        pred_inst, pred_type = postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device)
        meter = HoverNetMetrics(num_classes=Config.NUM_CLASSES, radius=Config.MATCH_RADIUS)
        for b in range(images.shape[0]):
            meter.update(
                gt_inst[b].cpu().numpy(),
                pred_inst[b].cpu().numpy(),
                masks[b].cpu().numpy(),
                pred_type[b].cpu().numpy(),
            )
        scores = meter.compute()
        print("Loss:", loss.item())
        print("AJI:", scores["aji"], "DQ:", scores["dq"], "SQ:", scores["sq"], "PQ:", scores["pq"])
        print("Detection F1:", scores["f1_d"])
        print("Per-class detection F1:", scores["f1_per_class"])

        visualize_hovernet_output(pred_inst.cpu().numpy()[0])