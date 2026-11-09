"""Instance metrics matched to HoVer-Net (Graham et al., 2019).

AJI is the MoNuSeg fast AJI used by HoVer-Net (`get_fast_aji`): a ground-truth
instance may reuse the same predicted instance. PQ / DQ / SQ follow Kirillov
panoptic quality with a unique IoU > 0.5 match (`get_fast_pq`). Per-class
detection F1 follows Eq. 8 with centroid matching at a fixed radius.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def remap_label(pred, by_size=False):
    """Rename instance ids so they are contiguous: 0 background, then 1..N."""
    pred = np.asarray(pred)
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return np.zeros(pred.shape, np.int32)
    if by_size:
        pred_size = [(pred == inst_id).sum() for inst_id in pred_id]
        order = np.argsort(pred_size)[::-1]
        pred_id = [pred_id[i] for i in order]

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def _foreground_ids(mask):
    """Instance ids other than background. Background is 0 and may be absent."""
    ids = np.unique(mask)
    return ids[ids != 0]


def _overlap_matrices(true, pred):
    """Pixel overlap of every foreground instance.

    Returns foreground ids, intersection, union, and each instance's area.
    Intersection and union have shape (n_true, n_pred). A side with no
    foreground instances contributes a zero-length axis.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    true_ids = _foreground_ids(true)
    pred_ids = _foreground_ids(pred)
    n_true = len(true_ids)
    n_pred = len(pred_ids)
    inter = np.zeros((n_true, n_pred), dtype=np.float64)
    union = np.zeros((n_true, n_pred), dtype=np.float64)
    true_area = np.zeros(n_true, dtype=np.float64)
    pred_area = np.zeros(n_pred, dtype=np.float64)
    if n_true == 0 or n_pred == 0:
        for ti, t in enumerate(true_ids):
            true_area[ti] = np.count_nonzero(true == t)
        for pi, p in enumerate(pred_ids):
            pred_area[pi] = np.count_nonzero(pred == p)
        return true_ids, pred_ids, inter, union, true_area, pred_area

    pred_masks = [pred == p for p in pred_ids]
    pred_index = {int(p): i for i, p in enumerate(pred_ids)}
    for pi, p_mask in enumerate(pred_masks):
        pred_area[pi] = np.count_nonzero(p_mask)
    for ti, t in enumerate(true_ids):
        t_mask = true == t
        true_area[ti] = np.count_nonzero(t_mask)
        for raw_id in np.unique(pred[t_mask]):
            pred_id = int(raw_id)
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_index[pred_id]]
            inter[ti, pred_index[pred_id]] = np.count_nonzero(t_mask & p_mask)
            union[ti, pred_index[pred_id]] = np.count_nonzero(t_mask | p_mask)
    return true_ids, pred_ids, inter, union, true_area, pred_area


def get_fast_aji(true, pred):
    """MoNuSeg / HoVer-Net AJI.

    Id 0 is background. A map that is entirely foreground is valid: the first
    unique id is an instance, not background. One ground-truth instance may
    reuse one predicted instance. Both sides empty scores 1; one side empty
    scores 0.
    """
    true_ids, pred_ids, inter, union, true_area, pred_area = _overlap_matrices(true, pred)
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    pairwise_iou = inter / (union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    best_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(best_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    overall_inter = inter[paired_true, paired_pred].sum()
    overall_union = union[paired_true, paired_pred].sum()

    paired_true_set = set(int(i) for i in paired_true)
    paired_pred_set = set(int(i) for i in paired_pred)
    for ti in range(len(true_ids)):
        if ti not in paired_true_set:
            overall_union += true_area[ti]
    for pi in range(len(pred_ids)):
        if pi not in paired_pred_set:
            overall_union += pred_area[pi]

    if overall_union == 0:
        return 1.0
    return float(overall_inter / overall_union)


def get_fast_pq(true, pred, match_iou=0.5):
    """Panoptic quality. Returns [dq, sq, pq] and the pairing lists.

    Pairing lists hold instance ids. Both sides empty scores 1; one side
    empty scores 0. Id 0 is background and may be absent from the map.
    """
    assert match_iou >= 0.0
    true_ids, pred_ids, inter, union, _, _ = _overlap_matrices(true, pred)
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return [1.0, 1.0, 1.0], [[], [], [], []]
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return [0.0, 0.0, 0.0], [
            [],
            [],
            [int(i) for i in true_ids],
            [int(i) for i in pred_ids],
        ]

    pairwise_iou = inter / np.maximum(union, 1.0e-6)
    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        keep = paired_iou > match_iou
        paired_true = paired_true[keep]
        paired_pred = paired_pred[keep]
        paired_iou = paired_iou[keep]

    paired_true_ids = [int(true_ids[i]) for i in paired_true]
    paired_pred_ids = [int(pred_ids[i]) for i in paired_pred]
    unpaired_true = [int(i) for i in true_ids if int(i) not in paired_true_ids]
    unpaired_pred = [int(i) for i in pred_ids if int(i) not in paired_pred_ids]
    tp = len(paired_true_ids)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = tp / denom if denom > 0 else 0.0
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0
    return [float(dq), float(sq), float(dq * sq)], [
        paired_true_ids,
        paired_pred_ids,
        unpaired_true,
        unpaired_pred,
    ]


def pair_coordinates(set_a, set_b, radius):
    """Hungarian matching of points. set_* are (N, 2) in XY order.

    Returns pairing indices (K, 2), unpaired indices in A, unpaired indices in B.
    """
    set_a = np.asarray(set_a, dtype=np.float32).reshape(-1, 2)
    set_b = np.asarray(set_b, dtype=np.float32).reshape(-1, 2)
    if set_a.shape[0] == 0 or set_b.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, np.arange(set_a.shape[0]), np.arange(set_b.shape[0])

    pair_distance = cdist(set_a, set_b, metric="euclidean")
    indices_a, paired_indices_b = linear_sum_assignment(pair_distance)
    pair_cost = pair_distance[indices_a, paired_indices_b]
    paired_a = indices_a[pair_cost <= radius]
    paired_b = paired_indices_b[pair_cost <= radius]
    pairing = np.stack([paired_a, paired_b], axis=1) if len(paired_a) else np.zeros((0, 2), np.int64)
    unpaired_a = np.delete(np.arange(set_a.shape[0]), paired_a)
    unpaired_b = np.delete(np.arange(set_b.shape[0]), paired_b)
    return pairing, unpaired_a, unpaired_b


def instance_centroids_and_types(inst, type_map):
    """Centres of mass (XY) and majority type for every non-zero instance."""
    inst = np.asarray(inst)
    type_map = np.asarray(type_map)
    ids = np.unique(inst)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int64)
    cents = []
    types = []
    for inst_id in ids:
        ys, xs = np.where(inst == inst_id)
        cents.append([xs.mean(), ys.mean()])
        vals, counts = np.unique(type_map[ys, xs], return_counts=True)
        types.append(int(vals[np.argmax(counts)]))
    return np.asarray(cents, np.float32), np.asarray(types, np.int64)


def score_instances(true_inst, pred_inst):
    """Per-image AJI, DQ, SQ, PQ. Empty vs empty scores 1; one side empty scores 0."""
    true_inst = remap_label(true_inst)
    pred_inst = remap_label(pred_inst)
    true_empty = int(true_inst.max()) == 0
    pred_empty = int(pred_inst.max()) == 0
    if true_empty and pred_empty:
        return {"aji": 1.0, "dq": 1.0, "sq": 1.0, "pq": 1.0}
    if true_empty or pred_empty:
        return {"aji": 0.0, "dq": 0.0, "sq": 0.0, "pq": 0.0}
    aji = get_fast_aji(true_inst, pred_inst)
    dq, sq, pq = get_fast_pq(true_inst, pred_inst, match_iou=0.5)[0]
    return {"aji": aji, "dq": dq, "sq": sq, "pq": pq}


class PixelComparisonMeter:
    """Published comparison scores on a semantic class map.

    AJI here is the mean foreground pixel IoU. F1 is the mean foreground pixel
    F1. PQ is the mean, over foreground classes, of the harmonic mean
    2 * IoU * F1 / (IoU + F1). Each class score is averaged over images first.
    An image where a class is absent from both maps scores 1 for that class.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.sum_iou = np.zeros(num_classes, dtype=np.float64)
        self.sum_f1 = np.zeros(num_classes, dtype=np.float64)
        self.n = 0

    def update(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.ndim == 2:
            pred = pred[None]
            target = target[None]
        for image_pred, image_true in zip(pred, target):
            for class_id in range(self.num_classes):
                pred_c = image_pred == class_id
                true_c = image_true == class_id
                tp = float(np.count_nonzero(pred_c & true_c))
                fp = float(np.count_nonzero(pred_c & ~true_c))
                fn = float(np.count_nonzero(~pred_c & true_c))
                iou_den = tp + fp + fn
                f1_den = 2.0 * tp + fp + fn
                self.sum_iou[class_id] += 1.0 if iou_den == 0.0 else tp / iou_den
                self.sum_f1[class_id] += 1.0 if f1_den == 0.0 else (2.0 * tp) / f1_den
            self.n += 1

    def compute(self):
        n = max(self.n, 1)
        iou_c = self.sum_iou / n
        f1_c = self.sum_f1 / n
        pq_c = 2.0 * iou_c * f1_c / (iou_c + f1_c + 1.0e-8)
        if self.num_classes > 1:
            iou_c = iou_c[1:]
            f1_c = f1_c[1:]
            pq_c = pq_c[1:]
        return {
            "aji": float(iou_c.mean()) if len(iou_c) else 0.0,
            "f1": float(f1_c.mean()) if len(f1_c) else 0.0,
            "pq": float(pq_c.mean()) if len(pq_c) else 0.0,
            "f1_per_class": [float(value) for value in f1_c],
        }


class BinaryDiceMeter:
    """Pixel-pooled Dice of nucleus versus background.

    For a binary mask this is the same quantity as the foreground F1:
    2 TP / (2 TP + FP + FN), summed over every pixel in the split.
    An empty prediction against an empty target scores 1.
    """

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred_fg, true_fg):
        pred_fg = np.asarray(pred_fg, dtype=bool)
        true_fg = np.asarray(true_fg, dtype=bool)
        self.tp += float(np.count_nonzero(pred_fg & true_fg))
        self.fp += float(np.count_nonzero(pred_fg & ~true_fg))
        self.fn += float(np.count_nonzero(~pred_fg & true_fg))

    def compute(self):
        denom = 2.0 * self.tp + self.fp + self.fn
        if denom == 0.0:
            return 1.0
        return float(2.0 * self.tp / denom)


class HoverNetMetrics:
    """Image-averaged AJI/DQ/SQ/PQ and dataset-pooled detection F1."""

    def __init__(self, num_classes, radius=12):
        self.num_classes = num_classes
        self.radius = radius
        self._aji = []
        self._dq = []
        self._sq = []
        self._pq = []
        self.tp_c = np.zeros(num_classes, dtype=np.float64)
        self.tn_c = np.zeros(num_classes, dtype=np.float64)
        self.fp_c = np.zeros(num_classes, dtype=np.float64)
        self.fn_c = np.zeros(num_classes, dtype=np.float64)
        self.fp_d = np.zeros(num_classes, dtype=np.float64)
        self.fn_d = np.zeros(num_classes, dtype=np.float64)
        self.det_tp = 0.0
        self.det_fp = 0.0
        self.det_fn = 0.0

    def update(self, true_inst, pred_inst, true_type, pred_type):
        scores = score_instances(true_inst, pred_inst)
        self._aji.append(scores["aji"])
        self._dq.append(scores["dq"])
        self._sq.append(scores["sq"])
        self._pq.append(scores["pq"])

        true_xy, true_cls = instance_centroids_and_types(true_inst, true_type)
        pred_xy, pred_cls = instance_centroids_and_types(pred_inst, pred_type)
        if len(true_xy) == 0 and len(pred_xy) == 0:
            return

        pairing, unpaired_t, unpaired_p = pair_coordinates(true_xy, pred_xy, self.radius)
        self.det_tp += len(pairing)
        self.det_fp += len(unpaired_p)
        self.det_fn += len(unpaired_t)

        if len(pairing):
            paired_true = true_cls[pairing[:, 0]]
            paired_pred = pred_cls[pairing[:, 1]]
        else:
            paired_true = np.zeros((0,), np.int64)
            paired_pred = np.zeros((0,), np.int64)
        unpaired_true_cls = true_cls[unpaired_t] if len(unpaired_t) else np.zeros((0,), np.int64)
        unpaired_pred_cls = pred_cls[unpaired_p] if len(unpaired_p) else np.zeros((0,), np.int64)

        for t in range(self.num_classes):
            self.tp_c[t] += np.count_nonzero((paired_true == t) & (paired_pred == t))
            self.tn_c[t] += np.count_nonzero((paired_true != t) & (paired_pred != t))
            self.fp_c[t] += np.count_nonzero((paired_true != t) & (paired_pred == t))
            self.fn_c[t] += np.count_nonzero((paired_true == t) & (paired_pred != t))
            self.fp_d[t] += np.count_nonzero(unpaired_pred_cls == t)
            self.fn_d[t] += np.count_nonzero(unpaired_true_cls == t)

    def compute(self):
        n = max(len(self._aji), 1)
        f1_per_class = []
        for t in range(1, self.num_classes):
            nom = 2.0 * (self.tp_c[t] + self.tn_c[t])
            den = nom + 2.0 * self.fp_c[t] + 2.0 * self.fn_c[t] + self.fp_d[t] + self.fn_d[t]
            f1_per_class.append(float(nom / den) if den > 0 else 0.0)
        det_den = 2.0 * self.det_tp + self.det_fp + self.det_fn
        f1_d = float(2.0 * self.det_tp / det_den) if det_den > 0 else 0.0
        return {
            "aji": float(np.mean(self._aji)) if self._aji else 0.0,
            "dq": float(np.mean(self._dq)) if self._dq else 0.0,
            "sq": float(np.mean(self._sq)) if self._sq else 0.0,
            "pq": float(np.mean(self._pq)) if self._pq else 0.0,
            "f1_d": f1_d,
            "f1_per_class": f1_per_class,
            "n_images": len(self._aji) if self._aji else n,
        }


def _self_check():
    inst = np.zeros((12, 16), np.int32)
    inst[1:5, 1:5] = 1
    inst[1:5, 8:12] = 2
    perfect = score_instances(inst, inst.copy())
    assert abs(perfect["aji"] - 1.0) < 1e-6, perfect
    assert abs(perfect["pq"] - 1.0) < 1e-5, perfect

    merged = np.zeros_like(inst)
    merged[1:5, 1:5] = 1
    merged[1:5, 8:12] = 1
    merged_scores = score_instances(inst, merged)
    assert merged_scores["aji"] < 1.0

    stats = HoverNetMetrics(num_classes=3, radius=12)
    stats.update(inst, inst, inst, inst)
    out = stats.compute()
    assert abs(out["f1_d"] - 1.0) < 1e-6, out
    assert abs(out["f1_per_class"][0] - 1.0) < 1e-6, out

    dice = BinaryDiceMeter()
    dice.update(inst > 0, inst > 0)
    assert abs(dice.compute() - 1.0) < 1e-6
    half = np.zeros_like(inst)
    half[:, :8] = inst[:, :8]
    dice = BinaryDiceMeter()
    dice.update(half > 0, inst > 0)
    assert 0.0 < dice.compute() < 1.0, dice.compute()
    empty = BinaryDiceMeter()
    assert empty.compute() == 1.0

    labels = np.zeros((4, 4), np.int32)
    labels[:2, :] = 1
    pixel = PixelComparisonMeter(2)
    pixel.update(labels, labels)
    perfect_px = pixel.compute()
    assert abs(perfect_px["aji"] - 1.0) < 1e-6, perfect_px
    assert abs(perfect_px["f1"] - 1.0) < 1e-6, perfect_px
    assert abs(perfect_px["pq"] - 1.0) < 1e-6, perfect_px
    partial_pred = np.zeros_like(labels)
    partial_pred[:1, :] = 1
    pixel = PixelComparisonMeter(2)
    pixel.update(partial_pred, labels)
    partial = pixel.compute()
    harmonic = (2.0 * partial["aji"] * partial["f1"]) / (partial["aji"] + partial["f1"])
    assert abs(partial["aji"] - 0.5) < 1e-6, partial
    assert abs(partial["pq"] - harmonic) < 1e-6, partial

    # A tile with no background pixel used to build a (0, 0) IoU matrix.
    solid = np.ones((8, 8), np.int32)
    solid_scores = score_instances(solid, solid.copy())
    assert abs(solid_scores["aji"] - 1.0) < 1e-6, solid_scores
    assert abs(solid_scores["pq"] - 1.0) < 1e-5, solid_scores

    tiled = np.ones((8, 8), np.int32)
    tiled[:, 4:] = 2
    tiled_scores = score_instances(tiled, tiled.copy())
    assert abs(tiled_scores["aji"] - 1.0) < 1e-6, tiled_scores
    assert abs(tiled_scores["pq"] - 1.0) < 1e-5, tiled_scores

    gt = np.zeros((8, 8), np.int32)
    gt[1:4, 1:4] = 1
    gt[1:4, 5:8] = 2
    blob = np.ones_like(gt)
    blob_scores = score_instances(gt, blob)
    assert 0.0 <= blob_scores["aji"] < 1.0, blob_scores
    assert blob_scores["pq"] == 0.0, blob_scores
    assert score_instances(gt, np.zeros_like(gt))["aji"] == 0.0
    print("metrics self-check ok", out)


if __name__ == "__main__":
    _self_check()
