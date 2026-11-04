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


def get_fast_aji(true, pred):
    """MoNuSeg / HoVer-Net AJI. Ids must be contiguous starting at 1."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))

    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap_id = list(np.unique(pred[t_mask > 0]))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    if overall_union == 0:
        return 1.0
    return float(overall_inter / overall_union)


def get_fast_pq(true, pred, match_iou=0.5):
    """Panoptic quality. Returns [dq, sq, pq] and the pairing lists."""
    assert match_iou >= 0.0
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))
    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap_id = list(np.unique(pred[t_mask > 0]))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_iou[true_id - 1, pred_id - 1] = inter / (total - inter)

    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true + 1)
        paired_pred = list(paired_pred + 1)
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        keep = paired_iou > match_iou
        paired_true = list(paired_true[keep] + 1)
        paired_pred = list(paired_pred[keep] + 1)
        paired_iou = paired_iou[keep]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = tp / denom if denom > 0 else 0.0
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0
    return [float(dq), float(sq), float(dq * sq)], [
        paired_true,
        paired_pred,
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
    print("metrics self-check ok", out)


if __name__ == "__main__":
    _self_check()
