"""
SlotDeconv: Utility functions for evaluation and data handling.
JSD uses standard definition: JSD = (JS_distance)^2, NOT 0.5 * (JS_distance)^2
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import average_precision_score
def spotwise_corr(G, P, eps=1e-12):
    """Spot-wise Pearson correlation."""
    A = G - G.mean(1, keepdims=True)
    B = P - P.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1)) + eps
    return num / den
def spotwise_cos(G, P, eps=1e-12):
    """Spot-wise cosine similarity."""
    num = (G * P).sum(1)
    den = np.sqrt((G * G).sum(1)) * np.sqrt((P * P).sum(1)) + eps
    return num / den
def spotwise_jsd(G, P, eps=1e-12):
    """
    Spot-wise Jensen-Shannon Divergence (standard definition).
    JSD = (JS_distance)^2, where JS_distance = scipy.jensenshannon()
    Note: scipy.jensenshannon returns JS distance (sqrt of divergence).
    """
    Gp = G + eps
    Pp = P + eps
    Gp = Gp / Gp.sum(1, keepdims=True)
    Pp = Pp / Pp.sum(1, keepdims=True)
    out = np.empty(Gp.shape[0], dtype=np.float64)
    for i in range(Gp.shape[0]):
        d = jensenshannon(Gp[i], Pp[i], base=2.0)
        out[i] = d * d
    return out
def celltype_corr(G, P, eps=1e-12):
    """Cell-type-wise Pearson correlation."""
    K = G.shape[1]
    out = np.empty(K, dtype=np.float64)
    for k in range(K):
        g, p = G[:, k], P[:, k]
        gm, pm = g.mean(), p.mean()
        num = ((g - gm) * (p - pm)).sum()
        den = np.sqrt(((g - gm) ** 2).sum() * ((p - pm) ** 2).sum()) + eps
        out[k] = num / den
    return out
def get_valid_types(true_props, cell_types, eps=1e-6):
    """
    Get cell types with non-constant ground truth (std > eps).
    Excludes types like Neutrophil where GT is all zeros.
    """
    true = np.asarray(true_props, dtype=np.float64)
    valid = []
    for j, ct in enumerate(cell_types):
        if np.std(true[:, j]) > eps:
            valid.append(ct)
    return valid
def compute_spot_metrics(G, P, eps=1e-12):
    """Compute per-spot metrics."""
    G = np.asarray(G, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    G = np.maximum(G, 0.0)
    P = np.maximum(P, 0.0)
    return dict(
        corr=spotwise_corr(G, P, eps),
        cosine=spotwise_cos(G, P, eps),
        rmse=np.sqrt(((G - P) ** 2).mean(1)),
        jsd=spotwise_jsd(G, P, eps),
        ae=np.abs(G - P).mean(1)
    )
def compute_metrics(true_props, pred_props, eps=1e-12):
    """
    Compute summary metrics (standard definitions).
    
    Returns dict with:
    - rmse: Mean of spot-wise RMSE
    - jsd: Mean of spot-wise JSD (standard: JS_distance^2, NOT 0.5*JS_distance^2)
    - corr_spot: Mean of spot-wise Pearson correlation
    - corr_type: Mean of cell-type-wise Pearson correlation
    - cosine: Mean of spot-wise cosine similarity
    - aupr: Area Under Precision-Recall curve
    """
    true = np.asarray(true_props, dtype=np.float64)
    pred = np.asarray(pred_props, dtype=np.float64)
    true = np.maximum(true, 0.0)
    pred = np.maximum(pred, 0.0)
    mask = ~(np.isnan(true).any(axis=1) | np.isnan(pred).any(axis=1))
    true, pred = true[mask], pred[mask]
    n_spots, n_types = true.shape
    if n_spots == 0 or n_types == 0:
        return dict(rmse=np.nan, jsd=np.nan, corr_spot=np.nan, corr_type=np.nan, cosine=np.nan, aupr=np.nan)
    spot = compute_spot_metrics(true, pred, eps)
    ctcorr = celltype_corr(true, pred, eps)
    rmse = float(np.nanmean(spot['rmse']))
    jsd = float(np.nanmean(spot['jsd']))
    corr_spot = float(np.nanmean(spot['corr']))
    corr_type = float(np.nanmean(ctcorr))
    cosine = float(np.nanmean(spot['cosine']))
    y_true_bin = (true > 0).astype(np.int32).ravel()
    y_score = pred.ravel()
    pos, neg = y_true_bin.sum(), len(y_true_bin) - y_true_bin.sum()
    aupr = np.nan if pos == 0 or neg == 0 else float(average_precision_score(y_true_bin, y_score))
    return dict(rmse=rmse, jsd=jsd, corr_spot=corr_spot, corr_type=corr_type, cosine=cosine, aupr=aupr)
def compute_metrics_both(true_props, pred_props, cell_types, eps=1e-12):
    """
    Compute metrics for both all types and valid types only.
    
    Returns dict with:
    - metrics_all: metrics on all cell types
    - metrics_valid: metrics on valid types only (GT std > eps)
    - valid_types: list of valid cell types
    - invalid_types: list of invalid cell types (GT constant/zero)
    """
    true = np.asarray(true_props, dtype=np.float64)
    pred = np.asarray(pred_props, dtype=np.float64)
    valid_types = get_valid_types(true, cell_types, eps)
    invalid_types = [ct for ct in cell_types if ct not in valid_types]
    metrics_all = compute_metrics(true, pred, eps)
    if len(valid_types) == len(cell_types):
        metrics_valid = metrics_all
    else:
        valid_idx = [cell_types.index(ct) for ct in valid_types]
        true_valid = true[:, valid_idx]
        pred_valid = pred[:, valid_idx]
        metrics_valid = compute_metrics(true_valid, pred_valid, eps)
    return dict(
        metrics_all=metrics_all,
        metrics_valid=metrics_valid,
        valid_types=valid_types,
        invalid_types=invalid_types
    )
def compute_celltype_metrics(true_props, pred_props, cell_types, eps=1e-12):
    """Compute per-cell-type metrics."""
    true = np.asarray(true_props, dtype=np.float64)
    pred = np.asarray(pred_props, dtype=np.float64)
    results = {}
    for j, ct in enumerate(cell_types):
        t, p = true[:, j], pred[:, j]
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() < 3:
            results[ct] = dict(corr=np.nan, rmse=np.nan, mae=np.nan, gt_std=0.0)
            continue
        t, p = t[mask], p[mask]
        gt_std = float(np.std(t))
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        mae = float(np.mean(np.abs(t - p)))
        if gt_std > eps and np.std(p) > eps:
            corr = float(pearsonr(t, p)[0])
        else:
            corr = np.nan
        results[ct] = dict(corr=corr, rmse=rmse, mae=mae, gt_std=gt_std)
    return results
def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  RMSE:        {metrics['rmse']:.4f}")
    print(f"  JSD:         {metrics['jsd']:.4f}")
    print(f"  Corr(spot):  {metrics['corr_spot']:.4f}")
    print(f"  Corr(type):  {metrics['corr_type']:.4f}")
    print(f"  Cosine:      {metrics['cosine']:.4f}")
    print(f"  AUPR:        {metrics['aupr']:.4f}")
    print(f"{'='*55}")
def print_metrics_both(result, title_prefix=""):
    """Print both all-types and valid-types metrics."""
    if result['invalid_types']:
        print(f"\n  Note: {len(result['invalid_types'])} types excluded (GT constant): {result['invalid_types']}")
    print_metrics(result['metrics_all'], f"{title_prefix} (All {len(result['valid_types'])+len(result['invalid_types'])} types)")
    if result['invalid_types']:
        print_metrics(result['metrics_valid'], f"{title_prefix} (Valid {len(result['valid_types'])} types)")
def print_celltype_metrics(ct_metrics, top_n=None):
    """Print per-cell-type metrics sorted by correlation."""
    print(f"\n{'='*60}")
    print(f"  Per-cell-type Metrics")
    print(f"{'='*60}")
    sorted_items = sorted(ct_metrics.items(), key=lambda x: -x[1]['corr'] if np.isfinite(x[1]['corr']) else -999)
    if top_n:
        sorted_items = sorted_items[:top_n]
    for ct, m in sorted_items:
        if m['gt_std'] < 1e-6:
            print(f"  {ct:25s}: Corr=  N/A   (GT constant)")
        else:
            print(f"  {ct:25s}: Corr={m['corr']:.4f}  RMSE={m['rmse']:.4f}")
    print(f"{'='*60}")
def metrics_to_dataframe(results_dict):
    """Convert multiple methods' metrics to DataFrame."""
    return pd.DataFrame(results_dict).T
def load_data(data_dir):
    """Load data from standard directory structure with spot id alignment."""
    p = lambda fn: os.path.join(data_dir, f"{fn}.csv")
    cell_types = [x.strip() for x in open(os.path.join(data_dir, "eval_types.txt")) if len(x.strip()) > 0]
    sc_count = pd.read_csv(p("sc_count"), index_col=0)
    st_count = pd.read_csv(p("st_count"), index_col=0)
    spatial = pd.read_csv(p("spatial_location"), index_col=0)
    sc_meta = pd.read_csv(p("sc_meta"), index_col=0)
    true_fp = p("true_props")
    true_props = pd.read_csv(true_fp, index_col=0) if os.path.exists(true_fp) else None
    st_count.columns = st_count.columns.astype(str)
    spatial.index = spatial.index.astype(str)
    if true_props is not None:
        true_props.index = true_props.index.astype(str)
    common = set(st_count.columns).intersection(set(spatial.index))
    if true_props is not None:
        common = common.intersection(set(true_props.index))
    common = sorted(list(common))
    if len(common) == 0:
        raise ValueError("No common spot ids among st_count.columns, spatial.index, true_props.index")
    st_count = st_count.loc[:, common]
    spatial = spatial.loc[common]
    if true_props is not None:
        true_props = true_props.loc[common]
    return dict(sc_count=sc_count, st_count=st_count, spatial=spatial, sc_meta=sc_meta, cell_types=cell_types, true_props=true_props)
def align_genes(sc_count, st_count):
    """Align genes between scRNA-seq and ST data."""
    common = sc_count.index.intersection(st_count.index)
    return sc_count.loc[common], st_count.loc[common]
def set_seed(seed=42):
    """Set random seed for full reproducibility."""
    import random
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False