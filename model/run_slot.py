"""
SlotDeconv: Run script for spatial transcriptomics deconvolution.

Usage:
    python run_slot.py --data_dir /path/to/data --output_dir /path/to/output
"""
import argparse
import os
import numpy as np
import pandas as pd
from slot_model import SlotDeconv, DEFAULT_CONFIG
from slot_utility import (load_data, align_genes, compute_metrics, compute_metrics_both,
                          compute_celltype_metrics, print_metrics, print_metrics_both,
                          print_celltype_metrics, metrics_to_dataframe)
def run_slotdeconv(data_dir, output_dir=None, use_spatial=True, seed=42, verbose=True, fit_kwargs=None, transform_kwargs=None):
    """
    Run SlotDeconv with default configuration.
    
    Returns dict with predictions and metrics (both all-types and valid-types).
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    if transform_kwargs is None:
        transform_kwargs = {}
    dat = load_data(data_dir)
    sc_count, st_count = align_genes(dat["sc_count"], dat["st_count"])
    model = SlotDeconv(random_state=seed, verbose=verbose, use_default_config=True)
    model.fit(sc_count, dat["sc_meta"], dat["cell_types"], **fit_kwargs)
    pred_nnls = model.transform(st_count, dat["spatial"], use_spatial=False, **transform_kwargs)
    pred = model.transform(st_count, dat["spatial"], use_spatial=use_spatial, **transform_kwargs) if use_spatial else pred_nnls
    result = {'pred': pred, 'pred_nnls': pred_nnls, 'model': model}
    if verbose:
        cfg = model.get_config()
        last_fit = cfg.get("last_fit", None)
        last_tr = cfg.get("last_transform", None)
        print(f"\n{'='*60}")
        print(f"  SlotDeconv: Spatial Transcriptomics Deconvolution")
        print(f"{'='*60}")
        print(f"  Data: {data_dir}")
        if isinstance(last_fit, dict):
            print(f"  Fit: n_genes={last_fit.get('n_genes')} max_cells={last_fit.get('max_cells_per_type')} λ_div={last_fit.get('lambda_div')} margin={last_fit.get('margin')}")
        if isinstance(last_tr, dict):
            print(f"  Transform: λ_sp={last_tr.get('lambda_sp')} backend={last_tr.get('spatial_backend_used')} gene_w={last_tr.get('use_gene_weight_in_kl')} mixed={last_tr.get('use_mixed_loss')}")
        print(f"{'='*60}\n")
        print(f"Data: {sc_count.shape[0]} genes, {sc_count.shape[1]} cells, {st_count.shape[1]} spots, {len(dat['cell_types'])} types\n")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred.to_csv(os.path.join(output_dir, "predicted_props.csv"))
        pred_nnls.to_csv(os.path.join(output_dir, "predicted_props_nnls.csv"))
        if verbose:
            print(f"\nPredictions saved to {output_dir}/")
    if 'true_props' in dat and dat['true_props'] is not None:
        true = dat["true_props"].loc[st_count.columns, dat["cell_types"]]
        result_nnls = compute_metrics_both(true, pred_nnls, dat["cell_types"])
        result_spatial = compute_metrics_both(true, pred, dat["cell_types"])
        ct_metrics = compute_celltype_metrics(true.values, pred.values, dat["cell_types"])
        result.update({
            'metrics_nnls': result_nnls['metrics_all'],
            'metrics_spatial': result_spatial['metrics_all'],
            'metrics_nnls_valid': result_nnls['metrics_valid'],
            'metrics_spatial_valid': result_spatial['metrics_valid'],
            'valid_types': result_spatial['valid_types'],
            'invalid_types': result_spatial['invalid_types'],
            'ct_metrics': ct_metrics,
            'true': true
        })
        if verbose:
            print_metrics_both(result_nnls, "NNLS (without spatial)")
            print_metrics_both(result_spatial, "SlotDeconv (with spatial)")
            m_nnls = result_nnls['metrics_valid']
            m_sp = result_spatial['metrics_valid']
            base = float(m_nnls.get('corr_spot', np.nan))
            new = float(m_sp.get('corr_spot', np.nan))
            if np.isfinite(base) and np.isfinite(new) and abs(base) > 1e-12:
                improvement = (new - base) / abs(base) * 100.0
                print(f"\n  Improvement (valid types): +{improvement:.1f}%\n")
            else:
                print(f"\n  Improvement (valid types): N/A\n")
            print_celltype_metrics(ct_metrics, top_n=10)
        if output_dir:
            pd.DataFrame([result_spatial['metrics_all']]).to_csv(os.path.join(output_dir, "metrics_all.csv"), index=False)
            pd.DataFrame([result_spatial['metrics_valid']]).to_csv(os.path.join(output_dir, "metrics_valid.csv"), index=False)
            pd.DataFrame(ct_metrics).T.to_csv(os.path.join(output_dir, "celltype_metrics.csv"))
    return result
def main():
    parser = argparse.ArgumentParser(description="Run SlotDeconv")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save results")
    parser.add_argument("--no_spatial", action="store_true", help="Disable spatial refinement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_slotdeconv(args.data_dir, args.output_dir, use_spatial=not args.no_spatial, seed=args.seed)
if __name__ == "__main__":
    main()