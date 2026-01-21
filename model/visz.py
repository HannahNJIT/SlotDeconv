"""
SlotDeconv Validation v3 - 改进配色版本
========================================
改进:
1. 使用scanpy风格的配色 (更好看)
2. 优化UMAP可视化
3. 添加legend
"""
from wsgiref.validate import validator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import scanpy as sc
import anndata as ad
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr, ttest_ind, hypergeom
from scipy.optimize import nnls
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
    
SCANPY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
    '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
]
NATURE_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
    '#E4E9B2', '#F9D423', '#FC913A', '#FF4E50', '#F9CDAD',
    '#C8C8A9', '#83AF9B', '#FE4365', '#FC9D9A', '#F9CDAD',
]
def get_palette(n, style='scanpy'):
    """获取配色方案"""
    if style == 'scanpy':
        colors = SCANPY_COLORS
    elif style == 'nature':
        colors = NATURE_COLORS
    else:
        colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    if n <= len(colors):
        return colors[:n]
    return [plt.cm.nipy_spectral(i / n) for i in range(n)]
def plot_validation_figures(validator, save_path=None, palette_style='scanpy'):
    """生成验证图表 - 改进配色版"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.rcParams['font.size'] = 11
    ax = axes[0, 0]
    B = validator.B_prob
    B_plot = B / (B.max(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(B_plot, cmap='magma', ax=ax, xticklabels=False,
                yticklabels=validator.cell_types if len(validator.cell_types) <= 20 else False,
                cbar_kws={'shrink': 0.6})
    ax.set_xlabel('Genes', fontsize=11)
    ax.set_ylabel('Cell Types', fontsize=11)
    ax.set_title('A. Learned Reference Matrix (B)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax = axes[0, 1]
    if 'B_separability' in validator.results:
        sim = validator.results['B_separability']['cos_sim_matrix']
        mask = np.triu(np.ones_like(sim, dtype=bool), k=0)
        sns.heatmap(sim, mask=mask, cmap='coolwarm', center=0.5, ax=ax, square=True,
                    xticklabels=validator.cell_types if len(validator.cell_types) <= 17 else False,
                    yticklabels=validator.cell_types if len(validator.cell_types) <= 17 else False,
                    vmin=0, vmax=1, cbar_kws={'shrink': 0.6})
        ax.tick_params(axis='both', labelsize=7, rotation=45)
    ax.set_title('B. Signature Similarity', fontsize=12, fontweight='bold')
    ax = axes[0, 2]
    if 'marker_de_validation' in validator.results:
        de_val = validator.results['marker_de_validation']
        cts = list(de_val.keys())
        overlaps = [de_val[ct]['overlap'] for ct in cts]
        sorted_idx = np.argsort(overlaps)[::-1]
        cts = [cts[i] for i in sorted_idx]
        overlaps = [overlaps[i] for i in sorted_idx]
        palette = get_palette(len(cts), palette_style)
        colors = [palette[validator.cell_types.index(ct)] if ct in validator.cell_types else '#888888' for ct in cts]
        y_pos = np.arange(len(cts))
        bars = ax.barh(y_pos, overlaps, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cts, fontsize=8)
        expected = de_val[cts[0]]['expected']
        ax.axvline(x=expected, color='#d62728', linestyle='--', linewidth=2, label=f'Expected ({expected:.1f})')
        ax.set_xlabel('Overlap with DE genes (top 50)', fontsize=10)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim(0, max(overlaps) * 1.1)
    ax.set_title('C. Marker Validation (DE-based)', fontsize=12, fontweight='bold')
    ax = axes[1, 0]
    if 'prototype_embedding' in validator.results:
        pe = validator.results['prototype_embedding']
        x = np.arange(2)
        width = 0.35
        sil = [pe['silhouette_original'], pe['silhouette_prototype']]
        ari = [pe['ari_original'], pe['ari_prototype']]
        bars1 = ax.bar(x - width/2, sil, width, label='Silhouette', color='#4DBBD5', edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, ari, width, label='ARI', color='#E64B35', edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars1, sil):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)
        for bar, val in zip(bars2, ari):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['Original\n(PCA)', 'Prototype\n(NNLS weights)'], fontsize=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, max(max(sil), max(ari)) * 1.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_title('D. Clustering Quality', fontsize=12, fontweight='bold')
    ax = axes[1, 1]
    if 'prototype_embedding' in validator.results:
        pe = validator.results['prototype_embedding']
        metrics = ['Silhouette', 'ARI']
        improvements = [pe['silhouette_improvement'], pe['ari_improvement']]
        colors = ['#00A087', '#3C5488']
        bars = ax.bar(metrics, improvements, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement (Δ)', fontsize=11)
        for bar, val in zip(bars, improvements):
            y_pos = val + 0.015 if val > 0 else val - 0.025
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'+{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(improvements) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_title('E. Clustering Improvement', fontsize=12, fontweight='bold')
    ax = axes[1, 2]
    if 'assignment_accuracy' in validator.results:
        acc = validator.results['assignment_accuracy']
        per_type = acc['per_type_accuracy']
        cts = list(per_type.keys())
        accs = [per_type[ct] for ct in cts]
        sorted_idx = np.argsort(accs)[::-1]
        cts = [cts[i] for i in sorted_idx]
        accs = [accs[i] for i in sorted_idx]
        palette = get_palette(len(cts), palette_style)
        colors = [palette[validator.cell_types.index(ct)] if ct in validator.cell_types else '#888888' for ct in cts]
        y_pos = np.arange(len(cts))
        bars = ax.barh(y_pos, accs, color=colors, edgecolor='white', linewidth=0.5)
        ax.axvline(x=acc['top1_accuracy'], color='#d62728', linestyle='--', linewidth=2, 
                   label=f"Mean = {acc['top1_accuracy']:.2f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cts, fontsize=8)
        ax.set_xlabel('Top-1 Accuracy', fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=9, loc='lower right')
    ax.set_title('F. Per-type Assignment Accuracy', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    return fig
def plot_umap_scanpy(validator, n_top_genes=2000, batch_key="source", use_harmony=True,
                        n_neighbors=15, min_dist=0.3, random_state=42, save_path=None,
                        palette_style='scanpy', point_size=8, alpha=0.8):
    """
    UMAP对比 - Scanpy风格 + 改进配色
    """
    import scanpy as sc
    import anndata as ad
    if "prototype_weights" not in validator.results:
        raise ValueError("Run analyze_prototype_embedding() first")
    if "prototype_sample_idx" not in validator.results:
        raise ValueError("prototype_sample_idx missing")
    idx = validator.results["prototype_sample_idx"]
    W = validator.results["prototype_weights"]
    y = validator.results["prototype_labels"]
    ct_names = [validator.cell_types[i] for i in y]
    palette = get_palette(len(validator.cell_types), palette_style)
    pal_map = {ct: palette[i] for i, ct in enumerate(validator.cell_types)}
    X = validator.sc_df.T.iloc[idx].to_numpy(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs = validator.sc_meta.iloc[idx].copy()
    adata.obs["cellType"] = adata.obs["cellType"].astype(str)
    if batch_key in adata.obs.columns:
        adata.obs[batch_key] = adata.obs[batch_key].astype(str).fillna("Unknown")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_genes_available = adata.n_vars
    if n_genes_available >= 200:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, n_genes_available), flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    rep = "X_pca"
    if use_harmony and batch_key in adata.obs.columns:
        try:
            sc.external.pp.harmony_integrate(adata, key=batch_key)
            rep = "X_pca_harmony"
        except:
            print("  Harmony failed, using PCA")
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    adataW = ad.AnnData(X=W.astype(np.float32))
    adataW.obs = pd.DataFrame({"cellType": ct_names})
    sc.pp.neighbors(adataW, use_rep="X", n_neighbors=min(n_neighbors, len(W)-1))
    sc.tl.umap(adataW, min_dist=min_dist, random_state=random_state)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (A, title) in zip(axes, [
        (adata, "A. Original scRNA-seq (Scanpy + Harmony)"),
        (adataW, "B. Prototype Embedding (NNLS weights)")
    ]):
        emb = A.obsm["X_umap"]
        cts = A.obs["cellType"].astype(str).values
        for ct in validator.cell_types:
            mask = cts == ct
            if mask.sum() > 0:
                ax.scatter(emb[mask, 0], emb[mask, 1], c=[pal_map[ct]], s=point_size, 
                          alpha=alpha, label=ct, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP1", fontsize=12)
        ax.set_ylabel("UMAP2", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [mpatches.Patch(facecolor=pal_map[ct], edgecolor='white', label=ct) 
               for ct in validator.cell_types]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.12, 0.5),
               fontsize=9, frameon=False, ncol=1 if len(validator.cell_types) <= 20 else 2)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"UMAP comparison saved to {save_path}")
    return fig
def plot_summary_stats(validator, save_path=None):
    """生成结果摘要统计图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    if 'B_separability' in validator.results:
        sep = validator.results['B_separability']
        sim_matrix = sep['cos_sim_matrix']
        n = len(validator.cell_types)
        off_diag = sim_matrix[~np.eye(n, dtype=bool)]
        ax.hist(off_diag, bins=30, color='#4DBBD5', edgecolor='white', alpha=0.8)
        ax.axvline(x=sep['cos_mean'], color='#E64B35', linestyle='--', linewidth=2, 
                   label=f"Mean = {sep['cos_mean']:.3f}")
        ax.axvline(x=sep['cos_max'], color='#00A087', linestyle=':', linewidth=2,
                   label=f"Max = {sep['cos_max']:.3f}")
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('A. Signature Similarity Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax = axes[1]
    if 'marker_de_validation' in validator.results:
        de_val = validator.results['marker_de_validation']
        enrichments = [v['enrichment'] for v in de_val.values()]
        ax.hist(enrichments, bins=20, color='#00A087', edgecolor='white', alpha=0.8)
        ax.axvline(x=np.mean(enrichments), color='#E64B35', linestyle='--', linewidth=2,
                   label=f"Mean = {np.mean(enrichments):.1f}x")
        ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, label='Random (1x)')
        ax.set_xlabel('Enrichment (fold over random)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('B. Marker Enrichment Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax = axes[2]
    if 'assignment_accuracy' in validator.results:
        acc = validator.results['assignment_accuracy']
        per_type = list(acc['per_type_accuracy'].values())
        ax.hist(per_type, bins=20, color='#3C5488', edgecolor='white', alpha=0.8)
        ax.axvline(x=acc['top1_accuracy'], color='#E64B35', linestyle='--', linewidth=2,
                   label=f"Mean = {acc['top1_accuracy']:.2f}")
        ax.set_xlabel('Top-1 Accuracy', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('C. Per-type Accuracy Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Summary stats saved to {save_path}")
    return fig
def print_validation_summary(validator):
    """打印验证结果摘要 (用于论文)"""
    print("\n" + "=" * 60)
    print("SlotDeconv Validation Summary (for ISMB paper)")
    print("=" * 60)
    if 'B_separability' in validator.results:
        sep = validator.results['B_separability']
        print(f"\n📊 B-matrix Separability:")
        print(f"   Cosine similarity (off-diagonal): mean={sep['cos_mean']:.3f}, max={sep['cos_max']:.3f}")
    if 'marker_de_validation' in validator.results:
        de_val = validator.results['marker_de_validation']
        enrichments = [v['enrichment'] for v in de_val.values()]
        sig_count = sum(1 for v in de_val.values() if v['p_value'] < 0.05)
        print(f"\n🧬 Marker Validation (DE-based):")
        print(f"   Avg enrichment: {np.mean(enrichments):.1f}x over random")
        print(f"   Significant (p<0.05): {sig_count}/{len(de_val)} cell types ({100*sig_count/len(de_val):.0f}%)")
    if 'prototype_embedding' in validator.results:
        pe = validator.results['prototype_embedding']
        print(f"\n📈 Clustering Quality (Prototype Embedding):")
        print(f"   Silhouette: {pe['silhouette_original']:.3f} → {pe['silhouette_prototype']:.3f} (Δ=+{pe['silhouette_improvement']:.3f})")
        print(f"   ARI: {pe['ari_original']:.3f} → {pe['ari_prototype']:.3f} (Δ=+{pe['ari_improvement']:.3f})")
    if 'assignment_accuracy' in validator.results:
        acc = validator.results['assignment_accuracy']
        print(f"\n🎯 Assignment Accuracy:")
        print(f"   Top-1 accuracy: {acc['top1_accuracy']:.3f}")
        per_type = acc['per_type_accuracy']
        print(f"   Range: [{min(per_type.values()):.3f}, {max(per_type.values()):.3f}]")
    print("\n" + "=" * 60)
# 快速修复：直接在notebook中运行这个修正版函数

def plot_umap_scanpy(validator, n_top_genes=2000, batch_key="source", use_harmony=True,
                        n_neighbors=15, min_dist=0.3, random_state=42, save_path=None,
                        palette_style='scanpy', point_size=8, alpha=0.8):
    """UMAP对比 - Scanpy风格 + 改进配色 (修复版)"""
    SCANPY_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
        '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    ]
    def get_palette(n, style='scanpy'):
        if style == 'scanpy':
            colors = SCANPY_COLORS
        else:
            colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
        if n <= len(colors):
            return colors[:n]
        return [plt.cm.nipy_spectral(i / n) for i in range(n)]
    if "prototype_weights" not in validator.results:
        raise ValueError("Run analyze_prototype_embedding() first")
    if "prototype_sample_idx" not in validator.results:
        raise ValueError("prototype_sample_idx missing")
    idx = validator.results["prototype_sample_idx"]
    W = validator.results["prototype_weights"]
    y = validator.results["prototype_labels"]
    ct_names = [validator.cell_types[i] for i in y]
    palette = get_palette(len(validator.cell_types), palette_style)
    pal_map = {ct: palette[i] for i, ct in enumerate(validator.cell_types)}
    X = validator.sc_df.T.iloc[idx].to_numpy(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs = validator.sc_meta.iloc[idx].copy()
    adata.obs["cellType"] = adata.obs["cellType"].astype(str)
    if batch_key in adata.obs.columns:
        adata.obs[batch_key] = adata.obs[batch_key].astype(str).fillna("Unknown")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_genes_available = int(adata.shape[1])
    if n_genes_available >= 200:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(int(n_top_genes), n_genes_available), flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception as e:
            print(f"  HVG selection failed: {e}, using all genes")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    rep = "X_pca"
    if use_harmony and batch_key in adata.obs.columns:
        try:
            sc.external.pp.harmony_integrate(adata, key=batch_key)
            rep = "X_pca_harmony"
        except Exception as e:
            print(f"  Harmony failed: {e}, using PCA")
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    adataW = ad.AnnData(X=W.astype(np.float32))
    adataW.obs = pd.DataFrame({"cellType": ct_names})
    sc.pp.neighbors(adataW, use_rep="X", n_neighbors=min(n_neighbors, len(W)-1))
    sc.tl.umap(adataW, min_dist=min_dist, random_state=random_state)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (A, title) in zip(axes, [
        (adata, "A. Original scRNA-seq (Scanpy + Harmony)"),
        (adataW, "B. Prototype Embedding (NNLS weights)")
    ]):
        emb = A.obsm["X_umap"]
        cts = A.obs["cellType"].astype(str).values
        for ct in validator.cell_types:
            mask = cts == ct
            if mask.sum() > 0:
                ax.scatter(emb[mask, 0], emb[mask, 1], c=[pal_map[ct]], s=point_size, 
                          alpha=alpha, label=ct, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP1", fontsize=12)
        ax.set_ylabel("UMAP2", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [mpatches.Patch(facecolor=pal_map[ct], edgecolor='white', label=ct) 
               for ct in validator.cell_types]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.12, 0.5),
               fontsize=9, frameon=False, ncol=1 if len(validator.cell_types) <= 20 else 2)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"UMAP comparison saved to {save_path}")
    return fig
def _ismb_palette(n):
    cols=list(plt.cm.tab20.colors)+list(plt.cm.tab20b.colors)+list(plt.cm.tab20c.colors)
    cols=cols[:n]
    return [tuple(c) for c in cols]
def plot_top2_confusion(rows,cell_types,save_path=None,top_m=5):
    pal=_ismb_palette(len(cell_types))
    labs=[];vals=[];cs=[]
    for r in rows:
        ct=r["cell_type"]
        for t,c,fr in r["top2"][:top_m]:
            labs.append(f"{ct}→{t}")
            vals.append(fr)
            cs.append(pal[cell_types.index(t)] if t in cell_types else (0.3,0.3,0.3))
    plt.figure(figsize=(9,0.28*len(vals)+1.4))
    y=np.arange(len(vals))[::-1]
    plt.barh(y,vals,color=cs,edgecolor="black",linewidth=0.6)
    plt.yticks(y,labs,fontsize=8)
    plt.xlim(0,1)
    plt.xlabel("Fraction within true type")
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path,dpi=300,bbox_inches="tight")
    return plt.gcf()
def plot_umap_scanpy_comparison(validator,n_top_genes=2000,batch_key="source",use_harmony=True,n_neighbors=15,min_dist=0.3,random_state=42,save_path=None):
    import scanpy as sc
    import anndata as ad
    if "prototype_weights" not in validator.results or "prototype_labels" not in validator.results: raise ValueError("Run analyze_prototype_embedding() first")
    if "prototype_sample_idx" not in validator.results: raise ValueError("prototype_sample_idx missing; modify analyze_prototype_embedding to store sampled indices then rerun")
    idx=validator.results["prototype_sample_idx"]
    W=validator.results["prototype_weights"]
    y=validator.results["prototype_labels"]
    ct_names=[validator.cell_types[i] for i in y]
    pal=_ismb_palette(len(validator.cell_types))
    pal_map={ct:pal[i] for i,ct in enumerate(validator.cell_types)}
    X=validator.sc_df.T.iloc[idx].to_numpy(np.float32)
    adata=ad.AnnData(X=X)
    adata.obs=validator.sc_meta.iloc[idx].copy()
    adata.obs["cellType"]=adata.obs["cellType"].astype(str)
    if batch_key in adata.obs.columns:
        adata.obs[batch_key]=adata.obs[batch_key].astype(str).fillna("Unknown")
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata,n_top_genes=min(n_top_genes,adata.n_vars),flavor="seurat_v3")
    adata=adata[:,adata.var["highly_variable"]].copy()
    sc.pp.scale(adata,max_value=10)
    sc.tl.pca(adata,svd_solver="arpack")
    rep="X_pca"
    if use_harmony and batch_key in adata.obs.columns:
        sc.external.pp.harmony_integrate(adata,key=batch_key)
        rep="X_pca_harmony"
    sc.pp.neighbors(adata,use_rep=rep,n_neighbors=n_neighbors)
    sc.tl.umap(adata,min_dist=min_dist,random_state=random_state)
    adataW=ad.AnnData(X=W.astype(np.float32))
    adataW.obs=pd.DataFrame({"cellType":ct_names})
    sc.pp.neighbors(adataW,use_rep="X",n_neighbors=n_neighbors)
    sc.tl.umap(adataW,min_dist=min_dist,random_state=random_state)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,(A,title) in zip(axes,[(adata,"A. Original scRNA-seq (Scanpy pipeline)"),(adataW,"B. Prototype Embedding (NNLS weights)")]):
        emb=A.obsm["X_umap"]
        c=[pal_map[t] for t in A.obs["cellType"].astype(str).values]
        ax.scatter(emb[:,0],emb[:,1],c=c,s=6,alpha=0.75,linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_xlabel("UMAP1");ax.set_ylabel("UMAP2")
    if len(validator.cell_types)<=15:
        handles=[plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=pal_map[ct],markersize=8) for ct in validator.cell_types]
        fig.legend(handles,validator.cell_types,loc="center right",bbox_to_anchor=(1.15,0.5),frameon=False)
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path,dpi=300,bbox_inches="tight")
    return fig
class SlotDeconvValidator:
    """严谨的SlotDeconv验证实验"""
    def __init__(self, model, sc_df, sc_meta, cell_types, gene_names=None):
        self.model = model
        self.sc_df = sc_df
        self.sc_meta = sc_meta
        self.cell_types = cell_types
        self.gene_names = gene_names if gene_names is not None else list(sc_df.index)
        self.B_prob = model.B_prob_
        self.results = {}
    def run_all_validations(self, save_dir=None):
        """运行所有验证实验"""
        print("=" * 60)
        print("SlotDeconv Validation (ISMB Ready)")
        print("=" * 60)
        print("\n[1/5] B-matrix Separability...")
        self.analyze_B_separability()
        print("\n[2/5] Self-consistent Marker Validation (DE-based)...")
        self.validate_markers_de_based(top_k=50)
        print("\n[3/5] Prototype-based Embedding...")
        self.analyze_prototype_embedding()
        print("\n[4/5] Top-1 Assignment Accuracy...")
        self.analyze_assignment_accuracy()
        print("\n[5/5] Slot Embeddings (if available)...")
        self.analyze_slot_embeddings()
        if save_dir:
            self.save_results(save_dir)
        return self.results
    def analyze_B_separability(self):
        """
        证据A: B矩阵的可分离性
        展示不同cell type有distinct的signatures
        """
        from sklearn.metrics.pairwise import cosine_similarity
        B = self.B_prob
        cos_sim = cosine_similarity(B)
        corr_sim = np.corrcoef(B)
        n = len(self.cell_types)
        mask = ~np.eye(n, dtype=bool)
        off_diag_cos = cos_sim[mask]
        off_diag_corr = corr_sim[mask]
        self.results['B_separability'] = {
            'cos_mean': off_diag_cos.mean(),
            'cos_max': off_diag_cos.max(),
            'cos_median': np.median(off_diag_cos),
            'corr_mean': off_diag_corr.mean(),
            'corr_max': off_diag_corr.max(),
            'cos_sim_matrix': cos_sim,
            'corr_sim_matrix': corr_sim,
        }
        print(f"  Cosine similarity (off-diag): mean={off_diag_cos.mean():.3f}, max={off_diag_cos.max():.3f}")
        print(f"  Correlation (off-diag): mean={off_diag_corr.mean():.3f}, max={off_diag_corr.max():.3f}")
        return self.results['B_separability']
    def validate_markers_de_based(self, top_k=50):
        """
        证据B: 自洽Marker验证 (不需要外部数据库)
        方法: 在scRNA上做DE, 看B的top genes是否富集到DE genes
        """
        X = self.sc_df.values  # genes x cells
        ct_map = {ct: i for i, ct in enumerate(self.cell_types)}
        labels = np.array([ct_map.get(ct, -1) for ct in self.sc_meta['cellType'].values])
        valid_mask = labels >= 0
        X = X[:, valid_mask]
        labels = labels[valid_mask]
        de_ranks = {}
        for i, ct in enumerate(self.cell_types):
            in_group = labels == i
            out_group = ~in_group
            if in_group.sum() < 3 or out_group.sum() < 3:
                continue
            mean_in = X[:, in_group].mean(axis=1) + 1e-8
            mean_out = X[:, out_group].mean(axis=1) + 1e-8
            logfc = np.log2(mean_in / mean_out)
            de_ranks[ct] = np.argsort(logfc)[::-1]
        B = self.B_prob
        B_specificity = B / (B.sum(axis=0, keepdims=True) + 1e-8)
        validation_results = {}
        for i, ct in enumerate(self.cell_types):
            if ct not in de_ranks:
                continue
            de_top = set(de_ranks[ct][:top_k])
            b_top_idx = np.argsort(B_specificity[i])[::-1][:top_k]
            b_top = set(b_top_idx)
            overlap = len(de_top & b_top)
            n_genes = len(self.gene_names)
            p_val = hypergeom.sf(overlap - 1, n_genes, top_k, top_k)
            validation_results[ct] = {
                'overlap': overlap,
                'expected': top_k * top_k / n_genes,
                'enrichment': overlap / (top_k * top_k / n_genes + 1e-8),
                'p_value': p_val,
                'top_b_genes': [self.gene_names[j] for j in b_top_idx[:10]],
            }
        self.results['marker_de_validation'] = validation_results
        if validation_results:
            avg_overlap = np.mean([v['overlap'] for v in validation_results.values()])
            avg_enrichment = np.mean([v['enrichment'] for v in validation_results.values()])
            sig_count = sum(1 for v in validation_results.values() if v['p_value'] < 0.05)
            print(f"  Avg overlap@{top_k}: {avg_overlap:.1f} genes")
            print(f"  Avg enrichment: {avg_enrichment:.1f}x over random")
            print(f"  Significant (p<0.05): {sig_count}/{len(validation_results)} cell types")
        return validation_results
    def analyze_prototype_embedding(self, n_sample=3000):
        """
        证据C: Prototype-based Embedding
        用NNLS得到每个细胞的K维权重w, 在w空间做UMAP/聚类
        比硬替换B[labels]更严谨
        """
        X = self.sc_df.T.values.astype(np.float32)
        X[X < 0] = 0
        ct_map = {ct: i for i, ct in enumerate(self.cell_types)}
        labels = []
        valid_idx = []
        for i, ct in enumerate(self.sc_meta['cellType'].values):
            if ct in ct_map:
                labels.append(ct_map[ct])
                valid_idx.append(i)
        labels = np.array(labels)
        X = X[valid_idx]
        if len(X)>n_sample:
            np.random.seed(42)
            idx=np.random.choice(len(X),n_sample,replace=False)
            X=X[idx]
            labels=labels[idx]
            self.results["prototype_sample_idx"]=np.array(valid_idx,dtype=np.int64)[idx]
        else:
            self.results["prototype_sample_idx"]=np.array(valid_idx,dtype=np.int64)
        X_norm = X / (X.sum(axis=1, keepdims=True) + 1e-8)
        B = self.B_prob
        W = np.zeros((len(X), len(self.cell_types)))
        for i in range(len(X)):
            w, _ = nnls(B.T, X_norm[i])
            w = w / (w.sum() + 1e-8)
            W[i] = w
        self.results['prototype_weights'] = W
        self.results['prototype_labels'] = labels
        sil_w = silhouette_score(W, labels)
        pca_orig = PCA(n_components=min(50, X_norm.shape[1]))
        X_pca = pca_orig.fit_transform(X_norm)
        sil_orig = silhouette_score(X_pca, labels)
        km = KMeans(n_clusters=len(self.cell_types), random_state=42, n_init=10)
        pred_w = km.fit_predict(W)
        pred_orig = km.fit_predict(X_pca)
        ari_w = adjusted_rand_score(labels, pred_w)
        ari_orig = adjusted_rand_score(labels, pred_orig)
        self.results['prototype_embedding'] = {
            'silhouette_original': sil_orig,
            'silhouette_prototype': sil_w,
            'silhouette_improvement': sil_w - sil_orig,
            'ari_original': ari_orig,
            'ari_prototype': ari_w,
            'ari_improvement': ari_w - ari_orig,
        }
        print(f"  Silhouette: original={sil_orig:.3f} → prototype={sil_w:.3f} (Δ={sil_w-sil_orig:+.3f})")
        print(f"  ARI: original={ari_orig:.3f} → prototype={ari_w:.3f} (Δ={ari_w-ari_orig:+.3f})")
        return self.results['prototype_embedding']
    def analyze_assignment_accuracy(self, n_sample=5000):
        """
        Top-1 Assignment Accuracy
        检查argmax(w)是否等于真实cell type标签
        """
        X = self.sc_df.T.values.astype(np.float32)
        X[X < 0] = 0
        ct_map = {ct: i for i, ct in enumerate(self.cell_types)}
        labels = []
        valid_idx = []
        for i, ct in enumerate(self.sc_meta['cellType'].values):
            if ct in ct_map:
                labels.append(ct_map[ct])
                valid_idx.append(i)
        labels = np.array(labels)
        X = X[valid_idx]
        if len(X) > n_sample:
            idx = np.random.choice(len(X), n_sample, replace=False)
            X = X[idx]
            labels = labels[idx]
        X_norm = X / (X.sum(axis=1, keepdims=True) + 1e-8)
        B = self.B_prob
        W = np.zeros((len(X), len(self.cell_types)))
        for i in range(len(X)):
            w, _ = nnls(B.T, X_norm[i])
            W[i] = w
        pred_labels = W.argmax(axis=1)
        top1_acc = (pred_labels == labels).mean()
        per_type_acc = {}
        for i, ct in enumerate(self.cell_types):
            mask = labels == i
            if mask.sum() > 0:
                per_type_acc[ct] = (pred_labels[mask] == i).mean()
        self.results['assignment_accuracy'] = {
            'top1_accuracy': top1_acc,
            'per_type_accuracy': per_type_acc,
        }
        print(f"  Top-1 accuracy: {top1_acc:.3f}")
        print(f"  Per-type range: [{min(per_type_acc.values()):.3f}, {max(per_type_acc.values()):.3f}]")
        return self.results['assignment_accuracy']
    def analyze_slot_embeddings(self):
        """分析Slot Embeddings (如果有)"""
        if not hasattr(self.model, 'slots_') or self.model.slots_ is None:
            print("  [Skip] slots_ not stored in model")
            print("  To enable: add 'self.slots_ = model.slots.detach().cpu().numpy()' in _train_reference")
            return None
        slots = self.model.slots_
        slot_sim = np.corrcoef(slots)
        b_sim = np.corrcoef(self.B_prob)
        corr, _ = pearsonr(slot_sim.flatten(), b_sim.flatten())
        self.results['slot_embedding'] = {
            'slot_B_correlation': corr,
            'slot_similarity': slot_sim,
        }
        print(f"  Slot-B similarity correlation: {corr:.3f}")
        return self.results['slot_embedding']
    def get_marker_df(self, top_k=50):
        """导出marker genes为DataFrame"""
        B = self.B_prob
        B_spec = B / (B.sum(axis=0, keepdims=True) + 1e-8)
        rows = []
        for i, ct in enumerate(self.cell_types):
            top_idx = np.argsort(B_spec[i])[::-1][:top_k]
            for rank, j in enumerate(top_idx):
                rows.append({
                    'cell_type': ct,
                    'rank': rank + 1,
                    'gene': self.gene_names[j],
                    'specificity': B_spec[i, j],
                    'expression': B[i, j],
                })
        return pd.DataFrame(rows)
    def save_results(self, save_dir):
        """保存结果"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        marker_df = self.get_marker_df(top_k=50)
        marker_df.to_csv(f"{save_dir}/learned_markers.csv", index=False)
        b_df = pd.DataFrame(self.B_prob, index=self.cell_types, columns=self.gene_names)
        b_df.to_csv(f"{save_dir}/B_matrix.csv")
        if 'marker_de_validation' in self.results:
            de_df = pd.DataFrame(self.results['marker_de_validation']).T
            de_df.to_csv(f"{save_dir}/marker_de_validation.csv")
        summary = {}
        if 'B_separability' in self.results:
            summary['B_cos_mean'] = self.results['B_separability']['cos_mean']
            summary['B_cos_max'] = self.results['B_separability']['cos_max']
        if 'prototype_embedding' in self.results:
            for k, v in self.results['prototype_embedding'].items():
                summary[k] = v
        if 'assignment_accuracy' in self.results:
            summary['top1_accuracy'] = self.results['assignment_accuracy']['top1_accuracy']
        pd.Series(summary).to_csv(f"{save_dir}/validation_summary.csv")
        print(f"\n  Results saved to {save_dir}/")
def plot_umap_comparison(validator, n_sample=3000, save_path=None):
    """
    UMAP对比: 原始scRNA vs Prototype weights
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. pip install umap-learn")
        return None
    if 'prototype_weights' not in validator.results:
        print("Run analyze_prototype_embedding() first")
        return None
    W = validator.results['prototype_weights']
    labels = validator.results['prototype_labels']
    X = validator.sc_df.T.values.astype(np.float32)
    X[X < 0] = 0
    ct_map = {ct: i for i, ct in enumerate(validator.cell_types)}
    valid_idx = []
    for i, ct in enumerate(validator.sc_meta['cellType'].values):
        if ct in ct_map:
            valid_idx.append(i)
    X = X[valid_idx]
    if len(X) > n_sample:
        np.random.seed(42)
        idx = np.random.choice(len(X), n_sample, replace=False)
        X = X[idx]
    X_norm = X / (X.sum(axis=1, keepdims=True) + 1e-8)
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_norm)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_orig = reducer.fit_transform(X_pca)
    umap_proto = reducer.fit_transform(W)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_types = len(validator.cell_types)
    cmap = 'tab20' if n_types <= 20 else 'nipy_spectral'
    ax = axes[0]
    scatter = ax.scatter(umap_orig[:, 0], umap_orig[:, 1], c=labels, cmap=cmap, s=5, alpha=0.6)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('A. Original scRNA-seq (PCA → UMAP)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    scatter = ax.scatter(umap_proto[:, 0], umap_proto[:, 1], c=labels, cmap=cmap, s=5, alpha=0.6)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('B. Prototype Embedding (NNLS weights → UMAP)')
    ax.set_xticks([])
    ax.set_yticks([])
    if n_types <= 15:
        cmap_obj = plt.cm.get_cmap(cmap)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_obj(i/n_types), markersize=8)
                   for i in range(n_types)]
        fig.legend(handles, validator.cell_types, loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP comparison saved to {save_path}")
    return fig
