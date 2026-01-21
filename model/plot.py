import os,argparse,numpy as np,pandas as pd,matplotlib.pyplot as plt,scanpy as sc
from scipy.optimize import nnls
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from slot_model import SlotDeconv,set_seed
from slot_utility import load_data,align_genes
def _mkdir(p): 
    os.makedirs(p,exist_ok=True)
def _norm_log1p_df(X):
    X=X.copy()
    lib=np.maximum(X.sum(1).values.astype(np.float64),1e-8)
    X=(X.T/lib).T*1e4
    X=np.log1p(X)
    return X
def _mantel_spearman(D1,D2,n_perm=200,seed=0):
    rng=np.random.RandomState(seed)
    v1=D1[np.triu_indices_from(D1,k=1)]
    v2=D2[np.triu_indices_from(D2,k=1)]
    r0=float(spearmanr(v1,v2).correlation)
    cnt=0
    n=D1.shape[0]
    for _ in range(int(n_perm)):
        p=rng.permutation(n)
        D2p=D2[p][:,p]
        v2p=D2p[np.triu_indices_from(D2p,k=1)]
        rp=float(spearmanr(v1,v2p).correlation)
        if np.isfinite(rp) and rp>=r0: cnt+=1
    pval=(cnt+1)/(n_perm+1)
    return r0,pval
def _knn_overlap(A,B,k=15):
    nnA=NearestNeighbors(n_neighbors=min(k+1,A.shape[0])).fit(A)
    nnB=NearestNeighbors(n_neighbors=min(k+1,B.shape[0])).fit(B)
    IA=nnA.kneighbors(A,return_distance=False)[:,1:]
    IB=nnB.kneighbors(B,return_distance=False)[:,1:]
    ov=[]
    for i in range(A.shape[0]):
        sa=set(IA[i].tolist());sb=set(IB[i].tolist())
        ov.append(len(sa&sb)/max(len(sa),1))
    return float(np.mean(ov))
def _markers_by_logfc(X,labels,ct_names,top_k=50):
    Xn=_norm_log1p_df(X)
    genes=Xn.columns.tolist()
    y=labels.astype(int)
    out={}
    for i,ct in enumerate(ct_names):
        m=y==i
        if m.sum()<3 or (~m).sum()<3: continue
        mu_in=Xn.loc[m].mean(0).values
        mu_out=Xn.loc[~m].mean(0).values
        logfc=mu_in-mu_out
        idx=np.argsort(-logfc)[:min(top_k,len(logfc))]
        out[ct]=[genes[j] for j in idx]
    return out
def _marker_enrichment_overlap(markers_a,markers_b,universe_n,top_k=50):
    rows=[]
    exp=top_k*top_k/max(universe_n,1)
    for ct in markers_a:
        if ct not in markers_b: continue
        a=set(markers_a[ct][:top_k]);b=set(markers_b[ct][:top_k])
        ov=len(a&b)
        rows.append((ct,ov,exp,ov/(exp+1e-8)))
    df=pd.DataFrame(rows,columns=["cell_type",f"overlap@{top_k}","expected","enrichment_x"])
    return df
def _compute_W_nnls(sc_sub,B,cell_types,n_cells=5000,seed=0):
    rng=np.random.RandomState(seed)
    cells=sc_sub.columns.astype(str).tolist()
    if len(cells)>n_cells:
        pick=rng.choice(len(cells),size=n_cells,replace=False)
        cells=[cells[i] for i in pick]
    X=sc_sub[cells].T.values.astype(np.float32)
    X=np.maximum(X,0)
    Xn=X/(X.sum(1,keepdims=True)+1e-8)
    W=np.zeros((Xn.shape[0],len(cell_types)),dtype=np.float32)
    Bt=B.T.astype(np.float64)
    for i in range(Xn.shape[0]):
        w,_=nnls(Bt,Xn[i].astype(np.float64))
        s=w.sum()
        if s<=0: w=np.ones(len(cell_types),dtype=np.float64)/len(cell_types)
        else: w=w/s
        W[i]=w.astype(np.float32)
    return cells,W
def _plot_spatial_gene(coords,vals,title,fp):
    x=coords[:,0];y=coords[:,1]
    plt.figure(figsize=(5.2,4.6))
    plt.scatter(x,y,c=vals,s=6,alpha=0.9,linewidths=0,cmap="viridis")
    plt.xticks([]);plt.yticks([])
    plt.title(title,fontsize=12)
    plt.tight_layout()
    plt.savefig(fp,dpi=300,bbox_inches="tight")
    plt.close()
def _auto_pick_marker_genes(B,genes,cell_types,n_each=1,max_genes=8):
    Bspec=B/(B.sum(0,keepdims=True)+1e-8)
    picks=[]
    for i,ct in enumerate(cell_types):
        idx=int(np.argmax(Bspec[i]))
        picks.append(genes[idx])
    uniq=[]
    for g in picks:
        if g not in uniq: uniq.append(g)
    return uniq[:max_genes]
def _marker_auc(X,labels,markers,ct_names):
    Xn=_norm_log1p_df(X)
    y=labels.astype(int)
    res=[]
    for i,ct in enumerate(ct_names):
        m=y==i
        if m.sum()<5 or (~m).sum()<5: continue
        gs=[g for g in markers.get(ct,[]) if g in Xn.columns]
        if len(gs)==0: continue
        aucs=[]
        yy=m.astype(int)
        for g in gs:
            s=Xn[g].values
            if np.std(s)<1e-10: continue
            try:
                a=roc_auc_score(yy,s)
                if np.isfinite(a): aucs.append(float(a))
            except Exception:
                pass
        if len(aucs)>0: res.append((ct,float(np.mean(aucs)),len(aucs)))
    return pd.DataFrame(res,columns=["cell_type","mean_auc","n_genes"])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,required=True)
    ap.add_argument("--output_dir",type=str,required=True)
    ap.add_argument("--device",type=str,default=None)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--n_sc",type=int,default=5000)
    ap.add_argument("--n_perm",type=int,default=200)
    ap.add_argument("--k_nn",type=int,default=15)
    ap.add_argument("--marker_genes",type=str,default="")
    args=ap.parse_args()
    set_seed(args.seed)
    _mkdir(args.output_dir)
    dat=load_data(args.data_dir)
    sc_count,st_count=align_genes(dat["sc_count"],dat["st_count"])
    model=SlotDeconv(device=args.device,random_state=args.seed,verbose=True,use_default_config=True)
    model.fit(sc_count,dat["sc_meta"],dat["cell_types"])
    pred=model.transform(st_count,dat["spatial"],use_spatial=True)
    genes=list(model.selected_genes_)
    sc_sub=sc_count.loc[genes]
    st_sub=st_count.loc[[g for g in genes if g in st_count.index]]
    gene_idx=[model._gene2idx[g] for g in st_sub.index]
    B_sub=model.B_prob_[:,gene_idx]
    Xhat_st=pred.values@B_sub
    Xhat_st=pd.DataFrame(Xhat_st,index=pred.index.astype(str),columns=st_sub.index.astype(str))
    coords=dat["spatial"].loc[Xhat_st.index][["x","y"]].to_numpy(np.float32)
    raw_st=st_sub.T.loc[Xhat_st.index]
    raw_st_n=_norm_log1p_df(raw_st)
    xhat_st_n=_norm_log1p_df(Xhat_st)
    mg=[x.strip() for x in args.marker_genes.split(",") if len(x.strip())>0]
    if len(mg)==0:
        mg=_auto_pick_marker_genes(model.B_prob_,genes,dat["cell_types"],max_genes=8)
    mg=[g for g in mg if g in raw_st_n.columns and g in xhat_st_n.columns]
    out_sp=os.path.join(args.output_dir,"spatial_markers")
    _mkdir(out_sp)
    for g in mg:
        _plot_spatial_gene(coords,raw_st_n[g].values,f"Raw ST log1p({g})",os.path.join(out_sp,f"raw_{g}.png"))
        _plot_spatial_gene(coords,xhat_st_n[g].values,f"Denoised VB log1p({g})",os.path.join(out_sp,f"vb_{g}.png"))
    sc_meta=dat["sc_meta"].copy()
    sc_meta.index=sc_meta.index.astype(str)
    ct_map={ct:i for i,ct in enumerate(dat["cell_types"])}
    cells_all=sc_sub.columns.astype(str)
    sc_meta=sc_meta.loc[cells_all.intersection(sc_meta.index)]
    cells,W=_compute_W_nnls(sc_sub.loc[:,sc_meta.index],model.B_prob_,dat["cell_types"],n_cells=args.n_sc,seed=args.seed)
    sc_meta_s=sc_meta.loc[cells]
    y=np.array([ct_map.get(x,-1) for x in sc_meta_s["cellType"].astype(str).values],dtype=int)
    keep=y>=0
    cells=np.array(cells)[keep].tolist()
    W=W[keep]
    y=y[keep]
    X_raw=sc_sub[cells].T
    Xhat_sc=pd.DataFrame(W@model.B_prob_,index=cells,columns=genes)
    markers_raw=_markers_by_logfc(X_raw,y,dat["cell_types"],top_k=50)
    markers_den=_markers_by_logfc(Xhat_sc,y,dat["cell_types"],top_k=50)
    ov_df=_marker_enrichment_overlap(markers_raw,markers_den,universe_n=len(genes),top_k=50)
    ov_df.to_csv(os.path.join(args.output_dir,"marker_overlap_raw_vs_denoised.csv"),index=False)
    auc_raw=_marker_auc(X_raw,y,markers_raw,dat["cell_types"])
    auc_den=_marker_auc(Xhat_sc,y,markers_raw,dat["cell_types"])
    auc_raw=auc_raw.rename(columns={"mean_auc":"mean_auc_raw","n_genes":"n_genes_raw"})
    auc_den=auc_den.rename(columns={"mean_auc":"mean_auc_denoised","n_genes":"n_genes_used"})
    auc_df=auc_raw.merge(auc_den,on="cell_type",how="outer")
    auc_df.to_csv(os.path.join(args.output_dir,"marker_auc_raw_vs_denoised.csv"),index=False)
    Xr=_norm_log1p_df(X_raw).values
    Xd=_norm_log1p_df(Xhat_sc).values
    pr=PCA(n_components=min(50,Xr.shape[1])).fit_transform(Xr)
    pdn=PCA(n_components=min(50,Xd.shape[1])).fit_transform(Xd)
    Dr=squareform(pdist(pr,metric="euclidean"))
    Dd=squareform(pdist(pdn,metric="euclidean"))
    mantel_r,mantel_p=_mantel_spearman(Dr,Dd,n_perm=args.n_perm,seed=args.seed)
    knn_ov=_knn_overlap(pr,pdn,k=args.k_nn)
    summ=pd.Series({"mantel_spearman_r":mantel_r,"mantel_perm_p":mantel_p,"knn_overlap":knn_ov,"n_cells_used":len(y),"n_genes":len(genes),"k_nn":args.k_nn,"n_perm":args.n_perm})
    summ.to_csv(os.path.join(args.output_dir,"denoise_quality_summary.csv"))
    plt.figure(figsize=(6,3.2))
    top=ov_df.sort_values("enrichment_x",ascending=False).head(15)
    plt.barh(np.arange(top.shape[0])[::-1],top["overlap@50"].values,color="#4c78a8",edgecolor="black",linewidth=0.6)
    plt.yticks(np.arange(top.shape[0])[::-1],top["cell_type"].values,fontsize=8)
    plt.xlabel("Overlap@50 (raw DE vs denoised DE)")
    plt.title("Self-consistent marker agreement")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,"fig_marker_overlap.png"),dpi=300,bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(6,3.2))
    auc_df2=auc_df.dropna(subset=["mean_auc_raw","mean_auc_denoised"])
    auc_df2=auc_df2.sort_values("mean_auc_denoised",ascending=False).head(15)
    yv=np.arange(auc_df2.shape[0])[::-1]
    plt.barh(yv,auc_df2["mean_auc_raw"].values,color="#9ecae1",edgecolor="black",linewidth=0.6,label="raw")
    plt.barh(yv,auc_df2["mean_auc_denoised"].values,color="#3182bd",edgecolor="black",linewidth=0.6,label="denoised",alpha=0.9)
    plt.yticks(yv,auc_df2["cell_type"].values,fontsize=8)
    plt.xlabel("Mean marker AUROC")
    plt.legend(frameon=False,fontsize=9)
    plt.title("Marker separability: raw vs denoised")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,"fig_marker_auc.png"),dpi=300,bbox_inches="tight")
    plt.close()
if __name__=="__main__":
    main()