"""
SlotDeconv: Spatial Transcriptomics Deconvolution via Slot-based Reference Learning
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from scipy.optimize import nnls
warnings.filterwarnings("ignore")
DEFAULT_CONFIG = {
    'n_genes': 3000,
    'max_cells_per_type': 750,
    'lambda_div': 4.0,
    'margin': 0.1,
    'lambda_sp': 15.0,
    'sp_epochs': 1500,
    'sp_lr': 0.01,
    'b_epochs': 2000,
    'b_lr': 1e-3,
    'pow_w': 0.8,
    'knn': 15,
    'd_slot':128,
    'dec_hidden':(256,512),
    'dec_dropout':0.1
}
class _SlotDecoder(nn.Module):
    def __init__(self, n_genes, n_types, d_slot=128, margin=0.1,hidden=(256,512),dropout=0.1):
        super().__init__()
        self.n_types = n_types
        self.n_genes = n_genes
        self.margin = margin
        self.slots = nn.Parameter(torch.randn(n_types, d_slot) * 0.1)
        dims=[d_slot]+list(hidden)+[n_genes]
        H=max(len(dims)-2,0)
        if isinstance(dropout,(list,tuple,np.ndarray)):
            drops=[float(x) for x in list(dropout)]
            if len(drops)<H: drops=drops+[0.0]*(H-len(drops))
            drops=drops[:H]
        else:
            d=float(dropout) if dropout is not None else 0.0
            drops=([d]+[0.0]*(H-1)) if H>0 else []
        layers=[]
        for i in range(H):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            if drops[i]>0: layers.append(nn.Dropout(drops[i]))
        layers.append(nn.Linear(dims[-2],dims[-1]))
        self.decoder=nn.Sequential(*layers)      
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(torch.ones(n_genes) * 5.0)))
    def alpha_disp(self):
        return F.softplus(self.alpha_raw) + 1e-6
    def forward(self, labels, size):
        logits = torch.clamp(self.decoder(self.slots[labels]), -20, 20)
        return (F.softplus(logits) * size).clamp(1e-8, 1e8)
    def get_reference_matrix(self):
        with torch.no_grad():
            logits = torch.clamp(self.decoder(self.slots), -20, 20)
            B = F.softplus(logits)
            B = B / (B.sum(dim=1, keepdim=True) + 1e-8)
        return B.cpu().numpy()
    def diversity_loss(self):
        logits = torch.clamp(self.decoder(self.slots), -20, 20)
        B = F.softplus(logits)
        B = B / (B.sum(dim=1, keepdim=True) + 1e-8)
        Bn = F.normalize(B, dim=-1)
        sim = Bn @ Bn.t()
        mask = torch.eye(self.n_types, device=sim.device)
        off_diag = sim * (1 - mask)
        return F.relu(off_diag - self.margin).max(), off_diag.max().item()
def _nb_nll(y, mu, theta):
    mu = mu.clamp(1e-8, 1e8)
    theta = theta.clamp(1e-4, 1e4)
    t1 = torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
    t2 = theta * torch.log(theta / (theta + mu) + 1e-12)
    t3 = y * torch.log(mu / (mu + theta) + 1e-12)
    return -(t1 + t2 + t3).mean()
def set_seed(seed=42):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def build_knn_graph(coords, knn=15):
    """Build KNN graph with RBF weights. O(NK) memory instead of O(N²)."""
    n = len(coords)
    knn = min(knn, n - 1)
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm="auto").fit(coords)
        dist, idx = nbrs.kneighbors(coords)
        idx = idx[:, 1:]
        dist = dist[:, 1:]
    except Exception:
        D = cdist(coords, coords)
        idx = np.argsort(D, axis=1)[:, 1:knn + 1]
        dist = np.take_along_axis(D, idx, axis=1)
    sigma = np.median(dist[dist > 0]) + 1e-8
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
    w = w / (w.sum(1, keepdims=True) + 1e-8)
    return idx.astype(np.int64), w.astype(np.float32)
def build_dense_graph(coords_norm):
    """Build dense spatial graph. Expects already normalized coords."""
    dist = cdist(coords_norm, coords_norm)
    sigma = np.median(dist[dist > 0])
    W = np.exp(-dist ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W.astype(np.float32)
class SlotDeconv:
    """
    SlotDeconv: Spatial-aware deconvolution with slot-based reference learning.
    
    Parameters
    ----------
    device : str, optional
        Device for computation. Default: auto-detect
    random_state : int, optional
        Random seed. Default: 42
    verbose : bool, optional
        Print progress. Default: True
    use_default_config : bool, optional
        Use optimized parameters from default configuration. Default: True
    """
    def __init__(self, device=None, random_state=42, verbose=True, use_default_config=True):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.use_default_config = use_default_config
        self.B_prob_ = None
        self.cell_types_ = None
        self.selected_genes_ = None
        self.gene_weights_ = None
        self._gene2idx = None
        self._last_fit_params = None
        self._last_transform_params = None
        set_seed(random_state)
    def _log(self, msg):
        if self.verbose:
            print(msg)
    def get_config(self):
        """Return current configuration including last used parameters."""
        d = dict(DEFAULT_CONFIG)
        d["use_default_config"] = bool(self.use_default_config)
        d["last_fit"] = self._last_fit_params
        d["last_transform"] = self._last_transform_params
        return d
    def fit(self, sc_count, sc_meta, cell_types, celltype_col='cellType',
            n_genes=None, max_cells_per_type=None, lambda_div=None, margin=None,
            b_epochs=None, b_lr=None,d_slot=None,dec_hidden=None,dec_dropout=None):
        """Learn reference matrix B from single-cell RNA-seq data."""
        set_seed(self.random_state)
        self.cell_types_ = list(cell_types)
        n_types = len(cell_types)
        if self.use_default_config:
            n_genes = n_genes if n_genes is not None else DEFAULT_CONFIG['n_genes']
            max_cells_per_type = max_cells_per_type if max_cells_per_type is not None else DEFAULT_CONFIG['max_cells_per_type']
            lambda_div = lambda_div if lambda_div is not None else DEFAULT_CONFIG['lambda_div']
            margin = margin if margin is not None else DEFAULT_CONFIG['margin']
            b_epochs = b_epochs if b_epochs is not None else DEFAULT_CONFIG['b_epochs']
            b_lr = b_lr if b_lr is not None else DEFAULT_CONFIG['b_lr']
            d_slot=d_slot if d_slot is not None else DEFAULT_CONFIG['d_slot']
            dec_hidden=dec_hidden if dec_hidden is not None else DEFAULT_CONFIG['dec_hidden']
            dec_dropout=dec_dropout if dec_dropout is not None else DEFAULT_CONFIG['dec_dropout']            
        else:
            n_genes = n_genes if n_genes is not None else self._auto_n_genes(sc_count.shape[0], n_types)
            max_cells_per_type = max_cells_per_type if max_cells_per_type is not None else self._auto_max_cells(sc_meta, cell_types, celltype_col)
            lambda_div = lambda_div if lambda_div is not None else self._adaptive_diversity_weight(n_types)
            margin = margin if margin is not None else self._adaptive_margin(n_types)
            b_epochs = b_epochs if b_epochs is not None else 2000
            b_lr = b_lr if b_lr is not None else 1e-3
            d_slot=d_slot if d_slot is not None else DEFAULT_CONFIG['d_slot']
            dec_hidden=dec_hidden if dec_hidden is not None else DEFAULT_CONFIG['dec_hidden']
            dec_dropout=dec_dropout if dec_dropout is not None else DEFAULT_CONFIG['dec_dropout']
        if isinstance(dec_hidden,int): dec_hidden=(dec_hidden,)    
        self._log(f"[SlotDeconv] Fitting: {n_types} types, {n_genes} genes, max {max_cells_per_type} cells/type")
        self._log(f"[SlotDeconv] Config: λ_div={lambda_div}, margin={margin}")
        self._last_fit_params = {"n_genes":n_genes,"max_cells_per_type":max_cells_per_type,"lambda_div":lambda_div,"margin":margin,"b_epochs":b_epochs,"b_lr":b_lr,"d_slot":int(d_slot),"dec_hidden":tuple(dec_hidden),"dec_dropout":float(dec_dropout)}
        sc_df, sc_meta_bal = self._balance_cells(sc_count, sc_meta, cell_types, celltype_col, max_cells_per_type)
        genes, self.gene_weights_ = self._select_discriminative_genes(sc_df, sc_meta_bal, cell_types, celltype_col, n_genes)
        self.selected_genes_ = genes
        self._gene2idx = {g: i for i, g in enumerate(genes)}
        sc_df = sc_df.loc[genes]
        self._log(f"[SlotDeconv] Training reference matrix...")
        self.B_prob_ = self._train_reference(sc_df,sc_meta_bal,cell_types,celltype_col,lambda_div=lambda_div,margin=margin,epochs=b_epochs,lr=b_lr,d_slot=d_slot,dec_hidden=dec_hidden,dec_dropout=dec_dropout)
        self._log(f"[SlotDeconv] Reference matrix learned")
        return self
    def transform(self, st_count, spatial_coords, use_spatial=True,
                  lambda_sp=None, sp_epochs=None, sp_lr=None, pow_w=None,
                  knn=None, spatial_backend='auto', use_gene_weight_in_kl=False,
                  use_mixed_loss=False, mix_alpha=0.2, w_clip=(0.5, 2.0)):
        """
        Deconvolve spatial transcriptomics data.
        
        Parameters
        ----------
        st_count : pd.DataFrame
            Spatial count matrix, shape (genes, spots)
        spatial_coords : pd.DataFrame or np.ndarray
            Spatial coordinates
        use_spatial : bool, optional
            Use spatial refinement. Default: True
        lambda_sp : float, optional
            Spatial regularization weight. Default: from config
        sp_epochs : int, optional
            Spatial refinement epochs. Default: from config
        sp_lr : float, optional
            Learning rate. Default: from config
        pow_w : float, optional
            Gene weight power for NNLS. Default: from config
        knn : int, optional
            K for KNN graph. Default: from config
        spatial_backend : str, optional
            'auto': dense if N<=8000, else KNN
            'dense': force dense graph (may OOM for large data)
            'knn': force KNN graph
        use_gene_weight_in_kl : bool, optional
            Use gene weights in KL loss (Stage3). Default: False
        use_mixed_loss : bool, optional
            Use mixed loss: (1-α)*KL + α*weighted_KL. Default: False
            Only effective when use_gene_weight_in_kl=True
        mix_alpha : float, optional
            Mixing weight for weighted KL (0-1). Default: 0.2
        w_clip : tuple, optional
            Clip range for gene weights in Stage3. Default: (0.5, 2.0)
        """
        if self.B_prob_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.use_default_config:
            lambda_sp = lambda_sp if lambda_sp is not None else DEFAULT_CONFIG['lambda_sp']
            sp_epochs = sp_epochs if sp_epochs is not None else DEFAULT_CONFIG['sp_epochs']
            sp_lr = sp_lr if sp_lr is not None else DEFAULT_CONFIG['sp_lr']
            pow_w = pow_w if pow_w is not None else DEFAULT_CONFIG['pow_w']
            knn = knn if knn is not None else DEFAULT_CONFIG['knn']
        else:
            if isinstance(spatial_coords, pd.DataFrame):
                coords_tmp = spatial_coords[['x', 'y']].values.astype(np.float32)
            else:
                coords_tmp = np.asarray(spatial_coords, dtype=np.float32)
            lambda_sp = lambda_sp if lambda_sp is not None else self._adaptive_spatial_weight(coords_tmp)
            sp_epochs = sp_epochs if sp_epochs is not None else 1500
            sp_lr = sp_lr if sp_lr is not None else 0.01
            pow_w = pow_w if pow_w is not None else 0.8
            knn = knn if knn is not None else 15
        common_genes = [g for g in self.selected_genes_ if g in st_count.index]
        if len(common_genes) < len(self.selected_genes_) * 0.5:
            warnings.warn(f"Only {len(common_genes)}/{len(self.selected_genes_)} genes found")
        gene_idx = np.array([self._gene2idx[g] for g in common_genes])
        st_df = st_count.loc[common_genes]
        B_prob = self.B_prob_[:, gene_idx]
        B_prob = B_prob / (B_prob.sum(axis=1, keepdims=True) + 1e-8)
        w = self.gene_weights_[gene_idx]
        self._log(f"[SlotDeconv] NNLS deconvolution...")
        pred_nnls = self._deconv_nnls(st_df, B_prob, w, pow_w)
        if not use_spatial:
            return pred_nnls
        if isinstance(spatial_coords, pd.DataFrame):
            coords = spatial_coords.loc[st_df.columns][['x', 'y']].values.astype(np.float32)
        else:
            coords = np.asarray(spatial_coords, dtype=np.float32)
        n_spots = len(coords)
        if spatial_backend == 'auto':
            use_knn = n_spots > 8000
        elif spatial_backend == 'knn':
            use_knn = True
        else:
            use_knn = False
        gene_w = w if use_gene_weight_in_kl else None
        backend_str = 'knn' if use_knn else 'dense'
        self._last_transform_params = {"lambda_sp": lambda_sp, "sp_epochs": sp_epochs, "sp_lr": sp_lr, "pow_w": pow_w, "knn": knn, "spatial_backend_used": backend_str, "use_gene_weight_in_kl": bool(use_gene_weight_in_kl), "use_mixed_loss": bool(use_mixed_loss), "mix_alpha": float(mix_alpha), "w_clip": tuple(w_clip)}
        self._log(f"[SlotDeconv] Spatial refinement (λ_sp={lambda_sp}, backend={backend_str})...")
        pred_kl = self._deconv_spatial(st_df, B_prob, coords, pred_nnls, lambda_sp, sp_epochs, sp_lr, 
                                        use_knn, knn, gene_w, pow_w, use_mixed_loss, mix_alpha, w_clip)
        self._log(f"[SlotDeconv] Done")
        return pred_kl
    def fit_transform(self, sc_count, sc_meta, cell_types, st_count, spatial_coords,
                      celltype_col='cellType', use_spatial=True, **kwargs):
        fit_keys = ['n_genes','max_cells_per_type','lambda_div','margin','b_epochs','b_lr','d_slot','dec_hidden','dec_dropout']
        transform_keys = ['lambda_sp', 'sp_epochs', 'sp_lr', 'pow_w', 'knn', 'spatial_backend', 
                         'use_gene_weight_in_kl', 'use_mixed_loss', 'mix_alpha', 'w_clip']
        fit_kwargs = {k: kwargs[k] for k in fit_keys if k in kwargs}
        transform_kwargs = {k: kwargs[k] for k in transform_keys if k in kwargs}
        self.fit(sc_count, sc_meta, cell_types, celltype_col, **fit_kwargs)
        return self.transform(st_count, spatial_coords, use_spatial, **transform_kwargs)
    def _auto_n_genes(self, total_genes, n_types):
        base = min(3000, total_genes)
        if n_types > 20:
            return min(int(base * 1.2), total_genes)
        elif n_types > 10:
            return min(base, total_genes)
        else:
            return min(int(base * 0.8), total_genes)
    def _auto_max_cells(self, sc_meta, cell_types, celltype_col):
        ct = sc_meta[celltype_col].astype(str).str.strip()
        counts = [sum(ct == t) for t in cell_types]
        median_count = np.median([c for c in counts if c > 0])
        return int(min(750, max(500, median_count)))
    def _adaptive_diversity_weight(self, n_types):
        if n_types <= 10:
            return 2.0
        elif n_types <= 20:
            return 3.0
        else:
            return 4.0
    def _adaptive_margin(self, n_types):
        if n_types <= 10:
            return 0.2
        elif n_types <= 20:
            return 0.15
        else:
            return 0.1
    def _adaptive_spatial_weight(self, coords):
        coords_norm = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
        dist = cdist(coords_norm, coords_norm)
        median_dist = np.median(dist[dist > 0])
        n_spots = len(coords)
        if n_spots < 1000:
            base_weight = 10.0
        elif n_spots < 5000:
            base_weight = 15.0
        else:
            base_weight = 20.0
        if median_dist < 0.5:
            return base_weight * 1.2
        elif median_dist > 1.5:
            return base_weight * 0.8
        return base_weight
    def _balance_cells(self, sc_df, sc_meta, cell_types, celltype_col, max_per_type):
        rng = np.random.RandomState(self.random_state)
        ct = sc_meta[celltype_col].astype(str).str.strip()
        keep = []
        for t in cell_types:
            idx = sc_meta.index[ct == t].astype(str).values
            if len(idx) == 0:
                continue
            if max_per_type and len(idx) > max_per_type:
                idx = rng.choice(idx, size=max_per_type, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep).astype(str)
        keep = pd.Index(keep).intersection(sc_df.columns.astype(str)).intersection(sc_meta.index.astype(str))
        return sc_df.loc[:, keep], sc_meta.loc[keep]
    def _select_discriminative_genes(self, sc_df, sc_meta, cell_types, celltype_col, n_genes):
        X = sc_df.T.to_numpy(np.float32)
        ct = sc_meta[celltype_col].astype(str).values
        mp = {c: i for i, c in enumerate(cell_types)}
        idx = [i for i, x in enumerate(ct) if x in mp]
        X, ct = X[idx], ct[idx]
        K, G = len(cell_types), X.shape[1]
        mu = np.zeros((K, G), dtype=np.float32)
        var = np.zeros((K, G), dtype=np.float32)
        n = np.zeros(K, dtype=np.float32)
        for i, x in enumerate(ct):
            k = mp[x]
            mu[k] += X[i]
            var[k] += X[i] * X[i]
            n[k] += 1.0
        n = np.maximum(n, 1.0)
        mu = mu / n[:, None]
        var = np.maximum(var / n[:, None] - mu * mu, 0.0)
        mu_all = (mu * n[:, None]).sum(0) / n.sum()
        between_var = ((mu - mu_all[None, :]) ** 2 * n[:, None]).sum(0) / n.sum()
        within_var = (var * n[:, None]).sum(0) / n.sum()
        f_score = between_var / (within_var + 1e-6)
        top_idx = np.argsort(-f_score)[:min(n_genes, len(f_score))]
        return sc_df.index[top_idx], f_score[top_idx].astype(np.float32)
    def _train_reference(self, sc_df, sc_meta, cell_types, celltype_col, lambda_div, margin, epochs, lr, d_slot=None, dec_hidden=None, dec_dropout=None):
        if d_slot is None: d_slot=DEFAULT_CONFIG['d_slot']
        if dec_hidden is None: dec_hidden=DEFAULT_CONFIG['dec_hidden']
        if dec_dropout is None: dec_dropout=DEFAULT_CONFIG['dec_dropout']
        if isinstance(dec_hidden,int): dec_hidden=(dec_hidden,)
        cells=sc_meta.index.astype(str)
        X=sc_df.loc[:,cells].T.values.astype(np.float32)
        X[X<0]=0
        size=X.sum(axis=1,keepdims=True).astype(np.float32)
        size=np.maximum(size,1e-8)/np.median(size[size>0])
        mp={ct:i for i,ct in enumerate(cell_types)}
        ct=sc_meta.loc[cells,celltype_col].astype(str).str.strip().values
        mask=np.array([x in mp for x in ct],dtype=bool)
        X=X[mask]
        size=size[mask]
        labels=np.array([mp[x] for x in ct[mask]],dtype=np.int64)
        X_t=torch.tensor(X,device=self.device)
        S_t=torch.tensor(size,device=self.device)
        L_t=torch.tensor(labels,dtype=torch.long,device=self.device)
        model=_SlotDecoder(X.shape[1],len(cell_types),d_slot=int(d_slot),margin=margin,hidden=tuple(dec_hidden),dropout=float(dec_dropout)).to(self.device)
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
        for ep in range(epochs):
            mu=model(L_t,S_t)
            recon=_nb_nll(X_t,mu,model.alpha_disp()[None,:])
            div,_=model.diversity_loss()
            loss=recon+lambda_div*div
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            sch.step()
        self.slots_=model.slots.detach().cpu().numpy()    
        self.alpha_=model.alpha_disp().detach().cpu().numpy()
        return model.get_reference_matrix()
    def _deconv_nnls(self, st_df, B_prob, w, pow_w):
        X = st_df.T.values.astype(np.float32)
        X[X < 0] = 0
        size = X.sum(axis=1, keepdims=True).astype(np.float32)
        size = np.maximum(size, 1e-8) / np.median(size[size > 0])
        Xu = X / size
        ww = np.clip((w / (np.median(w) + 1e-8)) ** pow_w, 0.2, 5.0).astype(np.float64)
        sw = np.sqrt(ww)
        Bt = (B_prob * sw[None, :]).T.astype(np.float64)
        Xu_w = Xu * sw[None, :]
        V = np.zeros((Xu.shape[0], B_prob.shape[0]), dtype=np.float32)
        for i in range(Xu.shape[0]):
            v, _ = nnls(Bt, Xu_w[i].astype(np.float64))
            s = v.sum()
            V[i] = (v / s if s > 0 else np.ones(B_prob.shape[0]) / B_prob.shape[0]).astype(np.float32)
        return pd.DataFrame(V, index=st_df.columns, columns=self.cell_types_)
    def _deconv_spatial(self, st_df, B_prob, coords, V_init, lambda_sp, epochs, lr, 
                         use_knn, knn, gene_w, pow_w, use_mixed_loss, mix_alpha, w_clip):
        X = st_df.T.values.astype(np.float32)
        X[X < 0] = 0
        X_prob = X / (X.sum(axis=1, keepdims=True) + 1e-8)
        X_t = torch.tensor(X_prob, device=self.device)
        B_t = torch.tensor(B_prob, device=self.device)
        V_logits = nn.Parameter(torch.tensor(np.log(V_init.values + 1e-6), device=self.device))
        coords_norm = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
        if use_knn:
            idx, w = build_knn_graph(coords_norm, knn)
            idx_t = torch.tensor(idx, device=self.device, dtype=torch.long)
            w_t = torch.tensor(w, device=self.device)
        else:
            W = build_dense_graph(coords_norm)
            W_t = torch.tensor(W, device=self.device)
        if gene_w is not None:
            gw = np.asarray(gene_w, dtype=np.float64)
            gw = (gw / (np.median(gw) + 1e-12)) ** pow_w
            gw = np.clip(gw, w_clip[0], w_clip[1])
            gw = gw / (np.mean(gw) + 1e-12)
            gw_t = torch.tensor(gw, device=self.device, dtype=torch.float32)
        opt = optim.Adam([V_logits], lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
        for ep in range(epochs):
            V = F.softmax(V_logits, dim=1)
            pred = V @ B_t
            pred_prob = pred / (pred.sum(dim=1, keepdim=True) + 1e-8)
            recon_base = (X_t * (torch.log(X_t + 1e-8) - torch.log(pred_prob + 1e-8))).sum(dim=1).mean()
            if gene_w is not None:
                recon_weighted = (gw_t[None, :] * X_t * (torch.log(X_t + 1e-8) - torch.log(pred_prob + 1e-8))).sum(dim=1).mean()
                if use_mixed_loss:
                    recon = (1 - mix_alpha) * recon_base + mix_alpha * recon_weighted
                else:
                    recon = recon_weighted
            else:
                recon = recon_base
            if use_knn:
                V_neighbor = (V[idx_t] * w_t[..., None]).sum(dim=1)
            else:
                V_neighbor = W_t @ V
            spatial = ((V - V_neighbor) ** 2).mean()
            loss = recon + lambda_sp * spatial
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            V_final = F.softmax(V_logits, dim=1).cpu().numpy()
        return pd.DataFrame(V_final, index=st_df.columns, columns=self.cell_types_)