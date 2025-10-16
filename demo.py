# Full, runnable PyTorch demo: "EM as a Neural Network Layer"
# - Defines a GMMLayer that unrolls T EM iterations (fully differentiable)
# - Builds a tiny model and trains it unsupervised by maximizing log-likelihood
# - Generates a 2D synthetic dataset, fits, and saves a plot + script file
#
# You can run this cell to see it work, or download the script at the end.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from textwrap import dedent
import torch.nn.functional as F


torch.set_default_dtype(torch.float64)  # for better numeric stability in Cholesky ops

def _log_gauss(X, mu, Sigma, eps=1e-6):
    # X: [B,N,D], mu: [B,K,D], Sigma: [B,K,D,D]
    B,N,D = X.shape
    eye = torch.eye(D, device=X.device, dtype=X.dtype)[None,None,:,:]
    L = torch.linalg.cholesky(Sigma + eps*eye)              # [B,K,D,D]
    diff = X[:, :, None, :] - mu[:, None, :, :]             # [B,N,K,D]
    y = torch.cholesky_solve(diff.unsqueeze(-1), L)         # [B,N,K,D,1]
    quad = (diff.unsqueeze(-1) * y).sum(dim=(-2, -1))       # [B,N,K]
    logdet = 2.0*torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # [B,K]
    const = D * torch.log(torch.tensor(2.0*torch.pi, device=X.device, dtype=X.dtype))
    return -0.5 * (quad + logdet[:,None,:] + const)


# ----- Tiny MLP blocks -----
class MLP1D(nn.Module):
    """Maps (B, N) -> (B, N)."""
    def __init__(self, n_inout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inout, 16),
            nn.ReLU(),
            nn.Linear(16, n_inout),
        )
    def forward(self, x):  # x: (B, N)
        return self.net(x)

class MLPLastDim(nn.Module):
    """Applies an MLP to the last dim: (..., D) -> (..., D)."""
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 16)
        self.fc2 = nn.Linear(16, d)
    def forward(self, x):
        orig = x.shape
        y = x.reshape(-1, orig[-1])        # (prod(..., D-1), D)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y.reshape(orig)

class MLP2x2(nn.Module):
    """Flattens the last two dims (2x2=4), applies MLP, then reshapes back."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):  # x: (..., 2, 2)
        *lead, h, w = x.shape
        assert (h, w) == (2, 2), "Expected last two dims = (2,2)"
        y = x.reshape(-1, 4)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y.reshape(*lead, 2, 2)

class GMMLayer(nn.Module):
    """
    Differentiable GMM-EM layer.
    Modes:
      - learnable_init=True: global learnable {alpha, mu, Sigma}
      - learnable_init=False: take inits from caller (amortized EM scenario)
    Forward returns: responsibilities r, refined (alpha, mu, Sigma), and per-batch log-likelihood.
    """
    def __init__(self, K, D, T=5, learnable_init=True, eps=1e-6):
        super().__init__()
        self.K, self.D, self.T, self.eps = K, D, T, eps
        self.mlp_x1 = MLP1D(n_inout=3)   # (1,3) -> (1,3)
        self.mlp_x2 = MLPLastDim(d=2)    # (1,3,2) -> (1,3,2)
        self.mlp_x3 = MLP2x2()           # (1,3,2,2) -> (1,3,2,2)
        if learnable_init:
            self.alpha0 = nn.Parameter(torch.zeros(1, K, dtype=torch.get_default_dtype()))  # logits
            self.mu0    = nn.Parameter(torch.randn(1, K, D, dtype=torch.get_default_dtype())*0.01)
            Sigma0 = torch.eye(D, dtype=torch.get_default_dtype())[None,None,:,:].repeat(1, K, 1, 1)
            self.Sigma0 = nn.Parameter(Sigma0)
        else:
            self.register_parameter('alpha0', None)
            self.register_parameter('mu0', None)
            self.register_parameter('Sigma0', None)

    @staticmethod
    def _mstep(X, r, eps):
        # X: [B,N,D], r: [B,N,K]
        B,N,D = X.shape
        K = r.shape[-1]
        Nk = r.sum(dim=1) + eps                                    # [B,K]
        mu = (r.transpose(1,2) @ X) / Nk[:,:,None]                  # [B,K,D]
        Xc = X[:, :, None, :] - mu[:, None, :, :]                   # [B,N,K,D]
        cov_num = torch.einsum('bnk,bnkd,bnke->bkde', r, Xc, Xc)    # [B,K,D,D]
        Sigma = cov_num / Nk[:,:,None,None] + eps*torch.eye(D, device=X.device, dtype=X.dtype)[None,None,:,:]
        alpha = torch.log(Nk) - torch.log(Nk.sum(dim=1, keepdim=True))
        return alpha, mu, Sigma

    def forward(self, X, alpha_init=None, mu_init=None, Sigma_init=None):
        """
        X: [B,N,D]
        If learnable_init=False, provide alpha_init [B,K] (logits), mu_init [B,K,D], Sigma_init [B,K,D,D].
        """
        if self.alpha0 is not None:
            B,N,D = X.shape
            alpha = self.alpha0.expand(B, self.K)
            mu    = self.mu0.expand(B, self.K, self.D)
            Sigma = self.Sigma0.expand(B, self.K, self.D, self.D)
        else:
            assert alpha_init is not None and mu_init is not None and Sigma_init is not None
            alpha, mu, Sigma = alpha_init, mu_init, Sigma_init

        for _ in range(self.T):
            w_log = torch.log_softmax(alpha, dim=-1)                # [B,K]
            lg = _log_gauss(X, mu, Sigma, eps=self.eps)             # [B,N,K]
            log_r = lg + w_log[:,None,:]
            r = torch.softmax(log_r, dim=-1)                        # [B,N,K]
            alpha, mu, Sigma = self._mstep(X, r, self.eps)

        # batch log-likelihood
        w_log = torch.log_softmax(alpha, dim=-1)
        #a_out = self.mlp_x1(X)
        mu = F.relu(self.mlp_x2(mu)) + mu
        Sigma = F.relu(self.mlp_x3(Sigma)) + Sigma 
        ll = torch.logsumexp(_log_gauss(X, mu, Sigma, eps=self.eps) + w_log[:,None,:], dim=-1).sum(dim=1)  # [B]
        return r, (alpha, mu, Sigma), ll

class TinyBackbone(nn.Module):
    """Optional feature extractor; identity by default for D=2."""
    def __init__(self, D_in, D_feat=None):
        super().__init__()
        if D_feat is None or D_feat == D_in:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(D_in, D_feat), nn.ReLU(),
                nn.Linear(D_feat, D_feat)
            )
    def forward(self, x):
        return self.net(x)

class EMNet(nn.Module):
    def __init__(self, D_in, K, T=5, D_feat=None, learnable_init=True):
        super().__init__()
        self.backbone = TinyBackbone(D_in, D_feat)
        D_eff = D_in if isinstance(self.backbone.net, nn.Identity) else D_feat
        self.em = GMMLayer(K=K, D=D_eff, T=T, learnable_init=learnable_init)

    def forward(self, X):
        feats = self.backbone(X)          # [B,N,D_feat]
        r, params, ll = self.em(feats)
        return r, params, ll

# ----- Synthetic data (2D, 3 clusters) -----
rng = np.random.default_rng(7)
true_means = np.array([[0.0, 0.0],[5.0, 0.0],[0.0, 5.0]])
true_covs = np.array([[[1.0, 0.3],[0.3, 1.5]],[[1.2, -0.2],[-0.2, 0.8]],[[0.5, 0.0],[0.0, 1.8]]])
true_weights = np.array([0.4, 0.35, 0.25])
N = 1200
z = rng.choice(3, size=N, p=true_weights)
X_np = np.vstack([rng.multivariate_normal(true_means[k], true_covs[k], size=(z==k).sum())
                  for k in range(3)])

# Put the data into a batch of size 1: [B=1, N, D]
X = torch.from_numpy(X_np).to(torch.get_default_dtype()).unsqueeze(0)

# ----- Build and train the model -----
K = 3
model = EMNet(D_in=2, K=K, T=40, D_feat=None, learnable_init=True)
opt = optim.Adam(model.parameters(), lr=5e-2)

epochs = 0

opt.zero_grad()
r, (alpha, mu, Sigma), ll = model(X)       # ll: [B]
print(mu)
loss = -ll.mean()                           # maximize likelihood
loss.backward()
opt.step()
print(f"Epoch NLL: {loss.item():.3f}  "
              f"weights≈ {torch.softmax(alpha, -1).detach().cpu().numpy().round(3)}")

# ----- Extract results -----
with torch.no_grad():
    r, (alpha, mu, Sigma), ll = model(X)
w = torch.softmax(alpha, -1).squeeze(0).cpu().numpy()      # [K]
mu_ = mu.squeeze(0).cpu().numpy()                          # [K,2]
Sigma_ = Sigma.squeeze(0).cpu().numpy()                    # [K,2,2]

print("\nLearned weights:", np.round(w, 4))
print("Learned means:\n", np.round(mu_, 3))

# ----- Plot data + learned ellipses -----
def draw_ellipse(ax, mean, cov, nsig=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nsig * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False)
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_np[:,0], X_np[:,1], s=5, alpha=0.5)
for k in range(K):
    draw_ellipse(ax, mu_[k], Sigma_[k], nsig=2.0)
    ax.scatter(mu_[k,0], mu_[k,1], marker='x', s=80)
ax.set_title("EM-as-a-Layer: learned 2σ Gaussians")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
plt.tight_layout()
plt.show() 