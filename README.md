# EM as a Neural Operator Layer

This project implements a **differentiable EM algorithm** for a Gaussian Mixture Model (GMM) as a **PyTorch layer**.  
It treats EM as a **neural operator**: data is transformed into a latent responsibility field, iteratively refined, and decoded back into a reconstructed data distribution using `torch.logsumexp`.

---

## 🧠 Concept Overview

The architecture draws an analogy with **Fourier Neural Operators (FNOs)**:

| Stage | FNO | EM-as-Layer |
|--------|-----|--------------|
| Transform (Encoder) | FFT — move data to spectral space | E-step — compute responsibilities (posterior mixture weights) |
| Operator Core | Spectral multipliers | EM iterations + learnable MLP refinements |
| Inverse Transform (Decoder) | IFFT — reconstruct spatial data | `torch.logsumexp` — reconstruct distribution from μ, Σ |

The result is a **statistical neural operator**, where EM is seen as a data-to-distribution mapping.

---

## 🔢 Mathematical Formulation

### 1. Log-Likelihood of GMM

For data \( X = \{x_n\}_{n=1}^N \) and mixture parameters \( \Theta = \{\alpha_k, \mu_k, \Sigma_k\}_{k=1}^K \):

\[
\log p(X | \Theta) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \alpha_k \, \mathcal{N}(x_n | \mu_k, \Sigma_k) \right)
\]

---

### 2. Transform (E-step)

Compute responsibilities \( r_{nk} \):

\[
r_{nk} = \frac{\alpha_k \, \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \alpha_j \, \mathcal{N}(x_n | \mu_j, \Sigma_j)}
\]

Implemented as:
```python
lg = _log_gauss(X, mu, Sigma)           # [B, N, K]
w_log = torch.log_softmax(alpha, -1)    # [B, K]
log_r = lg + w_log[:, None, :]          # [B, N, K]
r = torch.softmax(log_r, dim=-1)        # [B, N, K]
```

This acts as the **transform**: mapping data into a latent mixture-coefficient space.

---

### 3. Operator Core (Iterative EM with Learnable Residuals)

Each EM iteration computes:

\[
N_k = \sum_n r_{nk}, \quad
\mu_k = \frac{1}{N_k} \sum_n r_{nk} x_n, \quad
\Sigma_k = \frac{1}{N_k} \sum_n r_{nk} (x_n - \mu_k)(x_n - \mu_k)^\top
\]

Then applies **learnable residual corrections**:

```python
mu = F.relu(self.mlp_x2(mu)) + mu
Sigma = F.relu(self.mlp_x3(Sigma)) + Sigma
```

These MLPs act as **nonlinear correction operators**, enabling richer adaptation than pure EM.

---

### 4. Decoder (Mixture Reconstruction via `torch.logsumexp`)

After \( T \) refinement steps, the **decoder** reconstructs the data distribution using the refined parameters:

\[
\log p(x_n) = \log \sum_k \exp(\log \pi_k + \log \mathcal{N}(x_n | \mu_k, \Sigma_k))
\]

Implementation:
```python
w_log = torch.log_softmax(alpha, dim=-1)
log_probs = _log_gauss(X, mu, Sigma)
ll = torch.logsumexp(log_probs + w_log[:, None, :], dim=-1).sum(dim=1)
```

This stage maps the latent mixture parameters **back to a data-space distribution**.  
Hence, the decoder ≠ M-step — the M-step is part of the internal operator, while the decoder reconstructs the **observable density**.

---

## 🧮 Data Flow Summary

| Symbol | Meaning | Shape |
|---------|----------|--------|
| \( X \) | Input data batch | [B, N, D] |
| \( \alpha \) | Mixture logits | [B, K] |
| \( \mu \) | Means | [B, K, D] |
| \( \Sigma \) | Covariances | [B, K, D, D] |
| \( r \) | Responsibilities | [B, N, K] |

---

## 🧱 Architecture

```
EMNet
├── TinyBackbone       # optional feature extractor
└── GMMLayer
     ├── E-step (transform)
     ├── M-step (operator)
     ├── Learnable corrections (MLPs)
     └── logsumexp (decoder)
```

---

## 🔁 Neural Operator View

\[
X
\;\xrightarrow{\text{E-step}}\;
r
\;\xrightarrow{\text{Iterative EM + Learnable MLPs}}\;
(\alpha, \mu, \Sigma)
\;\xrightarrow{\text{logsumexp}}\;
\log p(X)
\]

- **Encoder:** E-step (projects into responsibility space)  
- **Operator:** Iterative EM refinement (latent evolution)  
- **Decoder:** `logsumexp` (reconstructs data likelihood)  

This structure directly parallels neural operators in PDE modeling.

---

## 🧩 Training Objective

Maximize the log-likelihood of observed data:

\[
\mathcal{L} = \mathbb{E}[\log p(X | \Theta)]
\]

Implementation:
```python
loss = -ll.mean()
loss.backward()
```

Everything (Cholesky, softmax, logsumexp, MLPs) is differentiable, so the entire EM pipeline supports gradient-based training.

---

## 🧪 Example Output

```
Epoch NLL: 1683.427
weights ≈ [0.39, 0.36, 0.25]
Learned means:
[[0.01, -0.02],
 [4.97, 0.04],
 [0.05, 4.99]]
```

Visualization:

![EM as a Layer: Learned Gaussians](./em_layer_output.png)

---


## 🧭 Summary

**EM-as-a-Layer** = *Neural Operator over probability distributions*

| Role | Function |
|------|-----------|
| Transform | E-step: projects data → latent |
| Operator | Unrolled EM + learnable MLPs |
| Decoder | logsumexp: reconstructs distribution from μ, Σ |
| Training | Maximize log-likelihood end-to-end |

This framework generalizes EM into a continuous, trainable operator that bridges statistical inference and deep representation learning.
