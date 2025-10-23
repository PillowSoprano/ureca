# kovae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(x_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    def forward(self, x):              # x: [B,T,x_dim]
        h,_ = self.gru(x)              # [B,T,h_dim]
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std             # reparameterization
        return z, (mu, logvar)

class GRUPrior(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    def forward(self, T, B, device):
        z0 = torch.zeros(B, 1, self.mu.out_features, device=device)  # start token
        h,_ = self.gru(z0.repeat(1,T,1))                             # dummy roll
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        zbar = mu + eps * std                                        # \bar z_{1:T}
        return zbar, (mu, logvar)

def koopman_A_from_zbar(zbar):
    # zbar: [B,T,z_dim] -> 以 batch 维拼接成一个大的最小二乘
    B,T,k = zbar.shape
    Z0 = zbar[:, :-1, :].reshape(-1, k)   # [(B*(T-1)), k]
    Z1 = zbar[:, 1:,  :].reshape(-1, k)   # same
    # A = argmin ||AZ0 - Z1||_F
    # 用 lstsq 比 pinv 更稳，且可反传
    A = torch.linalg.lstsq(Z0, Z1).solution.T   # [k,k]
    return A

class GRUDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(h_dim, x_dim)
    def forward(self, z):
        h,_ = self.gru(z)
        xhat = self.out(h)
        return xhat

class KoVAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, h_dim=64, layers=1, eig_target=None, eig_margin=0.0):
        super().__init__()
        self.enc = GRUEncoder(x_dim, h_dim, z_dim, layers)
        self.pri = GRUPrior(z_dim, h_dim, layers)
        self.dec = GRUDecoder(z_dim, h_dim, x_dim, layers)
        self.eig_target = eig_target     # None / 1.0 / "<=1"
        self.eig_margin = eig_margin

    def forward(self, x, alpha=0.1, beta=1e-3, gamma=0.0):
        B,T,X = x.shape
        # posterior
        z, (mu_q, logvar_q) = self.enc(x)           # [B,T,k]
        # prior (stochastic)
        zbar, (mu_p, logvar_p) = self.pri(T, B, x.device)  # [B,T,k]
        # Koopman A from zbar
        A = koopman_A_from_zbar(zbar.detach() if gamma==0 else zbar)  # 需要谱约束就别 detach
        # 用 A 推进一个先验序列（从 t=1 开始对齐）
        zhat = []
        zprev = zbar[:,0,:]                          # [B,k]
        zhat.append(zprev)
        for t in range(1, T):
            zprev = (zprev @ A.T)                    # [B,k]
            zhat.append(zprev)
        zhat = torch.stack(zhat, dim=1)              # [B,T,k]

        # 解码
        xhat = self.dec(z)

        # losses
        L_rec  = F.mse_loss(xhat, x)
        L_pred = F.mse_loss(z, zbar)                 # 后验与先验的一致性？
        # KL：逐时刻高斯闭式
        KL = -0.5 * (1 + (logvar_q - logvar_p) - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        L_kl = KL.mean()

        L_eig = 0.0
        if gamma > 0:
            evals = torch.linalg.eigvals(A)          # complex
            mod = evals.abs().real                   # |λ|
            if self.eig_target == 1.0:
                L_eig = ((mod - 1.0).abs() - self.eig_margin).clamp_min(0).pow(2).mean()
            elif self.eig_target == "<=1":
                L_eig = (mod - (1.0 - self.eig_margin)).clamp_min(0).pow(2).mean()

        loss = L_rec + alpha*L_pred + beta*L_kl + gamma*L_eig
        aux = dict(L_rec=L_rec.item(), L_pred=L_pred.item(), L_kl=L_kl.item(), L_eig=float(L_eig), 
                   max_eig=float(mod.max()), A=A.detach())
        return loss, xhat, aux
