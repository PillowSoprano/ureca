# kovae.py
# 核心模型实现！需要导入一些基本的包。torch 是 PyTorch 的核心库，是用来构建深度学习模型的
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_activation(name: str):
    name = (name or "tanh").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name in ("none", "linear"):
        return None
    return nn.Tanh()

# 编码器：把输入时间序列 x 映射到潜空间 z
# 感觉这部分就是论文里说的 posterior q(z|x)？？
class GRUEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_layers=1, dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        # 用 GRU 来提取时间依赖特征！
        self.gru = nn.GRU(x_dim, h_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # 映射成均值和方差，这俩参数后面用来做重参数化（reparameterization）？
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.layer_norm = nn.LayerNorm(h_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = _build_activation(activation)
    def forward(self, x):              # x: [B,T,x_dim]
        # 前向传播，取出每个时间步的 hidden state
        h,_ = self.gru(x)              # [B,T,h_dim]
        if self.layer_norm:
            h = self.layer_norm(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        mu, logvar = self.mu(h), self.logvar(h)
        # reparameterization trick！
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  
        z = mu + eps * std             # reparameterization
        # 返回 z 以及它的参数（mu, logvar）
        return z, (mu, logvar)

# GRUPrior: 先验模型
# 是论文里那个 “Koopman prior” 的一部分，
# 它自己生成潜变量的时间演化，不依赖真实输入

class GRUPrior(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers=1):
        super().__init__()
        # 同样用 GRU，但输入是自己生成的 token
        self.gru = nn.GRU(z_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    def forward(self, T, B, device):                                 # x: [B,T,x_dim]
        # 初始化 z0，全零开始（论文里提到的 start token）
        z0 = torch.zeros(B, 1, self.mu.out_features, device=device)  # start token！
        # 把它复制 T 次，生成序列长度的“假输入”
        h,_ = self.gru(z0.repeat(1,T,1))                             # dummy roll
        mu, logvar = self.mu(h), self.logvar(h)
        # 一样的 reparameterization……
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        zbar = mu + eps * std                                        # \bar z_{1:T}
        return zbar, (mu, logvar)

# 能不能别报错了 我真的改不动了……
# koopman_A_from_zbar: 求 Koopman A
# 论文里，如果我没理解错：A 是在潜空间里拟合的线性变换
# 满足 A * z_t ≈ z_{t+1}。
def koopman_A_from_zbar(zbar):
    # zbar: [B,T,z_dim]，这里把 batch 和时间拼起来当样本点。
    B,T,k = zbar.shape
    Z0 = zbar[:, :-1, :].reshape(-1, k)   # [(B*(T-1)), k]
    Z1 = zbar[:, 1:,  :].reshape(-1, k)   # same
    # 最小二乘求解 A，使得 AZ0 ≈ Z1
    # A = argmin ||AZ0 - Z1||_F
    # 用 lstsq 比 pinv 稳定，还能反向传播（这点论文里也有提）
    A = torch.linalg.lstsq(Z0, Z1).solution.T   # [k,k]
    return A

# GRUDecoder: 解码器部分
# 从潜空间 z 恢复回原始的观测序列 x
# 用来对应论文里的 decoder p(x|z)
class GRUDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_layers=1, dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Linear(h_dim, x_dim)
        self.layer_norm = nn.LayerNorm(h_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = _build_activation(activation)
    def forward(self, z):
        h,_ = self.gru(z)
        if self.layer_norm:
            h = self.layer_norm(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        xhat = self.out(h)
        return xhat
        
# KoVAE的主体模型
# 也是论文的核心结构
# 这一块把 Encoder, Prior, Decoder 结合起来
# 同时通过 koopman_A_from_zbar 求线性动力系统矩阵 A
# 能不能别报错了 求你了
class KoVAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, h_dim=64, layers=1, eig_target=None, eig_margin=0.0,
                 dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        self.enc = GRUEncoder(x_dim, h_dim, z_dim, layers, dropout=dropout, layer_norm=layer_norm, activation=activation)
        self.pri = GRUPrior(z_dim, h_dim, layers)
        self.dec = GRUDecoder(z_dim, h_dim, x_dim, layers, dropout=dropout, layer_norm=layer_norm, activation=activation)
        # 这俩参数跟论文里谱约束有关，用来控制 Koopman 矩阵的特征值范围
        self.eig_target = eig_target     # None / 1.0 / "<=1"？（比如可以设成 1.0 或 “<=1”）
        self.eig_margin = eig_margin

    def forward(self, x, alpha=0.1, beta=1e-3, gamma=0.0):
        B,T,X = x.shape
        # posterior: 编码得到 z
        z, (mu_q, logvar_q) = self.enc(x)           # [B,T,k]
        # prior: 从随机 token 得到 zbar
        zbar, (mu_p, logvar_p) = self.pri(T, B, x.device)  # [B,T,k]
        # 从 zbar 估计 Koopman A
        # 如果 gamma==0，就不反传梯度；否则参与更新（控制谱约束）
        A = koopman_A_from_zbar(zbar.detach() if gamma==0 else zbar)  # 需要谱约束就别 detach
        # 用 A 推进一个先验序列（从 t=1 开始对齐）
        zhat = []
        zprev = zbar[:,0,:]                          # [B,k]
        zhat.append(zprev)
        for t in range(1, T):
            zprev = (zprev @ A.T)                    # [B,k]
            zhat.append(zprev)
        zhat = torch.stack(zhat, dim=1)              # [B,T,k]
        # 这就是论文里提到的「隐空间线性动力学」重构部分？

        # 解码器部分，从 posterior 的 z 生成重构的 x
        xhat = self.dec(z)

        # 计算损失函数
        # 重构损失：x vs xhat，衡量生成质量
        L_rec  = F.mse_loss(xhat, x)
        # 预测损失：z vs zbar，确保 posterior 和 prior 的时间一致性
        # 论文里叫 “consistency loss” 或 “Koopman prediction loss”
        L_pred = F.mse_loss(z, zbar)                 # 后验与先验的一致性？
        # KL 散度：逐时刻高斯的闭式解，保持 posterior 不偏离 prior
        KL = -0.5 * (1 + (logvar_q - logvar_p) - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        L_kl = KL.mean()
        # 特征值正则项（谱约束）：控制 Koopman 矩阵 A 的稳定性
        L_eig = 0.0
        mod = torch.tensor([0.0], device=x.device)
        if gamma > 0:
            evals = torch.linalg.eigvals(A)          # complex？复数？
            mod = evals.abs().real                   # 取模 |λ|
            if self.eig_target == 1.0:
                # 约束特征值尽量在单位圆附近
                L_eig = ((mod - 1.0).abs() - self.eig_margin).clamp_min(0).pow(2).mean()
            elif self.eig_target == "<=1":
                # 限制特征值模长不超过 1（系统稳定）
                L_eig = (mod - (1.0 - self.eig_margin)).clamp_min(0).pow(2).mean()
                
        # 总损失函数：对应论文 Eq. (14)
        # 各项权重 α、β、γ 分别控制不同约束的重要性
        loss = L_rec + alpha*L_pred + beta*L_kl + gamma*L_eig
        # 把中间结果都存起来方便调试（auxiliary dict）
        aux = dict(L_rec=L_rec.item(), L_pred=L_pred.item(), L_kl=L_kl.item(), L_eig=float(L_eig), 
                   max_eig=float(mod.max()), A=A.detach())
        # 返回总损失、重构结果和辅助信息
        return loss, xhat, aux

# 重点在 “Koopman A” 的求法，用最小二乘去拟合隐空间动力系统
# 论文里后面还加了谱正则项（限制特征值在单位圆上）
# 那部分应该在 loss 里体现？

