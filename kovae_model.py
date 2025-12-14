# kovae_model.py
# 这是 KoVAE 的核心实现！尝试用 PyTorch 复现它
# 论文关键：VAE + Koopman 线性动力学（Eq.6: z_t = A * z_{t-1}）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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

# 这部分是论文的核心实现！
# KoVAE = Koopman Variational Autoencoder
# 主要思想：用线性的 Koopman 算子来建模潜在空间的动态，而不是传统 VAE 的静态高斯先验
class GRUEncoder(nn.Module):
# 后验编码器 q(z|x) - 论文 4.1 节
# 作用：把观测序列 x_{1:T} 编码成潜在表示 z_{1:T}
# 用 GRU是因为时序数据有前后依赖关系，RNN 能捕捉这种依赖！
# 输出两个东西：
# 1. 采样的 z (通过重参数化技巧)
# 2. 均值 mu 和方差 logvar (用于计算 KL 散度)

    def __init__(self, x_dim, h_dim, z_dim, layers=1, dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        # GRU 用来提取时序特征，batch_first=True 意味着输入是 [B,T,x_dim]
        self.gru = nn.GRU(x_dim, h_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # 两个线性层分别输出均值和对数方差
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.layer_norm = nn.LayerNorm(h_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = _build_activation(activation)
    def forward(self, x):              # [B,T,x_dim] - B是batch，T是时间步，x_dim是输入维度
        # GRU 处理整个序列，得到每个时间步的隐状态
        h,_ = self.gru(x)              # h: [B,T,h_dim]
        if self.layer_norm:
            h = self.layer_norm(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        # 计算后验分布的参数
        mu, logvar = self.mu(h), self.logvar(h)
        # 重参数化技巧！这是 VAE 的关键
        # 不能直接从 N(mu, sigma^2) 采样（不可导），所以用 z = mu + sigma * epsilon
        # 其中 epsilon ~ N(0,1)，这样反向传播时梯度可以流过 mu 和 sigma
        std = torch.exp(0.5 * logvar)    # sigma = exp(0.5 * log(sigma^2))
        z = mu + torch.randn_like(std) * std    # 采样 z_{1:T}
        return z, (mu, logvar)    # 返回采样结果和分布参数

class GRUPrior(nn.Module):
# 先验分布 p(z) - 论文 4.2 节的核心创新！
# 传统 VAE：p(z) = N(0, I)，就是标准正态分布
# KoVAE：p(z_{1:T}) 是时序的，用 GRU 生成一个"先验序列"
# 这个先验序列 \bar{z}_{1:T} 后面会用来拟合 Koopman 算子 A
# 使得 A * \bar{z}_t ≈ \bar{z}_{t+1}，即线性动态系统
    def __init__(self, z_dim, h_dim, layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.z_dim = z_dim
    def forward(self, T, B, device):
        # 用零向量"打拍子"生成长度为 T 的序列
        # 因为我想要一个长度为 T 的先验，但不依赖具体的输入!
        # GRU 会根据它自己学到的隐状态来生成合理的时序先验
        z0 = torch.zeros(B, T, self.z_dim, device=device)
        h,_ = self.gru(z0)    # GRU "展开"T 步
        
        # 生成先验分布的参数
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        
        # 采样得到 \bar{z}_{1:T}，这是我们的"先验序列"
        zbar = mu + torch.randn_like(std) * std
        return zbar, (mu, logvar)

def koopman_A_from_zbar(zbar):         # zbar: [B,T,k]
# 计算 Koopman 算子 A - 论文最核心的部分！
# Koopman 理论：非线性系统可以用无穷维线性算子表示
# 这里用有限维近似：找一个矩阵 A，使得 A * z_t ≈ z_{t+1}
# 方法：最小二乘法 (类似 DMD - Dynamic Mode Decomposition)
# min ||A * Z_0 - Z_1||_F^2
# 其中 Z_0 = [z_0, z_1, ..., z_{T-2}], Z_1 = [z_1, z_2, ..., z_{T-1}]
# 1. 线性系统容易分析（特征值、稳定性等）
# 2. 可以施加物理约束（如 |λ| ≤ 1 保证稳定）
# 3. 更好的长期预测能力

    B,T,k = zbar.shape
    # 构造 Z_0 和 Z_1
    # Z_0: 前 T-1 个时间步，展平成 [(B*(T-1)), k]
    Z0 = zbar[:, :-1, :].reshape(-1, k)   # [(B*(T-1)), k]
    # Z_1: 后 T-1 个时间步（对应 Z_0 的下一时刻）
    Z1 = zbar[:,  1:, :].reshape(-1, k)
    # 求解 A^T * Z0^T = Z1^T，即 Z0 * A^T = Z1
    # lstsq 比 pinv 更稳定，而且可以反向传播
    A = torch.linalg.lstsq(Z0, Z1).solution.T   # [k,k]
    return A

class GRUDecoder(nn.Module):
    # 解码器 p(x|z) - 论文 4.1 节
    # 作用：从潜在表示 z_{1:T} 重构观测序列 x_{1:T}
    def __init__(self, z_dim, h_dim, x_dim, layers=1, dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(h_dim, x_dim)
        self.layer_norm = nn.LayerNorm(h_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = _build_activation(activation)
    def forward(self, z):               # [B,T,z_dim]
        h,_ = self.gru(z)
        if self.layer_norm:
            h = self.layer_norm(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        xhat = self.out(h)              # 重构的序列
        return xhat

class KoVAE(nn.Module):
    # 完整的 KoVAE 模型 - 整合所有组件
    # 训练目标（论文公式 8）：
    # L = E[log p(x|z)] + α*L_pred - β*KL[q(z|x) || p(z)] + γ*L_eig
    # 各项含义（防止自己忘了）：
    # - log p(x|z): 重构损失，让生成的 x̂ 接近真实 x
    # - L_pred: 预测损失，让后验 z 接近先验 \bar{z}（增强一致性）
    # - KL: 正则化，让后验分布不要偏离先验太远
    # - L_eig: 特征值约束，控制 Koopman 算子的动态特性（可选）
    def __init__(self, x_dim, z_dim=16, h_dim=64, layers=1, eig_target=None, eig_margin=0.0,
                 dropout=0.0, layer_norm=False, activation="tanh"):
        super().__init__()
        self.enc = GRUEncoder(x_dim, h_dim, z_dim, layers, dropout=dropout, layer_norm=layer_norm, activation=activation)
        self.pri = GRUPrior(z_dim, h_dim, layers)
        self.dec = GRUDecoder(z_dim, h_dim, x_dim, layers, dropout=dropout, layer_norm=layer_norm, activation=activation)
        # 特征值约束参数（论文 4.3 节）
        # eig_target: None（无约束）, 1.0（单位圆上）, "<=1"（单位圆内，保证稳定性）
        self.eig_target = eig_target
        self.eig_margin = eig_margin

    def forward(self, x, alpha=0.1, beta=1e-3, gamma=0.0):
        # 前向传播 + 计算损失
        # 流程：
        # 1. 编码：x -> z (后验)
        # 2. 生成先验：\bar{z}
        # 3. 计算 Koopman 算子 A
        # 4. 解码：z -> x̂
        # 5. 计算各项损失
        B,T,X = x.shape
        # 步骤 1: 后验分布 q(z|x)
        z, (mu_q, logvar_q) = self.enc(x)                  # posterior
        # 步骤 2: 先验分布 p(z)
        
        zbar, (mu_p, logvar_p) = self.pri(T, B, x.device)  # prior
        
        # 步骤 3: 计算 Koopman 算子 A
        # 注意：如果 gamma=0（不做谱约束），就 detach，避免影响 A 的学习
        # 如果 gamma>0，就让梯度流过，以便优化特征值
        
        A = koopman_A_from_zbar(zbar.detach() if gamma == 0 else zbar)
        # 用 A 线性推进先验（仅用于监控/可视化，不入损失）
        with torch.no_grad():
            zlin = []
            zprev = zbar[:,0,:]
            zlin.append(zprev)
            for t in range(1,T):
                zprev = (zprev @ A.T)                      # 矩阵乘法：[B,k] @ [k,k] = [B,k]
                zlin.append(zprev)
            zlin = torch.stack(zlin, dim=1)                # [B,T,k]
        
        # 步骤 4: 解码
        xhat = self.dec(z)

        # 损失计算
        # 别报错了 我真的改不动了(╥﹏╥)
        # 1. 重构损失：生成的 x̂ 要接近真实 x!
        L_rec  = F.mse_loss(xhat, x)
        # 2. 预测损失：后验 z 要接近先验 \bar{z}
        # 这是论文公式 7，增强后验和先验的一致性
        L_pred = F.mse_loss(z, zbar)                       # posterior vs prior 一致性

        # 3. KL 散度：正则化项
        # KL[q(z|x) || p(z)] 的闭式解（两个高斯分布之间）
        # 公式：-0.5 * (1 + log(σ_q^2/σ_p^2) - (μ_q-μ_p)^2/σ_p^2 - σ_q^2/σ_p^2)
        KL = -0.5 * (1 + (logvar_q - logvar_p) - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        L_kl = KL.mean()

        # 4. 特征值约束损失（论文 4.3 节，公式 9）
        L_eig = 0.0
        max_eig = 0.0
        if gamma > 0:
            # 计算 A 的特征值
            evals = torch.linalg.eigvals(A)
            mod = evals.abs().real
            max_eig = float(mod.max())
            # 根据目标施加不同的约束
            if self.eig_target == 1.0:
                # 目标：|λ| = 1（单位圆上，保持能量）
                # 损失：||λ| - 1| > margin 的部分才惩罚
                L_eig = ((mod - 1.0).abs() - self.eig_margin).clamp_min(0).pow(2).mean()
            elif self.eig_target == "<=1":
                # 目标：|λ| ≤ 1（单位圆内，保证稳定性）
                # 损失：|λ| > (1 - margin) 的部分才惩罚
                L_eig = (mod - (1.0 - self.eig_margin)).clamp_min(0).pow(2).mean()

        # 总损失（论文公式 8）
        loss = L_rec + alpha*L_pred + beta*L_kl + gamma*L_eig
        aux = dict(L_rec=L_rec.item(), L_pred=L_pred.item(), L_kl=L_kl.item(),
                   L_eig=float(L_eig), max_eig=max_eig, A=A.detach())
        # 返回辅助信息用于监控
        return loss, xhat, aux

# 适配工程接口的“外壳”，和现有的训练框架（train.py）兼容

class Koopman_Desko:
# 这个类的名字与现有工程保持一致，供 train.py 按 method 动态 import。
# 封装了 KoVAE 模型，提供统一的训练/测试接口
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 选择输入维度：默认只用状态 x；如果 args['use_action']=True，则拼 [x,u]
        # 这是为了支持有控制输入的动态系统
        self.use_action = bool(args.get('use_action', False))
        x_dim = args['state_dim']
        if self.use_action:
            x_dim = args['state_dim'] + args['act_dim']

        # 从配置读取超参数
        z_dim = int(args.get('z_dim', 64))    # 潜在空间维度
        h_dim = int(args.get('h_dim', 256))    # GRU 隐状态维度
        layers = int(args.get('layers', 1))    # GRU 层数
        eig_target = args.get('eig_target', None)  # 可设为 1.0 或 "<=1"
        eig_margin = float(args.get('eig_margin', 0.0))
        dropout = float(args.get('dropout', 0.0))
        layer_norm = bool(args.get('layer_norm', False))
        activation = args.get('activation', 'tanh')

        # 创建模型
        self.model = KoVAE(x_dim, z_dim, h_dim, layers, eig_target, eig_margin,
                           dropout=dropout, layer_norm=layer_norm, activation=activation).to(self.device)

        # 优化器：AdamW 带权重衰减（正则化）
        lr = float(args.get('learning_rate', 1e-3))
        wd = float(args.get('weight_decay', 1e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler_type = args.get('lr_scheduler', 'none')
        self.scheduler = None
        if scheduler_type != 'none':
            step_size = int(args.get('scheduler_step', 50))
            gamma = float(args.get('scheduler_gamma', 0.5))
            min_lr = float(args.get('scheduler_min_lr', 1e-6))
            if scheduler_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=int(args.get('num_epochs', 100)), eta_min=min_lr)
            elif scheduler_type == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=step_size)

        # 训练相关参数
        self.grad_clip = float(args.get('grad_clip', 1.0))
        self.alpha = float(args.get('alpha', 0.1))
        self.beta  = float(args.get('beta', 1e-3))
        self.gamma = float(args.get('eig_gamma', 0.0))

        # 损失记录（给 train.py 存盘用的！）
        self.loss_store = 0.0     # 训练损失
        self.loss_store_t = 0.0   # 验证/测试损失

    def _make_batch(self, batch):
# ReplayMemory 数据集迭代返回的是 (x, u)。
# HybridDataset 则返回 (x, u, sim, mask)。
        if len(batch) == 4:
            x, u, sim, mask = batch
            sim = sim.float().to(self.device)
            mask = mask.float().to(self.device)
        else:
            x, u = batch
            sim = None
            mask = None
        x = x.float().to(self.device)
        u = u.float().to(self.device)
        if self.use_action:
            # 如果 u 是 [B,act_dim]，扩展成 [B,T,act_dim]
            if u.dim() == 2 and x.dim() == 3:
                u = u[:, None, :].expand(x.size(0), x.size(1), u.size(-1))
            # 拼接 [x, u]
            xin = torch.cat([x, u], dim=-1)
        else:
            xin = x
        return x, u, xin, sim, mask

    @torch.no_grad()
    def pred_forward_test(self, x, u, is_draw, args, epoch):
# 给 train.py 调用的测试接口，这里简单做一次前向并返回预测！
# 若要画图，可以在 is_draw 为 True 时保存图片到与工程一致的目录！

        self.model.eval()
        x, u, xin, _, _ = self._make_batch((x, u))
        loss, xhat, aux = self.model(xin, self.alpha, self.beta, self.gamma)
        return xhat.detach().cpu(), aux

    def learn(self, epoch, x_train, x_val, x_test, args):
# 单个 epoch 的训练与验证
# 流程：
# 1. 训练：遍历 train_loader，更新参数
# 2. 验证：遍历 val_loader，计算验证损失
        batch_size = int(args.get('batch_size', 64))
        train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(x_val,   batch_size=batch_size, shuffle=False, drop_last=False)

        # 训练
        self.model.train()
        running = 0.0    # 累计损失
        n = 0    # 样本数
        cw = float(args.get('control_weight', 0.0))
        sw = float(args.get('sim_weight', 0.0))
        for batch in train_loader:
            x, u, xin, sim, mask = self._make_batch(batch)
            # 前向传播
            loss, xhat, aux = self.model(xin, self.alpha, self.beta, self.gamma)
            if cw > 0:
                control_penalty = (u ** 2).mean()
                loss = loss + cw * control_penalty
                aux['L_ctrl'] = float(control_penalty)
            if sw > 0 and sim is not None and mask is not None:
                sim_loss = nn.functional.mse_loss(xhat, sim, reduction='none')
                sim_loss = (sim_loss.mean(dim=(1, 2)) * mask.squeeze(-1)).mean()
                loss = loss + sw * sim_loss
                aux['L_sim'] = float(sim_loss)

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸，RNN 常用）
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # 更新参数
            self.optimizer.step()

            # 累积损失
            running += loss.item() * x.size(0)
            n += x.size(0)
        self.loss_store = running / max(1, n)    # 平均训练损失

        # 验证
        self.model.eval()
        running_v = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                x, u, xin, sim, mask = self._make_batch(batch)
                loss, xhat, _ = self.model(xin, self.alpha, self.beta, self.gamma)
                if sw > 0 and sim is not None and mask is not None:
                    sim_loss = nn.functional.mse_loss(xhat, sim, reduction='none')
                    sim_loss = (sim_loss.mean(dim=(1, 2)) * mask.squeeze(-1)).mean()
                    loss = loss + sw * sim_loss
                running_v += loss.item() * x.size(0)
                n_v += x.size(0)
        self.loss_store_t = running_v / max(1, n_v)

        # 调整学习率
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.loss_store_t)
            else:
                self.scheduler.step()

    # 保存/恢复，与工程一致

    def parameter_store(self, args):
        # 保存模型和优化器状态
        path_m = args['save_model_path']
        path_o = args['save_opti_path']
        torch.save(self.model.state_dict(), path_m)
        torch.save(self.optimizer.state_dict(), path_o)

    def parameter_restore(self, args):
        # 加载模型和优化器状态
        path_m = args['save_model_path']
        path_o = args['save_opti_path']

        # 自动检测设备：优先 CUDA，其次 MPS，再 CPU
        if torch.cuda.is_available():
            map_loc = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            map_loc = torch.device("mps")
        else:
            map_loc = torch.device("cpu")

        # 加载模型和优化器
        state = torch.load(path_m, map_location=map_loc)
        opti  = torch.load(path_o, map_location=map_loc)
        self.model.load_state_dict(state)
        self.optimizer.load_state_dict(opti)

        # 把模型搬到正确的设备上
        self.model.to(map_loc)
        self.device = map_loc
        print(f"^_^ Model restored to device: {self.device}")


