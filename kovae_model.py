# kovae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ========= KoVAE 核心 =========

class GRUEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, layers=1):
        super().__init__()
        self.gru = nn.GRU(x_dim, h_dim, num_layers=layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    def forward(self, x):              # [B,T,x_dim]
        h,_ = self.gru(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, (mu, logvar)

class GRUPrior(nn.Module):
    def __init__(self, z_dim, h_dim, layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=layers, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.z_dim = z_dim
    def forward(self, T, B, device):
        # 用零向量“打拍子”生成长度为 T 的先验参数
        z0 = torch.zeros(B, T, self.z_dim, device=device)
        h,_ = self.gru(z0)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        zbar = mu + torch.randn_like(std) * std
        return zbar, (mu, logvar)

def koopman_A_from_zbar(zbar):         # zbar: [B,T,k]
    B,T,k = zbar.shape
    Z0 = zbar[:, :-1, :].reshape(-1, k)   # [(B*(T-1)), k]
    Z1 = zbar[:,  1:, :].reshape(-1, k)
    A = torch.linalg.lstsq(Z0, Z1).solution.T   # [k,k]
    return A

class GRUDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, h_dim, num_layers=layers, batch_first=True)
        self.out = nn.Linear(h_dim, x_dim)
    def forward(self, z):               # [B,T,z_dim]
        h,_ = self.gru(z)
        xhat = self.out(h)
        return xhat

class KoVAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, h_dim=64, layers=1, eig_target=None, eig_margin=0.0):
        super().__init__()
        self.enc = GRUEncoder(x_dim, h_dim, z_dim, layers)
        self.pri = GRUPrior(z_dim, h_dim, layers)
        self.dec = GRUDecoder(z_dim, h_dim, x_dim, layers)
        self.eig_target = eig_target
        self.eig_margin = eig_margin

    def forward(self, x, alpha=0.1, beta=1e-3, gamma=0.0):
        B,T,X = x.shape
        z, (mu_q, logvar_q) = self.enc(x)                  # posterior
        zbar, (mu_p, logvar_p) = self.pri(T, B, x.device)  # prior
        # Koopman A（若不做谱约束，detach 稳一点）
        A = koopman_A_from_zbar(zbar.detach() if gamma == 0 else zbar)
        # 用 A 线性推进先验（仅用于监控/可视化，不入损失）
        with torch.no_grad():
            zlin = []
            zprev = zbar[:,0,:]
            zlin.append(zprev)
            for t in range(1,T):
                zprev = (zprev @ A.T)
                zlin.append(zprev)
            zlin = torch.stack(zlin, dim=1)                # [B,T,k]

        xhat = self.dec(z)

        L_rec  = F.mse_loss(xhat, x)
        L_pred = F.mse_loss(z, zbar)                       # posterior vs prior 一致性
        KL = -0.5 * (1 + (logvar_q - logvar_p) - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        L_kl = KL.mean()

        L_eig = 0.0
        max_eig = 0.0
        if gamma > 0:
            evals = torch.linalg.eigvals(A)
            mod = evals.abs().real
            max_eig = float(mod.max())
            if self.eig_target == 1.0:
                L_eig = ((mod - 1.0).abs() - self.eig_margin).clamp_min(0).pow(2).mean()
            elif self.eig_target == "<=1":
                L_eig = (mod - (1.0 - self.eig_margin)).clamp_min(0).pow(2).mean()

        loss = L_rec + alpha*L_pred + beta*L_kl + gamma*L_eig
        aux = dict(L_rec=L_rec.item(), L_pred=L_pred.item(), L_kl=L_kl.item(),
                   L_eig=float(L_eig), max_eig=max_eig, A=A.detach())
        return loss, xhat, aux

# ========= 适配你工程接口的“外壳” =========

class Koopman_Desko:
    """
    这个类的名字与现有工程保持一致，供 train.py 按 method 动态 import。
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 选择输入维度：默认只用状态 x；如果 args['use_action']=True，则拼 [x,u]
        self.use_action = bool(args.get('use_action', False))
        x_dim = args['state_dim']
        if self.use_action:
            x_dim = args['state_dim'] + args['act_dim']

        z_dim = int(args.get('z_dim', 16))
        h_dim = int(args.get('h_dim', 64))
        layers = int(args.get('layers', 1))
        eig_target = args.get('eig_target', None)  # 可设为 1.0 或 "<=1"
        eig_margin = float(args.get('eig_margin', 0.0))

        self.model = KoVAE(x_dim, z_dim, h_dim, layers, eig_target, eig_margin).to(self.device)

        lr = float(args.get('learning_rate', 1e-3))
        wd = float(args.get('weight_decay', 1e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        self.grad_clip = float(args.get('grad_clip', 1.0))
        self.alpha = float(args.get('alpha', 0.1))
        self.beta  = float(args.get('beta', 1e-3))
        self.gamma = float(args.get('eig_gamma', 0.0))

        self.loss_store = 0.0     # 训练损失（给 train.py 存盘）
        self.loss_store_t = 0.0   # 验证/测试损失（给 train.py 存盘）

    def _make_batch(self, batch):
        """
        你的 ReplayMemory 数据集迭代返回的是 (x, u)。
        这里把它搬到 device，并按需要拼接。
        约定形状：x:[B,T,state_dim], u:[B,T,act_dim] 或 [B,act_dim]（按你的数据而定）
        """
        x, u = batch
        x = x.float().to(self.device)
        u = u.float().to(self.device)
        if self.use_action:
            # 若 u 是 [B,act_dim]，扩成 [B,T,act_dim]
            if u.dim() == 2 and x.dim() == 3:
                u = u[:, None, :].expand(x.size(0), x.size(1), u.size(-1))
            xin = torch.cat([x, u], dim=-1)
        else:
            xin = x
        return x, u, xin

    @torch.no_grad()
    def pred_forward_test(self, x, u, is_draw, args, epoch):
        """
        给 train.py 调用的测试接口。这里简单做一次前向并返回预测。
        若要画图，你可以在 is_draw 为 True 时保存图片到与工程一致的目录。
        """
        self.model.eval()
        x, u, xin = self._make_batch((x, u))
        loss, xhat, aux = self.model(xin, self.alpha, self.beta, self.gamma)
        return xhat.detach().cpu(), aux

    def learn(self, epoch, x_train, x_val, x_test, args):
        """
        单个 epoch 的训练与验证。x_train/x_val 都是 Dataset。
        """
        batch_size = int(args.get('batch_size', 64))
        train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(x_val,   batch_size=batch_size, shuffle=False, drop_last=False)

        # 训练
        self.model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x, u, xin = self._make_batch(batch)
            loss, xhat, aux = self.model(xin, self.alpha, self.beta, self.gamma)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        self.loss_store = running / max(1, n)

        # 验证
        self.model.eval()
        running_v = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                x, u, xin = self._make_batch(batch)
                loss, _, _ = self.model(xin, self.alpha, self.beta, self.gamma)
                running_v += loss.item() * x.size(0)
                n_v += x.size(0)
        self.loss_store_t = running_v / max(1, n_v)

    # ====== 保存/恢复，与工程一致 ======

    def parameter_store(self, args):
        path_m = args['save_model_path']
        path_o = args['save_opti_path']
        torch.save(self.model.state_dict(), path_m)
        torch.save(self.optimizer.state_dict(), path_o)

    def parameter_restore(self, args):
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
