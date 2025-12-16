# MamKO vs KoVAE 对比报告 (v1 - 重训前)

**日期**: 2025-12-16
**目的**: 记录重训前的基线性能，分析 KoVAE 失败原因

---

## 📊 执行摘要

| 模型 | 测试损失 | 预测误差 (关键维度) | 训练时长 | 状态 |
|------|---------|---------------------|---------|------|
| **MamKO** | 0.159 | **14.3-14.8%** ✓ | ~8 小时 | 优秀 |
| **KoVAE v1** | 0.004 | **4391.7%** ❌ | ~8 小时 | 失败 |

**结论**: MamKO 性能优于 KoVAE v1 约 **300 倍**

---

## 🔍 问题发现：测试损失悖论

### 悖论现象

```
KoVAE 测试损失: 0.004   (比 MamKO 低 40 倍) ✓
KoVAE 预测误差: 4392%   (比 MamKO 高 300 倍) ❌
```

**为什么损失低但预测差？**

---

## 🐛 根本原因分析

### 1. 损失函数差异

#### MamKO 的损失函数
```python
# MamKO.py line 365-373
for i in range(L):
    z_t = A * z_{t-1} + B * u_t
    y_t = C @ z_t
    loss += MSE(y_t, x[O+i])  # 直接比较预测 vs 真实未来
```

**特点**: 直接测量多步预测能力 ✓

#### KoVAE v1 的损失函数
```python
# kovae_model.py line 219
loss = L_rec + alpha*L_pred + beta*L_kl + gamma*L_eig

其中:
L_rec  = MSE(xhat, x)    # 重构损失：解码器 vs 输入
L_pred = MSE(z, zbar)    # 潜在空间一致性
L_kl   = KL 散度          # 正则化
L_eig  = 特征值约束       # (gamma=0, 未启用)
```

**问题**:
- ❌ L_rec 只测试 encoder+decoder 的重构能力
- ❌ 不测试 Koopman 算子 A 的预测能力
- ❌ 测试损失 = 重构损失，不是预测损失

---

### 2. KoVAE v1 超参数分析

```python
# train.py line 89-102 (旧版)
z_dim = 16          # 潜在空间太小
h_dim = 64          # GRU 隐层太小
alpha = 0.1         # 预测一致性权重太低 ❌
beta = 0.001        # KL 权重太低
gamma = 0.0         # 没有特征值约束 ❌
use_action = False  # 没有使用动作信息
```

**权重分配**:
```
loss = 1.0*L_rec + 0.1*L_pred + 0.001*L_kl + 0.0*L_eig
       ^^^^^^^^    ^^^^^^^^^
       重构占主导   预测权重小
```

**结果**: 模型优先学习重构（压缩/解压），忽略动力学预测

---

## 📈 详细性能对比

### 训练损失曲线

#### MamKO
```
初始损失: 2.2269
最终损失: 0.0206
下降比例: 99.1%
趋势: 稳定下降 ✓
```

#### KoVAE v1
```
训练损失: 0.0287 → 0.0117 (59.2% 下降)
测试损失: 0.0108 → 0.0040 (63.0% 下降)
趋势: 稳定下降 ✓ (但优化的是错误的目标)
```

---

### 关键维度预测误差

**测试环境**:
- 数据: 废水处理系统 (159 维状态空间)
- 预测时域: 20 步
- 关键维度: 标准差最大的 5 个维度 [73, 69, 71, 70, 72]

#### MamKO 性能
```
维度 73: 相对误差 14.3%
维度 69: 相对误差 14.8%
维度 71: 相对误差 14.5%
维度 70: 相对误差 14.6%
维度 72: 相对误差 14.7%

平均: 14.6% ✓ 优秀
```

#### KoVAE v1 性能 (修复预测方法后)
```
维度 73: 相对误差 5356.59%
维度 69: 相对误差 5377.55%
维度 71: 相对误差 3377.99%
维度 70: 相对误差 3519.66%
维度 72: 相对误差 4326.64%

平均: 4391.7% ❌ 灾难性失败
```

**对比**: KoVAE v1 比 MamKO 差了 **300 倍**

---

## 🔧 预测方法修复过程

### 问题 1: 预测方法错误

**旧代码** (kovae_model.py 原版):
```python
def pred_forward_test(self, x, u, is_draw, args, epoch):
    loss, xhat, aux = self.model(xin, self.alpha, self.beta, self.gamma)
    return xhat.detach().cpu(), aux
```

**问题**: 只做了一次前向传播 (x → encode → decode → xhat)
- 这是**重构**，不是**预测**
- 没有使用 Koopman 算子 A 做递推

---

**新代码** (修复后):
```python
def pred_forward_test(self, x, u, is_draw, args, epoch):
    # 1. 编码初始状态
    z_init, _ = self.model.enc(x_init)
    z_0 = z_init[:, -1, :]

    # 2. 计算 Koopman 算子 A
    zbar, _ = self.model.pri(T, B, x.device)
    A = koopman_A_from_zbar(zbar)

    # 3. 递推预测: z_t = A @ z_{t-1}
    z_preds = []
    z_t = z_0
    for t in range(pred_horizon):
        z_t = z_t @ A.T
        z_preds.append(z_t)

    # 4. 解码预测
    xhat_pred = self.model.dec(z_preds)

    # 5. 计算预测损失 (vs 真实未来)
    pred_loss = mse_loss(xhat_pred, x_target)
    return xhat_pred, aux
```

**改进**:
- ✅ 使用 Koopman 算子 A 做线性递推
- ✅ 每步解码得到状态预测
- ✅ 与真实未来状态对比
- ✅ 返回真正的预测损失

---

### 问题 2: 维度不匹配

**错误信息**:
```
RuntimeError: The size of tensor a (161) must match the size of tensor b (159)
```

**原因**:
- 解码器输出: 161 维 (状态 159 + 动作 2)
- 目标状态: 159 维 (只有状态)

**修复** (kovae_model.py line 358-361):
```python
# 如果解码器输出维度不匹配，只取状态部分
if xhat_pred.shape[-1] != x_target.shape[-1]:
    xhat_pred = xhat_pred[:, :, :x_target.shape[-1]]
```

---

## 📊 测试结果详情

### 测试配置
```python
测试数据形状: x=torch.Size([1, 520, 159]), u=torch.Size([1, 519, 2])
预测输出形状: torch.Size([1, 40, 159])
预测损失 (标准化): 2.333014
```

### 反标准化后的误差 (RMSE)

| 维度 | 真实值均值 | RMSE | 相对RMSE |
|-----|-----------|------|---------|
| 73 | 3387.9 | 181463.66 | 5356.59% |
| 69 | 3366.4 | 181027.22 | 5377.55% |
| 71 | 5399.7 | 182412.18 | 3377.99% |
| 70 | 5178.0 | 182264.99 | 3519.66% |
| 72 | 4205.5 | 181927.68 | 4326.64% |

**观察**:
- 预测值完全偏离真实值
- RMSE 比真实值大 30-50 倍
- Koopman 算子 A 没有学到正确的动力学

---

## 🎯 失败原因总结

### 1. 训练目标错误
```
优化目标: 重构损失 (L_rec)
实际需要: 预测损失
结果: 学会了压缩/解压，但没学会预测
```

### 2. 超参数不当
```
alpha = 0.1  → 预测一致性权重太小
gamma = 0.0  → 没有特征值约束
z_dim = 16   → 潜在空间太小
```

### 3. Koopman 算子未学习
```
Koopman 算子 A 的特征值可能不稳定
线性递推 z_t = A @ z_{t-1} 发散
导致预测完全失控
```

---

## 🔄 改进方案 (v2 计划)

### 超参数调整

| 参数 | v1 (失败) | v2 (计划) | 变化 |
|------|----------|----------|------|
| alpha | 0.1 | 1.0 | 10x ↑ |
| beta | 0.001 | 0.01 | 10x ↑ |
| eig_gamma | 0.0 | 0.1 | 启用 |
| eig_target | None | '<=1' | 稳定性约束 |
| z_dim | 16 | 64 | 4x ↑ |
| h_dim | 64 | 256 | 4x ↑ |
| dropout | 0.0 | 0.1 | 正则化 |
| layer_norm | False | True | 归一化 |

### 期望效果

**损失函数权重**:
```
v1: loss = 1.0*L_rec + 0.1*L_pred + 0.001*L_kl + 0.0*L_eig
v2: loss = 1.0*L_rec + 1.0*L_pred + 0.01*L_kl  + 0.1*L_eig
            ^^^^^^^^    ^^^^^^^^^                  ^^^^^^^
            重构         预测同等重要              稳定性约束
```

**目标**:
- ✅ 强制模型学习 Koopman 动力学
- ✅ 保证 Koopman 算子 A 的稳定性
- ✅ 增大模型容量以捕捉复杂动力学

---

## 📝 实验记录

### MamKO 训练
```
命令: python train.py mamba waste_water
开始时间: 下午 3:00
结束时间: 晚上 11:00
训练时长: 8 小时
Epochs: 401
最终性能: 14.6% 相对误差 ✓
```

### KoVAE v1 训练
```
命令: python train.py kovae waste_water
训练时长: ~8 小时
Epochs: 401
训练损失: 0.0287 → 0.0117
测试损失: 0.0108 → 0.0040 (重构损失，不是预测损失)
最终性能: 4391.7% 相对误差 ❌
```

---

## 🎓 经验教训

### 1. 损失函数设计至关重要
```
教训: 低损失 ≠ 好性能
原因: 必须测试真正关心的指标（预测，不是重构）
```

### 2. 超参数权重影响巨大
```
教训: alpha=0.1 vs alpha=1.0 可能导致完全不同的结果
原因: 权重决定了模型优化的优先级
```

### 3. Koopman 算子需要约束
```
教训: 无约束的 Koopman 算子可能不稳定
解决: 启用谱约束 (|λ| ≤ 1) 保证系统稳定性
```

### 4. 预测方法必须正确
```
教训: 使用正确的预测方法才能评估真实性能
修复: 从"重构"改为"Koopman 递推预测"
```

---

## 📊 可视化对比

### 损失曲线对比
```
MamKO:   快速下降，稳定收敛 ✓
KoVAE v1: 平滑下降，但优化错误的目标 ❌
```

### 预测误差对比
```
MamKO:    14.6%   █
KoVAE v1: 4391.7% ████████████████████████████████████ (300x)
```

---

## 🔮 展望

### 成功标准 (v2)
```
保守: 预测误差 < 100%  (比 v1 改善 40x)
乐观: 预测误差 < 30%   (可接受水平)
理想: 预测误差 < 20%   (接近 MamKO)
```

### 如果 v2 仍然失败
```
可能原因:
1. KoVAE 架构不适合废水处理系统
2. 线性 Koopman 算子不足以捕捉非线性动力学
3. 需要更复杂的模型 (如非线性 Koopman)

备选方案:
- 使用 MamKO (已验证有效)
- 尝试其他方法 (Transformer, Neural ODE 等)
```

---

## 📁 相关文件

```
代码修改:
- kovae_model.py (line 311-374): 修复预测方法
- train.py (line 89-105): 准备 v2 超参数

测试脚本:
- test_kovae_prediction.py: 独立测试脚本
- compare_models_fixed.py: 完整对比脚本
- diagnose_kovae_loss.py: 损失悖论诊断

文档:
- KoVAE_FIX_SUMMARY.md: 修复总结
- RETRAIN_KOVAE.md: 重训指南
- REPORT_MamKO_vs_KoVAE_v1.md: 本报告

模型:
- save_model/mamba/waste_water/*.pt: MamKO 模型
- save_model/kovae/waste_water/*.pt: KoVAE v1 模型
- save_model/kovae/waste_water_backup/*.pt: 备份 (待创建)
```

---

## 🏁 结论

**当前状态**:
- ✅ MamKO 性能优秀 (14.6% 误差)
- ❌ KoVAE v1 完全失败 (4391.7% 误差)
- ✅ 已诊断问题根源
- ✅ 已准备 v2 改进方案

**下一步**:
1. 备份 KoVAE v1 模型
2. 使用新超参数重新训练
3. 测试 v2 性能
4. 对比 v1 vs v2 改善

**风险评估**:
- 中等风险: v2 可能仍然不如 MamKO
- 低风险: v2 应该比 v1 显著改善
- 时间成本: 8+ 小时训练时间

---

**报告生成时间**: 2025-12-16
**作者**: Claude (Anthropic)
**状态**: 准备开始 KoVAE v2 训练
