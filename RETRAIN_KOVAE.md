# KoVAE 重训指南 🔥

## 📊 超参数变化

### 之前的超参数（重构优先）
```python
z_dim = 16          # 潜在空间太小
h_dim = 64          # GRU 隐层太小
alpha = 0.1         # 预测权重太低 ❌
beta = 0.001        # KL 权重太低
gamma = 0.0         # 没有谱约束 ❌
```

**问题**：模型主要学习重构，Koopman 算子没学好

---

### 现在的超参数（动力学优先）
```python
z_dim = 64          # 4x 增大潜在空间
h_dim = 256         # 4x 增大 GRU 隐层
alpha = 1.0         # 10x 增大预测权重 ✓
beta = 0.01         # 10x 增大 KL 权重
eig_gamma = 0.1     # 启用谱约束 ✓
eig_target = '<=1'  # 保证系统稳定性 ✓
dropout = 0.1       # 正则化
layer_norm = True   # 层归一化
```

**目标**：强制模型学习正确的 Koopman 动力学

---

## 🎯 核心改进

### 1. 增大预测一致性权重 (alpha: 0.1 → 1.0)
```python
loss = L_rec + 1.0*L_pred + 0.01*L_kl + 0.1*L_eig
              ^^^^
              现在预测和重构同等重要！
```

### 2. 启用谱约束 (gamma: 0 → 0.1)
```python
# 约束 Koopman 算子 A 的特征值 |λ| ≤ 1
# 保证系统稳定性，避免预测爆炸
```

### 3. 增大模型容量
```python
# 更大的潜在空间和隐层
# 更好地捕捉复杂的废水系统动力学
```

---

## 🚀 开始训练

### 步骤 1: 备份旧模型
```bash
cd /workspaces/ureca
mkdir -p save_model/kovae/waste_water_backup
cp save_model/kovae/waste_water/*.pt save_model/kovae/waste_water_backup/
cp save_model/kovae/waste_water/*.txt save_model/kovae/waste_water_backup/
```

### 步骤 2: 拉取新代码
```bash
git pull origin claude/wastewater-treatment-modeling-FE9ay
```

### 步骤 3: 启动训练
```bash
# 方式 1: 前台训练（可以看到进度）
python train.py kovae waste_water

# 方式 2: 后台训练（推荐，可以关闭终端）
nohup python train.py kovae waste_water > training_kovae_v2.log 2>&1 &

# 查看进程
ps aux | grep train.py

# 实时查看日志
tail -f training_kovae_v2.log
```

### 步骤 4: 监控训练
```bash
# 查看损失曲线（每 10 分钟运行一次）
tail -20 training_kovae_v2.log | grep "epoch"

# 或者用 Python 绘制损失曲线
python -c "
import numpy as np
import matplotlib.pyplot as plt
loss_train = np.loadtxt('loss/kovae/waste_water/0/loss_.txt')
loss_val = np.loadtxt('loss/kovae/waste_water/0/loss_t.txt')
plt.figure(figsize=(10, 5))
plt.plot(loss_train, label='train')
plt.plot(loss_val, label='val')
plt.yscale('log')
plt.legend()
plt.savefig('kovae_v2_loss.png')
print(f'训练损失: {loss_train[0]:.4f} -> {loss_train[-1]:.4f}')
print(f'验证损失: {loss_val[0]:.4f} -> {loss_val[-1]:.4f}')
"
```

---

## ⏱️ 预期时间

- **总轮数**: 401 epochs
- **预计时间**: 8-10 小时
- **检查点**: 每 10 epochs 保存一次模型

---

## 📈 预期结果

### 乐观情况 (成功) ✅
```
训练损失: 稳定下降
验证损失: 稳定下降
预测误差: < 30% (可接受)
```
→ KoVAE 学到了正确的动力学！

### 中等情况 (部分改善) ⚠️
```
训练损失: 稳定下降
验证损失: 稳定下降
预测误差: 30-100% (一般)
```
→ 有改善但不够好，可能需要进一步调参

### 悲观情况 (仍然失败) ❌
```
训练损失: 不稳定或很高
验证损失: 不稳定
预测误差: > 1000% (糟糕)
```
→ KoVAE 架构可能不适合这个问题

---

## 🔍 训练完成后测试

### 步骤 1: 运行测试脚本
```bash
git pull origin claude/wastewater-treatment-modeling-FE9ay
python test_kovae_prediction.py
```

### 步骤 2: 查看结果
```bash
# 应该看到类似：
平均相对误差: XX.X%

# 对比 MamKO:
# MamKO: 14-15% ✓
# KoVAE v1: 4392% ❌
# KoVAE v2: ??? (期待 < 30%)
```

---

## 📊 完整对比

训练完成后运行：
```bash
python compare_models_fixed.py
```

会生成：
- 详细误差对比
- 可视化对比图: `comparison_mamko_vs_kovae_fixed.png`

---

## 🛠️ 如果训练出问题

### 问题 1: 损失爆炸 (NaN)
```bash
# 降低学习率
# 修改 args_new.py:
args['learning_rate'] = 1e-4  # 原来 1e-3
```

### 问题 2: 损失不下降
```bash
# 检查数据是否正确加载
# 检查 GPU/CPU 使用情况
nvidia-smi  # 如果有 GPU
```

### 问题 3: 内存不足
```bash
# 减小 batch size
# 修改 args_new.py:
args['batch_size'] = 128  # 原来 256
```

---

## 📝 训练日志示例

正常的训练日志应该像这样：
```
[epoch 0] training...
epoch 0: loss_traning data 1.2345 loss_val data 1.3456 test_data 1.4567 ...
store!!!

[epoch 10] training...
epoch 10: loss_traning data 0.8234 loss_val data 0.9123 test_data 1.0234 ...
store!!!

[epoch 50] training...
draw...
epoch 50: loss_traning data 0.3456 loss_val data 0.4123 test_data 0.5678 ...
store!!!
```

---

## 💡 小贴士

1. **耐心等待**: 训练需要 8+ 小时，不要急
2. **定期检查**: 每小时看一次日志，确保没出错
3. **备份模型**: 每次重要的检查点都备份
4. **对比结果**: 训练完立即测试，对比改善程度

---

## 🎯 成功标准

如果新训练的 KoVAE 满足以下任意一条，就算成功：
- ✅ 预测误差 < 30%
- ✅ 预测误差比旧版本降低 100x
- ✅ 预测误差接近 MamKO (14-15%)

---

**现在开始训练吧！** 🚀

祝你好运！我会等你的结果 😊
