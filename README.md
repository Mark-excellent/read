softmax线性分类器

---

##  目录
- [ 项目亮点](#-项目亮点)
- [ 核心实现](#️-核心实现)
- [ 数值稳定性](#-数值稳定性)
- [ 梯度推导](#-梯度推导)
- [ 快速开始](#-快速开始)
- [ 超参数说明](#️-超参数说明)
- [ 实验结果](#-实验结果)
- [ 何时用优化版？](#-何时用优化版)
- [ 文件结构](#-文件结构)
- [ 参与贡献](#-参与贡献)
- [ 许可证](#-许可证)

---

##  项目亮点
| 维度 | 基础版（批量 GD） | 优化版（Mini-batch + Adam） |
| --- | --- | --- |
| **算法** | 朴素矩阵求导 | 随机采样 + 动量 |
| **优化器** | 固定学习率 | 自适应学习率（Adam） |
| **早停** | ❌ | ✅（省 30-45% 迭代） |
| **大数据** | 易内存爆炸 | 恒定内存（99%+ 节省） |
| **困难数据** | 易过拟合 | 过拟合 ↓22-58% |
| **准确率** | baseline | **+1.5%** |

---

##  核心实现
1. **前向传播**：线性分数 → Softmax 概率  
2. **损失函数**：交叉熵 + L2 正则  
3. **反向传播**：手工推导梯度（见下方公式）  
4. **参数更新**：支持 GD / SGD / Momentum / Adam  

---

## 数值稳定性
```python
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
```

---

## 梯度推导
```python
dscores = probs.copy()
dscores[range(N), y] -= 1
dscores /= N
dW = X.T @ dscores + reg * W
db = np.sum(dscores, axis=0)
```

---

## 快速开始
```bash
git clone https://github.com/Mark-excellent/softmax-classifier.git
cd softmax-classifier
python -m venv venv && source venv/bin/activate  
pip install -r requirements.txt
```

## 基础版Demo
```python
from Softmax线性分类器 import SoftmaxClassifier
clf = SoftmaxClassifier(learning_rate=0.5, n_iterations=1000)
clf.fit(X_train, y_train)
print("test acc:", clf.score(X_test, y_test))
```

## 优化版Demo
```python
from 优化版softmax import OptimizedSoftmaxClassifier
clf = OptimizedSoftmaxClassifier(optimizer='adam', early_stopping=True)
clf.fit(X_train, y_train, X_val, y_val)
print("test acc:", clf.score(X_test, y_test))
```

---

## 超参数说明
| 变量                               | 示例值  | 含义                       |
| -------------------------------- | ---- | ------------------------ |
| `SIMPLE_MODEL_LEARNING_RATE`     | 0.5  | 基础版学习率                   |
| `OPTIMIZED_MODEL_BATCH_SIZE`     | 32   | Mini-batch 大小            |
| `OPTIMIZED_MODEL_OPTIMIZER`      | adam | 可选 sgd / momentum / adam |
| `OPTIMIZED_MODEL_EARLY_STOPPING` | True | 是否早停                     |
| `DATASET_NOISE_LEVEL`            | 0.4  | 噪声强度（对比实验用）              |

---

## 实验结果
场景: 超高难度-噪声大 (5000×60×8, 噪声=0.6)
原始版(BGD):     测试准确率=69.20%  训练/测试差距=3.57%
优化版(Adam):    测试准确率=70.70%  训练/测试差距=1.50%
⇒ 准确率 +1.5%，过拟合 ↓58%，收敛轮数 ↓45%

在小/简单数据集上：两版本准确率均≈100%，基础版更快（无额外开销）。

---

## 何时用优化版？
 数据复杂（高维、类别重叠、高噪声）
 大数据集（>10 k 样本，内存受限）
 易过拟合（特征 >> 样本）
 生产部署（需要早停 + 可扩展）
 追求 SOTA（+1~2% 准确率提升）

 ---

 ## 文件结构
  softmax-classifier

├─ Softmax线性分类器.py           # 基础版（批量 GD）[Softmax线性分类器.py](https://github.com/user-attachments/files/23136095/Softmax.py)
├─ 优化版softmax.py              # 优化版（Mini-batch + Adam + 早停）[优化版so[comparison.py](https://github.com/user-attachments/files/23136065/comparison.py)
├─ comparison.py                 # 对比实验脚本[comparison.py](https://github.com/user-attachments/files/23136096/comparison.py)
├─ 性能差异详细说明.md           # 深度性能报告[性能差异详细说明.md](https://github.com/user-attachments/files/23136098/default.md)
├─ .env                          # 统一超参数配置
├─ requirements.txt              # 依赖（numpy/scipy）[requirements.txt](https://github.com/user-attachments/files/23136119/requirements.txt)
├─ logs/                         # 训练日志
├─ README.md                     # 本文件

## 参与贡献
1. 新增优化器（RMSprop、AdaGrad）
2. 学习率调度器
3. 多线程数据加载
4. 可视化决策边界

## 许可证
MIT License — 详见 LICENSE





