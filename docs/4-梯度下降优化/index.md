# 梯度下降优化 (Gradient Descent)

## 1. 如何最小化损失函数？

交叉熵损失函数 $E_{in}(\mathbf{w})$ 是一个连续、可微的凸函数。这意味着它有一个全局最低点（"山谷"的谷底），我们可以通过求导来找到它。当函数的梯度为零时，就达到了最低点。

- **目标**：找到 $\mathbf{w}$ 使得梯度 $\nabla E_{in}(\mathbf{w}) = 0$。

## 2. 梯度下降算法

直接解方程 $\nabla E_{in}(\mathbf{w}) = 0$ 可能很复杂，所以我们采用一种迭代的优化方法——梯度下降。

### 核心思想

从一个初始点出发，沿着当前位置**梯度下降最快**的方向（即梯度的反方向）走一小步，如此反复，最终就能到达谷底。

### 迭代更新规则

$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t - \eta \nabla E_{in}(\mathbf{w}_t)$$

其中：
- $\mathbf{w}_t$ 是第 $t$ 次迭代的权重。
- $\eta$ 是**学习率 (learning rate)**，控制每一步走多大。
- $\nabla E_{in}(\mathbf{w}_t)$ 是损失函数在 $\mathbf{w}_t$ 处的梯度。

## 3. 逻辑回归的梯度

我们需要计算交叉熵损失函数的梯度。对于逻辑回归，交叉熵损失函数为：

$$E_{in}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$$

其中 $y_n \in \{-1, +1\}$ 是标签，$\mathbf{x}_n$ 是第 $n$ 个样本的特征向量。

### 梯度推导过程

对损失函数关于权重 $\mathbf{w}$ 求偏导：

$$\frac{\partial E_{in}(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$$

使用链式法则：

$$\frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}) = \frac{1}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} \cdot \frac{\partial}{\partial \mathbf{w}} (1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$$

$$= \frac{1}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} \cdot e^{-y_n \mathbf{w}^T \mathbf{x}_n} \cdot \frac{\partial}{\partial \mathbf{w}} (-y_n \mathbf{w}^T \mathbf{x}_n)$$

$$= \frac{e^{-y_n \mathbf{w}^T \mathbf{x}_n}}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} \cdot (-y_n \mathbf{x}_n)$$

注意到 $\frac{e^{-y_n \mathbf{w}^T \mathbf{x}_n}}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} = \frac{1}{1 + e^{y_n \mathbf{w}^T \mathbf{x}_n}} = \theta(-y_n \mathbf{w}^T \mathbf{x}_n)$，其中 $\theta(x) = \frac{e^x}{1+e^x}$ 是 sigmoid 函数。

因此，最终的梯度为：

$$\nabla E_{in}(\mathbf{w}) = \frac{\partial E_{in}(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{N} \sum_{n=1}^{N} \theta(-y_n \mathbf{w}^T \mathbf{x}_n) (-y_n \mathbf{x}_n)$$

### 应用到优化

将此梯度公式代入梯度下降的更新规则中，我们就可以通过迭代的方式，一步步地找到使交叉熵损失最小化的最优权重 $\mathbf{w}$，从而完成逻辑回归模型的训练。

## 4. 梯度下降的要点

- **学习率的选择**：学习率太大可能导致无法收敛（振荡），太小则收敛速度慢。
- **初始化**：权重的初始值会影响收敛速度和最终结果。
- **停止条件**：通常设置最大迭代次数或当梯度足够小时停止。
