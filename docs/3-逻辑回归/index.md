# 逻辑回归 (Logistic Regression)

## 1. 从硬分类到软分类（概率）

感知机做的是"硬分类"，输出是确定的 $+1$ 或 $-1$。但在很多场景下，我们更希望得到一个概率，例如"一个病人有多大概率患有心脏病？"

- **目标**：我们希望模型的输出 $h(\mathbf{x})$ 能够估计目标概率 $f(\mathbf{x}) = P(+1|\mathbf{x}) \in [0, 1]$。

## 2. 逻辑函数 (Logistic Function)

我们需要一个函数，能将线性模型计算出的分数 $s = \mathbf{w}^T \mathbf{x}$（其范围是整个实数域）映射到 $[0, 1]$ 区间来表示概率。**逻辑函数**（也称 **Sigmoid 函数**）正好能满足这个需求。

### 标准逻辑函数 $\theta(s)$

$$\theta(s) = \frac{1}{1 + e^{-s}} = \frac{e^s}{1 + e^s}$$

### 函数特性

这是一个平滑、单调递增的S形曲线，可以将任何实数映射到 $(0, 1)$ 区间。

## 3. 逻辑回归的假设函数

将线性模型与逻辑函数结合，就得到了逻辑回归的假设函数：

$$h(\mathbf{x}) = \theta(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{w}^T \mathbf{x})}$$

这个 $h(\mathbf{x})$ 就作为对目标概率 $P(y=+1|\mathbf{x})$ 的估计。

## 4. 逻辑回归的损失函数推导（交叉熵损失）

如何衡量 $h(\mathbf{x})$ 的好坏？我们使用**最大似然估计 (Maximum Likelihood Estimation)** 的思想来推导其损失函数。

### 步骤 1：构建似然函数

假设数据点的标签 $y_n \in \{+1, -1\}$。

- 当 $y_n = +1$ 时，我们希望 $h(\mathbf{x}_n)$ 尽可能大。
- 当 $y_n = -1$ 时，我们希望 $1-h(\mathbf{x}_n)$ 尽可能大。

整个训练集的似然函数（即模型产生这些观测数据的概率）为所有样本点概率的乘积：

$$\text{Likelihood}(h) \propto \prod_{n=1}^{N} P(y_n|\mathbf{x}_n)$$

### 步骤 2：数学简化

利用 $1 - \theta(s) = \theta(-s)$ 的性质，我们可以将 $P(y_n|\mathbf{x}_n)$ 统一写成 $\theta(y_n \mathbf{w}^T \mathbf{x}_n)$。

因此：

$$\text{Likelihood}(h) \propto \prod_{n=1}^{N} \theta(y_n \mathbf{w}^T \mathbf{x}_n)$$

### 步骤 3：最大化对数似然

我们的目标是找到 $\mathbf{w}$ 来最大化这个似然函数。直接优化乘积很困难，所以我们转而最大化其对数（$\ln$）形式：

$$\max_{\mathbf{w}} \sum_{n=1}^{N} \ln\left(\theta(y_n \mathbf{w}^T \mathbf{x}_n)\right)$$

### 步骤 4：转化为最小化损失函数

最大化一个函数等价于最小化其相反数。因此，我们的优化目标变为：

$$\min_{\mathbf{w}} \frac{1}{N} \sum_{n=1}^{N} -\ln\left(\theta(y_n \mathbf{w}^T \mathbf{x}_n)\right)$$

代入 $\theta(s)$ 的定义，我们得到：

$$\min_{\mathbf{w}} E_{in}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} \ln(1 + \exp(-y_n \mathbf{w}^T \mathbf{x}_n))$$

这个损失函数 $E_{in}(\mathbf{w})$ 就是著名的 **交叉熵损失 (Cross-Entropy Error)**。
