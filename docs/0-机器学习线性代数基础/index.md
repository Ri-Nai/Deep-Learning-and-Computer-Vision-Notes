# 机器学习线性代数基础 (Linear Algebra Basics for Machine Learning)

## 1. 向量的表示约定

在机器学习的上下文中，向量的表示法有非常一致的约定。

### 核心约定

**在机器学习的惯例中，当您看到 `w` 和 `x` 这样的向量时，它们通常默认是列向量。**

因此，对于内积（点积），我们不能直接将两个列向量相乘。我们需要将其中一个转置，变成行向量。

### 向量内积的两种表示方法

让我们来详细分解一下 `w·x` 和公式中的 `w^T x`：

#### 1. 点积符号: `w · x`

这是一种比较抽象的数学表示法，读作 "w dot x"。它直接表示 `w` 和 `x` 两个向量之间的内积运算。

- **定义**：如果 `w` = $[w_1, w_2, \dots, w_d]$ 并且 `x` = $[x_1, x_2, \dots, x_d]$，那么它们的内积是一个**标量**（一个数字），计算方式如下：

$$ \mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{d} w_i x_i = w_1 x_1 + w_2 x_2 + \dots + w_d x_d $$

- **几何意义**：内积等于两个向量的模长之积再乘以它们之间夹角的余弦值 ($\|\mathbf{w}\| \|\mathbf{x}\| \cos\theta$)。

- **向量方向**：在使用点积符号 `·` 时，我们通常不太关心向量是行向量还是列向量，因为它是一个抽象的运算符号。

#### 2. 矩阵乘法表示法: `w^T x`

这是在机器学习和线性代数中**更常见、更具体**的写法。它利用矩阵乘法的规则来计算内积。

- **约定**：在机器学习中，向量（如特征向量 `x` 和权重向量 `w`）默认被视为**列向量**（dimension: $d \times 1$）。

$$
\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}
$$

- **运算**：
  - 为了计算内积，我们需要将第一个向量 `w` **转置 (Transpose)**，使其从一个 $d \times 1$ 的列向量变成一个 $1 \times d$ 的**行向量**。

    $$ \mathbf{w}^T = \begin{bmatrix} w_1 & w_2 & \dots & w_d \end{bmatrix} $$

  - 现在，我们可以按照矩阵乘法的规则进行计算。一个 $(1 \times d)$ 的矩阵乘以一个 $(d \times 1)$ 的矩阵，结果是一个 $(1 \times 1)$ 的矩阵，也就是一个标量。

    $$
    \mathbf{w}^T \mathbf{x} = \begin{bmatrix} w_1 & w_2 & \dots & w_d \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d
    $$

### 结论

在机器学习公式中，更精确的写法是 `w^T x`。

- 在这种表示法下，**`w` 和 `x` 都被默认视为列向量**。
- 为了进行内积运算，我们将 `w` 转置为行向量 `w^T`，然后与列向量 `x` 进行矩阵乘法。

**一言以蔽之：`w` 和 `x` 都是列向量。内积是通过将第一个向量转置成行向量，然后与第二个列向量做矩阵乘法来实现的。**

## 2. 向量求导法则

向量求导，也称为矩阵求导或多元微积分，本质上是将标量微积分中的概念扩展到多维空间。

### 核心思想

**一个函数对一个向量求导，结果是该函数对向量中每个分量分别求导后组成的向量。**

### 2.1 标量对向量求导

这是向量求导中最核心的法则。假设有一个标量函数 $f(\mathbf{w})$，其中 $\mathbf{w}$ 是一个 $d \times 1$ 的列向量，即 $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$。

那么，函数 $f$ 对向量 $\mathbf{w}$ 的导数（梯度）被定义为：

$$
\frac{\partial f(\mathbf{w})}{\partial \mathbf{w}} = \nabla f(\mathbf{w}) = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \\ \vdots \\ \frac{\partial f}{\partial w_d} \end{bmatrix}
$$

这个结果是一个与 $\mathbf{w}$ 维度相同的列向量。

### 2.2 线性形式的求导

在机器学习中，经常需要计算 $\frac{\partial}{\partial \mathbf{w}} (\mathbf{w}^T \mathbf{x})$ 这样的导数。

让我们详细推导：

- 首先，$\mathbf{w}^T \mathbf{x}$ 是一个标量，等于 $w_1 x_1 + w_2 x_2 + \dots + w_d x_d$。

- 对于函数 $f(\mathbf{w}) = \mathbf{w}^T \mathbf{x} = \sum_{i=1}^{d} w_i x_i$，我们计算它对向量 $\mathbf{w}$ 的每一个分量 $w_j$ 求偏导：

$$
\frac{\partial f}{\partial w_j} = \frac{\partial}{\partial w_j} (\sum_{i=1}^{d} w_i x_i) = x_j
$$

因为只有当 $i=j$ 时，$\frac{\partial w_i}{\partial w_j} = 1$，其余项的导数都为0。

- 最后，将所有偏导数重新组合成一个向量：

$$
\frac{\partial (\mathbf{w}^T \mathbf{x})}{\partial \mathbf{w}} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} = \mathbf{x}
$$

### 2.3 带系数的线性形式

对于 $f(\mathbf{w}) = -y_n \mathbf{w}^T \mathbf{x}_n$，其中 $y_n$ 是标量：

$$
\frac{\partial}{\partial \mathbf{w}} (-y_n \mathbf{w}^T \mathbf{x}_n) = -y_n \frac{\partial (\mathbf{w}^T \mathbf{x}_n)}{\partial \mathbf{w}} = -y_n \mathbf{x}_n
$$

### 2.4 链式法则

向量求导同样遵循链式法则。当一个函数是复合函数时，可以先对外层函数求导，再乘以内层函数对变量的导数。

**示例**：计算 $\frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$

- 令 $u = -y_n \mathbf{w}^T \mathbf{x}_n$
- 令 $g(u) = \ln(1 + e^u)$

那么，$\frac{\partial g}{\partial \mathbf{w}} = \frac{dg}{du} \cdot \frac{\partial u}{\partial \mathbf{w}}$。

- $\frac{dg}{du} = \frac{1}{1 + e^u} \cdot e^u = \frac{e^u}{1 + e^u}$
- $\frac{\partial u}{\partial \mathbf{w}} = \frac{\partial}{\partial \mathbf{w}} (-y_n \mathbf{w}^T \mathbf{x}_n) = -y_n \mathbf{x}_n$

因此：

$$
\frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}) = \frac{e^{-y_n \mathbf{w}^T \mathbf{x}_n}}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} \cdot (-y_n \mathbf{x}_n)
$$

注意到 $\frac{e^{-y_n \mathbf{w}^T \mathbf{x}_n}}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} = \frac{1}{1 + e^{y_n \mathbf{w}^T \mathbf{x}_n}} = \theta(-y_n \mathbf{w}^T \mathbf{x}_n)$，其中 $\theta(x) = \frac{e^x}{1+e^x}$ 是 sigmoid 函数。

所以最终结果：

$$
\frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}) = \theta(-y_n \mathbf{w}^T \mathbf{x}_n) (-y_n \mathbf{x}_n)
$$

### 2.5 常用向量求导法则总结

假设 $\mathbf{w}$ 和 $\mathbf{x}$ 是列向量，$\mathbf{A}$ 是一个矩阵。

#### 线性法则

- $\frac{\partial (\mathbf{w}^T \mathbf{x})}{\partial \mathbf{w}} = \mathbf{x}$
- $\frac{\partial (\mathbf{A} \mathbf{w})}{\partial \mathbf{w}} = \mathbf{A}^T$
- $\frac{\partial (c \mathbf{w}^T \mathbf{x})}{\partial \mathbf{w}} = c \mathbf{x}$（其中 $c$ 是标量）

#### 二次型法则

- $\frac{\partial (\mathbf{w}^T \mathbf{A} \mathbf{w})}{\partial \mathbf{w}} = (\mathbf{A} + \mathbf{A}^T) \mathbf{w}$
- 如果 $\mathbf{A}$ 是对称矩阵，则 $\frac{\partial (\mathbf{w}^T \mathbf{A} \mathbf{w})}{\partial \mathbf{w}} = 2\mathbf{A} \mathbf{w}$

#### 逐元素运算

如果有一个函数 $f$ 作用于向量 $\mathbf{w}$ 的每一个元素，那么求导也是逐元素的。例如，对于 sigmoid 函数 $\theta(\mathbf{z})$，其输入是一个向量，那么 $\frac{\partial \theta(\mathbf{z})}{\partial \mathbf{z}}$ 的结果会是一个雅可比矩阵 (Jacobian matrix)。

## 3. 实际应用示例

### 交叉熵损失函数的梯度推导

在逻辑回归中，交叉熵损失函数为：

$$E_{in}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$$

利用向量求导法则，我们可以推导其梯度：

$$\nabla E_{in}(\mathbf{w}) = \frac{\partial E_{in}(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n})$$

使用链式法则：

$$\frac{\partial}{\partial \mathbf{w}} \ln(1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}) = \frac{e^{-y_n \mathbf{w}^T \mathbf{x}_n}}{1 + e^{-y_n \mathbf{w}^T \mathbf{x}_n}} \cdot (-y_n \mathbf{x}_n) = \theta(-y_n \mathbf{w}^T \mathbf{x}_n) (-y_n \mathbf{x}_n)$$

因此，最终的梯度为：

$$\nabla E_{in}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} \theta(-y_n \mathbf{w}^T \mathbf{x}_n) (-y_n \mathbf{x}_n)$$

这个过程巧妙地运用了**标量对向量求导的链式法则**，是理解机器学习中梯度计算的关键。

