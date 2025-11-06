# 卷积神经网络 (Convolutional Neural Networks)

## 1. 背景：全连接网络处理图像的局限

如果使用传统的多层感知机（全连接网络）来处理图像，会面临**参数爆炸**的问题。例如，一张100x100像素的彩色（3通道）图片，其输入维度高达30,000。如果第一个隐藏层有1000个神经元，仅这一层就需要 `30,000 * 1000 = 3000万` 个权重参数，这使得模型极难训练且容易过拟合。

**CNN的核心任务就是简化神经网络的架构，使其更适合处理图像这类网格状数据。**

---

## 2. CNN核心思想

### 2.1 局部连接 (Local Connectivity) / 感受野 (Receptive Field)

- **思想**：图像中的空间联系是局部的，即一个像素与其周围的像素关系最密切，与远处像素关系较弱。因此，**每个神经元没有必要连接到输入图像的每一个像素**。

- **实现**：CNN中的神经元只与输入的一个**局部区域**相连接。这个局部区域被称为该神经元的**感受野 (Receptive Field)**。

- **效果**：这种局部连接的方式极大地减少了参数数量。

### 2.2 参数共享 (Parameter Sharing) / 权值共享

- **思想**：一个在图像某个位置学习到的特征（如一条垂直边缘、一个鸟嘴），在图像的其他位置也同样适用。因此，我们**不需要为图像的每个位置都学习一个单独的特征检测器**。

- **实现**：在同一张特征图 (feature map) 中，所有神经元**共享同一组权重**。这组共享的权重被称为**卷积核 (Kernel)** 或**滤波器 (Filter)**。卷积核在整个图像上滑动，以检测特定特征。

- **效果**：参数共享进一步**急剧减少**了模型的参数量。例如，无论图像多大，一个10x10的卷积核都只有100个参数。

### 2.3 多卷积核 (Multiple Filters)

- **思想**：单一的卷积核只能检测一种类型的特征。为了提取图像的丰富信息，我们需要检测多种特征。

- **实现**：一个卷积层通常包含**多个不同的卷积核**，每个卷积核负责学习和检测一种特定的特征（如不同的方向、颜色、纹理等）。

- **效果**：每个卷积核都会生成一个对应的特征图，这些特征图堆叠起来，构成了下一层的输入。

---

## 3. 卷积层详解 (The Convolution Layer Explained)

### 3.1 组成部分

- **输入 (Input Volume)**：通常是一个三维的张量，尺寸为 `[height, width, depth]`。例如，一张32x32的RGB图像，其尺寸为 `[32, 32, 3]`。

- **滤波器/卷积核 (Filters/Kernels)**：一组学习到的权重。每个滤波器也是一个三维张量，尺寸为 `[filter_height, filter_width, input_depth]`。

  - **关键点**：滤波器的**深度 (depth) 必须与输入数据的深度相同**。

### 3.2 卷积运算过程

卷积运算的核心是**滑动窗口和点积计算**。

1. **选择一个滤波器** (例如，一个 `5x5x3` 的滤波器)。

2. 将这个滤波器放置在输入数据的左上角。

3. 计算滤波器和其覆盖的输入数据区域（一个 `5x5x3` 的小块）之间的**点积 (dot product)**，并加上一个偏置项 (bias)。这个计算会得到一个**单一的数值**。

   - `w^T x + b`

4. 这个数值就是输出**激活图 (Activation Map)** 左上角的第一个元素。

5. 将滤波器在输入数据上**按步长 (stride) 滑动**（向右，然后向下），在每个位置重复步骤3，直到遍历完所有空间位置。

6. 所有位置的点积计算结果共同构成了一个二维的**激活图**。这个激活图代表了该滤波器在图像不同位置检测到的特征的响应强度。

### 3.3 多滤波器与输出

- 一个卷积层通常有多个滤波器（例如，6个 `5x5x3` 的滤波器）。

- **每个滤波器**都会通过上述卷积运算生成一个独立的二维激活图。

- 如果一个层有6个滤波器，它就会生成6个激活图。

- 最后，这6个激活图在深度维度上**堆叠 (stack)** 起来，形成该卷积层的最终输出，即一个**新的三维数据体**。例如，如果每个激活图尺寸为 `28x28`，那么最终输出的尺寸就是 `[28, 28, 6]`。这个输出将作为下一层的输入。

---

## 4. 卷积层超参数 (Convolutional Layer Hyperparameters)

卷积层的行为和输出尺寸由几个关键的超参数控制。

### 4.1 步长 (Stride)

- **定义**：步长指的是滤波器在输入数据上每次滑动的距离（步长为1则移动1个像素，步长为2则移动2个像素）。

- **作用**：步长会影响输出特征图的空间尺寸。**步长越大，输出的尺寸越小**。

### 4.2 填充 (Padding)

- **问题**：如果不进行填充，每次卷积操作都会使输出的空间尺寸比输入小，导致深层网络中的特征图迅速缩小。同时，图像边缘的像素被计算的次数远少于中心像素，可能导致边缘信息丢失。

- **定义**：填充 (通常是**零填充 Zero-Padding**) 是指在输入数据的边界周围添加一圈或多圈的零值像素。

- **作用**：

  1. **控制输出尺寸**：通过合理设置填充，可以使卷积后的输出尺寸与输入尺寸保持一致（"same" convolution）。

  2. **保留边界信息**：确保图像边缘的像素能被滤波器充分处理。

### 4.3 输出尺寸计算

输出特征图的空间尺寸可以通过以下公式计算：

$$ O = \frac{N - F + 2P}{S} + 1 $$

其中：

- $O$：输出尺寸 (高或宽)
- $N$：输入尺寸
- $F$：滤波器尺寸
- $P$：填充大小 (例如，填充1个像素则 P=1)
- $S$：步长

**注意**：如果 $(N - F + 2P)$ 不能被 $S$ 整除，则通常意味着配置不当。

### 4.4 参数数量计算

一个卷积层中的可训练参数数量取决于滤波器的尺寸和数量。

$$ \text{参数数量} = (F_h \times F_w \times D_{in} + 1) \times K $$

其中：

- $F_h, F_w$：滤波器的高和宽
- $D_{in}$：输入数据的深度（通道数）
- $+ 1$：每个滤波器对应的一个偏置项 (bias)
- $K$：滤波器的数量（也决定了输出的深度）

**示例**：输入为 `32x32x3`，使用10个 `5x5` 的滤波器。

参数数量 = `(5 * 5 * 3 + 1) * 10 = 760` 个。可见，参数量与输入图像的尺寸无关，这是参数共享的巨大优势。

---

## 5. 池化层 (Pooling Layer)

### 5.1 什么是池化层？

池化层（也称下采样层）通常紧跟在卷积层之后，其主要目的是**逐步减小特征图的空间尺寸**，并使特征表示更加紧凑和易于管理。

### 5.2 池化层的作用

- **降低计算量**：通过减小特征图的尺寸，显著减少了后续层的参数数量和计算开销。

- **增强平移不变性**：使模型对特征在图像中的微小位置变化不那么敏感，提高了模型的鲁棒性。例如，无论一只猫的眼睛在图像的哪个小区域内，最大池化都倾向于提取出"有眼睛"这个特征。

- **防止过拟合**：通过减少特征维度，在一定程度上起到了正则化的作用。

### 5.3 最大池化 (Max Pooling)

最大池化是目前最常用的池化操作。

- **工作原理**：

  1. 定义一个池化窗口（例如 `2x2`）和一个步长（例如 `2`）。

  2. 将这个窗口在输入特征图上滑动。

  3. 在每个窗口覆盖的区域内，**只取最大值**作为该区域的输出。

- **特点**：它独立地作用于输入的**每一个深度切片（通道）**。如果输入是 `112x112x64`，经过 `2x2` 的最大池化（步长为2）后，输出将是 `56x56x64`。

---

## 6. CNN架构与特征层次 (CNN Architecture and Feature Hierarchy)

### 6.1 典型的CNN架构

一个现代的卷积神经网络 (ConvNet) 通常是由一系列特定功能的层堆叠而成，形成一个端到端的模型。

- **基本构建块**：`CONV -> RELU -> POOL`

  - **卷积层 (CONV)**：负责通过滤波器学习和提取局部特征。

  - **激活层 (ReLU)**：引入非线性，使网络能够学习更复杂的模式。

  - **池化层 (POOL)**：进行下采样，减小数据尺寸，增强特征的平移不变性。

- **整体结构**：网络通常由多个这样的构建块串联组成。随着网络层数的加深，特征图的**空间尺寸 (高和宽) 逐渐减小**，而**深度 (通道数) 逐渐增加**。

- **分类部分**：在网络的末端，通常会有一到多个**全连接层 (Fully Connected Layer, FC)**。它们接收由卷积和池化层提取出的高级特征，并执行最终的分类任务。

### 6.2 特征层次 (Feature Hierarchy)

CNN能够自动地、层次化地学习特征，这是其强大能力的核心。

- **浅层 (Low-Level Features)**：靠近输入的卷积层学习到的特征比较基础和通用，例如**边缘、角点、颜色块**等。

- **中层 (Mid-Level Features)**：中间的层会组合浅层的特征，形成更复杂的模式，例如**纹理、眼睛、鼻子**等物体部件。

- **深层 (High-Level Features)**：靠近输出的层学习到的特征更具抽象性和语义性，能够识别出**完整的物体或场景**。

这个过程与生物视觉系统（如Hubel & Wiesel对猫的视觉皮层研究）处理信息的方式非常相似，从简单的刺激响应开始，逐步构建出对复杂物体的感知。

---

## 7. 里程碑CNN架构

### 7.1 LeNet-5 (1998)

- **贡献**：由Yann LeCun提出，是第一个成功应用于商业的卷积神经网络，主要用于手写数字识别。它奠定了现代CNN `CONV-POOL-CONV-POOL-FC` 的基本架构。

- **结构**：包含2个卷积层、2个池化（子采样）层和2个全连接层。使用了5x5的卷积核。

### 7.2 AlexNet (2012)

- **贡献**：在ImageNet LSVRC-2012竞赛中以巨大优势夺冠，引爆了深度学习在计算机视觉领域的革命。

- **关键创新点**：

  1. **ReLU激活函数**：首次大规模使用ReLU，有效解决了深度网络中的梯度消失问题，训练速度远超tanh。

  2. **数据增强 (Data Augmentation)**：采用了大量的镜像、裁剪等数据增强技术来减少过拟合。

  3. **Dropout**：在全连接层使用Dropout来防止过拟合。

  4. **重叠池化 (Overlapping Pooling)**：使用`3x3`的池化窗口和步长`2`，产生了重叠的感受野，提升了性能。

  5. **多GPU训练**：由于当时GPU显存限制，将网络拆分到两个GPU上进行并行训练。

  6. **局部响应归一化 (LRN)**：一种侧抑制的归一化层，后被证明效果不如批量归一化（BN）。

### 7.3 VGGNet (2014)

- **贡献**：探索了网络**深度**对性能的影响，证明了通过堆叠非常小的卷积核可以构建出性能优异的深度网络。

- **核心思想**：

  - **只使用3x3的小型卷积核**：通过堆叠多个`3x3`的卷积层，可以获得与一个较大卷积核（如`5x5`或`7x7`）相同的感受野，但参数更少，非线性变换更多，学习能力更强。

  - **结构规整**：整个网络结构非常简洁、统一，都是由`3x3`卷积和`2x2`池化组成。

- **影响**：VGGNet的深度（16或19层）和规整的结构使其成为后续许多研究的基础模型和特征提取器。其主要缺点是参数量巨大（约1.38亿）。

### 7.4 GoogLeNet (Inception v1, 2014)

- **贡献**：在加深网络的同时，关注提升计算效率。它与VGGNet并列成为ILSVRC 2014的冠军。

- **核心思想：Inception模块**

  - **并行结构**：在一个模块内，并行地使用多种不同尺寸的卷积核（`1x1`, `3x3`, `5x5`）和池化操作，然后将所有输出在深度维度上拼接起来。这使得网络可以捕捉到不同尺度的特征。

  - **1x1卷积降维**：在计算开销大的`3x3`和`5x5`卷积之前，先使用`1x1`的卷积来减少输入的深度（通道数），形成一个"瓶颈层"(bottleneck)，极大地降低了计算量。

- **影响**：GoogLeNet在大幅减少参数量（约500万）的情况下，获得了比AlexNet和VGGNet更优的性能，开启了高效网络设计的新思路。

### 7.5 ResNet (Residual Network, 2015)

- **贡献**：成功解决了**深度网络退化 (Degradation)** 的问题，使得训练数百甚至上千层的超深网络成为可能。在ILSVRC 2015中取得了压倒性胜利。

- **退化问题**：当网络深度增加到一定程度后，其性能（训练和测试准确率）反而会下降。这并非由过拟合导致，而是因为深度模型难以优化。

- **核心思想：残差学习 (Residual Learning)**

  - **残差块 (Residual Block)**：ResNet引入了"短路连接"或"跳跃连接"(skip connection)。它不让网络层直接学习目标映射 $H(x)$，而是学习一个**残差映射** $F(x) = H(x) - x$。原始的输入 $x$ 通过一条"短路"直接加到 $F(x)$ 的输出上，即最终输出为 $F(x) + x$。

  - **优势**：如果某一层的恒等映射（即输出=输入）是最优的，那么模型只需将残差 $F(x)$ 学成0即可，这比直接学习一个恒等映射要容易得多。这种结构极大地缓解了梯度消失问题，使得信息和梯度可以在网络中更顺畅地传播。

---

## 8. 迁移学习 (Transfer Learning)

### 8.1 背景问题

从零开始训练一个强大的卷积神经网络（CNN）需要海量的标注数据（如ImageNet的上百万张图片）和巨大的计算资源，这对于大多数应用场景来说是不现实的。

### 8.2 什么是迁移学习？

迁移学习是一种强大的技术，它允许我们将一个在**大型、通用数据集（如ImageNet）上预训练好的模型**的知识，"迁移"到我们自己的、通常数据量较小的特定任务上。

### 8.3 核心思想

一个在ImageNet上训练好的CNN，其浅层和中层已经学会了如何提取非常通用的图像特征（如边缘、纹理、形状、物体部件等）。这些特征对于许多其他的视觉任务同样是有效的。因此，我们可以将这个预训练好的网络作为一个**固定的特征提取器**。

### 8.4 如何实施？

1. **获取预训练模型**：选择一个在ImageNet上预训练好的著名模型，如VGGNet, ResNet, Inception等。

2. **改造模型**：

   - 去掉模型的原始全连接分类层（例如，ImageNet的1000类分类器）。

   - 在模型的卷积基座之上，添加一个新的、为我们自己任务定制的分类器（通常是规模较小的全连接层）。

3. **"冻结"权重**：将预训练的卷积层的权重固定住，使其在训练中不更新。

4. **训练**：只训练我们新添加的分类器部分。由于这部分参数量很少，所以即使在小数据集上也能快速、有效地进行训练。

5. **（可选）微调 (Fine-tuning)**：在完成上述步骤后，可以"解冻"预训练模型的一部分或全部卷积层，然后用一个非常小的学习率对整个网络进行训练，以微调特征，使其更适应新任务。

---

## 9. PyTorch 代码实现 (PyTorch Implementation)

### 9.1 卷积层和池化层的 PyTorch 实现

#### 9.1.1 卷积层

```python
import torch
import torch.nn as nn

# 创建一个卷积层
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
conv_layer = nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB图像为3）
    out_channels=64,   # 输出通道数（滤波器数量）
    kernel_size=3,     # 卷积核大小（3x3）
    stride=1,          # 步长
    padding=1          # 填充（padding=1表示在四周各填充1个像素）
)

# 输入：batch_size=1, channels=3, height=32, width=32
x = torch.randn(1, 3, 32, 32)

# 前向传播
output = conv_layer(x)
print(f"输入尺寸: {x.shape}")      # torch.Size([1, 3, 32, 32])
print(f"输出尺寸: {output.shape}")  # torch.Size([1, 64, 32, 32])
```

#### 9.1.2 池化层

```python
# 最大池化层
# nn.MaxPool2d(kernel_size, stride=None, padding=0)
max_pool = nn.MaxPool2d(
    kernel_size=2,  # 池化窗口大小（2x2）
    stride=2       # 步长（通常等于kernel_size，实现不重叠池化）
)

# 输入：batch_size=1, channels=64, height=32, width=32
x = torch.randn(1, 64, 32, 32)

# 前向传播
output = max_pool(x)
print(f"输入尺寸: {x.shape}")      # torch.Size([1, 64, 32, 32])
print(f"输出尺寸: {output.shape}")  # torch.Size([1, 64, 16, 16])
```

### 9.2 完整的 CNN 模型示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    一个简单的CNN模型
    对应理论中的架构：CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积块：CONV -> RELU -> POOL
        # 输入: 3通道，输出: 32个特征图
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块：CONV -> RELU -> POOL
        # 输入: 32通道，输出: 64个特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        # 假设输入图像是32x32，经过两次池化后变为8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        前向传播
        对应理论中的逐层计算过程
        """
        # 第一个卷积块
        x = self.conv1(x)        # CONV: 特征提取
        x = F.relu(x)            # RELU: 非线性激活
        x = self.pool1(x)        # POOL: 下采样
        
        # 第二个卷积块
        x = self.conv2(x)        # CONV: 更深层的特征
        x = F.relu(x)            # RELU: 非线性激活
        x = self.pool2(x)        # POOL: 下采样
        
        # 展平：将特征图展平成一维向量
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

# 使用示例
model = SimpleCNN(num_classes=10)
input_tensor = torch.randn(4, 3, 32, 32)  # batch_size=4, 3通道, 32x32图像
output = model(input_tensor)
print(f"输入尺寸: {input_tensor.shape}")  # torch.Size([4, 3, 32, 32])
print(f"输出尺寸: {output.shape}")         # torch.Size([4, 10])
```

### 9.3 计算输出尺寸和参数数量

```python
def calculate_output_size(input_size, kernel_size, stride=1, padding=0):
    """
    计算卷积层的输出尺寸
    对应公式: O = (N - F + 2P) / S + 1
    """
    output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    return output_size

def calculate_conv_params(in_channels, out_channels, kernel_size):
    """
    计算卷积层的参数数量
    对应公式: 参数数量 = (F_h * F_w * D_in + 1) * K
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    params = (kernel_size[0] * kernel_size[1] * in_channels + 1) * out_channels
    return params

# 示例计算
input_size = 32
kernel_size = 3
stride = 1
padding = 1

output_size = calculate_output_size(input_size, kernel_size, stride, padding)
print(f"输出尺寸: {output_size}")  # (32 - 3 + 2*1) / 1 + 1 = 32

# 参数数量计算
in_channels = 3
out_channels = 64
params = calculate_conv_params(in_channels, out_channels, kernel_size)
print(f"参数数量: {params}")  # (3*3*3 + 1) * 64 = 1792
```

### 9.4 更复杂的 CNN 架构示例（类似 VGG）

```python
class VGGLikeCNN(nn.Module):
    """
    类似VGG的CNN架构
    使用多个小的3x3卷积核堆叠
    """
    def __init__(self, num_classes=10):
        super(VGGLikeCNN, self).__init__()
        
        # 第一组：两个3x3卷积层
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二组：两个3x3卷积层
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 9.5 迁移学习的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ========== 方法1：固定特征提取器 ==========

# 加载预训练模型（如ResNet18）
pretrained_model = models.resnet18(pretrained=True)

# 冻结所有卷积层的参数
for param in pretrained_model.parameters():
    param.requires_grad = False

# 替换分类器（假设我们的任务有5个类别）
num_classes = 5
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

# 现在只有fc层的参数需要训练
trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params}")  # 只有fc层的参数

# ========== 方法2：微调（Fine-tuning） ==========

# 加载预训练模型
fine_tune_model = models.resnet18(pretrained=True)

# 替换分类器
fine_tune_model.fc = nn.Linear(fine_tune_model.fc.in_features, num_classes)

# 使用不同的学习率
# 新添加的层使用较大的学习率，预训练层使用较小的学习率
optimizer = torch.optim.SGD([
    {'params': fine_tune_model.conv1.parameters(), 'lr': 1e-5},  # 浅层：小学习率
    {'params': fine_tune_model.layer1.parameters(), 'lr': 1e-5},
    {'params': fine_tune_model.layer2.parameters(), 'lr': 1e-4},  # 中层：中等学习率
    {'params': fine_tune_model.fc.parameters(), 'lr': 1e-3}      # 新层：大学习率
], momentum=0.9)

# ========== 完整的迁移学习训练流程 ==========

def train_transfer_learning(model, train_loader, num_epochs=10):
    """
    迁移学习的训练函数
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 使用示例
# train_transfer_learning(model, train_loader, num_epochs=10)
```

### 9.6 使用预训练模型进行特征提取

```python
import torchvision.models as models
from torchvision import transforms

# 加载预训练的ResNet（不包含分类层）
resnet = models.resnet18(pretrained=True)
# 移除最后的分类层
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# 图像预处理（ImageNet的标准化）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 提取特征
def extract_features(image_tensor):
    """
    使用预训练模型提取特征
    """
    feature_extractor.eval()
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)  # 展平
    return features

# 示例：提取特征
# image = preprocess(image)  # 预处理图像
# image_batch = image.unsqueeze(0)  # 添加batch维度
# features = extract_features(image_batch)
```

### 9.7 完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ========== 1. 数据准备 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 假设使用CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

# ========== 2. 创建模型 ==========
model = SimpleCNN(num_classes=10)

# ========== 3. 定义损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ========== 4. 训练循环 ==========
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print("训练完成！")
```

### 9.8 关键点说明

#### 9.8.1 卷积层参数对应关系

| 理论概念 | PyTorch 参数 | 说明 |
|---------|-------------|------|
| 滤波器数量 $K$ | `out_channels` | 输出的通道数 |
| 输入通道数 $D_{in}$ | `in_channels` | 输入的通道数 |
| 卷积核尺寸 $F$ | `kernel_size` | 滤波器的高和宽 |
| 步长 $S$ | `stride` | 滤波器滑动步长 |
| 填充 $P$ | `padding` | 边界填充大小 |

#### 9.8.2 输出尺寸验证

```python
# 验证输出尺寸计算公式
# 公式: O = (N - F + 2P) / S + 1

conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 32, 32)
output = conv(x)

# 理论计算: (32 - 3 + 2*1) / 1 + 1 = 32
# 实际输出: output.shape = [1, 64, 32, 32] ✓
print(f"输出尺寸匹配: {output.shape[2] == 32}")  # True
```

#### 9.8.3 参数共享的体现

在PyTorch中，参数共享是自动实现的。一个卷积层的所有空间位置共享同一组权重：

```python
conv = nn.Conv2d(3, 64, kernel_size=3)
print(f"权重形状: {conv.weight.shape}")  # torch.Size([64, 3, 3, 3])
# 只有64个3x3x3的滤波器，无论输入图像多大，参数数量都相同
```

这展示了CNN参数共享的优势：**参数量与输入图像尺寸无关**！
