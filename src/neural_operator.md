# 算子

## flatten
$$
shape(Input) = (N, C, H, W) \\\\
Output = flatten(Input) \\\\
shape(Output) = (N, C \times H \times W)
$$
flatten算子只是复制输入tensor，底层数据(包括data和grad数据)不做修改，只修改它的shape属性，所以它的backword只是把上游的导数复制到输入端即可。

## add
$$
c = a + b \\\\
\frac{\partial c}{\partial a} = 1 \\\\
\frac{\partial c}{\partial b} = 1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * 1 \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * 1
$$

## sub
$$
c = a - b \\\\
\frac{\partial c}{\partial a} = 1 \\\\
\frac{\partial c}{\partial b} = -1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * 1 \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * (-1)
$$

## neg
$$
c = -a \\\\
\frac{\partial c}{\partial a} = -1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * (-1)
$$

## element-wise mul
$$
c[k] = a[k] * b[k] \\\\
\frac{\partial c[k]}{\partial a[k]} = b[k] \\\\
\frac{\partial c[k]}{\partial b[k]} = a[k] \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * b \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * a 
$$
这里 \\(c, a, b\\) 的维度相同即可。

## relu
$$
dout =
\begin{cases}
0 & \text{if } din < 0 \\\\
din & \text{if } din \geq 0
\end{cases}
$$

$$
\frac{\partial dout}{\partial din} = 
\begin{cases}
0 &\quad din < 0 \\\\
1 &\quad din \geq 0
\end{cases}
$$

## sigmod
$$
dout = \frac{1}{1+e^{-din}} \\\\
dout = {(1+e^{-din})}^{-1} \\\\
\frac{\partial dout}{\partial din} = -1 * {(1+e^{-din})}^{-2} * (-1) * e^{-din} \\\\
\frac{\partial dout}{\partial din} = {(1+e^{-din})}^{-2} * e^{-din} = \frac{1}{1+e^{-din}} * \frac{e^{din}}{1+e^{-din}} \\\\
\frac{\partial dout}{\partial din} = dout * (1-dout)
$$

## tanh
$$
dout = \tanh(din) \\\\
\tanh(din) = \frac{\sinh(din)}{\cosh(din)} \\\\
\sinh(din) = \frac{e^{din}-e^{-din}}{2} \\\\
\cosh(din) = \frac{e^{din}+e^{-din}}{2} \\\\
\tanh(din) = \frac{e^{din}-e^{-din}}{e^{din}+e^{-din}} \\\\
\frac{\partial dout}{\partial din} = \frac{(e^{din}+e^{-din})(e^{din}+e^{-din}) - (e^{din}-e^{-din})(e^{din}-e^{-din})}{(e^{din}+e^{-din})^2} \\\\
\frac{\partial dout}{\partial din} = \frac{2*e^{din} * 2 e^{-din}}{(e^{din}+e^{-din})^2} = \frac{4}{(e^{din}+e^{-din})^2} = \frac{4}{4 * \cosh^2(din)} = \frac{1}{\cosh^2(din)} \\\\
\frac{\partial dout}{\partial din} = \frac{\cosh^2(din) - \sinh^2(din)}{\cosh^2(din)} = 1 - \tanh^2(din) = 1 - dout^2
$$

## matmul
$$
𝐴_{𝑚×k}𝐵_{k×n} = 𝐶_{m×n}
$$

展开表示：
$$
\underset{m \times k}{\begin{bmatrix}
a_{11} & \cdots & a_{1k} \\\\
\vdots &        & \vdots \\\\
a_{m1} & \cdots & a_{mk}
\end{bmatrix}}
\*
\underset{k \times n}{\begin{bmatrix}
b_{11} & \cdots & b_{1n} \\\\
\vdots &        & \vdots \\\\
b_{k1} & \cdots & b_{kn}
\end{bmatrix}}
\=
\underset{m \times n}{\begin{bmatrix}
c_{11} & \cdots & c_{1n} \\\\
\vdots &        & \vdots \\\\
c_{m1} & \cdots & c_{mn}
\end{bmatrix}}
$$

已知：
$$
c_{ij} = \sum_{p=1}^{k} a_{ip} * b_{pj}
$$

对于 \\(a_{11}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{1n}\\) 这一行，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{n} \frac {\partial {c_{1p}}}{\partial a_{11}} * c_{1p}^{'} = \sum_{p=1}^{n} b_{p1} * c_{1p}^{'}
$$

其中 \\(c_{1p}^{'}\\) 表示上游梯度。
对于 \\(a_{12}\\) 来说，它对结果 \\(c\\) 的贡献也是仅仅局限在 \\(c_{11} \cdots c_{1n}\\) 这一行，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{n} \frac {\partial {c_{1p}}}{\partial a_{12}} * c_{1p}^{'} = \sum_{p=1}^{n} b_{p2} * c_{1p}^{'}
$$

推广到所有的 \\(a_{ij}\\) 上，得到：
$$
\sum_{p=1}^{n} \frac {\partial {c_{ip}}}{\partial a_{ij}} * c_{ip}^{'} = \sum_{p=1}^{n} b_{pj} * c_{ip}^{'}
$$

所以得到：
$$
\frac {\partial {Loss}}{\partial A} = \frac {\partial {Loss}}{\partial C} * B^{T} \\\\
A \in \mathbb{R}^{m \times k}, \frac {\partial {Loss}}{\partial A} \in \mathbb{R}^{m \times k} \\\\
\frac {\partial {Loss}}{\partial C} \in \mathbb{R}^{m \times n}, B^{T} \in \mathbb{R}^{n \times k}
$$

类似的，得到：

对于 \\(b_{11}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{m1}\\) 这一列，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{m} \frac {\partial {c_{p1}}}{\partial b_{11}} * c_{p1}^{'} = \sum_{p=1}^{m} a_{1p} * c_{p1}^{'}
$$
对于 \\(b_{21}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{m1}\\) 这一列，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{m} \frac {\partial {c_{p1}}}{\partial b_{21}} * c_{p1}^{'} = \sum_{p=1}^{m} a_{2p} * c_{p1}^{'}
$$

推广到所有的 \\(b_{ij}\\) 上，得到：
$$
\sum_{p=1}^{m} \frac {\partial {c_{pj}}}{\partial b_{ij}} * c_{pj}^{'} = \sum_{p=1}^{m} a_{ip} * c_{pj}^{'}
$$

所以得到：
$$
\frac {\partial {Loss}}{\partial B} = A^{T} * \frac {\partial {Loss}}{\partial C}  \\\\
B \in \mathbb{R}^{k \times n}, \frac {\partial {Loss}}{\partial B} \in \mathbb{R}^{k \times n} \\\\
A^{T} \in \mathbb{R}^{k \times m}, \frac {\partial {Loss}}{\partial C} \in \mathbb{R}^{m \times n}
$$

## Linear
全连接层，一般在flatten层后面，这个时候输入的`dim=2`
$$ 
shape(Input) = (N, C \times H \times W) = (N, M) \\\\
Output = Input * Weight_{M \times P} + Bias \\\\
shape(Output) = (N, P) \\\\
shape(Bias) = (P,)
$$
求导参考matmul算子：
$$
\frac{\partial Loss}{\partial weight} = \frac{\partial Loss}{\partial output} * \frac{\partial output}{\partial weight} = Input^T * \frac{\partial Loss}{\partial output} \\\\
Input^T \in \mathbb{R}^{M \times 1}, \frac{\partial Loss}{\partial output} \in \mathbb{R}^{1 \times P} \\\\
\frac{\partial Output}{\partial bais} = 1
$$

## MaxPool2D
对于 \\(N \times C \times Hin \times Win \\) 排列的输入数据来说，经过maxpool2d的输出是 \\(N \times C \times Hout \times Wout\\) ，我们先把Batch维度`N`省略，从单一输入数据看计算过程。
每个输出点 (\\(i,j\\)) 对应输入上一个由核位置 (\\(p,q\\)) 决定的采样点集合（dilation 控制核内采样间隔，不连续）：
$$
Out(C, i, j) = \max_{p \in [0, ksize[0]), q \in [0, ksize[1])} Input\big(C, h(i,p), w(j,q)\big) \\\\
h(i,p) = i\cdot stride[0] - padding[0] + p\cdot dilation[0],\qquad p \in [0, ksize[0]) \\\\
w(j,q) = j\cdot stride[1] - padding[1] + q\cdot dilation[1],\qquad q \in [0, ksize[1]) \\\\
Out \in \mathbb{R}^{C \times Hout \times Wout} \\\\
Input \in \mathbb{R}^{C \times Hin \times Win}
$$
注意 \\(h(i,p), w(j,q)\\) 可能落到输入边界外（由 padding 引起）。padding 模式为 **zeros**（补 0，与 PyTorch 默认一致）：越界位置当作 0 参与 max 比较。当窗口内所有真实值都为负时，输出为 0 而非最大负值；但梯度不路由到任何真实输入（padding 位无对应输入）。

影响最终输出`Hout * Wout`的因素包括：`ksize, stride, padding, dilation`，输出`Hout * Wout`的计算逻辑：
$$
Hout = \frac{Hin + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1}{stride[0]} + 1 \\\\
Wout = \frac{Win + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1}{stride[1]} + 1
$$

求导。先定义窗口 \\(i,j\\) 的 argmax 位置 \\( p^\*, q^\* \\) (若 padding 赢出则记为空)：

$$
(p^\*, q^\*) = \arg\max_{p,q} Input\big(C, h(i,p), w(j,q)\big)
$$

梯度只在 argmax 处为 1，其余位置为 0：
$$
\frac{\partial Out(C, i, j)}{\partial Input(C, p, q)} = \begin{cases}
1 &\quad (p, q) = (p^\*, q^\*) \\\\
0 &\quad \text{otherwise}
\end{cases}
$$

当 `stride < ksize` 时，同一个输入位置 \\(p,q\\) 可能被多个输出窗口采样。若它是多个窗口的 argmax，梯度应从所有这些窗口**累加**：
$$
\frac{\partial L}{\partial Input(C, p, q)} = \sum_{\substack{(i,j) : (p,q) = \arg\max\, window(i,j)}} \frac{\partial L}{\partial Out(C, i, j)}
$$
实现上：前向记录每个窗口的 argmax 对应的输入扁平索引，反向把上游梯度 scatter-add 到该索引。

## sum
$$
dout = \sum_{i=0}^{K} din_{i} \\\\
\frac{\partial dout}{\partial din_{i}} = 1
$$

## cross_entropy
cross_entropy 损失函数计算分为两步计算，第一步计算每一个类别的 \\(p\\) :
$$
p(lable=i) = softmax(h_{i}) = \frac{\exp(h_{i})}{\sum_{j=0}^{k}\exp(h_{j})}
$$
这里为了防止分母太大或者太小，做一个等比例缩放：
$$
h_{max} = max([h_{0}...h_{k}]) \\\\
p(lable=i) = softmax(h_{i}) = \frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})}
$$
第二步计算 \\(loss\\) :
$$
loss = -\log(p(lable=y)) = -\log(\frac{\exp(h_{y} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})}) = -(h_{y} - h_{max}) + \log(\sum_{j=0}^{k}\exp(h_{j} - h_{max}))
$$

然后它的导数就很明了了，我们需要对所有的 \\(h_{i}\\) 求导：
$$
\frac{\partial loss}{\partial h_{i}} = \begin{cases}
-1 + \frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})} &\quad i = y \\\\
\frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})} &\quad i \neq y
\end{cases}
$$

## batch normalization
先对mini batch求均值和方差：
$$
\mu_{b} = \frac{1}{m}\sum_{i=1}^{m}x_{i} \tag{0}
$$
其中 \\(m = B \times C \times H \times W\\)
$$
\sigma_{b}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_{i} - \mu_{b})^2 \tag{1}
$$

然后有两个可训练参数 \\(\gamma, \beta\\) ：
$$
y_{i} = \gamma * \frac{x_{i} - \mu_{b}}{\sqrt{\sigma_{b}^2 + \epsilon}} + \beta \tag{2}
$$

\\(\epsilon\\) 是极小值防止除0。

它的导数：
$$
\frac{\partial y_{i}}{\partial \gamma} = \frac{x_{i} - \mu_{b}}{\sqrt{\sigma_{b}^2 + \epsilon}} \tag{3}
$$
$$
\frac{\partial y_{i}}{\partial \beta} = 1 \tag{4}
$$
对 \\(x_i\\) 求导有点麻烦，问题在于 \\(\mu_b, \sigma_b\\) 也是关于 \\(x_i\\) 的函数，所以这里先令
$$
\sigma = \sqrt{\sigma_{b}^2 + \epsilon} \tag{6}
$$
$$
y_{i} = \gamma * \frac{x_i - \mu_b}{\sigma} + \beta \tag{7}
$$
根据复合函数求导规则，得到
$$
\frac{\partial y_{i}}{\partial x_i} = \gamma * (\frac{1}{\sigma} * \frac{m-1}{m} - (x_i - \mu_b) * \frac{1}{\sigma^2} * \frac{\partial \sigma}{\sigma_b^2} * \frac{\partial \sigma_b^2}{x_i}) \tag{8}
$$
其中：
$$
\frac{\partial (x_i - \mu_b)}{\partial x_i} = \frac{m-1}{m} \tag{9}
$$

$$
\frac{\partial \sigma_b^2}{\partial x_i} = \frac{1}{m}\sum_{j=1}^{m} 2*(x_j-\mu_b) * \frac{\partial (x_j - \mu_b)}{\partial x_i}
$$
$$
\frac{\partial \sigma_b^2}{\partial x_i} = \frac{2}{m} ((x_i - \mu_b)*(\frac{m-1}{m}) + \sum_{i \neq j}^{} (x_j - \mu_b) * (-\frac{1}{m})) \tag{10}
$$

继续化简，注意到：
$$
\sum_{i \neq j}^{} (x_j - \mu_b) = \sum_{j=1}^{m}(x_j-\mu_b) - (x_i - \mu_b) = 0 - (x_i - \mu_b)  \tag{11}
$$
所以得到：

$$
\frac{\partial \sigma_{b}^{2}}{\partial x_{i}} = \frac{2}{m} ((x_{i} - \mu_{b}) * (\frac{m-1}{m}) + \sum_{i \neq j} (x_j - \mu_b) * (-\frac{1}{m})) \\\\
= \frac{2}{m} ((x_i - \mu_b)*(\frac{m-1}{m}) + (x_i - \mu_b) * \frac{1}{m}) 
$$

$$
\frac{\partial \sigma_b^2}{\partial x_i} = \frac{2}{m}*(x_i - \mu_b) \tag{12}
$$

代入公式8，可以得到 \\(\frac{\partial y_{i}}{\partial x_i}\\)

但是**所有的输出 \\(y_j\\) 都对 \\(x_i\\) 都有依赖**，所以不能简单的用这个结果作为导数，我们需要遍历所有 \\(y_j\\) 对 \\(x_i\\) 的导数并和上游导数相乘，然后累加得到最终的导数：
$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} * \frac{\partial y_j}{\partial x_i})
$$
其中 \\(\frac{\partial L}{\partial y_j}\\) 是上游导数，是一个已知量，求 \\(\frac{\partial y_j}{\partial x_i} = ? \\)
$$ y_j = \gamma \frac{x_j-\mu_b}{\sigma} + \beta $$

令归一化值
$$
\hat{x}_{i} = \frac{x_i - \mu_b}{\sigma}
$$

则
$$
y_i = \gamma \hat{x}_{i} + \beta
$$

\\( \beta \\) 是加法常数求导为0，\\( \gamma \\) 是乘法常数需保留，因此先计算 \\( \frac{\partial \hat{x}_{j}}{\partial x_i} \\)

$$
\sigma = \sqrt{\sigma_{b}^2 + \epsilon}
$$

$$
\frac{\partial \hat{x}_{j}}{\partial x_i} = \frac{1}{\sigma} \frac{\partial (x_j-\mu_b)}{\partial x_i} - \frac{x_j-\mu_b}{\sigma^2} \frac{\partial \sigma}{\partial x_i}
$$

先求第一部分：
$$
\frac{\partial (x_j-\mu_b)}{\partial x_i} = \begin{cases}
\frac{m-1}{m} &\quad i = j \\\\
-\frac{1}{m} &\quad i \neq j
\end{cases}
$$
第二部分：
$$
\frac{\partial \sigma}{\partial x_i} = \frac{1}{2\sigma} \frac{\partial \sigma_b^2}{\partial x_i}
$$
根据公式（12），可知：
$$
\frac{\partial \sigma}{\partial x_i} = \frac{1}{2\sigma} \frac{2}{m}*(x_i - \mu_b) = \frac{x_i - \mu_b}{m \sigma}
$$

合并起来得到：
$$
\frac{\partial \hat{x}_j}{\partial x_i} = \frac{1}{\sigma} \frac{\partial (x_j-\mu_b)}{\partial x_i} - \frac{x_j-\mu_b}{\sigma^2} \frac{x_i - \mu_b}{m \sigma}
$$

得到最终 \\(Loss\\) 关于 \\(x_i\\) 的导数：

$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} * \frac{\partial y_j}{\partial x_i}) = \gamma \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} * \frac{\partial \hat{x}_{j}}{\partial x_i}) 
$$

$$
\frac{\partial L}{\partial x_i} = \gamma \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} * \frac{1}{\sigma} \frac{\partial (x_j-\mu_b)}{\partial x_i}) - \gamma \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} * \frac{x_j-\mu_b}{\sigma^2} \frac{x_i - \mu_b}{m \sigma})
$$

第一部分：

$$
\gamma \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \frac{1}{\sigma} \frac{\partial (x_j-\mu_b)}{\partial x_i}) = \frac{\gamma}{\sigma} \sum_{j=1}^{m} (\frac{\partial L}{\partial y_j} \cdot \frac{\partial (x_j-\mu_b)}{\partial x_i}) = \frac{\gamma}{\sigma}(\frac{\partial L}{\partial y_i} \cdot \frac{m-1}{m} + \sum_{j \neq i}^{m}\frac{\partial L}{\partial y_j} \cdot \frac{-1}{m}) = \frac{\gamma}{\sigma}(\frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_j})
$$

第二部分：

$$
\gamma \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \frac{x_j-\mu_b}{\sigma^2} \frac{x_i - \mu_b}{m \sigma}) = \gamma \frac{x_i - \mu_b}{m \sigma} \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \frac{x_j-\mu_b}{\sigma^2})
$$

$$
= \gamma \frac{x_{i} - \mu_b}{m \sigma} \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \frac{\hat{x_{j}}}{\sigma}) = \gamma \frac{\hat{x_{i}}}{m \sigma} \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \hat{x_{j}})
$$

最终合并得到：

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sigma}(\frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}) - \frac{\gamma \hat{x_{i}}}{m \sigma} \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \hat{x_{j}}) = \frac{\gamma}{m \sigma}(m \frac{\partial L}{\partial y_i} - \sum_{j=1}^{m}\frac{\partial L}{\partial y_j} - \hat{x_{i}} \sum_{j=1}^{m}(\frac{\partial L}{\partial y_j} \cdot \hat{x_{j}}))
$$

这里 \\(m\\) 是常数，\\(\sigma = \sqrt{\sigma_{b}^2 + \epsilon}\\) 、\\(\hat{x}_{i} = \frac{x_i - \mu_b}{\sigma}\\) 通过前向计算可得到，\\(\frac{\partial L}{\partial y_i}\\) 是上游导数已知，\\(\gamma\\) 是可训练参数已知，所以反向求导的过程计算复杂度是`O(m)`

### 推理模式
训练阶段可以计算得到mini batch的均值和方差，在推理阶段因为输入数据是单个data，没有mini batch，所以推理阶段需要切换到一个预置的均值和方差，这个预置全局的均值和方差也是训练阶段（滑动平均）得到的：
$$
\mu_{global} = (1 - \alpha) * \mu_{global} + \alpha * \mu_{i}  \\\\
\sigma_{global}^2 = (1 - \alpha) * \sigma_{global}^2 + \alpha * \sigma_{i}^2
$$

## conv2D
对于 $N \times Cin \times Hin \times Win$ 排列的输入数据来说，经过conv2d的输出是 $N \times Cout \times Hout \times Wout$ ，我们先把Batch维度`N`省略，从单一输入数据看计算过程：
$$
out(Cout_{i}) = bias(Cout_{i}) + \sum_{k=0}^{Cin-1} input(k) * weight(Cout_{i}, k) \\\\
out \in \mathbb{R}^{Cout \times Hout \times Wout} \\\\
input \in \mathbb{R}^{Cin \times Hin \times Win} \\\\
bias \in \mathbb{R}^{Cout \times Hout \times Wout} \\\\
weight \in \mathbb{R}^{Cout \times Cin \times ksize[0] \times ksize[1]} \\\\
weight(Cout_{i}, k) \in \mathbb{R}^{ksize[0] \times ksize[1]}
$$

影响卷积计算和最终输出`Hout * Wout`的因素包括：`ksize, stride, padding, dilation`，输出`Hout * Wout`的计算逻辑：
$$
Hout = \frac{Hin + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1}{stride[0]} + 1 \\\\
Wout = \frac{Win + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1}{stride[1]} + 1
$$

padding模式也会影响（部分边缘）输出。

需要求导的有：
$$
\frac{\partial Out}{\partial In} \\\\
\\
\frac{\partial Out}{\partial bias} = 1 \\\\
\\
\frac{\partial Out}{\partial weight}
$$

我们从Out局部`(cout, i, j)`展开计算过程得到：
$$
Out(cout,i,j) = \sum_{ch=0}^{Cin-1}\sum_{p=0}^{k_0-1}\sum_{q=0}^{k_1-1} weight(cout,ch,p,q)\cdot In\big(ch,\;\; m(i,p),\;\; n(j,q)\big) \\\\
m(i,p) = i\cdot stride[0] - padding[0] + p\cdot dilation[0],\qquad n(j,q) = j\cdot stride[1] - padding[1] + q\cdot dilation[1]
$$
这里需要根据输出坐标`i, j`反推输入坐标`m, n`的值范围，第一个卷积运算的起始点是`-padding[0], -padding[1]`，然后第一个卷积计算的结束点是`-padding[0] + dilation[0] * (ksize[0] - 1), -padding[1] + dilation[1] * (ksize[1] - 1)`，以此类推得到：
$$
m0 = i * stride[0] - padding[0] \\\\
m1 = i * stride[0] - padding[0] + dilation[0] * (ksize[0] - 1) \\\\
n0 = j * stride[1] - padding[1] \\\\
n1 = j * stride[1] - padding[1] + dilation[1] * (ksize[1] - 1)
$$
所以结果就明了了，每一个输出点对weight，input做一次求导，然后把相同坐标的梯度累加起来即可得到最终梯度：
$$
\frac{\partial Out_{cout, i, j}}{\partial weight_{cout, ch, p, q}} = In(ch, m(i,p), n(j,q)) \\\\
\frac{\partial Out_{cout, i, j}}{\partial In_{ch, m(i,p), n(j,q)}} = weight_{cout, ch, p, q}
$$
