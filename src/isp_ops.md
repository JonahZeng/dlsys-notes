# ISP常用算子

## PWL(piecewise linear)
对于给定的递增`x`节点 \\([x_{1}, x_{2} \dots x_{k}]\\) ，有对应的`y`节点 \\([y_{1}, y_{2} \dots y_{k}]\\) :
$$
Out =
\begin{cases}
    y_{1}, & if\text{ }In < x_{1},\\\\
    \frac{y_{2} - y_{1}}{x_{2} - x_{1}} \cdot (In - x_{1}) + y_{1}, & if\text{ } x_{1} \leq In < x_{2},\\\\
    ... \\\\
    \frac{y_{k} - y_{k-1}}{x_{k} - x_{k-1}} \cdot (In - x_{k-1}) + y_{k-1}, & if\text{ } x_{k-1} \leq In < x_{k},\\\\
    y_{k}, & otherwise
\end{cases}
$$

反向传播求导，因为是输入输出是1对1的关系，所以对最终`Loss`的导数就是
$$ \frac{\partial Loss}{\partial Out_{i}} * \frac{\partial Out_{i}}{\partial In_{i}} $$
展开单独看后半部分：
$$
\frac{\partial Out}{\partial In} =
\begin{cases}
    0, & if\text{ }In < x_{1},\\\\
    \frac{y_{2} - y_{1}}{x_{2} - x_{1}}, & if\text{ } x_{1} \leq In < x_{2},\\\\
    ... \\\\
    \frac{y_{k} - y_{k-1}}{x_{k} - x_{k-1}}, & if\text{ } x_{k-1} \leq In < x_{k},\\\\
    0, & otherwise
\end{cases}
$$
如果要求`x`和`y`参数可训：
$$
\frac{\partial Loss}{\partial x_{j}} = \sum_{i=1}^{m} \frac{\partial Loss}{\partial Out_{i}} * \frac{\partial Out_{i}}{\partial x_{j}} \\\\
\frac{\partial Loss}{\partial y_{j}} = \sum_{i=1}^{m} \frac{\partial Loss}{\partial Out_{i}} * \frac{\partial Out_{i}}{\partial y_{j}} \\\\
j \in \\{1,2,3...k \\}
$$
接下来，对每一个 \\(In_{i}\\) 找到对应的`j`索引，如果 `j=0`（比最左侧x节点还小）：
$$
\frac{\partial Out_{i}}{\partial x_{1}} = 0 \\\\
\frac{\partial Out_{i}}{\partial y_{1}} = 1
$$
如果`j=k`（比最右侧x节点还大）
$$
\frac{\partial Out_{i}}{\partial x_{k}} = 0 \\\\
\frac{\partial Out_{i}}{\partial y_{k}} = 1
$$
如果是正常情况 \\( 1 \leq j < k \\) :
$$
Out_{i} = \frac{y_{j+1} - y_{j}}{x_{j+1} - x_{j}} \cdot (In_{i} - x_{j}) + y_{j} \\\\
\frac{\partial Out_{i}}{\partial x_{j}} = \frac{In_{i}-x_{j+1}}{(x_{j+1}-x_{j})^2} * (y_{j+1} - y_{j}) \\\\
\frac{\partial Out_{i}}{\partial x_{j+1}} = \frac{x_{j} - In_{i}}{(x_{j+1}-x_{j})^2} * (y_{j+1} - y_{j}) \\\\
\frac{\partial Out_{i}}{\partial y_{j}} = 1 - \frac{In_{i} - x_{j}}{x_{j+1}-x_{j}} \\\\
\frac{\partial Out_{i}}{\partial y_{j+1}} = \frac{In_{i} - x_{j}}{x_{j+1}-x_{j}}
$$

## 3D lut

### 正向
3D lookup table同样是点对点关系，要求`channels=3`，为了让输入全部可用，先对输入做`clip(0, 1)`处理。先设3d lut的3维坐标节点分别为 \\([x_{1}, x_{2} \dots x_{t}], [y_{1}, y_{2} \dots y_{t}], [z_{1}, z_{2} \dots z_{t}]\\)，首先找到输入点对应的节点索引：
$$
i = when(x_{i} \leq In_{1} < x_{i+1}) \\\\
j = when(y_{j} \leq In_{2} < y_{j+1}) \\\\
k = when(z_{k} \leq In_{3} < z_{k+1})
$$
输入点周围最近的8个节点分别为
$$
p000 = (x_{i}, y_{j}, z_{k}), p100 = (x_{i+1}, y_{j}, z_{k}) \\\\
p010 = (x_{i}, y_{j+1}, z_{k}), p110 = (x_{i+1}, y_{j+1}, z_{k}) \\\\
p001 = (x_{i}, y_{j}, z_{k+1}), p101 = (x_{i+1}, y_{j}, z_{k+1}) \\\\
p011 = (x_{i}, y_{j+1}, z_{k+1}), p111 = (x_{i+1}, y_{j+1}, z_{k+1})
$$
然后计算8个点的权重，它们的累加和为1：
$$
w000 = w_{x} * w_{y} * w_{z}, w100 = (1 - w_{x}) * w_{y} * w_{z} \\\\
w010 = w_{x} * (1 - w_{y}) * w_{z}, w110 = (1 - w_{x}) * (1 - w_{y}) * w_{z} \\\\
w001 = w_{x} * w_{y} * (1- w_{z}), w101 = (1 - w_{x}) * w_{y} * (1- w_{z}) \\\\
w011 = w_{x} * (1 - w_{y}) * (1- w_{z}), w111 = (1 - w_{x}) * (1 - w_{y}) * (1- w_{z})
$$
其中：
$$
w_{x} = \frac{x_{i+1} - In_{1}}{x_{i+1} - x_{i}} \\\\
w_{y} = \frac{y_{j+1} - In_{2}}{y_{j+1} - y_{j}} \\\\
w_{z} = \frac{z_{k+1} - In_{3}}{z_{k+1} - z_{k}}
$$
插值权重决定偏移矢量 \\(X_{s}, Y_{s}, Z_{s}\\)：
$$
X_{s} = w000 * X_{000} + w100 * X_{100} + w010 * X_{010} + w110 * X_{110} + w001 * X_{001} + w101 * X_{101} + w011 * X_{011} + w111 * X_{111} \\\\
Y_{s} = w000 * Y_{000} + w100 * Y_{100} + w010 * Y_{010} + w110 * Y_{110} + w001 * Y_{001} + w101 * Y_{101} + w011 * Y_{011} + w111 * Y_{111} \\\\
Z_{s} = w000 * Z_{000} + w100 * Z_{100} + w010 * Z_{010} + w110 * Z_{110} + w001 * Z_{001} + w101 * Z_{101} + w011 * Z_{011} + w111 * Z_{111}
$$
然后输出：
$$
Out_{1} = In_{1} + X_{s} \\\\
Out_{2} = In_{2} + Y_{s} \\\\
Out_{3} = In_{3} + Z_{s}
$$

### 反向
这里有9个必须的求导:
$$
\frac{\partial Out_{1}}{\partial In_{1}} = 1 + \frac{\partial X_{s}}{\partial In_{1}} = 1 + X_{000} * \frac{\partial w000}{\partial In_{1}} + X_{100} * \frac{\partial w100}{\partial In_{1}} + \dots X_{111} * \frac{\partial w111}{\partial In_{1}} \\\\
\frac{\partial Out_{1}}{\partial In_{2}} = \frac{\partial X_{s}}{\partial In_{2}} = X_{000} * \frac{\partial w000}{\partial In_{2}} + X_{100} * \frac{\partial w100}{\partial In_{2}} + \dots X_{111} * \frac{\partial w111}{\partial In_{2}} \\\\
\frac{\partial Out_{1}}{\partial In_{3}} = \frac{\partial X_{s}}{\partial In_{3}} = X_{000} * \frac{\partial w000}{\partial In_{3}} + X_{100} * \frac{\partial w100}{\partial In_{3}} + \dots X_{111} * \frac{\partial w111}{\partial In_{3}} \\\\
\frac{\partial Out_{2}}{\partial In_{1}} = \frac{\partial Y_{s}}{\partial In_{1}} = Y_{000} * \frac{\partial w000}{\partial In_{1}} + Y_{100} * \frac{\partial w100}{\partial In_{1}} + \dots Y_{111} * \frac{\partial w111}{\partial In_{1}} \\\\
\frac{\partial Out_{2}}{\partial In_{2}} = 1 + \frac{\partial Y_{s}}{\partial In_{2}} = 1 + Y_{000} * \frac{\partial w000}{\partial In_{2}} + Y_{100} * \frac{\partial w100}{\partial In_{2}} + \dots Y_{111} * \frac{\partial w111}{\partial In_{2}} \\\\
\frac{\partial Out_{2}}{\partial In_{3}} = \frac{\partial Y_{s}}{\partial In_{3}} = Y_{000} * \frac{\partial w000}{\partial In_{3}} + Y_{100} * \frac{\partial w100}{\partial In_{3}} + \dots Y_{111} * \frac{\partial w111}{\partial In_{3}} \\\\
\frac{\partial Out_{3}}{\partial In_{1}} = \frac{\partial Z_{s}}{\partial In_{1}} = Z_{000} * \frac{\partial w000}{\partial In_{1}} + Z_{100} * \frac{\partial w100}{\partial In_{1}} + \dots Z_{111} * \frac{\partial w111}{\partial In_{1}} \\\\
\frac{\partial Out_{3}}{\partial In_{2}} = \frac{\partial Z_{s}}{\partial In_{2}} = Z_{000} * \frac{\partial w000}{\partial In_{2}} + Z_{100} * \frac{\partial w100}{\partial In_{2}} + \dots Z_{111} * \frac{\partial w111}{\partial In_{2}} \\\\
\frac{\partial Out_{3}}{\partial In_{3}} = 1 + \frac{\partial Z_{s}}{\partial In_{3}} = 1 + Z_{000} * \frac{\partial w000}{\partial In_{3}} + Z_{100} * \frac{\partial w100}{\partial In_{3}} + \dots Z_{111} * \frac{\partial w111}{\partial In_{3}}
$$
接下来只要求插值权重对输入的导数：
$$
\frac{\partial w000}{\partial In_{1}} = \frac{-1}{x_{i+1} - x_{i}} * w_{y} * w_{z}, 
\frac{\partial w000}{\partial In_{2}} = w_{x} * \frac{-1}{y_{j+1} - y_{j}} * w_{z}, 
\frac{\partial w000}{\partial In_{3}} = w_{x} * w_{y} * \frac{-1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w100}{\partial In_{1}} = \frac{1}{x_{i+1} - x_{i}} * w_{y} * w_{z}, 
\frac{\partial w100}{\partial In_{2}} = (1 - w_{x}) * \frac{-1}{y_{j+1} - y_{j}} * w_{z}, 
\frac{\partial w100}{\partial In_{3}} = (1 - w_{x}) * w_{y} * \frac{-1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w010}{\partial In_{1}} = \frac{-1}{x_{i+1} - x_{i}} * (1 - w_{y}) * w_{z}, 
\frac{\partial w010}{\partial In_{2}} = w_{x} * \frac{1}{y_{j+1} - y_{j}} * w_{z}, 
\frac{\partial w010}{\partial In_{3}} = w_{x} * (1 - w_{y}) * \frac{-1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w110}{\partial In_{1}} = \frac{1}{x_{i+1} - x_{i}} * (1 - w_{y}) * w_{z}, 
\frac{\partial w110}{\partial In_{2}} = (1 - w_{x}) * \frac{1}{y_{j+1} - y_{j}} * w_{z}, 
\frac{\partial w110}{\partial In_{3}} = (1 - w_{x}) * (1 - w_{y}) * \frac{-1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w001}{\partial In_{1}} = \frac{-1}{x_{i+1} - x_{i}} * w_{y} * (1 - w_{z}), 
\frac{\partial w001}{\partial In_{2}} = w_{x} * \frac{-1}{y_{j+1} - y_{j}} * (1 - w_{z}), 
\frac{\partial w001}{\partial In_{3}} = w_{x} * w_{y} * \frac{1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w101}{\partial In_{1}} = \frac{1}{x_{i+1} - x_{i}} * w_{y} *(1 - w_{z}), 
\frac{\partial w101}{\partial In_{2}} = (1 - w_{x}) * \frac{-1}{y_{j+1} - y_{j}} * (1 - w_{z}), 
\frac{\partial w101}{\partial In_{3}} = (1 - w_{x}) * w_{y} * \frac{1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w011}{\partial In_{1}} = \frac{-1}{x_{i+1} - x_{i}} * (1 - w_{y}) * (1 - w_{z}), 
\frac{\partial w011}{\partial In_{2}} = w_{x} * \frac{1}{y_{j+1} - y_{j}} * (1 - w_{z}), 
\frac{\partial w011}{\partial In_{3}} = w_{x} * (1 - w_{y}) * \frac{1}{z_{k+1} - z_{k}} \\\\
\frac{\partial w111}{\partial In_{1}} = \frac{1}{x_{i+1} - x_{i}} * (1 - w_{y}) * (1 - w_{z}), 
\frac{\partial w111}{\partial In_{2}} = (1 - w_{x}) * \frac{1}{y_{j+1} - y_{j}} * (1 - w_{z}), 
\frac{\partial w111}{\partial In_{3}} = (1 - w_{x}) * (1 - w_{y}) * \frac{1}{z_{k+1} - z_{k}}
$$


还有可选的训练参数求导：
$$
\frac{\partial Out_{1}}{\partial X_{000}}, \frac{\partial Out_{1}}{\partial X_{100}} \dots \frac{\partial Out_{1}}{\partial X_{111}} \\\\
\frac{\partial Out_{2}}{\partial Y_{000}}, \frac{\partial Out_{2}}{\partial Y_{100}} \dots \frac{\partial Out_{2}}{\partial Y_{111}} \\\\
\frac{\partial Out_{3}}{\partial Z_{000}}, \frac{\partial Out_{3}}{\partial Z_{100}} \dots \frac{\partial Out_{3}}{\partial Z_{111}}
$$
因为 \\(X_{s}\\) 是偏移量和权重的线性组合，权重又是常数（对表格参数而言），所以每个导数就等于对应的插值权重：
$$
\frac{\partial Out_{1}}{\partial X_{c}} = w_{c}, \\\\
\frac{\partial Out_{2}}{\partial Y_{c}} = w_{c}, \\\\
\frac{\partial Out_{3}}{\partial Z_{c}} = w_{c}, \\\\
c \in \\{000, 100, 010, 110, 001, 101, 011, 111\\}
$$