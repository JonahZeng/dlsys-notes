# dlsys-notes

一个具备多年软件开发经验的 AI 初学者，在学习 [Deep Learning Systems（CMU 10-414）](https://dlsyscourse.org/) 过程中的系统化总结笔记。

从传统的应用/系统开发转入 DL system 领域，我希望不只是"会调 PyTorch"，而是真正理解深度学习系统从上层模型到底层硬件的完整链路：**模型是怎么定义的 → 算子是怎么实现的 → 计算是怎么在 GPU 上跑起来的**。这个仓库就是这条学习路径的记录。

## 内容

笔记按学习顺序组织，大致分为三个层次：

### 1. 基础：从零理解深度学习

- [softmax 线性回归](src/softmax_regression.md) — 从 MNIST 分类问题出发，理解深度学习三要素：模型、损失函数、优化方法
- [手工神经网络](src/manual_neural_nets.md) — 不借助框架，手写前向传播与反向传播，理解 computational graph 与 autograd 的本质

### 2. 算子：深度学习的积木

- [神经网络算子](src/neural_operator.md) — 常见网络算子的实现与推导
- [ISP 常用算子](src/isp_ops.md) — 图像信号处理中的常用算子，如 3D LUT、PWL 等

### 3. CUDA：让计算真正跑起来

- [CUDA 异步 API、Event 与 Block/Thread 模型](src/cuda_async_event_block_thread_model.md) — GPU 编程的执行模型与同步机制
- [tile based 矩阵乘法](src/cuda_matmul.md) — 从朴素 CUDA matmul 到 shared memory 优化的分块实现

## 在线阅读

笔记通过 mdBook 构建并部署在 GitHub Pages：

**[https://jonahzeng.github.io/dlsys-notes/](https://jonahzeng.github.io/dlsys-notes/)**

## 本地构建

```bash
# 安装 mdBook
cargo install mdbook

# 本地预览
mdbook serve

# 构建
mdbook build
```

## 说明

- 笔记以中文撰写，夹杂必要的英文术语
- 数学公式使用 MathJax 渲染
- 所有内容基于课程讲义与自己的动手实践整理，欢迎在 [Issues](https://github.com/JonahZeng/dlsys-notes/issues) 指正错误

## License

仅供学习交流使用。
