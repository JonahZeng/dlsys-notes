# CUDA Basic

## Async api and Event
cuda可以通过异步API来实现kerne并发执行，即不堵塞host，host端通过指定cuda stream执行异步API来实现并发。在host端不同步的情况下，如果需要对kernel，Async API计时，则需要引入cuda event。
```cpp
// 创建event handler
cudaEvent_t start, stop;
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));
// 录制gpu上的事件
cudaEventRecord(start, 0);
// Async API，不堵塞host
cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
// 继续录制gpu上事件
cudaEventRecord(stop, 0);
// 等待stop事件完成，通常不需要，cudaEventElapsedTime会保证stop事件完成
checkCudaErrors(cudaEventSynchronize(stop));
// 计算gpu事件时间差
checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
printf("time spent executing by the GPU: %.2f\n", gpu_time);
// 销毁event handler
checkCudaErrors(cudaEventDestroy(start));
checkCudaErrors(cudaEventDestroy(stop));
```

## 并发执行多个kernel
下面一个实际示例：
```cpp
// allocate and initialize an array of stream handles
cudaStream_t *streams =
    (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

for (int i = 0; i < nstreams; i++) {
  checkCudaErrors(cudaStreamCreate(&(streams[i])));
}
// kernel evnet，用于stream同步
cudaEvent_t *kernelEvent;
kernelEvent = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));

for (int i = 0; i < nkernels; i++) {
  checkCudaErrors(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
}
// 并发stream执行kernel
for (int i = 0; i < nkernels; ++i) {
  clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
  checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));

  // 如果需要最后一个stream处理善后，则添加依赖关系，让最后一个stream等待到最后才执行
  checkCudaErrors(cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0));
}

for (int i = 0; i < nkernels; i++) {
  cudaStreamDestroy(streams[i]);
  cudaEventDestroy(kernelEvent[i]);
}
```

## cuda thread/block执行模型
1. thread block 会被发送到同一个stream multiprocessor去执行，thread block共享shared memory；
2. thread block 内的thread被拆分为若干个warp，每个warp 32个thread，这些warp会尽量常驻在sm上（如果sm资源足够容纳所有这些warp的情况下），由sm调度器来并发交错执行（非cpu那种排队执行），通过warp切换执行来隐藏memory访问延迟。
3. 如果sm资源不能容纳所有warp常驻，那么只有部分warp常驻，剩余warp在排队，但是，排队的warp也不是完全串行，在sm上的资源释放后，这些warp立即加入仍然并发交错执行。
4. 如果sm被分配多个block，sm仍然会尽量让block常驻，仍然用并发交错执行warp的方式来隐藏memory延迟，这是gpu设计的核心理念之一：高吞吐；

获取gpu的资源属性：
```cpp
void print_gpu_info(int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("=== Device %d: %s ===\n", device, prop.name);

    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);

    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);

    printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);
    printf("Shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);

    int maxWarpsPerSM;
    cudaDeviceGetAttribute(&maxWarpsPerSM,
        cudaDevAttrMaxWarpsPerMultiprocessor, device);
    printf("Max warps per SM: %d\n", maxWarpsPerSM);

    int maxBlocksPerSM;
    cudaDeviceGetAttribute(&maxBlocksPerSM,
        cudaDevAttrMaxBlocksPerMultiprocessor, device);
    printf("Max blocks per SM: %d\n", maxBlocksPerSM);
}
```

思考：
1. 如果thread的寄存器和block的shared memory占用过大，导致sm驻留的warp数量偏小，如何优化？
   驻留warp过小，导致memory延迟被暴漏出来，先通过Nsight Compute来确定：

    寄存器受限（Register Limited）--- 优化寄存器占用，减小临时变量，减小内联优化导致的寄存器占用，减小大数组放在寄存器上等措施
    shared memory 受限（Shared Memory Limited）--- 减小tile，较小double buffer，等
    warp 数量受限（Warp Limited）--- 
    block 数量受限（Block Limited）

    使用API查询最大blockDim:
    ```cpp
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks,
      kernel,
      blockSize,
      sharedMemSize);
    ```

2. 如果thread占用寄存器很小，没有shared memory占用，应该怎么设置gridDim，blockDim？
    最佳实践：
    blockDim = 128 或 256
    gridDim = (N + blockDim - 1) / blockDim
    gridDim ≥ SM 数 × 4（保证 SM 不空泡）
    memory-bound → blockDim 越大越好（256–512）
    compute-bound → blockDim 128–256 最佳

## lanuch kernel configure
cuda提供API: `cudaOccupancyMaxPotentialBlockSize` 获取高占用率的block size配置和最小的grid size配置：
```cpp
/**
* minGridSize：返回满足高占用率的最小grid size
* blockSize：返回高占用率的block size
* func：kernel函数
* dynamicSMemSize：block内的动态shared memory占用
* blockSizeLimit：kernel函数的最大block size，比如说kernel函数需要处理1000个数据，每个thread处理一个数据，那么block size最大限制就是1000
*/
template < class T >
__host__​cudaError_t cudaOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int  blockSizeLimit = 0 ) [inline]
```
计算硬件利用率，硬件利用率是指：驻留thread warp数量 / sm最大可驻留warp数量。这里需要用到一个api:
```cpp
/**
* numBlocks: 返回sm可驻留的block数量
* func：kernel函数
* blockSize：block size配置
* dynamicSMemSize：block内的动态shared memory占用
*/
template < class T >
__host__​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, T func, int  blockSize, size_t dynamicSMemSize ) [inline]
```
理论利用率计算：
```cpp
activeWarps = numBlocks * blockSize / prop.warpSize;
maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

occupancy = (double)activeWarps / maxWarps;
```

## fp16
fp16是AI领域常用的基础数据类型，cuda硬件支持fp16的运算指令，以及simd指令
```cpp
half2 value = __float2half2_rn(0.f); // 一次创建并初始化2个fp16
value = __hfma2(a[i], b[i], value);  // 一个指令完成两个fp16的乘加运算， simd
half2 result;
float f_result = __low2float(result) + __high2float(result); // half2拆分为两个fp16然后相加

__hadd2(v[threadIdx.x], v[threadIdx.x + 64]); // half2，两个fp16逐元素相加
```

## 原子指令
原子指令用于多thread共同读写基础变量，为了防止数据竞争导致的错误。
```cpp
  // Atomic addition
  atomicAdd(&g_odata[0], 10);

  // Atomic subtraction (final should be 0)
  atomicSub(&g_odata[1], 10);

  // Atomic exchange
  atomicExch(&g_odata[2], tid);

  // Atomic maximum
  atomicMax(&g_odata[3], tid);

  // Atomic minimum
  atomicMin(&g_odata[4], tid);

  // Atomic increment (modulo 17+1)
  atomicInc((unsigned int *)&g_odata[5], 17);

  // Atomic decrement
  atomicDec((unsigned int *)&g_odata[6], 137);

  // Atomic compare-and-swap
  atomicCAS(&g_odata[7], tid - 1, tid);

  // Bitwise atomic instructions

  // Atomic AND
  atomicAnd(&g_odata[8], 2 * tid + 7);

  // Atomic OR
  atomicOr(&g_odata[9], 1 << tid);

  // Atomic XOR
  atomicXor(&g_odata[10], tid);
```

## zero copy
通常来说host只能操作host的内存数据，device只能操作device端的内存数据，如果device要操作host的内存数据，需要通过`cudaMemcpy`把数据复制到device端；但是device端也可以通过memory map来直接操作host的数据：
```cpp
size_t bytes;
float *a, *b, *c;           // Pinned memory allocated on the CPU
float *a_UA, *b_UA, *c_UA;  // Non-4K Aligned Pinned memory on the CPU
float *d_a, *d_b, *d_c;     // Device pointers for mapped memory

checkCudaErrors(cudaHostAlloc((void **)&a, bytes, cudaHostAllocMapped));
checkCudaErrors(cudaHostAlloc((void **)&b, bytes, cudaHostAllocMapped));
checkCudaErrors(cudaHostAlloc((void **)&c, bytes, cudaHostAllocMapped));
// host的内存地址被映射到device内存地址
checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
```