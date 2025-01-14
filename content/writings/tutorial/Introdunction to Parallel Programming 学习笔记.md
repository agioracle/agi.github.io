---
title: <Introdunction to Parallel Programming> 学习笔记
date: 2024-08-19
draft: false
tags:
  - tutorials
  - parallel-programming
---

*课程视频链接*： https://www.youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2

[TOC]

## 1. Unit 1
### 1.1 typical CUDA Program

![screenshot-124.390.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191009340.jpg)

![screenshot-32.070.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191009547.jpg)

### 1.2 parallel communication patterns

![screenshot-43.120.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191010957.jpg)



### 1.3 GPU allocate blocks to SMs
![screenshot-78.208.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191010445.jpg)


### 1.3 GPU memory hierarchy
![screenshot-80.277.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191010321.jpg)


![screenshot-63.013.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191010247.jpg)


### 1.4 high level strategies of optimizing performance
![screenshot-106.397.jpg](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202408191010273.jpg)

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291520982.png)

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291522885.png)

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291523996.png)


## 2. Unit 2
### 2.1 parallel communication patters - Map
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291510578.png)
### 2.2 parallel communication patters - Gather
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291511604.png)

### 2.3 parallel communication patters - Scatter
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291512776.png)
### 2.4 parallel communication patters - Stencil
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291513102.png)

### 2.5  parallel communication patters - Transpose
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291515249.png)

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291515484.png)


### 2.6 parallel communication patters recap
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291516574.png)

### 2.7 SM, Kernel, Thread Blocks, Thread 
![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291517262.png)

![](https://futurelog-1251943639.cos.accelerate.myqcloud.com/img/202409291518974.png)