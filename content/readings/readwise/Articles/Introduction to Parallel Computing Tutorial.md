---
author: llnl.gov
title: Introduction to Parallel Computing Tutorial
category: #articles
url: https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial
---
![rw-book-cover](https://readwise-assets.s3.amazonaws.com/static/images/article0.00998d930354.png)

## Metadata
- Author: llnl.gov
- Full Title: Introduction to Parallel Computing Tutorial
- Category: #articles
- Summary: This document titled "Introduction to Parallel Computing Tutorial" provides a brief overview of parallel computing and is intended as an introduction to the subject. It covers the basics of parallel computing, including concepts and terminology, parallel memory architectures, and programming models. The document also discusses the reasons for using parallel computing, such as saving time and money, solving larger and more complex problems, and taking advantage of non-local resources. It concludes by highlighting the use of parallel computing in science and engineering, industrial and commercial applications, and global applications.
- URL: https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial

## Highlights
- Parallel computers still follow this basic design, just multiplied in units. The basic, fundamental architecture remains the same. ([View Highlight](https://read.readwise.io/read/01hym7hfq34bgfhmr0xg5a78tn))
    - Note: is the von Neumann architecture the best for AI computing?
- Flynn's taxonomy distinguishes multi-processor computer architectures according to how they can be classified along the two independent dimensions of ***Instruction Stream*** and ***Data Stream***. Each of these dimensions can have only one of two possible states: ***Single*** or ***Multiple***. ([View Highlight](https://read.readwise.io/read/01hym7nw32t2p8qa0qf8efjjfb))
    - Note: most common categories: SIMD, MIMD
- Single Instruction, Multiple Data (SIMD) ([View Highlight](https://read.readwise.io/read/01hym7qqepsmh80p67jqdbvba4))
- Multiple Instruction, Multiple Data (MIMD) ([View Highlight](https://read.readwise.io/read/01hym854hyqfnrg4rtcj4yq85j))
- Symmetric Multi-Processor (SMP) 
  对称多处理器 (SMP)
  Shared memory hardware architecture where multiple processors share a single address space and have equal access to all resources - memory, disk, etc. ([View Highlight](https://read.readwise.io/read/01hymc0dcf8wwsmg4s11er8jtd))
- Amdahl's Law states that potential program speedup is defined by the fraction of code (P) that can be parallelized ([View Highlight](https://read.readwise.io/read/01hymb29798gpycr7k3krr3g7v))
- Historically, shared memory machines have been classified as ***UMA*** and ***NUMA***, based upon memory access times. ([View Highlight](https://read.readwise.io/read/01hymmddmpwtwvtxhxhwtdbeaz))
- Uniform Memory Access (UMA) ([View Highlight](https://read.readwise.io/read/01hymnhw1jpv8dj75zdq7rk0t9))
- Non-Uniform Memory Access (NUMA) ([View Highlight](https://read.readwise.io/read/01hymnj0vgznxrwn9r8qc45pd8))
