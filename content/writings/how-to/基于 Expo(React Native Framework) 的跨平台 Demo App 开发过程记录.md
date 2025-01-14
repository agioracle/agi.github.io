---
title: 基于 Expo(React Native Framework) 的跨平台 Demo App 开发过程记录
date: 2024-12-24
draft: true
tags:
  - React Native Framework
  - Expo
  - multiple platform app
---


## 问题记录
### 1. 在 mac 上创建完项目后，首次执行 `npx expo start`，碰到报错
```
Your macOS system limit does not allow enough watchers for Metro, install Watchman instead. Learn more: [https://facebook.github.io/watchman/docs/install](https://facebook.github.io/watchman/docs/install) Error: EMFILE: too many open files, watch
```

- 解决： 由于 hot reloading 等功能需要持续监测项目文件的变化，通过执行 `breww install watchman` 解决。

