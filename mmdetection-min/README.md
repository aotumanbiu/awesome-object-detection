# mmdetection-mini
#### 方便学习目标检测流程 删除其他暂时用不到的文件
#### 依赖和环境需求参考 [mmdetection仓库](https://github.com/open-mmlab/mmdetection)

====================================分割线====================================

#####  相关问题

**1. WIN10 怎么安装mmcv_full ?????**

在之前的mmcv安装过程中，直接 ```pip install mmcv_full``` 总是安装失败，因此需要自己的手动编译，十分麻烦。

这里则提供一个十分简介安装方法：

```python
# cu_version 为CUDA版本 torch_version为pytorch版本
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/11.3/1.12.0/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

**2. mmdetection 运行过程中提示 Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"**

解决方法参考: [Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"的解决办法](https://zhuanlan.zhihu.com/p/471661231)

