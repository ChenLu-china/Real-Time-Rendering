# Real-Time-Rendering

# 这一模块用来学习记录实时渲染的相关内容和具体实现（包括CUDA编程学习）
## 基础知识
 所有的预前知识和笔记都记录在util里
### 相机坐标
我们使用右手坐标系， 不同于迪卡尔坐标系，我们对于相机坐标系轴的描述为right(x-axis), up(y-axis), forward(z-axis), 同样为了用一个4x4矩阵表示一个cameraToworld的转换矩阵，我们还需要一个from点(光心)和to点(相机看向)，具体的计算和实现在camera.h文件里。

### Curve曲面和Mesh网格

### BRDF

### 光线的相关理论知识

## Ray Marching
当学完世界栅格化+栅隔化光线追踪+光线协函数 = 基本上完全掌握Pleonexls文章的大部分知识点  
     [14/Dec/22] 补充世界栅格化细节及背景(MSI)技术补充
     [06/Jan/23] 局部世界栅格化细节

## 英伟达Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
优点：秒级的重建速度 甚至有可能到毫秒级
     高质量的渲染效果（十分利于自动驾驶的相关应用）

