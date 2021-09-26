[视频回放丨AI技术沙龙—语音增强 (qq.com)](https://mp.weixin.qq.com/s/TUSBZXCm0M0-lYZYgCttWg)

## 一

- 1.1 单通道：[[2105.02436\] DBNet: A Dual-branch Network Architecture Processing on Spectrum and Waveform for Single-channel Speech Enhancement (arxiv.org)](https://arxiv.org/abs/2105.02436)
  - [driving-behavior/DBNet: DBNet: A Large-Scale Dataset for Driving Behavior Learning, CVPR 2018 (github.com)](https://github.com/driving-behavior/DBNet)
  - **动机：**不同噪声在时域和频域的表现不同
    - 对于冲击信息，时域更合适
    - 对于单频信号，频域更合适
  - 网络结构：同时做时域和频域（Bridge Layer）
  - 实验基线：WSJ0
    - CRN，时域，幅度谱，4.5M
    - GCRN，频域，复数谱，9.67M
    - AECNN，时域，时频波形，18M
- 1.2 多通道：[[2107.11968\] Inplace Gated Convolutional Recurrent Neural Network For Dual-channel Speech Enhancement (arxiv.org)](https://arxiv.org/abs/2107.11968)
  - 暂时跳过
- 应用：骨传导耳机

> 综述：
>
> - 基于深度学习语音分离技术的研究现状与进展，自动化学报
> - Supervised speech separation based on deep learning: An overview