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
  - 多传感器融合：利用加速度传感器检测佩戴者说话时引起的震动，自动隔绝外界噪声
  - 麦克风信息+骨传导信号 -->  神经网络  --> 人声和噪声分离，得到纯净信号

> 综述：
>
> - 基于深度学习语音分离技术的研究现状与进展，自动化学报
> - Supervised speech separation based on deep learning: An overview

## 二、

![图片](https://mmbiz.qpic.cn/mmbiz_png/YtgMd9orlsRFOyqXL5OxCjNksRibV9VAZwyoQb46aSLkP9lNcLYFuYyUaUJtY27tLVJb3VbK1ZeHCK0jdVtsQ7A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- 宽带 vs 超宽带
- 若尔康

## 三、

- CRN
  - Unet-like
  - Estimate real & imaginary spectrum separately

- DCUNET
  - complex convolution
  - jointly estimate real & imaginary spectrum

- DCCRN

  - combine the advantage of CRN and DCUnet，as well as meet the small footprint and low-latency requirement

  - unet-structured complex-valued network with LSTM to model temporal context

  - 推导：

    - complex-valued convolution filter and inputs can be defined like
      $$
      W = W_r + jW_i
      $$

      $$
      X = X_r + jX_i
      $$

    - The convolution can easily  be defined as 
      $$
      F = (X_r*W_r-X_i*W_i)+j(X_r*W_i+X_i*W_r)
      $$

    - similarly,we can get a naive complex LSTM
      $$
      F_{rr}=LSTM_r(X_r)
      $$

      $$
      F_{ir}=LSTM_r(X_i)
      $$

      $$
      F_{ri}=LSTM_i(X_r)
      $$

      $$
      F_{ii}=LSTM_i(X_i)
      $$

      $$
      F_{out}= (F_{rr}-F_{ii}) + j(F_{ri}-F_{ir})
      $$

  - complex encoder/decoder

    - complex conv2d layer/complex convtransposed2d layer
    - complex batch normalization layer
    - real prelu

  - Training target of DCRNN is a Complex ratio mask
    $$
    CRM = \frac{Y_rS_r+Y_iS_i}{Y_r^2+Y_i^2}+j\frac{Y_rS_i-Y_iS_r}{Y_r^2+Y_i^2}
    $$

    - The estimated mask cartesian coordinate representation
      $$
      M^e=M^e_r+jM_i^e
      $$

    - it is also can be expressed in polar coordinates
      $$
      M^e_{mag} = \tanh(\sqrt{M_r^{e2} + M_i^{e2}} )
      $$

      $$
      M^e_{phase} = \arctan2({M_r^{e} + M_i^{e}} )
      $$

    - The clean speech can be estimated as
      $$
      DCRRN-R
      $$

      $$
      DCCRN-C
      $$

      $$
      DCCRN-E
      $$

    - Loss funcation
      $$
      Loss = -SISNR(istft(S^e),s)
      $$

  - Dynamic data augmentation

- DCCRN+
  - subband processing
    - Learnable subband and split and merge moudules to reduce model size and computational cost，实现了降采样的过程，语音里面的低频部分能量比较高（谐波成分比较明显），高频部分的能量比较低，但是是类噪的，因此高频的降噪是具有挑战的
    - neural network based learnable subband split and merge
      - split block
      - merge block
  - **Complex TF-LSTM Block**  --> attention是都能够继续改进 
    - Modeling frequency domain sequence and time domain sequence
    - 先频域建模，在时域建模，可见论文模块b
  - Convolution pathway
    - aggregate richer information from encoder output
  - SNR Esinmator
    - SNR estimation as MTL to maintain good speech quality while reducing noise
    - Problem：directly training neural noise suppressor --> certain amount of speech distortion
    - solution:add frame-level SNR estimator under MTL framework
      - network:1 LSTM + 1 Conv1D w/ sigmoid
      - label
        - extract the amplitude spectrum after STFT
        - Calcalate the log energy of noise and speech based on amplitude spectrum
  - post-processing
    - remove residual noise
  - https://imybo.github.io/dccrn-plus

- DESNet
  - Motivation
    - real-time environment:speech overlapping , directional/isotropic noise and reverberatrion may exist together
    - prior arts: Direct separation on noisy mixtures, cascaded/two-stage (enhancement-separation,separation-enhancement).recursive separation..
    - E2E-UFE and DCCRN show great potenial on multi-channel separation and single-channel enhancement
  - contribution
    - we propose an offline processing neural network for simltaneous speech Dereverberation ,Enhancement and Separation
    - we combine the DNN-WPE,E2E-UFE and DCCRN organically together with diffrerentiable STFT to form an end-to-end manner
    - Staged SNR Strategy
  - we evaluate the performance of the proposed model
    - three scenarios:SE、CSS、NSS
    - Two categories:dereverberated and non-dereverberated-
- DCUNET Front-end for Multi-channel ASR
  - https://arxiv.org/abs/2011.09081
  - cascade：http://arxiv.org/abs/2102.01547
  - Joint：https://github.com/wennet-e2e/wenet













































