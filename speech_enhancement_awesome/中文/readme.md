- [x] 1. 基于评价指标网络的语音增强优化研究（多目标）

  - 评价指标（**PESQ+STOI**）+增强模型：联合训练
  - 数据集：TIMIT+NoiseX92
  - 实验：不同信噪比、不同噪声。
    - *每个句子只在一个信噪比水平上被一个噪声类型破坏
    - 增强网络根据输入特征的不同分为频域模型和时域模型
    - *原始\不同评价指标\损失函数带来的频谱图的效果也不同（噪声残留、谐波结构清晰与否）
    - 对低频和高频的影响不同，MSE优化的是低频部分、高频对听觉影响较大

  > MATLAB measure：[dakenan1/Speech-measure-SDR-SAR-STOI-PESQ: Speech quality measure of SDR、SAR、STOI、ESTOI、PESQ via MATLAB (github.com)](https://github.com/dakenan1/Speech-measure-SDR-SAR-STOI-PESQ)

- [x] 2 基于复值掩蔽与扩张卷积的实时语音增强方法  （*IRM vs cIRM）

  - 实时语音增强、复值遮掩
  - encoder输入语音stft后的实部和虚部，decoder输出IR
  - 损失函数：PASE
  - 评价指标：PESQ+STOI
  - 实验：实验中对实时性进行了验证

  > [[2005.11611\] Exploring the Best Loss Function for DNN-Based Low-latency Speech Enhancement with Temporal Convolutional Networks (arxiv.org)](https://arxiv.org/abs/2005.11611)

- [x] 3 基于多目标联合优化的语音增强方法研究  

  - 过去的方法：通常是对单个输出目标进行损失函数的计算，多目标之间并行的，并没有充分利用多目标之间可能存在的关联。因此本文提出了**多目标联合，双输出（语音和噪声）**
  - 损失函数：SISDR、MSE
  - 数据集：THCHS-30+NOISEX-92+MS-SNSD
  - 实验分析：实验分析结合了ASR进行分析

  > 也许可以结合ASR

- [x] 4 基于区域自适应多尺度卷积的单声道语音增强算法  

  - 多尺度卷积堆叠：提升模型整体的信息感知能力
  - 数据集：TIMIT+NoiseX92
  
- [x] 5 基于动态选择机制的低信噪比单声道语音增强算法  

  - 形变卷积
  - 基于注意力的动态选择机制
  - **渐进学习**（Recursive Learning，Monaural Speech Enhancement with Recursive Learning in the Time Domain）
  - 数据集：TIMIT+NoiseX92

  > [haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement: A minimum unofficial implementation of the "A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement" (CRN) using PyTorch (github.com)](https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement)
  >
  > [Andong-Li-speech/RTNet: implementation of Monaural Speech Enhancement with Recursive Learning in the Time Domain (github.com)](https://github.com/Andong-Li-speech/RTNet)
  >
  > [Andong-Li-speech (AndongLi) / Repositories (github.com)](https://github.com/Andong-Li-speech?tab=repositories)







数据集：