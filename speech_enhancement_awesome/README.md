# Awesome Speech Enhancement

This repository summarizes the papers, codes and tools for single-/multi-channel speech enhancement/speech seperation task, which aims to create a list of open source projects rather than pursuing the completeness of the papers. You are kindly invited to pull requests. 
<!--TODO ...
datasets...
Tutorials...
https://github.com/topics/beamforming
-->

## Contents
- [Speech_Enhancement](#Speech_Enhancement)
- [Dereverberation](#Dereverberation)
- [Speech_Seperation](#Speech_Seperation)
- [Array_Signal_Processing](#Array_Signal_Processing)
- [Sound_Event_Detection](#Sound_Event_Detection)
- [Tools](#Tools)
- [Resources](#Resources)


## Speech_Enhancement
  ### Magnitude spectrogram

  #### spectral masking
  * 2014, On Training Targets for Supervised Speech Separation, Wang. [[Paper]](https://ieeexplore.ieee.org/document/6887314)  
  * 2018, A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement, [Valin](https://github.com/jmvalin). [[Paper]](https://ieeexplore.ieee.org/document/8547084/) [[RNNoise]](https://github.com/xiph/rnnoise) [[RNNoise16k]](https://github.com/YongyuG/rnnoise_16k)
  * 2020, A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech, [Valin](https://github.com/jmvalin). [Paper](https://arxiv.org/abs/2008.04259) [[PercepNet]](https://github.com/jzi040941/PercepNet)
  * 2021, RNNoise-Ex: Hybrid Speech Enhancement System based on RNN and Spectral Features. [[Paper]](https://arxiv.org/abs/2105.11813) [[RNNoise-Ex]](https://github.com/CedArctic/rnnoise-ex)
  * Other IRM-based SE repositories: [[IRM-SE-LSTM]](https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM) [[nn-irm]](https://github.com/zhaoforever/nn-irm) [[rnn-se]](https://github.com/amaas/rnn-speech-denoising) [[DL4SE]](https://github.com/miralv/Deep-Learning-for-Speech-Enhancement)

  #### spectral mapping
  * 2014, An Experimental Study on Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6665000)
  * 2014, A Regression Approach to Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/document/6932438) [[sednn]](https://github.com/yongxuUSTC/sednn) [[DNN-SE-Xu]](https://github.com/yongxuUSTC/DNN-Speech-enhancement-demo-tool) [[DNN-SE-Li]](https://github.com/hyli666/DNN-SpeechEnhancement) 
  * Other DNN magnitude spectrum mapping-based SE repositories: [[SE toolkit]](https://github.com/jtkim-kaist/Speech-enhancement) [[TensorFlow-SE]](https://github.com/linan2/TensorFlow-speech-enhancement-Chinese) [[UNetSE]](https://github.com/vbelz/Speech-enhancement)
  * 2015, Speech enhancement with LSTM recurrent neural networks and its application to noise-robust ASR, Weninger. [[Paper]](https://hal.inria.fr/hal-01163493/file/weninger_LVA15.pdf)
  * 2016, A Fully Convolutional Neural Network for Speech Enhancement, Park. [[Paper]](https://arxiv.org/abs/1609.07132) [[CNN4SE]](https://github.com/dtx525942103/CNN-for-single-channel-speech-enhancement)
  * 2017, Long short-term memory for speaker generalizationin supervised speech separation, Chen. [[Paper]](http://web.cse.ohio-state.edu/~wang.77/papers/Chen-Wang.jasa17.pdf)
  * 2018, A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf) [[CRN-Tan]](https://github.com/JupiterEthan/CRN-causal)
  * 2018, Convolutional-Recurrent Neural Networks for Speech Enhancement, Zhao. [[Paper]](https://arxiv.org/pdf/1805.00579.pdf) [[CRN-Hao]](https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement)
  * 2020, Online Monaural Speech Enhancement using Delayed Subband LSTM, Li. [[Paper]](https://isca-speech.org/archive/Interspeech_2020/pdfs/2091.pdf)
  * 2020, FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://arxiv.org/pdf/2010.15508.pdf) [[FullSubNet]](https://github.com/haoxiangsnr/FullSubNet)

- https://arxiv.org/pdf/2106.15813.pdf)

### Complex domain

  * 2017, Complex spectrogram enhancement by convolutional neural network with multi-metrics learning, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1704.08504.pdf)
      * 提出：**大家都搞相位和幅度，我搞实部虚部**
          * 过去很多研究只用到了幅度谱，但是相位谱也很重要
          * RI spectrograms与幅度谱相似
          * 借鉴：高信噪比下，相位谱相较干净的语音变化不大？**公式不懂**
      * 解决两个问题：1.phase估计困难；2.单一的目标损失函数并不能同时考虑多个指标
      * 方法：
          * **通过IR图制造wave图，**提出基于CNN在噪声频谱图中使用不同的通道去估计干净语音的实部和虚部（RI），然后直接用于合成语音
          * 多尺度学习（MML），针对Log-power spectrograms （LPS），提出新的目标函数（融合RI和LPS），等效于同时优化segmental signal-to-noise ratio（SSNR）and log-spectral distortion（LSD）
    * 数据集：TIMIT
    * 实验：不同信噪比下，phase的影响
  * 2017, Time-Frequency Masking in the Complex Domain for Speech Dereverberation and Denoising, Williamson. [[Paper]](https://ieeexplore.ieee.org/abstract/document/7906509)
        * 任务：单通道去混响的语音分离和去噪
        * 方法：
              * cIRM，可以恢复相位信息
              * 使用DNN去学习带混响或者噪声的spectral，并映射到cIRM（complex ideal ratio mask）
      * 实验
           * 这里的实验对比可以借鉴
                * 真实环境对比仿真环境
                * 不同的方式对比
                * 不同的噪声对比
      * 在一下篇论文中，将会指出，该论文存在问题
  * 2019, PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network, Yin. [[Paper]](https://arxiv.org/abs/1911.04697) [[PHASEN]](https://github.com/huyanxin/phasen)
        * 使用双流网络，而不是cIRM，其实就是换成了相位预测、幅度预测、通过谐波去重构phase
              * 两个流之间有交互，提高相位预测的质量*（在上一篇论文中指出，直接使用双流网络去预测相位谱并不可靠，独立的相位预测是不可靠的）
              * we propose frequency transformation blocks to catch long-range correlations along the frequency axis.  
              * learned transformation matrix可以捕获**谐波（harmonics）**相关系数，Visualization shows that the learned transformation matrix spontaneously captures the harmonic correlation, which has been proven to be helpful for T-F spectrogram reconstruction.   
                    * **Visualization of FTB weights  ：**to insert frequency transformation blocks (FTBs) to capture global correlations along the frequency axis.
                    * **Two-Stream Architecture**     
      * 数据集：ACSpeech+AudioSet、Voice Bank+DEMAND
  * 2019, Phase-aware Speech Enhancement with Deep Complex U-Net, Choi. [[Paper]](https://arxiv.org/abs/1903.03107) [[DC-UNet]](https://github.com/chanil1218/DCUnet.pytorch)
        * 主要思路：之前的工作主要为“使用幅度谱和重构噪声的相位谱”的方式去构造语音增强系统<--噪声的相位谱+去噪语音的幅度谱 --> 去估计纯净语音的相位谱
        * 解决方法：关于“相位重构困难的问题”
              * U-Net
              * polar coordinate-wise complex-valued masking  
              * a novel loss function, weighted source-to-distortion ratio (wSDR) loss  
      * 论文中的附录：
           * **实数值卷积和复数值卷积**，因为实部和虚部都很重要呀
           * **是否有限制的mark组合相位和幅度**，tanh的有限制更好
           * **Loss**
        * code:[Phase-aware Speech Enhancement with Deep Complex U-Net | Papers With Code](https://paperswithcode.com/paper/phase-aware-speech-enhancement-with-deep-1)
  * 2020, Learning Complex Spectral Mapping With GatedConvolutional Recurrent Networks forMonaural Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf) [[GCRN]](https://github.com/JupiterEthan/GCRN-complex)
        * 可视化：相位谱+平滑 vs 非平滑
        * Gated
  * 2020, DCCRN: Deep Complex Convolution Recurrent Network for Phase-AwareSpeech Enhancement, [Hu](https://github.com/huyanxin). [[Paper]](https://isca-speech.org/archive/Interspeech_2020/pdfs/2537.pdf) [[DCCRN]](https://github.com/huyanxin/DeepComplexCRN)
        * 参考模块
              * ConvSTFT
              * complex convolution
              * SI-SNR
  * 2020, T-GSA: Transformer with Gaussian-Weighted Self-Attention for Speech Enhancement, Kim. [[Paper]](https://ieeexplore.ieee.org/document/9053591) 
  * 2020, Phase-aware Single-stage Speech Denoising and Dereverberation with U-Net, Choi. [[Paper]](https://arxiv.org/abs/2006.00687)

### attention

- 2020,[[2002.05873\] Speech Enhancement using Self-Adaptation and Multi-Head Self-Attention (arxiv.org)](https://arxiv.org/abs/2002.05873
  - 方法：
    - 加上了speaker特征，未知说话者（generalization  vs ）
      - 使用辅助的向量
      - 或者直接使用自注意力进行抽取
    - 多任务学习Loss
  - 说话者的特征信息或许可以参考语音合成
  - 数据集：VoiceBank-DEMAND   
- 2020,[[2009.01941\] Dense CNN with Self-Attention for Time-Domain Speech Enhancement (arxiv.org)](https://arxiv.org/abs/2009.01941)
  - DCN with attention
  - **with a loss** based on the spectral magnitude of enhanced speech.  
  - **Causal convolution  ： real-time** vs Non-Causal 
- 2019,[Sci-Hub | [IEEE ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) - Brighton, United Kingdom (2019.5.12-2019.5.17)\] ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) - An Attention-based Neural Network Approach for Single Channel Speech Enhancement | 10.1109/ICASSP.2019.8683169](https://www.sci-hub.ren/10.1109/icassp.2019.8683169)
  - local causal attention
  - 可视化attention对齐
- [Sci-Hub | Monaural Speech Dereverberation Using Temporal Convolutional Networks with Self Attention | 10.1109/TASLP.2020.2995273 (hkvisa.net)](https://sci-hubtw.hkvisa.net/10.1109/TASLP.2020.2995273)
  - 使用自注意力机制进行去混响
- [Sci-Hub | [IEEE 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) - New Paltz, NY, USA (2019.10.20-2019.10.23)\] 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) - Attention Wave-U-Net for Speech Enhancement | 10.1109/WASPAA.2019.8937186 (hkvisa.net)](https://sci-hubtw.hkvisa.net/10.1109/WASPAA.2019.8937186)
  - We observe that the final layer attention mask has an interpretation as a soft Voice Activity Detector   
  - time domain UNet-wave
- DF-CONFORMER：[2106.15813.pdf (arxiv.org)](https://arxiv.org/pdf/2106.15813.pdf)

### loss

- [[1909.01019\] On Loss Functions for Supervised Monaural Time-Domain Speech Enhancement (arxiv.org)](https://arxiv.org/abs/1909.01019)

  - 

- [[2005.11611\] Exploring the Best Loss Function for DNN-Based Low-latency Speech Enhancement with Temporal Convolutional Networks (arxiv.org)](https://arxiv.org/abs/2005.11611)

  > [Performance comparison evaluation of speech enhancement using various loss functions -The Journal of the Acoustical Society of Korea | Korea Science](http://koreascience.or.kr/article/JAKO202113150644115.page)
  >
  > [sorenchiron/Awesome-Speech-Enhancement - githubmemory](https://githubmemory.com/repo/sorenchiron/Awesome-Speech-Enhancement)

- [Sci-Hub | Learning with Learned Loss Function: Speech Enhancement with Quality-Net to Improve Perceptual Evaluation of Speech Quality | 10.1109/LSP.2019.2953810](https://www.sci-hub.ren/10.1109/LSP.2019.2953810)

  - Quality-Net  

- [[1811.02508\] SDR - half-baked or well done? (arxiv.org)](https://arxiv.org/abs/1811.02508)

  - code:[mitmedialab/WetlandAvianSourceSeparation (github.com)](https://github.com/mitmedialab/WetlandAvianSourceSeparation)
  
  - SI-SAR  SI-SIR
  
- [2102.05109.pdf (arxiv.org)](https://arxiv.org/pdf/2102.05109.pdf)
  
  - contrastive learning:[pranaymanocha/PerceptualAudio: Perceptual Metrics of Audio - perceptually relevant loss function. DPAM and CDPAM (github.com)](https://github.com/pranaymanocha/PerceptualAudio)
  - [2001.04460.pdf (arxiv.org)](https://arxiv.org/pdf/2001.04460.pdf)

- [Sci-Hub | A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality | 10.1109/LSP.2018.2871419 (hkvisa.net)](https://sci-hubtw.hkvisa.net/10.1109/LSP.2018.2871419)
  
  - [Perceptual metric for speech quality evaluation (PMSQE): Source code and audio examples (ugr.es)](http://sigmat.ugr.es/PMSQE/)


### Time domain

  * 2018, Improved Speech Enhancement with the Wave-U-Net, Macartney. [[Paper]](https://arxiv.org/pdf/1811.11307.pdf) [[WaveUNet]](https://github.com/YosukeSugiura/Wave-U-Net-for-Speech-Enhancement-NNabla) 
  * 2019, A New Framework for CNN-Based Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8701652) 
  * 2019, TCNN: Temporal Convolutional Neural Network for Real-time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8683634)
  * 2020, Real Time Speech Enhancement in the Waveform Domain, Defossez. [[Paper]](https://arxiv.org/abs/2006.12847) [[facebookDenoiser]](https://github.com/facebookresearch/denoiser)
  * 2020, Monaural speech enhancement through deep wave-U-net, Guimarães. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417420304061) [[SEWUNet]](https://github.com/Hguimaraes/SEWUNet) 
  * 2020, Speech Enhancement Using Dilated Wave-U-Net: an Experimental Analysis, Ali. [[Paper]](https://ieeexplore.ieee.org/document/9211072)
  * 2020, Densely Connected Neural Network with Dilated Convolutions for Real-Time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ashutosh620.github.io/files/DDAEC_ICASSP_2020.pdf) [[DDAEC]](https://github.com/ashutosh620/DDAEC)
  * 2021, Dense CNN With Self-Attention for Time-Domain Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/9372863)
  * 2021, Dual-path Self-Attention RNN for Real-Time Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://arxiv.org/abs/2010.12713)

  ### GAN
  * 2017, SEGAN: Speech Enhancement Generative Adversarial Network, Pascual. [[Paper]](https://arxiv.org/pdf/1703.09452.pdfsegan_pytorch) [[SEGAN]](https://github.com/santi-pdp/segan_pytorch)
  * 2019, SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty, [Deepak Baby]((https://github.com/deepakbaby)). [[Paper]](https://biblio.ugent.be/publication/8613639/file/8646769.pdf) [[SERGAN]](https://github.com/deepakbaby/se_relativisticgan)
  * 2019, MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1905.04874.pdf) [[MetricGAN]](https://github.com/JasonSWFu/MetricGAN)
  * 2019, MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/abs/2104.03538) [[MetricGAN+]](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN)
  * 2020, HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks, Su. [[Paper]](https://arxiv.org/abs/2006.05694) [[HifiGAN]](https://github.com/rishikksh20/hifigan-denoiser)

  ### Hybrid SE 
  * 2019, Deep Xi as a Front-End for Robust Automatic Speech Recognition, [Nicolson](https://github.com/anicolson). [[Paper]](https://arxiv.org/abs/1906.07319) [[DeepXi]](https://github.com/anicolson/DeepXi)
  * 2019, Using Generalized Gaussian Distributions to Improve Regression Error Modeling for Deep-Learning-Based Speech Enhancement, [Li](https://github.com/LiChaiUSTC). [[Paper]](http://staff.ustc.edu.cn/~jundu/Publications/publications/chaili2019trans.pdf) [[SE-MLC]](https://github.com/LiChaiUSTC/Speech-enhancement-based-on-a-maximum-likelihood-criterion)
  * 2020, Deep Residual-Dense Lattice Network for Speech Enhancement, [Nikzad](https://github.com/nick-nikzad). [[Paper]](https://arxiv.org/pdf/2002.12794.pdf) [[RDL-SE]](https://github.com/nick-nikzad/RDL-SE)
  * 2020, DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation, [Zhang](https://github.com/yunzqq). [[Paper]](https://ieeexplore.ieee.org/document/9066933)
  * 2020, Speech Enhancement Using a DNN-Augmented Colored-Noise Kalman Filter, [Yu](https://github.com/Hongjiang-Yu). [[Paper]](https://www.sciencedirect.com/science/article/pii/S0167639320302831) [[DNN-Kalman]](https://github.com/Hongjiang-Yu/DNN_Kalman_Filter)

  <!--### NMF
  * Speech_Enhancement_DNN_NMF 
    [[Code]](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
  * gcc-nmf:Real-time GCC-NMF Blind Speech Separation and Enhancement 
    [[Code]](https://github.com/seanwood/gcc-nmf)-->

  ### Multi-stage
  * 2020, A Recursive Network with Dynamic Attention for Monaural Speech Enhancement, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2003.12973) [[DARCN]](https://github.com/Andong-Li-speech/DARCN)
  * 2020, Masking and Inpainting: A Two-Stage Speech Enhancement Approach for Low SNR and Non-Stationary Noise, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://ieeexplore.ieee.org/document/9053188/)
  * 2020, A Joint Framework of Denoising Autoencoder and Generative Vocoder for Monaural Speech Enhancement, Du. [[Paper]](https://ieeexplore.ieee.org/document/9082858)
  * 2020, Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression, [Westhausen](https://github.com/breizhn). [[Paper]](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf) [[DTLN]](https://github.com/breizhn/DTLN)
  * 2020, Listening to Sounds of Silence for Speech Denoising, [Xu](https://github.com/henryxrl). [[Paper]](http://www.cs.columbia.edu/cg/listen_to_the_silence/paper.pdf) [[LSS]](https://github.com/henryxrl/Listening-to-Sound-of-Silence-for-Speech-Denoising)
  * 2021, ICASSP 2021 Deep Noise Suppression Challenge: Decoupling Magnitude and Phase Optimization with a Two-Stage Deep Network, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2102.04198)

  ### Data collection
  * [Kashyap](https://arxiv.org/pdf/2104.03838.pdf)([[Noise2Noise]](https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data))
  ### Loss 
  * [[Quality-Net]](https://github.com/JasonSWFu/Quality-Net)
  ### Challenge
  * DNS Challenge [[DNS Interspeech2020]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/) [[DNS ICASSP2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2021/) [[DNS Interspeech2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2021/)

  ### Other repositories
  * Collection of papers, datasets and tools on the topic of Speech Dereverberation and Speech Enhancement 
    [[Link]](https://github.com/jonashaag/speech-enhancement)
  * nanahou's awesome speech enhancement [[Link]](https://github.com/nanahou/Awesome-Speech-Enhancement)

## Dereverberation
### Traditional method
  * SPENDRED [[Paper]](https://ieeexplore.ieee.org/document/7795155)
    [[SPENDRED]](https://github.com/csd111/dereverberation)
  * WPE(MCLP) [[Paper]](https://ieeexplore.ieee.org/document/6255769)[[nara-WPE]](https://github.com/fgnt/nara_wpe)
  * LP Residual [[Paper]](https://ieeexplore.ieee.org/abstract/document/1621193) [[LP_residual]](https://github.com/shamim-hussain/speech_dereverbaration_using_lp_residual)
  * dereverberate [[Paper]](https://www.aes.org/e-lib/browse.cfm?elib=15675) [[Code]](https://github.com/matangover/dereverberate)
  * NMF [[Paper]](https://ieeexplore.ieee.org/document/7471656/) [[NMF]](https://github.com/deepakbaby/dereverberation-and-denoising)
### Hybrid method
  * DNN_WPE [[Paper]](https://ieeexplore.ieee.org/document/7471656/) [[Code]](https://github.com/nttcslab-sp/dnn_wpe)
### NN-based Derev
  * Dereverberation-toolkit-for-REVERB-challenge [[Code]](https://github.com/hshi-speech/Dereverberation-toolkit-for-REVERB-challenge)
  * SkipConvNet [[Paper]](https://arxiv.org/pdf/2007.09131.pdf) [[Code]](https://github.com/zehuachenImperial/SkipConvNet)

## Speech Separation (single channel)
* Tutorial speech separation, like awesome series [[Link]](https://github.com/gemengtju/Tutorial_Separation)
### NN-based separation
* 2015, Deep-Clustering:Discriminative embeddings for segmentation and separation, Hershey and Chen.[[Paper]](https://arxiv.org/abs/1508.04306)
[[Code]](https://github.com/JusperLee/Deep-Clustering-for-Speech-Separation)
[[Code]](https://github.com/simonsuthers/Speech-Separation)
[[Code]](https://github.com/funcwj/deep-clustering)
* 2016, DANet:Deep Attractor Network (DANet) for single-channel speech separation, Chen.[[Paper]](https://arxiv.org/abs/1611.08930)
[[Code]](https://github.com/naplab/DANet)
* 2017, Multitalker speech separation with utterance-level permutation invariant training of deep recurrent, Yu.[[Paper]](https://ai.tencent.com/ailab/media/publications/Multi-talker_Speech_Separation_with_Utterance-level.pdf)
[[Code]](https://github.com/funcwj/uPIT-for-speech-separation)
* 2018, LSTM_PIT_Speech_Separation 
[[Code]](https://github.com/pchao6/LSTM_PIT_Speech_Separation)
* 2018, Tasnet: time-domain audio separation network for real-time, single-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1711.00541v2)
[[Code]](https://github.com/mpariente/asteroid/blob/master/egs/whamr/TasNet)
* 2019, Conv-TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation, Luo.[(Paper)](https://arxiv.org/pdf/1809.07454.pdf)
[[Code]](https://github.com/kaituoxu/Conv-TasNet)
* 2019, Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1910.06379v1)
[[Code1]](https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation) 
[[Code2]](https://github.com/JusperLee/Dual-Path-RNN-Pytorch)
* 2019, TAC end-to-end microphone permutation and number invariant multi-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1910.14104) 
[[Code]](https://github.com/yluo42/TAC)
* sound separation(Google) [[Code]](https://github.com/google-research/sound-separation)
* sound separation: Deep learning based speech source separation using Pytorch [[Code]](https://github.com/AppleHolic/source_separation)
* music-source-separation 
[[Code]](https://github.com/andabi/music-source-separation)
* Singing-Voice-Separation 
[[Code]](https://github.com/Jeongseungwoo/Singing-Voice-Separation)
* Comparison-of-Blind-Source-Separation-techniques[[Code]](https://github.com/TUIlmenauAMS/Comparison-of-Blind-Source-Separation-techniques)
### BSS/ICA method
* FastICA[[Code]](https://github.com/ShubhamAgarwal1616/FastICA)
* A localisation- and precedence-based binaural separation algorithm[[Download]](http://iosr.uk/software/downloads/PrecSep_toolbox.zip)
* Convolutive Transfer Function Invariant SDR [[Code]](https://github.com/fgnt/ci_sdr)
* 
## Array Signal Processing
* MASP:Microphone Array Speech Processing [[Code]](https://github.com/ZitengWang/MASP)
* BeamformingSpeechEnhancer 
[[Code]](https://github.com/hkmogul/BeamformingSpeechEnhancer)
* TSENet [[Code]](https://github.com/felixfuyihui/felixfuyihui.github.io)
* steernet [[Code]](https://github.com/FrancoisGrondin/steernet)
* DNN_Localization_And_Separation 
[[Code]](https://github.com/shaharhoch/DNN_Localization_And_Separation)
* nn-gev:Neural network supported GEV beamformer CHiME3 [[Code]](https://github.com/fgnt/nn-gev)
* chime4-nn-mask:Implementation of NN based mask estimator in pytorch（reuse some programming from nn-gev）[[Code]](https://github.com/funcwj/chime4-nn-mask)
* beamformit_matlab:A MATLAB implementation of CHiME4 baseline Beamformit  [[Code]](https://github.com/gogyzzz/beamformit_matlab)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)
* beamformit:麦克风阵列算法 [[Code]](https://github.com/592595/beamformit)
* Beamforming-for-speech-enhancement [[Code]](https://github.com/AkojimaSLP/Beamforming-for-speech-enhancement)
* deepBeam [[Code]](https://github.com/auspicious3000/deepbeam)
* NN_MASK [[Code]](https://github.com/ZitengWang/nn_mask)
* Cone-of-Silence [[Code]](https://github.com/vivjay30/Cone-of-Silence)
-----------------------------------------------------------------------
* binauralLocalization 
[[Code]](https://github.com/nicolasobin/binauralLocalization)
* robotaudition_examples:Some Robot Audition simplified examples (sound source localization and separation), coded in Octave/Matlab [[Code]](https://github.com/balkce/robotaudition_examples)
* WSCM-MUSIC
[[Code]](https://github.com/xuchenglin28/WSCM-MUSIC)
* doa-tools
[[Code]](https://github.com/morriswmz/doa-tools)
*  Regression and Classification for Direction-of-Arrival Estimation with Convolutional Recurrent Neural Networks
[[Code]](https://github.com/RoyJames/doa-release) [[PDF]](https://arxiv.org/pdf/1904.08452v3.pdf)
* messl:Model-based EM Source Separation and Localization 
[[Code]](https://github.com/mim/messl)
* messlJsalt15:MESSL wrappers etc for JSALT 2015, including CHiME3 [[Code]](https://github.com/speechLabBcCuny/messlJsalt15)
* fast_sound_source_localization_using_TLSSC:Fast Sound Source Localization Using Two-Level Search Space Clustering
[[Code]](https://github.com/LeeTaewoo/fast_sound_source_localization_using_TLSSC)
* Binaural-Auditory-Localization-System 
[[Code]](https://github.com/r04942117/Binaural-Auditory-Localization-System)
* Binaural_Localization:ITD-based localization of sound sources in complex acoustic environments [[Code]](https://github.com/Hardcorehobel/Binaural_Localization)
* Dual_Channel_Beamformer_and_Postfilter [[Code]](https://github.com/XiaoxiangGao/Dual_Channel_Beamformer_and_Postfilter)
* 麦克风声源定位 [[Code]](https://github.com/xiaoli1368/Microphone-sound-source-localization)
* RTF-based-LCMV-GSC [[Code]](https://github.com/Tungluai/RTF-based-LCMV-GSC)
* DOA [[Code]](https://github.com/wangwei2009/DOA)


## Sound Event Detection
* sed_eval - Evaluation toolbox for Sound Event Detection 
[[Code]](https://github.com/TUT-ARG/sed_eval)
* Benchmark for sound event localization task of DCASE 2019 challenge 
[[Code]](https://github.com/sharathadavanne/seld-dcase2019)
* sed-crnn DCASE 2017 real-life sound event detection winning method. 
[[Code]](https://github.com/sharathadavanne/sed-crnn)
* seld-net 
[[Code]](https://github.com/sharathadavanne/seld-net)

## Tools
* APS:A workspace for single/multi-channel speech recognition & enhancement & separation.  [[Code]](https://github.com/funcwj/aps)
* AKtools:the open software toolbox for signal acquisition, processing, and inspection in acoustics [[SVN Code]](https://svn.ak.tu-berlin.de/svn/AKtools)(username: aktools; password: ak)
* espnet [[Code]](https://github.com/espnet/espnet)
* asteroid:The PyTorch-based audio source separation toolkit for researchers[[PDF]](https://arxiv.org/pdf/2005.04132.pdf)[[Code]](https://github.com/mpariente/asteroid)
* pytorch_complex [[Code]](https://github.com/kamo-naoyuki/pytorch_complex)
* ONSSEN: An Open-source Speech Separation and Enhancement Library 
[[Code]](https://github.com/speechLabBcCuny/onssen)
* separation_data_preparation[[Code]](https://github.com/YongyuG/separation_data_preparation)
* MatlabToolbox [[Code]](https://github.com/IoSR-Surrey/MatlabToolbox)
* athena-signal [[Code]](https://github.com/athena-team/athena-signal）
* python_speech_features [[Code]](https://github.com/jameslyons/python_speech_features)
* speechFeatures:语音处理，声源定位中的一些基本特征 [[Code]](https://github.com/SusannaWull/speechFeatures)
* sap-voicebox [[Code]](https://github.com/ImperialCollegeLondon/sap-voicebox)
* Calculate-SNR-SDR [[Code]](https://github.com/JusperLee/Calculate-SNR-SDR)
* RIR-Generator [[Code]](https://github.com/ehabets/RIR-Generator)
* Python library for Room Impulse Response (RIR) simulation with GPU acceleration [[Code]](https://github.com/DavidDiazGuerra/gpuRIR)
* ROOMSIM:binaural image source simulation [[Code]](https://github.com/Wenzhe-Liu/ROOMSIM)
* binaural-image-source-model [[Code]](https://github.com/iCorv/binaural-image-source-model)
* PESQ [[Code]](https://github.com/vBaiCai/python-pesq)
* SETK: Speech Enhancement Tools integrated with Kaldi 
[[Code]](https://github.com/funcwj/setk)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)

## Resources
* Speech Signal Processing Course(ZH) [[Link]](https://github.com/veenveenveen/SpeechSignalProcessingCourse)
* Speech Algorithms(ZH) [[Link]](https://github.com/Ryuk17/SpeechAlgorithms)
* CCF语音对话与听觉专业组语音对话与听觉前沿研讨会(ZH) [[Link]](https://www.bilibili.com/video/BV1MV411k7iJ)

