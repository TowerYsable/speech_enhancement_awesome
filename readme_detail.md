## 一、 基础方法

### 1.1 维纳滤波

![image-20210924224901160](readme_detail.assets/image-20210924224901160.png)

- 通过最小均方误差计算得到：(推导详情见《1.语音增强维纳滤波》)
  $$
  H(w_k)= \frac {P^*_{yx}(w_k)}{P_{yy}(w_k)}
  $$

- 物理意义
  - 推导得$H(w_k) = \frac {\xi_k}{\xi_k +1}$，其中$\xi=\frac{P_{xx}w_k}{P_{nn}(w_k)}$为先验信噪比
  - 当信噪比大的时候，允许信号通过；当信噪比小的时候，抑制信号通过。（$\alpha_k$为先验信噪比）
- 变种
  - 平方根维纳滤波：保证增强后的能量谱与干净语音的能量谱相同
  
  - 参数型维纳滤波：根据噪声大小的不同选择不同的参数
    $$
    H(w_k)=(\frac{\xi_k}{\xi_k+\alpha})^\beta
    $$
    
  
- 代码：代码见《1.wiener》

### 1.2 子空间法

![image-20210925101656126](readme_detail.assets/image-20210925101656126.png)

- 本质：本质事寻找一个H能够将含噪信号y映射成干净的信号x(推导见《2.语音增强子空间法》)
- 参考论文：Hu, Y. and Loizou, P. (2003). A generalized subspace approach for enhancing speech corrupted by colored noise. IEEE Trans. on Speech and Audio Processing, 11, 334-341  
- 代码：《2.subspace》

### 1.3 谱减法

- 推导见《3.specsub推导》
- 具体步骤：
  - 对原始信号进行分帧、加窗处理，求每帧的FFT
  - 求噪声的FFT，并求均值
  - 进行谱减
    - 其中谱减部分具体有三种方法：
      （1）利用幅度谱进行谱减。 （推导中④式）
      （2）利用功率谱进行谱减。 （推导中⑤式，alpha取2）
      （3）alpha-beta法进行改进，可以减少音乐噪声（推导中⑤式）
  - 复原出原始声音信号
- 变种
  - 过减法（乘以系数加权）
  - 引入平滑机制（对于过小的值用相邻帧的最小值取代）
- 缺点：
  - 会产生 音乐噪声==噪声残留

### 1.4 MMSEE（计算和推导复杂）

![image-20210925105309373](readme_detail.assets/image-20210925105309373.png)

- 最小均方误差估计（Minimum Mean Square Error Estimation）  

![image-20210925105634721](readme_detail.assets/image-20210925105634721.png)

### 1.5 DNN频谱映射

<img src="readme_detail.assets/image-20210925111312301.png" alt="image-20210925111312301" style="zoom:50%;" />

### 1.6 DNN-IRM

![image-20210925122320974](readme_detail.assets/image-20210925122320974.png)

### 1.7 GAN



## 二、 complex domain

## 三、 attention

## 四、 loss

