## Learning Tips

### 1.Data Science & AI & Machine Learning & Deep Learning

* #### Artificial Intelligence

  Artificial Intelligence(人工智能)是以上眾多詞之中，被誤解最深的一個。由於近年 Data Science備受歡迎，AI也時時被人提起，權當作Data Science應用上的代名詞。 然而，AI的研究早於電腦初成形之時已經開始，在眾多不同方法探討之中，大量運用Data Science的 Machine Learning 只是其中一門人工智能的方法，其他方法例如 Logic-based Artificial Intelligence(基於邏輯的人工智能)，Fuzzy logic(模糊邏輯)等等其實亦是人工智能範圍。只是由於近年Machine Learning 得到長足進步，令人誤解Artificial Intelligence 等於 Machine Learning 的錯覺。

* #### Data Science

  而Data Science(數據科學)本身是一個跨學科的學問，泛指所有關於收集、處理、展示、分析數據的知識，顧名思義就是有關數據的知識。從事Data Science的人需要同時有數學、統計、電腦科學的知識，亦要理解起碼的編程技術。 數據科學涵蓋的範圍很大，包括 Data Visualisation(數據視覺化)、Data Engineering(數據工程)、Machine Learning(機器學習)、Data Warehouse(數據倉庫)等都是數據科學的領域，今時今日公司通常聘請 Data Scientist(數據科學家)、Data Analyst(數據分析師)、Business Intelligence Analyst(商業智能分析師)等，亦都屬於數據科學的領域。或者大家會好奇，為何數據科學此門學科，多年未曾聽聞，要直至近年才突然變得如 此受歡迎呢？ 這其實與Big Data(大數據)的普及不無關係：基礎的數據分析其實大家都曾經做過，用Microsoft Excel也綽綽有餘。但在大數據時代中，收集得來數據的Volume(資料量）、 Variety(變化)、Velocity(速度)都大大提高，使到Data Science一門學科於近年變得如此炙手可熱。 常用的library 包括 : Numpy , Pandas、 Seaborn

* #### Machine Learning

  Machine Learning(機器學習)是數據科學的一門，指的則是向一個模型輸入一大堆訓練數據，利用數據重覆訓練，遞歸出一個誤差低、準確度高的模型，再用測試數據重加測試。為何會 用訓練數據，而不是以輸入大量規則去令程式能夠判別問題呢？使用機器學習的情況通常是因為要解決的問題不容易以 Rule-based Knowledge(規則基礎知識) 解決。 邏輯問題、數學問題等等問題牽涉的是大量的規則，電腦無須使用機器學習亦迎刃有餘。 但例如人臉辨識、語音辨識一類關乎知覺(Perception)的問題，不論寫下多少條規則，都難以清晰表達何謂一張人臉。最直接的方法正如訓練嬰兒一樣，通過大量觀看、大量嘗試，從而理解何謂一張人臉。 常用的library 包括 Scikit-learn、Open CV

* #### Deep Learning
  Deep Learning(深度學習)又是機器學習的一門，特指使用neural network(神經網絡)作分析的機器學習方法。神經網絡本身是為嘗試模仿人腦神經元結構而建立。 因為如果人腦能夠輕易處理知覺一類的問題，而電腦卻未能遲遲未能達至相同的準確度，必然代表人腦的結構在解決此一類問題上，有優勝於電腦的地方。
  常見的Deep Learning應用範圍有： Speech Recognition(語音辨識)、Image Recognition(圖像辨識)、Language Understanding(語言理解)、Language Generation(語言生成)等等。常用的library 包括 Tensorflow、Pytorch、Keras.
  Alpha Go的理論是基於數據科學中的機器學習中的深度學習中的强化學習一科

### 2. Regression

* 找一个model （function set）：

   linear model: y=b+ wixi I  (x:feature, w:weight, b:bias)

* Goodness of Function 

  Loss Function: 损失函数  L(f)=L(w,b) 
  
  衡量参数的好坏：how bad it is  
  
  f* = arg min L(f), 找到是的loss function最小的w & b

* 找最小的loss方法： Gradient Descent

  随机选一个点求导：如果斜率为负，表示左高右低，需要增加w
  如果斜率为正，表示左低右高，需要减少w
  step size： 需要看微分值，常数项（learning rate：事先定好的数值v）

* Linear regression 是convex的函数，没有local optimal，只有global optimal

* 如果gradient descent符合，那么a more complex model yields lower error 【training data】，但在testing data上面是不符合的，这叫做overfitting 

* 如何处理ovefitting：用regularization来找有效的feature


### 3. Error 

* Error Source:  Bias & Variance
  Bias: E[f*]=f bar: 简单的model 会有小的variance，复杂的model会有较大的variance [simpler model is less influenced bby the sampled data ]
  Variance: depend on the number of samples
  
* Overfitting: error来源于bias很大
  Underfitting: error 来源于variance很大

* 常问问题：你觉得你的model是bias大还是variance大，才可以知道怎么改进model？

  如果model无法fit你的training examples，表示bia大 【underfitting】，需要redesign model，function不够好，考虑更多有效的features；
  如果model 在trainning data上的到小的error，但在testing data上有大的error，表示variance大【overfitting】， 需要增加data（真实或数据转换都可以：比如翻转，调角度等），或者regularization （正则化，overfitting是由high variance导致，high variance 是由特征太多，特征值过度敏感导致，regularizer能够减少特征数量和降低特征值敏感度，所以说是个好方法。）

* Model Selection
  trade-off between bias & variance
  Testing Model的：Cross Validation; N-fold Cross Validation 

### 4. Gradient Descent

* Learning Rate: 太小会走的很慢，太大会很难找到最低点。需要自动的方法找到合适的learning rate， 开始的时候learning rate较大，快接近最低点时，要把rate变小。

* 可以转换为Adaptive rate用AdaGrad算法: 设置全局学习率之后，每次通过，全局学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同，起到的效果是在参数空间更为平缓的方向，会取得更大的进步（因为平缓，所以历史梯度平方和较小，对应学习下降的幅度较小），并且能够使得陡峭的方向变得平缓，从而加快训练速度。

* Stochastic Gradient Descent:  gd是update after seeing all examples，sgd是update for each example

* Adam Gradient Descent

* Feature Scaling: 做feature scaling可以增加效率，类似等高线图

### Types of Gradient Descent Algorithms (BGD、SGD & MBGD)

Various variants of gradient descent are defined on the basis of how we use the data to calculate derivative of cost function in gradient descent. Depending upon the amount of data used, the time complexity and accuracy of the algorithms differs with each other.

* Batch Gradient Descent (BGD): 批梯度下降每次更新使用了所有的训练数据，最小化损失函数，如果只有一个极小值，那么批梯度下降是考虑了训练集所有数据，是朝着最小值迭代运动的，但是缺点是如果样本值很大的话，更新速度会很慢。

* Stochastic Gradient Descent (SGD): 随机梯度下降在每次更新的时候，只考虑了一个样本点，这样会大大加快训练数据，也恰好是批梯度下降的缺点，但是有可能由于训练数据的噪声点较多，那么每一次利用噪声点进行更新的过程中，就不一定是朝着极小值方向更新，但是由于更新多轮，整体方向还是大致朝着极小值方向更新，又提高了速度。

* Mini-Batch Gradient Descent (MBGD): 小批量梯度下降法是为了解决批梯度下降法的训练速度慢，以及随机梯度下降法的准确性综合而来，但是这里注意，不同问题的batch是不一样的，听师兄跟我说，我们nlp的parser训练部分batch一般就设置为10000，那么为什么是10000呢，我觉得这就和每一个问题中神经网络需要设置多少层，没有一个人能够准确答出，只能通过实验结果来进行超参数的调整。

### 5. Classification

同理，我们用极大似然估计法Maximum Likelihood在高斯函数上的公式计算出class 2的两个参数，得到的最终结果如下：

3个steps： 
* function set: 条件概率模型 Probability Distribution
* goodness of a function:  mean & convariance 
* find the best function: easy

Gaussian process: 什么是高斯过程？简单的说，就是一系列关于连续域（时间或空间）的随机变量的联合，而且针对每一个时间或是空间点上的随机变量都是服从高斯分布的。

### 6. Logistic Regression

* Step1: function set, if p>=0.5 output c1; else output c2, posterior probability，用sigmoid函数，结果介于0-1

#### Discriminative 判定模型 & Generative 生成模型：

监督学习方法又可分为生成方法和判别方法。所学到的模型分别称为生成模型和判别模型。      生成方法由数据学习联合概率分布，然后求出条件概率分布作为预测模型，即生成模型：这样的方法称为生成方法，是因为模型表示了给定输入X产生输出Y的生成关系。典型的生成模型有：朴素贝叶斯法、隐马尔科夫模型、混合高斯模型、AODE、Latent Dirichlet allocation（unsup）、Restricted Boltzmann Machine。

判别方法由数据直接学习决策函数[公式]或者条件概率分布[公式]作为预测的模型，即判别模型。判别方法关心的是对给定的输入X，应该预测什么样的输出Y。典型的判别方法包括，kNN，感知机，决策树，逻辑回归，最大熵模型，SVM，提升方法，条件随机场，神经网络等。


#### 决策函数和条件概率分布

决策函数Y=f(X)
决策函数Y=f(X)：你输入一个X，它就输出一个Y，这个Y与一个阈值比较，根据比较结果判定X属于哪个类别。例如两类（w1和w2）分类问题，如果Y大于阈值，X就属于类w1，如果小于阈值就属于类w2。这样就得到了该X对应的类别了。

条件概率分布P(Y|X)
你输入一个X，它通过比较它属于所有类的概率，然后输出概率最大的那个作为该X对应的类别。例如：如果P(w1|X)大于P(w2|X)，那么我们就认为X是属于w1类的。

生成模型是模拟这个结果是如何产生的,然后算出产生各个结果的概率，预先做了假设，假设你的data来自于某个几率模型。

判别模型是发现各个结果之间的不同,不关心产生结果的过程

* 优缺点

  generative model：有假设，受data影响小，对噪音可以忽视，
  discriminative model：没有任何假设，所以受data的影响大，function拆成2部分，可以来自不同的来源。

* Limitation of Logistic Regression: 4个点，对角线为一类，无法用regression划线

* 解决方式：1）Feature Transformation：不希望人工作，机器去找，用cascading logistic regression model

### 7. Deep Learning

Function Set: Neural Network, weight & bias, 有不同的连接方式，fully connect feedfoward network; Output layer: feature extractor replacing feature engineering
Goodness of function:  total loss, 用梯度下降，只是function复杂了而已

？ How many layers? how many neurons for each layer?
  
   trial & error + instuition

   ml & deep learning ：dl没有使得ml更容易，而是转换了问题。 ml关键在于feature engineer，dl关键在于design network structure。看哪个更容易，比如语音/影像，可能design network structure比feature engineer容易。nlp中dl的应用没有别的领域那么强可能是因为人类对于语言的理解更强。
   
？能否机器自动决定structure：可以，Evolutionary Artificaila Neural Networks

？能否自己去找structure：可以，cnn

？ Deeper is better? : more parameters, better performance, 
 

### 8. Backpropagation (反向传播)

目的：和梯度下降没有不同，只是由于dp需要neuron多，参数很多，bp是一个有效率的办法把这个计算出来

算法： 利用chain rule链式法则原理分别计算2个微分。首先计算forwards pass，计算a， 再乘以通过反向传播计算出来的微分。

Forward pass：看前面input的什么，微分就是什么。如果是中间，多加一个weight

Backward pass： 假设2个不同的case： 
  1）如果是整个network的output layer，就可以直接计算出来。 
  2）如果是中间层，后面还有计算，就需要一直计算到output层，但其实只要反向过来，就和forward pass类似，变成简单的计算。


### 9.Deep Learning Keras

梯度下降: 梯度下降就是我上面的推导，在梯度下降中，对于 [公式] 的更新，需要计算所有的样本然后求平均.其计算得到的是一个标准梯度。因而理论上来说一次更新的幅度是比较大的。

随机梯度下降: Stochastic Gradient Descent：可以看到多了随机两个字，随机也就是说我每次用样本中的一个例子来近似我所有的样本，用这一个例子来计算梯度并用这个梯度来更新[公式]。因为每次只用了一个样本因而容易陷入到局部最优解中

批量随机梯度下降 Mini-batch SGD：他用了一些小样本来近似全部的，其本质就是竟然1个样本的近似不一定准，那就用更大的30个或50个样本来近似。将样本分成m个mini-batch，每个mini-batch包含n个样本；在每个mini-batch里计算每个样本的梯度，然后在这个mini-batch里求和取平均作为最终的梯度来更新参数；然后再用下一个mini-batch来计算梯度，如此循环下去直到m个mini-batch操作完就称为一个epoch结束。

* Step1: Define a set of function
* Step2: Goodness of function
* Step3: Pick the best functioin
* Step4：检查在training data上是否有good results，如果没有，需要调整。比起knn之类的方法（在training data上的准确度就是100），dp不容易overfitting，还有可能在training data上正确率就很低。
* Step5: 如果在traning data上有一个较好的performance，下一步就是看在testing data上是否有好的结果，如果没有在testing上效果不好的的话，就是overfitting

?? 不要看到performance不好就认为是overfitting

  在deep learning 训练模型的时候，基本是2个问题：1).在traning data上的performance不好. 2).在testing data上的performance不好


* Early Stopping
  当training data的total loss减少，可能testing set loss在上升，需要找到一个stop点，需要用validation set来确认

* Regularization
  L2, regularization 时候不考虑bias，重要性没有那么高，weight decay，svm里面是直接放进去的。

* Dropout
  update参数之前，做sampling，每个neuron有几率被丢掉，会变thinner
  dropout之后会perforamnce会变差，是让training变差，但testing编号
  dropout rate * 0.5

? 为什么有用？

是一种ensemble方法：variance  & bias的问题。


* New Activation Function:

  Activation Function: 如果是sigmoid function，在input参数变化，如果layer越多，对output的影响越小。

  常用的activation function：Rectified Linear Unit (ReLu函数)，一个是运算更快，二是类似无穷多的sigmoid function 叠加的结果，三是可以handle Vanishing的问题。

  ReLu: Leaky ReLu; Parametric ReLu; Maxout Network (自动学activation function)，其中RuLu是Maxout的一个特殊case

* Vanishing Gradient Problem: 

什么是梯度不稳定问题：深度神经网络中的梯度不稳定性，前面层中的梯度或会消失，或会爆炸。

原因：前面层上的梯度是来自于后面层上梯度的乘乘积。当存在过多的层次时，就出现了内在本质上的不稳定场景，如梯度消失和梯度爆炸。

梯度消失与梯度爆炸其实是一种情况，看接下来的文章就知道了。两种情况下梯度消失经常出现，一是在深层网络中，二是采用了不合适的损失函数，比如sigmoid。梯度爆炸一般出现在深层网络和权值初始化值太大的情况下，下面分别从这两个角度分析梯度消失和爆炸的原因。

* Adaptive Learning Rate

  Adagrad: 每个parameter都有不同的learning rate，除以root，但可能还是不够

  RMSProp: 更进阶的方法，平方和

  Local Optimal 的问题

  network越大，local minium 概率会越小

  在Grdient Decent上加上惯性 Momentum

  添加后就是惯性（原来的方向） + gradient方向的中间 或者 可以理解为过去所有gd的总总和。只是weight不同。

  Adam: RMSProp + Momentum


### 10. Convolutional Neural Network (CNN)
  多用于影像处理，简化fully connected 的架构，拿掉一些参数。
  ? 为什么可以拿掉参数？
 * 一些pattern只需要看一小部分就可以，不需要看全图
 * 同样的pattern，可能出现在image不同的部分
 * subsampling对识别来说影响不大

  和dnn区别：需要把input format 从vector变成3-d tensor



Convolution (卷积)：

* 有很多filter，是一个matrix，与image做内积，需要设置stride决定每次移动的距离。 

* 每个filter都负责侦测一个不同的pattern，最后所有的filter都得到一个matrix，所有的合起来就是feature map 

* Convolution这件事情就是fully connected layer拿掉一些weight

Max pooling （池化）:

* 在四个值中选择最大的保留

* 结果：
  image 6*6 →  convolution 4*4  → max pooling 2*2 ，深度由filter决定，50个filter就有50维

  (image - filter + 1) = 6 - 3 + 1= 4

  eg: 28-3+1=26 ---- 26/2=13 ---13-3+1=11 --- 11/2=5

  eg: parameter: 3*3=9 --- 3*3*25=225


? deep learning是黑盒子？

  deep dream: 强化看到的东西
  deep style： deep dream进阶的部分，给一个照片，让machine 去修改起来看到的照片，考虑correaltion 得到相关的style。

? 什么时候用cnn？为什么alpha go用到cnn，因为有和影像相似的地方
  一些pattern只需要看一小部分就可以，不需要看全图，alpha go用到的filter是5*5
  同样的pattern，可能出现在image不同的部分，代表同样的意义。
  subsampling： 对围棋可以做max pooling吗？ 19*19*48，每个位置有48个value，第一个layer， alpha 没有用到max pooling
