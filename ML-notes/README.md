# 1.Data Science(数据科学)、AI(人工智能)、Machine Learning(机器学习)、 Deep Learning(深度学习)

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

# 2. Regression

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


# 3. Error 

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

# 4. Gradient Descent

* Learning Rate: 太小会走的很慢，太大会很难找到最低点。需要自动的方法找到合适的learning rate， 开始的时候learning rate较大，快接近最低点时，要把rate变小。

* 可以转换为Adaptive rate用AdaGrad算法: 设置全局学习率之后，每次通过，全局学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同，起到的效果是在参数空间更为平缓的方向，会取得更大的进步（因为平缓，所以历史梯度平方和较小，对应学习下降的幅度较小），并且能够使得陡峭的方向变得平缓，从而加快训练速度。

* Stochastic Gradient Descent:  gd是update after seeing all examples，sgd是update for each example

* Adam Gradient Descent

* Feature Scaling: 做feature scaling可以增加效率，类似等高线图

#### Types of Gradient Descent Algorithms (BGD、SGD & MBGD)

* Various variants of gradient descent are defined on the basis of how we use the data to calculate derivative of cost function in gradient descent. Depending upon the amount of data used, the time complexity and accuracy of the algorithms differs with each other.

* Batch Gradient Descent (BGD): 批梯度下降每次更新使用了所有的训练数据，最小化损失函数，如果只有一个极小值，那么批梯度下降是考虑了训练集所有数据，是朝着最小值迭代运动的，但是缺点是如果样本值很大的话，更新速度会很慢。

* Stochastic Gradient Descent (SGD): 随机梯度下降在每次更新的时候，只考虑了一个样本点，这样会大大加快训练数据，也恰好是批梯度下降的缺点，但是有可能由于训练数据的噪声点较多，那么每一次利用噪声点进行更新的过程中，就不一定是朝着极小值方向更新，但是由于更新多轮，整体方向还是大致朝着极小值方向更新，又提高了速度。

* Mini-Batch Gradient Descent (MBGD): 小批量梯度下降法是为了解决批梯度下降法的训练速度慢，以及随机梯度下降法的准确性综合而来，但是这里注意，不同问题的batch是不一样的，听师兄跟我说，我们nlp的parser训练部分batch一般就设置为10000，那么为什么是10000呢，我觉得这就和每一个问题中神经网络需要设置多少层，没有一个人能够准确答出，只能通过实验结果来进行超参数的调整。

