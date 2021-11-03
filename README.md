import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


X,y=datasets.make_moons(n_samples=1000,noise=0.2,random_state=100)#准备数据
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

'''def make_plot(X,y,plot_name):
    

    :param X: array（number，2）
    :param y: array（number，）
    :param plot_name: 图名
    :return: 散点图
    
    plt.figure(figsize=(12,8))
    plt.title(plot_name,fontsize=30)
    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    plt.show()
make_plot(X,y,'Classification Dataset Visualization')'''

class Layer:
    def __init__(self,n_input,n_output,activation=None,weights=None,bias=None):
        '''
        全连接网络层,实现一个网络层
        :param n_input: 输入节点数
        :param n_output: 输出节点数
        :param activation: 激活函数类型
        :param weights: 权值张量，默认类内生成
        :param bias: 偏置，默认类内生成
        '''
        self.weights=weights if weights is not None else np.random.randn(n_input,n_output)*np.sqrt(1/n_output)
        self.bias=bias if bias is not None else np.random.randn(n_output)*0.1
        self.activation=activation   #激活函数类型
        self.activation_output=None  #激活函数的输出值 o
        self.error=None  #用于计算当前层的delta变量的中间变量
        self.delta=None  #用于记录当前层的delta变量，用于计算梯度

    def activate(self,X):
        '''
        该函数调用后可以得到激活函数的输出值
        :param X: 网络没有与权重计算的输入值
        :return: 经过激活函数的计算得到输出值
        '''
        #z是激活函数的输入
        z=np.dot(X,self.weights)+self.bias
        #通过激活函数，得到全连接层的输出 o
        self.activation_output=self._apply_activation(z)
        return self.activation_output

    def _apply_activation(self,z):
        '''
        激活函数的选择以及适用过程
        :param z: 激活函数的输入
        :return: 激活函数的输出
        '''
        if self.activation is None:
            return z
        elif self.activation =='relu':
            return np.maximum(z,0)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation=='sigmoid':
            return 1/(1+np.exp(-z))
    def apply_activation_derivative(self,z):
        '''
        :param z: 激活函数的输入
        :return: 激活函数的导数值
        '''
        if self.activation is None:
            return np.ones_like(z)
        elif self.activation == "relu":
            grad = np.array(z,copy=True)
            grad[z>0]=1.   #这竟然不用遍历数组就可以实现赋值
            grad[z<=0]=0.
            return grad
        elif self.activation=='tanh':
            return 1-z**2
        elif self.activation=='sigmoid':
            return z*(1-z)

class NeuralNetwork:
    def __init__(self):
        self._layers=[]   #这里定义的是一个数列，因此无所谓向里面添加的是什么数据类型

    def add_layers(self,layer):
        self._layers.append(layer)   #在这里增加某一层的时候因为hi使用类的概念所以要Layer（等等等）
    def feed_forward(self,X):
        #向后传播
        '''
        :param X: 网络的输入值
        :return: 得到的是神经网络最后一层的输出值
        '''
        for layer in self._layers:
            X=layer.activate(X)
        return X

    def backpropagation(self,X,y,learning_rate):
        #向前计算
        output=self.feed_forward(X)   #网络的输出值output
        for i in reversed(range(len(self._layers))):#反向循环  #reversed输出一个反转的迭代器，也就是将后面的列表反转
            layer=self._layers[i]
            if layer==self._layers[-1]:
                layer.error=y-output
                #计算最后一层的delata，参考输出层的梯度公式
                layer.delta=layer.error*layer.apply_activation_derivative(output)
            else:
                next_layer=self._layers[i+1]
                layer.error=np.dot(next_layer.weights,next_layer.delta)
                layer.delta=layer.error*layer.apply_activation_derivative(layer.activation_output)
        #循环更新权值
        for i in range(len(self._layers)):
            layer=self._layers[i]
            # o_i为上一网络层的输出
            o_i=np.atleast_2d(X if i==0 else self._layers[i-1].activation_output)
            layer.weights+=layer.delta*o_i.T*learning_rate

    def train(self,X_train,X_test,y_train,y_test,learning_rate,max_epochs):
        #网络训练函数
        #one-hot编码
        y_onehot=np.zeros((y_train.shape[0],2))
        y_onehot[np.arange(y_train.shape[0]),y_train]=1  #这一步还是非常好的
        mses=[]
        for i in range(max_epochs):#训练100个周期
            for j in range(len(X_train)): #一次训练一个样本
                self.backpropagation(X_train[j],y_onehot[j],learning_rate)
                if i%10==0:
                    mse=np.mean(np.square(y_onehot-self.feed_forward(X_train)))
                    mses.append(mse)
                    print('Epoch: #%s, MSE: %f, Accuracy: %.2f%%' %
                          (i, float(mse), self.accuracy(self.predict(X_test), y_test.flatten()) * 100))
        return mses

    def accuracy(self,y_predict,y_test):
        return np.sum(y_predict==y_test)/len(y_test)

    def predict(self,X_predict):
        y_predict=self.feed_forward(X_predict)
        y_predict=np.argmax(y_predict,axis=1)
        return y_predict

#网络训练
nn=NeuralNetwork()
nn.add_layers(Layer(2,25,'sigmoid'))
nn.add_layers(Layer(25,50,'sigmoid'))
nn.add_layers(Layer(50,25,'sigmoid'))
nn.add_layers(Layer(25,2,'sigmoid'))
nn.train(X_train,X_test,y_train,y_test,0.01,100)

