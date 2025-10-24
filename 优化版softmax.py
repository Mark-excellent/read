"""
优化版Softmax分类器
实现了Mini-batch SGD、自适应学习率（Adam/Momentum）、早停法等优化技术
特性：
    - 支持批量梯度下降和Mini-batch SGD
    - 三种优化器：SGD、Momentum、Adam
    - 早停法防止过拟合
    - L2正则化
    - 数值稳定的Softmax实现
"""
import numpy as np


class OptimizedSoftmaxClassifier:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.01, 
                 batch_size=32, optimizer='sgd', early_stopping=True, patience=10):
        """
        初始化分类器
        
        参数:
            learning_rate: 初始学习率
            n_iterations: 最大迭代次数
            reg_lambda: L2正则化系数
            batch_size: Mini-batch大小 (None表示使用批量梯度下降)
            optimizer: 优化器类型 ('sgd', 'momentum', 'adam')
            early_stopping: 是否使用早停法
            patience: 早停耐心值
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss_history = []
        
        # Adam优化器参数
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None
        self.t = 0
        
        # Momentum优化器参数
        self.momentum = 0.9
        self.velocity_W = None
        self.velocity_b = None
        
    def _softmax(self, z):
        #计算Softmax函数
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss(self, X, y):
        #计算交叉熵损失函数
        n_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = self._softmax(scores)
        
        correct_log_probs = -np.log(probs[range(n_samples), y] + 1e-8)
        data_loss = np.sum(correct_log_probs) / n_samples
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.weights * self.weights)
        
        return data_loss + reg_loss
    
    def _compute_gradients(self, X, y):
        #计算梯度
        n_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = self._softmax(scores)
        
        dscores = probs.copy()
        dscores[range(n_samples), y] -= 1
        dscores /= n_samples
        
        dW = np.dot(X.T, dscores) + self.reg_lambda * self.weights
        db = np.sum(dscores, axis=0)
        
        return dW, db
    
    def _update_parameters_sgd(self, dW, db):
        #标准SGD参数更新
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
    
    def _update_parameters_momentum(self, dW, db):
        #带动量的SGD参数更新
        if self.velocity_W is None:
            self.velocity_W = np.zeros_like(self.weights)
            self.velocity_b = np.zeros_like(self.bias)
        
        self.velocity_W = self.momentum * self.velocity_W - self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db
        
        self.weights += self.velocity_W
        self.bias += self.velocity_b
    
    def _update_parameters_adam(self, dW, db):
        #Adam优化器参数更新
        if self.m_W is None:
            self.m_W = np.zeros_like(self.weights)
            self.v_W = np.zeros_like(self.weights)
            self.m_b = np.zeros_like(self.bias)
            self.v_b = np.zeros_like(self.bias)
        
        self.t += 1
        
        #一阶矩估计
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        #二阶矩估计
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        
        #偏差修正
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        #更新参数
        self.weights -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
    
    def _update_parameters(self, dW, db):
        #根据选择的优化器更新参数
        if self.optimizer == 'sgd':
            self._update_parameters_sgd(dW, db)
        elif self.optimizer == 'momentum':
            self._update_parameters_momentum(dW, db)
        elif self.optimizer == 'adam':
            self._update_parameters_adam(dW, db)
        else:
            raise ValueError(f"未知的优化器: {self.optimizer}")
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        训练Softmax分类器        
        参数:
            X: 训练特征矩阵
            y: 训练标签
            X_val: 验证集特征矩阵（用于早停）
            y_val: 验证集标签（用于早停）
            verbose: 是否打印训练信息
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        #初始化权重和偏置
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)
        
        #早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_bias = None
        
        #训练循环
        for iteration in range(self.n_iterations):
            #Mini-batch训练
            if self.batch_size is not None:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                n_batches = n_samples // self.batch_size
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = start_idx + self.batch_size
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    dW, db = self._compute_gradients(X_batch, y_batch)
                    self._update_parameters(dW, db)
            else:
                #批量梯度下降
                dW, db = self._compute_gradients(X, y)
                self._update_parameters(dW, db)
            
            #记录损失
            if iteration % 50 == 0:
                train_loss = self._compute_loss(X, y)
                self.loss_history.append(train_loss)
                
                if verbose:
                    print(f"迭代 {iteration}/{self.n_iterations}, 训练损失: {train_loss:.4f}", end="")
                
                #早停检查
                if self.early_stopping and X_val is not None and y_val is not None:
                    val_loss = self._compute_loss(X_val, y_val)
                    self.val_loss_history.append(val_loss)
                    
                    if verbose:
                        print(f", 验证损失: {val_loss:.4f}", end="")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self.weights.copy()
                        best_bias = self.bias.copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"\n早停触发于迭代 {iteration}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
                
                if verbose:
                    print()
        
        return self
    
    def predict_proba(self, X):
        #预测类别概率
        scores = np.dot(X, self.weights) + self.bias
        return self._softmax(scores)
    
    def predict(self, X):
        #预测类别
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        #计算准确率
        y_pred = self.predict(X)
        return np.mean(y_pred == y)



