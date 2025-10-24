import numpy as np

class SoftmaxClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.01):
        """
        初始化Softmax分类器
        
        参数:
        learning_rate: 学习率
        n_iterations: 迭代次数
        reg_lambda: L2正则化系数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        
        # 权重和偏置将在fit时初始化
        self.weights = None
        self.bias = None
        
        # 记录训练历史
        self.loss_history = []
    
    def _softmax(self, z):
        """
        计算Softmax函数
        
        参数:
            z: 输入数组，形状为 (n_samples, n_classes)
        
        返回:
            softmax概率，形状为 (n_samples, n_classes)
        """
        # 为了数值稳定性，减去每行的最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss(self, X, y):
        """
        计算交叉熵损失函数（包含L2正则化）
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 标签，形状为 (n_samples,)
        
        返回:
            损失值
        """
        n_samples = X.shape[0]
        
        # 计算预测分数
        scores = np.dot(X, self.weights) + self.bias
        
        # 计算softmax概率
        probs = self._softmax(scores)
        
        # 计算交叉熵损失
        correct_log_probs = -np.log(probs[range(n_samples), y] + 1e-8)
        data_loss = np.sum(correct_log_probs) / n_samples
        
        # 添加L2正则化
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.weights * self.weights)
        
        total_loss = data_loss + reg_loss
        
        return total_loss
    
    def _compute_gradients(self, X, y):
        """
        计算梯度
        
        参数:
            X: 特征矩阵
            y: 标签
        
        返回:
            权重梯度和偏置梯度
        """
        n_samples = X.shape[0]
        
        # 前向传播
        scores = np.dot(X, self.weights) + self.bias
        probs = self._softmax(scores)
        
        # 计算梯度
        dscores = probs.copy()
        dscores[range(n_samples), y] -= 1
        dscores /= n_samples
        
        # 反向传播
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0)
        
        # 添加正则化梯度
        dW += self.reg_lambda * self.weights
        
        return dW, db
    
    def fit(self, X, y):
        """
        使用批量梯度下降训练Softmax分类器
        
        参数:
            X: 训练特征矩阵，形状为 (n_samples, n_features)
            y: 训练标签，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 初始化权重和偏置
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)
        
        # 批量梯度下降
        for i in range(self.n_iterations):
            # 计算梯度
            dW, db = self._compute_gradients(X, y)
            
            # 更新参数
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # 记录损失
            if i % 100 == 0:
                loss = self._compute_loss(X, y)
                self.loss_history.append(loss)
                print(f"迭代 {i}/{self.n_iterations}, 损失: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
            X: 特征矩阵
        
        返回:
            概率矩阵，形状为 (n_samples, n_classes)
        """
        scores = np.dot(X, self.weights) + self.bias
        return self._softmax(scores)
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 特征矩阵
        
        返回:
            预测标签
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        """
        计算准确率
        
        参数:
            X: 特征矩阵
            y: 真实标签
        
        返回:
            准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=5, random_state=42):
    """
    手动生成合成数据集（不使用sklearn）
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        random_state: 随机种子
    
    返回:
        X: 特征矩阵
        y: 标签
    """
    np.random.seed(random_state)
    
    X_list = []
    y_list = []
    
    samples_per_class = n_samples // n_classes
    
    for class_id in range(n_classes):
        # 为每个类别生成不同的中心点
        center = np.random.randn(n_features) * 3
        # 生成该类别的样本
        X_class = np.random.randn(samples_per_class, n_features) + center
        y_class = np.full(samples_per_class, class_id)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    手动实现训练集和测试集划分（不使用sklearn）
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    split_idx = int((1 - test_size) * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def main():
    """
    示例：使用Softmax分类器进行多分类
    """
    print("=" * 60)
    print("Softmax分类器示例（纯Numpy实现）")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 生成模拟数据（5个类别）
    X, y = generate_synthetic_data(n_samples=1000, n_features=20, n_classes=5)
    
    print(f"\n数据集信息:")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建并训练Softmax分类器
    print(f"\n{'=' * 60}")
    print("开始训练Softmax分类器...")
    print(f"{'=' * 60}\n")
    
    classifier = SoftmaxClassifier(
        learning_rate=0.5,
        n_iterations=1000,
        reg_lambda=0.01
    )
    
    classifier.fit(X_train, y_train)
    
    # 评估模型
    print(f"\n{'=' * 60}")
    print("模型评估")
    print(f"{'=' * 60}")
    
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 显示一些预测示例
    print(f"\n{'=' * 60}")
    print("预测示例（前5个测试样本）")
    print(f"{'=' * 60}")
    
    y_pred = classifier.predict(X_test[:5])
    y_proba = classifier.predict_proba(X_test[:5])
    
    for i in range(5):
        print(f"\n样本 {i+1}:")
        print(f"  真实标签: {y_test[i]}")
        print(f"  预测标签: {y_pred[i]}")
        print(f"  预测概率: {y_proba[i]}")
    
    print(f"\n{'=' * 60}")
    print("完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()