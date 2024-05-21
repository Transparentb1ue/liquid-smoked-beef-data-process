import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 加载数据
file_path = r"C:\Users\20609\OneDrive\Desktop\毕业设计\数据分析\数据分析.xlsx"
data = pd.read_excel(file_path)

# 数据预处理
X = data.drop(columns=['实验名称', 'Cluster'])
y = data['Cluster']

# OneHotEncoder 对目标变量进行一热编码
one_hot_encoder = OneHotEncoder(sparse=False)
y_encoded = one_hot_encoder.fit_transform(y.values.reshape(-1, 1))

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_accuracies = []
test_accuracies = []
conf_matrices = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # 标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 支持向量机分类器
    svm = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))

    # 训练模型
    svm.fit(X_train_scaled, y_train)

    # 预测
    y_train_pred = svm.predict(X_train_scaled)
    y_test_pred = svm.predict(X_test_scaled)

    # 将预测值转换为类别
    y_train_pred_class = np.argmax(y_train_pred, axis=1)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)

    # 将一热编码的实际值转换为类别
    y_train_class = np.argmax(y_train, axis=1)
    y_test_class = np.argmax(y_test, axis=1)

    # 计算训练集和验证集的预测精度
    train_accuracy = accuracy_score(y_train_class, y_train_pred_class)
    test_accuracy = accuracy_score(y_test_class, y_test_pred_class)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 记录每次的混淆矩阵
    conf_matrix = confusion_matrix(y_test_class, y_test_pred_class)
    conf_matrices.append(conf_matrix)

# 计算平均精度和标准差
avg_train_accuracy = np.mean(train_accuracies)
avg_test_accuracy = np.mean(test_accuracies)
std_train_accuracy = np.std(train_accuracies)
std_test_accuracy = np.std(test_accuracies)

print(f'Average Training Accuracy: {avg_train_accuracy:.4f} ± {std_train_accuracy:.4f}')
print(f'Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}')

# 打印最后一次的混淆矩阵作为示例
print('Confusion Matrix (Last Fold Test Set):')
print(conf_matrices[-1])
