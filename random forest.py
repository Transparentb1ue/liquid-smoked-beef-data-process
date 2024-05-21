import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 设置matplotlib支持中文显示
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 确保你的系统中安装了此字体

# 加载数据
data = pd.read_excel(r'C:\Users\20609\OneDrive\Desktop\data.xlsx')  # 请替换为你的文件实际路径

# 准备数据集
X = data[['液熏浓度', '液熏时间', '液熏温度', '烧烤时间', '烧烤温度']]
y = data['Cluster']

# 划分训练集和测试集，测试集比例调整为10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 使用网格搜索找到的最佳参数
best_params = {
    'max_depth': 10,
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'n_estimators': 100
}

# 初始化带有最佳参数的随机森林模型
rf_model_optimized = RandomForestClassifier(**best_params, random_state=42)
rf_model_optimized.fit(X_train, y_train)

# 预测测试集
y_pred = rf_model_optimized.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_text)

# 特征重要性
feature_importances = rf_model_optimized.feature_importances_
features = X.columns

# 创建特征重要性图表
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
