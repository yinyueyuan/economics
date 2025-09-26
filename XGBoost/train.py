import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
# 读取 Excel 数据
file_path = r"C:\Users\ASUS\Desktop\shuju.xlsx"
df = pd.read_excel(file_path)
# 选择因变量（目标变量）
y = df['IneInvest']
# 选择特征变量（X）
features = ['Growth', 'Lev', 'CFO', 'Age', 'Asset', 'Return', 'ROA', 'TanRatio', 
            'Top1', 'Top10', 'Sep', 'SOE', 'TanAsset', 'ToAsset', 
            'TreatPost', 'Post', 'Treat', 'Post2', 'TreatPost2', 
            ]
X = df[features]
# 处理缺失值（如果有）
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 初始化 XGBoost 模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                             learning_rate=0.05, 
                             max_depth=5, 
                             subsample=0.8, 
                             n_estimators=200, 
                             reg_lambda=20, reg_alpha=10,
                             random_state=42)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测
y_pred = xgb_model.predict(X_test)
# 评估模型
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R2: {r2}")
# 变量重要性排名
xgb.plot_importance(xgb_model, max_num_features=10)
plt.show()
