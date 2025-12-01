import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib  
import eli5  
from eli5.sklearn import PermutationImportance

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_explore_data(file_path):
    print("=" * 50)
    print("=" * 50)
    df = pd.read_excel(file_path)
    print(f"数据集形状: {df.shape}")
    print(f"数据列名: {list(df.columns)}")
    print("\n数据基本信息:")
    print(df.info())
    print("\n描述性统计:")
    print(df.describe().round(3))
    print(f"\n缺失值统计:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    return df

def preprocess_data(df):
    print("\n" + "=" * 50)
    print("步骤2: 数据预处理")
    print("=" * 50)
        y = df['IneInvest']
        features = ['Growth', 'Lev', 'CFO', 'Age', 'Asset', 'Return', 'ROA', 'TanRatio', 
                'Top1', 'Top10', 'Sep', 'SOE', 'TanAsset', 'ToAsset', 
                'TreatPost', 'Post', 'Treat', 'Post2', 'TreatPost2']
     X = df[features]
        print("缺失值处理策略:")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            missing_rate = X[col].isnull().sum() / len(X) * 100
            print(f"  {col}: {X[col].isnull().sum()}个缺失值 ({missing_rate:.2f}%)")
                        if X[col].dtype in ['float64', 'int64']:
                fill_value = X[col].median()  
                X[col].fillna(fill_value, inplace=True)
                print(f"    使用中位数填充: {fill_value:.4f}")
    if y.isnull().sum() > 0:
        y.fillna(y.median(), inplace=True)
        print(f"目标变量缺失值已用中位数填充")
        print("\n异常值检测（基于IQR方法）:")
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers}个异常值 ({outliers/len(X)*100:.2f}%)")
    return X, y, features

def prepare_train_test(X, y):
    """
    准备训练集和测试集
    """
    print("\n" + "=" * 50)
    print("步骤3: 数据分割")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost模型并进行超参数调优
    """
    print("\n" + "=" * 50)
    print("步骤4: 模型训练与调优")
    print("=" * 50)
        initial_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'n_estimators': 200,
        'reg_lambda': 20,  # L2正则化
        'reg_alpha': 10,   # L1正则化
        'random_state': 42,
        'n_jobs': -1,      # 使用所有CPU核心
        'early_stopping_rounds': 10
    }
    
    print("训练基础XGBoost模型...")
    base_model = xgb.XGBRegressor(**initial_params)
    base_model.fit(X_train, y_train)
    
    y_pred_base = base_model.predict(X_test)
    mae_base = mean_absolute_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    print(f"基础模型性能:")
    print(f"  MAE: {mae_base:.4f}")
    print(f"  R²: {r2_base:.4f}")
    
    print("\n开始超参数调优（网格搜索）...")
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'reg_lambda': [1, 10, 20],
        'reg_alpha': [0, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        cv=3,  # 3折交叉验证
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    # grid_search.fit(X_train, y_train)
    # print(f"最佳参数: {grid_search.best_params_}")
    # print(f"最佳分数: {-grid_search.best_score_:.4f}")
    
    # 使用最佳参数训练模型
    best_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'n_estimators': 200,
        'reg_lambda': 20,
        'reg_alpha': 10,
        'random_state': 42,
        'n_jobs': -1 }
    
    print("\n训练最终模型...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    return final_model

#  模型评估 
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    全面评估模型性能
    """
    print("\n" + "=" * 50)
    print("步骤5: 模型评估")
    print("=" * 50)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算多个评估指标
    metrics = {
        'MAE': mean_absolute_error,
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score,
        'Explained Variance': explained_variance_score
    }
    
    print("训练集性能:")
    for name, metric_func in metrics.items():
        score = metric_func(y_train, y_train_pred)
        print(f"  {name}: {score:.4f}")
    
    print("\n测试集性能:")
    test_scores = {}
    for name, metric_func in metrics.items():
        score = metric_func(y_test, y_test_pred)
        test_scores[name] = score
        print(f"  {name}: {score:.4f}")
    
    # 交叉验证评估
    print("\n5折交叉验证:")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    print(f"  CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 可视化预测 vs 实际值
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测 vs 实际值')
    
    plt.subplot(2, 2, 2)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')
    
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.title('残差分布')
    
    plt.subplot(2, 2, 4)
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
    plt.xlabel('特征重要性')
    plt.title('Top 10 重要特征')
    
    plt.tight_layout()
    plt.show()
    
    return test_scores

# 模型解释性分析 
def explain_model(model, X_train, X_test):
    """
    使用SHAP和Permutation Importance解释模型
    """
    print("\n" + "=" * 50)
    print("步骤6: 模型可解释性分析")
    print("=" * 50)
    
    # 方法1: 内置特征重要性
    print("XGBoost内置特征重要性:")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=15, importance_type='weight')
    plt.title('XGBoost特征重要性 (基于weight)')
    plt.tight_layout()
    plt.show()
    
    # 方法2: SHAP值分析（需要较长时间，样本量大时建议使用子样本）
    print("\n计算SHAP值...")
    try:
        # 使用测试集子样本加速计算
        sample_indices = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # SHAP摘要图
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
        plt.title('SHAP特征重要性')
        plt.tight_layout()
        plt.show()
        
        # SHAP依赖图（对最重要特征）
        most_important_feature = importance_df.iloc[0]['feature']
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(most_important_feature, shap_values, X_sample, show=False)
        plt.title(f'SHAP依赖图 - {most_important_feature}')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP分析出错: {e}")
        print("跳过SHAP分析，继续其他解释方法...")
    
    # 方法3: 排列重要性
    print("\n计算排列重要性...")
    perm = PermutationImportance(model, random_state=42).fit(X_test, model.predict(X_test))
    perm_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': perm.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("排列重要性排名:")
    print(perm_df.head(10))

# 模型保存与应用 
def save_and_deploy_model(model, scaler, features, test_scores):
    """
    保存模型和相关信息
    """
    print("\n" + "=" * 50)
    print("步骤7: 模型保存与部署准备")
    print("=" * 50)
    
    # 保存模型
    model_path = 'xgboost_ineinvest_model.pkl'
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存标准化器
    scaler_path = 'feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存至: {scaler_path}")
    
    # 保存特征列表和性能指标
    model_info = {
        'features': features,
        'model_performance': test_scores,
        'model_params': model.get_params(),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
    info_path = 'model_info.json'
    pd.Series(model_info).to_json(info_path)
    print(f"模型信息已保存至: {info_path}")
    
    # 创建预测函数示例
    print("\n预测函数示例:")
    print("""
def predict_ineinvest(input_data):
    # input_data: DataFrame 或 dict，包含所有特征
    # 加载模型和标准化器
    model = joblib.load('xgboost_ineinvest_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    # 数据预处理
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    # 预测
    prediction = model.predict(input_scaled)
    return prediction[0]
    """)

# 主程序 
def main():
    """
    主函数：整合所有步骤
    """
    print("企业投资预测机器学习建模系统")
    print("=" * 50)
    
    # 文件路径
    file_path = r"C:\Users\ASUS\Desktop\shuju.xlsx"
    
    try:
        df = load_and_explore_data(file_path)
        X, y, features = preprocess_data(df)
        X_train, X_test, y_train, y_test, scaler = prepare_train_test(X, y)
        model = train_xgboost_model(X_train, y_train, X_test, y_test)
        test_scores = evaluate_model(model, X_train, y_train, X_test, y_test)
        explain_model(model, X_train, X_test)
        save_and_deploy_model(model, scaler, features, test_scores)
        
        print("\n" + "=" * 50)
        print("建模完成")
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到，请检查路径是否正确")
    except KeyError as e:
        print(f"错误: 数据中缺少必要的列: {e}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

