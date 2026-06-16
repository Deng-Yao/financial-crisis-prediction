# API 参考文档

## 📋 模块概览

本项目包含以下核心模块：

| 模块 | 功能 |
|------|------|
| `CSMARDataLoader` | 数据加载器 |
| `CSMARDataProcessor` | 数据处理器 |
| `CSMARFeatureEngineer` | 特征工程器 |
| `TimeSeriesModelTrainer` | 时序模型训练器 |
| `ModelEvaluator` | 模型评估器 |
| `ModelSaver` | 模型保存器 |
| `FinancialCrisisPredictor` | 财务危机预测器 |

---

## 🔧 CSMARDataLoader

### 功能
加载CSMAR数据库的原始数据文件。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import CSMARDataLoader

loader = CSMARDataLoader(data_path='./data')
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| data_path | str | 数据文件目录路径 | './data' |

### 方法

#### load_all_data()

加载所有CSMAR数据表。

```python
data_dict = loader.load_all_data()
```

**返回值**：
```python
{
    'company_info': DataFrame,      # 公司基本信息
    'balance_sheet': DataFrame,     # 资产负债表
    'income_statement': DataFrame,  # 利润表
    'cash_flow': DataFrame,         # 现金流量表
    'st_data': DataFrame            # ST变动文件
}
```

#### load_company_info()

加载公司基本信息表。

```python
company_info = loader.load_company_info()
```

**返回值**：DataFrame 或 None

#### load_balance_sheet()

加载资产负债表。

```python
balance_sheet = loader.load_balance_sheet()
```

**返回值**：DataFrame 或 None

---

## 🔧 CSMARDataProcessor

### 功能
处理和合并CSMAR数据。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import CSMARDataProcessor

processor = CSMARDataProcessor(data_dict)
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| data_dict | dict | 由CSMARDataLoader返回的数据字典 |

### 方法

#### merge_all_data(start_year, end_year)

合并所有数据表。

```python
merged_data = processor.merge_all_data(start_year=2003, end_year=2023)
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| start_year | int | 开始年份 | 2003 |
| end_year | int | 结束年份 | 2023 |

**返回值**：DataFrame 或 None

**返回的DataFrame包含**：
- 财务报表字段（标准化后的列名）
- `year`：年份
- `is_st`：ST标签（0或1）

---

## 🔧 CSMARFeatureEngineer

### 功能
计算财务指标特征。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import CSMARFeatureEngineer

feature_engineer = CSMARFeatureEngineer(df)
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| df | DataFrame | 合并后的数据 |

### 方法

#### calculate_financial_ratios()

计算财务比率指标。

```python
df = feature_engineer.calculate_financial_ratios()
```

**返回值**：DataFrame

**计算的指标**：
| 类别 | 指标 | 计算公式 |
|------|------|----------|
| 盈利能力 | ROA | 净利润/总资产 |
| | ROE | 净利润/净资产 |
| | operating_margin | 营业利润/营业收入 |
| | profit_margin | 利润总额/营业收入 |
| | net_margin | 净利润/营业收入 |
| 偿债能力 | debt_ratio | 总负债/总资产 |
| | current_ratio | 流动资产/流动负债 |
| | debt_to_equity | 总负债/净资产 |
| | short_debt_ratio | 短期借款/总负债 |
| | long_debt_ratio | 长期借款/总负债 |
| 运营能力 | asset_turnover | 营业收入/总资产 |
| | receivables_turnover | 营业收入/应收账款 |
| | payables_turnover | 营业成本/应付账款 |
| 成长能力 | revenue_growth | 营收增长率 |
| | profit_growth | 利润增长率 |
| | asset_growth | 资产增长率 |
| 现金流 | cash_flow_ratio | 经营现金流/营业收入 |
| | cash_flow_to_assets | 经营现金流/总资产 |
| | cash_flow_to_debt | 经营现金流/总负债 |
| | cash_flow_to_profit | 经营现金流/净利润 |
| 费用率 | selling_expense_ratio | 销售费用/营业收入 |
| | admin_expense_ratio | 管理费用/营业收入 |
| | financial_expense_ratio | 财务费用/营业收入 |
| 资产结构 | current_asset_ratio | 流动资产/总资产 |
| | fixed_asset_ratio | 固定资产/总资产 |

#### handle_missing_values(method)

处理缺失值。

```python
df = feature_engineer.handle_missing_values(method='median')
```

**参数**：
| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| method | str | 填充方法 | 'median', 'mean', 'drop' |

#### normalize_features(method)

特征标准化。

```python
df = feature_engineer.normalize_features(method='standard')
```

**参数**：
| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| method | str | 标准化方法 | 'standard', 'minmax' |

#### get_feature_matrix(target_col)

获取特征矩阵和目标变量。

```python
X, y = feature_engineer.get_feature_matrix(target_col='is_st')
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| target_col | str | 目标变量列名 | 'is_st' |

**返回值**：
- X: DataFrame (特征矩阵)
- y: Series (目标变量)

---

## 🔧 TimeSeriesModelTrainer

### 功能
使用时序交叉验证训练模型。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import TimeSeriesModelTrainer

trainer = TimeSeriesModelTrainer(X, y, n_splits=5)
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| X | DataFrame | 特征矩阵 | - |
| y | Series | 目标变量 | - |
| n_splits | int | 时序交叉验证折数 | 5 |

### 方法

#### train_logistic_regression()

训练逻辑回归模型。

```python
model = trainer.train_logistic_regression()
```

**返回值**：LogisticRegression

#### train_random_forest(n_estimators, max_depth)

训练随机森林模型。

```python
model = trainer.train_random_forest(n_estimators=200, max_depth=10)
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| n_estimators | int | 树数量 | 200 |
| max_depth | int | 最大深度 | 10 |

**返回值**：RandomForestClassifier

#### train_xgboost(n_estimators, max_depth, learning_rate)

训练XGBoost模型（加强正则化）。

```python
model = trainer.train_xgboost(n_estimators=200, max_depth=6, learning_rate=0.05)
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| n_estimators | int | 树数量 | 200 |
| max_depth | int | 最大深度 | 6 |
| learning_rate | float | 学习率 | 0.05 |

**返回值**：XGBClassifier

**XGBoost正则化参数**：
```python
{
    'min_child_weight': 5,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

#### find_optimal_threshold(model_name)

寻找最优预测阈值。

```python
best_threshold, threshold_df = trainer.find_optimal_threshold('xgboost')
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | str | 模型名称 |

**返回值**：
- best_threshold: float (最优阈值)
- threshold_df: DataFrame (阈值分析结果)

#### evaluate_overfitting(X_test, y_test, model_name)

验证模型是否过拟合。

```python
overfit_results = trainer.evaluate_overfitting(X_test, y_test, 'xgboost')
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| X_test | DataFrame | 测试集特征 |
| y_test | Series | 测试集标签 |
| model_name | str | 模型名称 |

**返回值**：
```python
{
    'train_auc': float,
    'test_auc': float,
    'overfit_ratio': float  # 百分比
}
```

#### get_cv_comparison_table()

获取交叉验证比较表。

```python
comparison_df = trainer.get_cv_comparison_table()
```

**返回值**：DataFrame

---

## 🔧 ModelEvaluator

### 功能
评估模型性能并生成可视化。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import ModelEvaluator

evaluator = ModelEvaluator()
```

### 方法

#### evaluate_model(model, X_test, y_test, model_name, threshold)

评估单个模型。

```python
results = evaluator.evaluate_model(model, X_test, y_test, 'xgboost', threshold=0.5)
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| model | object | 训练好的模型 | - |
| X_test | DataFrame | 测试集特征 | - |
| y_test | Series | 测试集标签 | - |
| model_name | str | 模型名称 | - |
| threshold | float | 预测阈值 | 0.5 |

**返回值**：
```python
{
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float,
    'auc': float,
    'ap': float  # Average Precision
}
```

#### create_comparison_table()

创建模型比较表。

```python
comparison_df = evaluator.create_comparison_table()
```

**返回值**：DataFrame

#### plot_roc_curves(models_dict, X_test, y_test, threshold)

绘制ROC曲线和PR曲线。

```python
evaluator.plot_roc_curves(trainer.models, X_test, y_test, threshold=0.5)
```

**输出**：保存 `roc_pr_curves.png`

#### plot_feature_importance(model, feature_names, top_n, model_name)

绘制特征重要性图。

```python
evaluator.plot_feature_importance(model, feature_names, top_n=20, model_name='xgboost')
```

**输出**：保存 `feature_importance_xgboost.png`

#### plot_overfitting_analysis(cv_results, test_results)

绘制过拟合分析图。

```python
evaluator.plot_overfitting_analysis(trainer.cv_results, evaluator.results)
```

**输出**：保存 `overfitting_analysis.png`

---

## 🔧 ModelSaver

### 功能
保存训练好的模型。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import ModelSaver

saver = ModelSaver(save_dir='./trained_models')
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| save_dir | str | 模型保存目录 | './trained_models' |

### 方法

#### save_model(model, model_name, feature_names, scaler, threshold, metadata)

保存模型及相关组件。

```python
saved_paths = saver.save_model(
    model=model,
    model_name='xgboost',
    feature_names=feature_names,
    scaler=scaler,
    threshold=0.35,
    metadata={'cv_auc': 0.92, 'test_auc': 0.90}
)
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| model | object | 训练好的模型 |
| model_name | str | 模型名称 |
| feature_names | list | 特征名称列表 |
| scaler | object | 标准化器 |
| threshold | float | 预测阈值 |
| metadata | dict | 额外元数据（可选） |

**返回值**：
```python
{
    'model_path': str,
    'feature_path': str,
    'scaler_path': str,
    'metadata_path': str
}
```

---

## 🔧 FinancialCrisisPredictor

### 功能
使用训练好的模型进行预测。

### 初始化

```python
from financial_crisis_prediction_lite_v2 import FinancialCrisisPredictor

predictor = FinancialCrisisPredictor(model_dir='./trained_models')
```

**参数**：
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| model_dir | str | 模型目录路径 | './trained_models' |

### 方法

#### load_latest_model(model_name)

加载最新的模型。

```python
success = predictor.load_latest_model('xgboost')
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | str | 模型名称 |

**返回值**：bool

**加载的组件**：
- 模型文件
- 特征名称
- 标准化器
- 元数据（包括阈值）

#### predict(data)

预测财务危机风险。

```python
results = predictor.predict(company_data)
```

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| data | DataFrame 或 dict | 待预测的数据 |

**返回值**：
```python
[
    {
        'risk_probability': float,  # 风险概率 (0-1)
        'risk_score': float,        # 风险得分 (0-100)
        'prediction': int,          # 预测结果 (0或1)
        'risk_level': str,          # 风险等级 ('低风险' 或 '高风险')
        'threshold_used': float     # 使用的阈值
    }
]
```

---

## 📝 使用示例

### 完整训练流程

```python
from financial_crisis_prediction_lite_v2 import (
    CSMARDataLoader, CSMARDataProcessor, CSMARFeatureEngineer,
    TimeSeriesModelTrainer, ModelEvaluator, ModelSaver
)

# 1. 加载数据
loader = CSMARDataLoader('./data')
data_dict = loader.load_all_data()

# 2. 处理数据
processor = CSMARDataProcessor(data_dict)
merged_data = processor.merge_all_data(2003, 2023)

# 3. 特征工程
fe = CSMARFeatureEngineer(merged_data)
featured_data = fe.calculate_financial_ratios()
featured_data = fe.handle_missing_values('median')
featured_data = fe.normalize_features('standard')
X, y = fe.get_feature_matrix('is_st')

# 4. 训练模型
trainer = TimeSeriesModelTrainer(X, y, n_splits=5)
trainer.train_xgboost(n_estimators=200, max_depth=6, learning_rate=0.05)
best_threshold, _ = trainer.find_optimal_threshold('xgboost')

# 5. 评估模型
evaluator = ModelEvaluator()
evaluator.evaluate_model(trainer.models['xgboost'], X_test, y_test, 'xgboost', best_threshold)
evaluator.plot_roc_curves(trainer.models, X_test, y_test, best_threshold)

# 6. 保存模型
saver = ModelSaver('./trained_models')
saver.save_model(
    trainer.models['xgboost'], 'xgboost',
    fe.feature_names, fe.scaler, best_threshold
)
```

### 预测新数据

```python
from financial_crisis_prediction_lite_v2 import FinancialCrisisPredictor

# 加载模型
predictor = FinancialCrisisPredictor('./trained_models')
predictor.load_latest_model('xgboost')

# 准备数据
company_data = {
    'ROA': 0.05,
    'ROE': 0.10,
    'debt_ratio': 0.45,
    'current_ratio': 1.5,
    # ... 其他特征
}

# 预测
results = predictor.predict(company_data)
print(results)
```

---

## ⚠️ 注意事项

1. **数据顺序**：时序数据必须按时间顺序排列，不能打乱
2. **特征一致性**：预测时的特征必须与训练时完全一致
3. **缺失值**：预测数据中的缺失值会用0填充
4. **阈值选择**：根据业务需求选择合适的阈值（高召回率 vs 高精确率）