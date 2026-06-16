# -*- coding: utf-8 -*-
"""
快速开始示例
演示如何使用财务危机预测系统
"""

import sys
sys.path.append('..')

from financial_crisis_prediction_lite_v2 import (
    CSMARDataLoader, CSMARDataProcessor, CSMARFeatureEngineer,
    TimeSeriesModelTrainer, ModelEvaluator, ModelSaver, FinancialCrisisPredictor
)


def train_example():
    """训练模型示例"""
    print("=" * 60)
    print("训练模型示例")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据")
    loader = CSMARDataLoader(data_path='../data')
    data_dict = loader.load_all_data()
    
    # 2. 处理数据
    print("\n2. 处理数据")
    processor = CSMARDataProcessor(data_dict)
    merged_data = processor.merge_all_data(start_year=2003, end_year=2023)
    
    if merged_data is None:
        print("错误：数据处理失败")
        return
    
    # 3. 特征工程
    print("\n3. 特征工程")
    fe = CSMARFeatureEngineer(merged_data)
    featured_data = fe.calculate_financial_ratios()
    featured_data = fe.handle_missing_values(method='median')
    featured_data = fe.normalize_features(method='standard')
    
    X, y = fe.get_feature_matrix(target_col='is_st')
    
    # 4. 训练模型
    print("\n4. 训练模型")
    trainer = TimeSeriesModelTrainer(X, y, n_splits=5)
    
    # 训练XGBoost
    trainer.train_xgboost(n_estimators=200, max_depth=6, learning_rate=0.05)
    
    # 寻找最优阈值
    best_threshold, _ = trainer.find_optimal_threshold('xgboost')
    
    # 5. 保存模型
    print("\n5. 保存模型")
    saver = ModelSaver(save_dir='../trained_models')
    saver.save_model(
        model=trainer.models['xgboost'],
        model_name='xgboost',
        feature_names=fe.feature_names,
        scaler=fe.scaler,
        threshold=best_threshold
    )
    
    print("\n训练完成！")


def predict_example():
    """预测示例"""
    print("=" * 60)
    print("预测示例")
    print("=" * 60)
    
    # 1. 创建预测器
    print("\n1. 创建预测器")
    predictor = FinancialCrisisPredictor(model_dir='../trained_models')
    
    # 2. 加载模型
    print("\n2. 加载模型")
    success = predictor.load_latest_model('xgboost')
    
    if not success:
        print("错误：模型加载失败，请先运行训练")
        return
    
    # 3. 准备数据
    print("\n3. 准备数据")
    # 这里需要提供完整的特征数据
    # 实际使用时，应该从财务报表计算这些特征
    company_data = {
        'ROA': 0.05,           # 总资产收益率
        'ROE': 0.10,           # 净资产收益率
        'operating_margin': 0.08,  # 营业利润率
        'profit_margin': 0.07,    # 利润率
        'net_margin': 0.05,       # 净利润率
        'debt_ratio': 0.45,       # 资产负债率
        'current_ratio': 1.5,     # 流动比率
        'debt_to_equity': 0.8,    # 产权比率
        'short_debt_ratio': 0.3,  # 短期债务比率
        'long_debt_ratio': 0.2,   # 长期债务比率
        'asset_turnover': 0.6,    # 总资产周转率
        'receivables_turnover': 8.0,  # 应收账款周转率
        'payables_turnover': 5.0,     # 应付账款周转率
        'revenue_growth': 0.15,       # 营收增长率
        'profit_growth': 0.10,        # 利润增长率
        'asset_growth': 0.12,         # 资产增长率
        'cash_flow_ratio': 0.15,      # 现金流比率
        'cash_flow_to_assets': 0.08,  # 现金流资产比
        'cash_flow_to_debt': 0.2,     # 现金流负债比
        'cash_flow_to_profit': 1.5,   # 现金流利润比
        'selling_expense_ratio': 0.1, # 销售费用率
        'admin_expense_ratio': 0.08,  # 管理费用率
        'financial_expense_ratio': 0.02,  # 财务费用率
        'current_asset_ratio': 0.6,   # 流动资产比率
        'fixed_asset_ratio': 0.3,     # 固定资产比率
    }
    
    # 4. 预测
    print("\n4. 预测")
    results = predictor.predict(company_data)
    
    # 5. 显示结果
    print("\n5. 预测结果")
    for result in results:
        print(f"  风险概率: {result['risk_probability']:.4f}")
        print(f"  风险得分: {result['risk_score']:.2f}%")
        print(f"  预测结果: {result['prediction']}")
        print(f"  风险等级: {result['risk_level']}")
        print(f"  使用阈值: {result['threshold_used']:.2f}")


def batch_predict_example():
    """批量预测示例"""
    print("=" * 60)
    print("批量预测示例")
    print("=" * 60)
    
    import pandas as pd
    
    # 1. 创建预测器
    predictor = FinancialCrisisPredictor(model_dir='../trained_models')
    predictor.load_latest_model('xgboost')
    
    # 2. 加载数据（假设已有特征数据）
    # 实际使用时，应该从财务报表计算特征
    # 这里模拟多条数据
    companies = [
        {'ROA': 0.05, 'ROE': 0.10, 'debt_ratio': 0.45, ...},  # 公司1
        {'ROA': -0.02, 'ROE': -0.05, 'debt_ratio': 0.85, ...},  # 公司2
        # ... 更多公司
    ]
    
    # 3. 批量预测
    for i, company_data in enumerate(companies):
        results = predictor.predict(company_data)
        print(f"公司 {i+1}: {results[0]['risk_level']} ({results[0]['risk_score']:.2f}%)")


if __name__ == '__main__':
    # 运行训练示例
    train_example()
    
    # 运行预测示例
    predict_example()