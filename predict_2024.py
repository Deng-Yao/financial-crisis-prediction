# -*- coding: utf-8 -*-
"""
使用已训练的XGBoost模型预测2024年数据，并验证准确率
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

MODEL_DIR = "D:/DENG Yao/2026/思路/trained_models"
DATA_PATH = "D:/DENG Yao/2026/思路/data"  # 原始数据路径

# 最新模型时间戳
TIMESTAMP = "20260615_094119"

# CSMAR字段映射
CSMAR_FIELD_MAPPING = {
    'STK_LISTEDCOINFOANL': {
        'stock_code': 'Symbol',
        'company_name': 'ShortName',
        'report_date': 'EndDate',
        'industry_code': 'IndustryCodeD',
        'industry_name': 'IndustryNameD',
    },
    'FS_Combas': {
        'stock_code': 'Stkcd',
        'report_date': 'Accper',
        'report_type': 'Typrep',
        'cash': 'A001101000',
        'accounts_receivable': 'A001111000',
        'other_receivables': 'A001121000',
        'current_assets': 'A001100000',
        'fixed_assets': 'A001212000',
        'total_assets': 'A001000000',
        'short_term_debt': 'A002101000',
        'accounts_payable': 'A002108000',
        'current_liabilities': 'A002100000',
        'long_term_debt': 'A002201000',
        'total_liabilities': 'A002000000',
        'total_equity': 'A003000000',
    },
    'FS_Comins': {
        'stock_code': 'Stkcd',
        'report_date': 'Accper',
        'report_type': 'Typrep',
        'revenue': 'B001100000',
        'operating_cost': 'B001201000',
        'selling_expense': 'B001209000',
        'admin_expense': 'B001210000',
        'financial_expense': 'B001211000',
        'operating_profit': 'B001300000',
        'total_profit': 'B001000000',
        'net_profit': 'B002000000',
    },
    'FS_Comscfd': {
        'stock_code': 'Stkcd',
        'report_date': 'Accper',
        'report_type': 'Typrep',
        'operating_cash_flow': 'C001000000',
        'investing_cash_flow': 'C002000000',
        'financing_cash_flow': 'C003000000',
    },
    'SPT_Trdchg': {
        'stock_code': 'Stkcd',
        'change_type': 'Chgtype',
        'announcement_date': 'Annoudt',
    }
}

ST_CHANGE_TYPES = {
    'AB': '从正常变为ST',
    'BA': '从ST恢复正常',
    'AD': '从正常变为*ST',
    'DA': '从*ST恢复正常',
    'BD': '从ST变为*ST',
    'DB': '从*ST变为ST',
}


# ============================================================
# 数据加载与处理
# ============================================================

def standardize_columns(df, table_name):
    """标准化列名"""
    df_std = df.copy()
    mapping = CSMAR_FIELD_MAPPING.get(table_name, {})
    rename_dict = {}
    for std_name, orig_name in mapping.items():
        if orig_name in df_std.columns:
            rename_dict[orig_name] = std_name
    df_std.rename(columns=rename_dict, inplace=True)
    return df_std


def load_and_process_2024_data():
    """加载并处理2024年数据"""
    print("=" * 60)
    print("加载2024年数据")
    print("=" * 60)
    
    # 加载原始数据
    print("\n加载原始数据...")
    
    balance_sheet = pd.read_csv(f"{DATA_PATH}/FS_Combas.csv", encoding='utf-8')
    income_statement = pd.read_csv(f"{DATA_PATH}/FS_Comins.csv", encoding='utf-8')
    cash_flow = pd.read_csv(f"{DATA_PATH}/FS_Comscfd.csv", encoding='utf-8')
    st_data = pd.read_csv(f"{DATA_PATH}/SPT_Trdchg.csv", encoding='utf-8')
    
    print(f"  资产负债表: {len(balance_sheet)} 条")
    print(f"  利润表: {len(income_statement)} 条")
    print(f"  现金流量表: {len(cash_flow)} 条")
    print(f"  ST变动文件: {len(st_data)} 条")
    
    # 标准化列名
    balance_df = standardize_columns(balance_sheet, 'FS_Combas')
    income_df = standardize_columns(income_statement, 'FS_Comins')
    cashflow_df = standardize_columns(cash_flow, 'FS_Comscfd')
    
    # 合并三大报表
    print("\n合并三大报表...")
    merge_keys = ['stock_code', 'report_date']
    
    merged_df = pd.merge(balance_df, income_df, on=merge_keys, how='inner', suffixes=('_bs', '_is'))
    merged_df = pd.merge(merged_df, cashflow_df, on=merge_keys, how='inner', suffixes=('', '_cf'))
    
    # 处理日期，筛选2024年数据
    merged_df['report_date'] = pd.to_datetime(merged_df['report_date'], errors='coerce')
    merged_df['year'] = merged_df['report_date'].dt.year
    merged_df['month'] = merged_df['report_date'].dt.month
    
    # 筛选2024年 + 合并报表 + 年报（12月）
    data_2024 = merged_df[
        (merged_df['year'] == 2024) & 
        (merged_df['report_type'] == 'A') &
        (merged_df['month'] == 12)
    ].copy()
    
    # 去重：每只股票只保留一条记录（取第一条）
    data_2024 = data_2024.drop_duplicates(subset=['stock_code'], keep='first')
    
    print(f"  2024年年报数据（去重后）: {len(data_2024)} 条")
    
    # 创建ST标签
    print("\n创建2024年ST标签...")
    st_std = standardize_columns(st_data, 'SPT_Trdchg')
    st_std['announcement_date'] = pd.to_datetime(st_std['announcement_date'], errors='coerce')
    st_std['year'] = st_std['announcement_date'].dt.year
    
    st_2024 = st_std[st_std['year'] == 2024].copy()
    
    labels = []
    for _, row in st_2024.iterrows():
        stock_code = row['stock_code']
        change_type = str(row.get('change_type', ''))
        
        if len(change_type) >= 2:
            after_status = change_type[1]
            is_st = 1 if after_status in ['B', 'D', 'S'] else 0
        else:
            is_st = 1 if change_type in ['B', 'D', 'S'] else 0
        
        labels.append({'stock_code': stock_code, 'is_st': is_st})
    
    if labels:
        labels_df = pd.DataFrame(labels)
        labels_df = labels_df.groupby('stock_code')['is_st'].max().reset_index()
        
        # 合并标签
        data_2024 = pd.merge(data_2024, labels_df, on='stock_code', how='left')
        data_2024['is_st'].fillna(0, inplace=True)
        data_2024['is_st'] = data_2024['is_st'].astype(int)
        
        print(f"  ST公司数量: {(data_2024['is_st'] == 1).sum()}")
        print(f"  ST公司占比: {data_2024['is_st'].mean()*100:.2f}%")
    else:
        print("  警告：未找到2024年ST标签")
        data_2024['is_st'] = 0
    
    return data_2024


def calculate_features(df):
    """计算财务特征"""
    print("\n计算财务特征...")
    
    feature_names = []
    
    # 盈利能力
    if 'net_profit' in df.columns and 'total_assets' in df.columns:
        df['ROA'] = df['net_profit'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('ROA')
    
    if 'net_profit' in df.columns and 'total_equity' in df.columns:
        df['ROE'] = df['net_profit'] / df['total_equity'].replace(0, np.nan)
        feature_names.append('ROE')
    
    if 'operating_profit' in df.columns and 'revenue' in df.columns:
        df['operating_margin'] = df['operating_profit'] / df['revenue'].replace(0, np.nan)
        feature_names.append('operating_margin')
    
    if 'total_profit' in df.columns and 'revenue' in df.columns:
        df['profit_margin'] = df['total_profit'] / df['revenue'].replace(0, np.nan)
        feature_names.append('profit_margin')
    
    if 'net_profit' in df.columns and 'revenue' in df.columns:
        df['net_margin'] = df['net_profit'] / df['revenue'].replace(0, np.nan)
        feature_names.append('net_margin')
    
    # 偿债能力
    if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
        df['debt_ratio'] = df['total_liabilities'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('debt_ratio')
    
    if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
        df['current_ratio'] = df['current_assets'] / df['current_liabilities'].replace(0, np.nan)
        feature_names.append('current_ratio')
    
    if 'total_liabilities' in df.columns and 'total_equity' in df.columns:
        df['debt_to_equity'] = df['total_liabilities'] / df['total_equity'].replace(0, np.nan)
        feature_names.append('debt_to_equity')
    
    if 'short_term_debt' in df.columns and 'total_liabilities' in df.columns:
        df['short_debt_ratio'] = df['short_term_debt'] / df['total_liabilities'].replace(0, np.nan)
        feature_names.append('short_debt_ratio')
    
    if 'long_term_debt' in df.columns and 'total_liabilities' in df.columns:
        df['long_debt_ratio'] = df['long_term_debt'] / df['total_liabilities'].replace(0, np.nan)
        feature_names.append('long_debt_ratio')
    
    # 运营能力
    if 'revenue' in df.columns and 'total_assets' in df.columns:
        df['asset_turnover'] = df['revenue'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('asset_turnover')
    
    if 'revenue' in df.columns and 'accounts_receivable' in df.columns:
        df['receivables_turnover'] = df['revenue'] / df['accounts_receivable'].replace(0, np.nan)
        feature_names.append('receivables_turnover')
    
    if 'operating_cost' in df.columns and 'accounts_payable' in df.columns:
        df['payables_turnover'] = df['operating_cost'] / df['accounts_payable'].replace(0, np.nan)
        feature_names.append('payables_turnover')
    
    # 成长能力（2024年单年数据无法计算同比，设为0）
    df['revenue_growth'] = 0
    df['profit_growth'] = 0
    df['asset_growth'] = 0
    feature_names.extend(['revenue_growth', 'profit_growth', 'asset_growth'])
    
    # 现金流
    if 'operating_cash_flow' in df.columns and 'revenue' in df.columns:
        df['cash_flow_ratio'] = df['operating_cash_flow'] / df['revenue'].replace(0, np.nan)
        feature_names.append('cash_flow_ratio')
    
    if 'operating_cash_flow' in df.columns and 'total_assets' in df.columns:
        df['cash_flow_to_assets'] = df['operating_cash_flow'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('cash_flow_to_assets')
    
    if 'operating_cash_flow' in df.columns and 'total_liabilities' in df.columns:
        df['cash_flow_to_debt'] = df['operating_cash_flow'] / df['total_liabilities'].replace(0, np.nan)
        feature_names.append('cash_flow_to_debt')
    
    if 'operating_cash_flow' in df.columns and 'net_profit' in df.columns:
        df['cash_flow_to_profit'] = df['operating_cash_flow'] / df['net_profit'].replace(0, np.nan)
        feature_names.append('cash_flow_to_profit')
    
    # 费用率
    if 'selling_expense' in df.columns and 'revenue' in df.columns:
        df['selling_expense_ratio'] = df['selling_expense'] / df['revenue'].replace(0, np.nan)
        feature_names.append('selling_expense_ratio')
    
    if 'admin_expense' in df.columns and 'revenue' in df.columns:
        df['admin_expense_ratio'] = df['admin_expense'] / df['revenue'].replace(0, np.nan)
        feature_names.append('admin_expense_ratio')
    
    if 'financial_expense' in df.columns and 'revenue' in df.columns:
        df['financial_expense_ratio'] = df['financial_expense'] / df['revenue'].replace(0, np.nan)
        feature_names.append('financial_expense_ratio')
    
    # 资产结构
    if 'current_assets' in df.columns and 'total_assets' in df.columns:
        df['current_asset_ratio'] = df['current_assets'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('current_asset_ratio')
    
    if 'fixed_assets' in df.columns and 'total_assets' in df.columns:
        df['fixed_asset_ratio'] = df['fixed_assets'] / df['total_assets'].replace(0, np.nan)
        feature_names.append('fixed_asset_ratio')
    
    print(f"  计算了 {len(feature_names)} 个特征")
    
    # 处理缺失值和无穷大
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf], np.nan)
    df[feature_names] = df[feature_names].fillna(0)
    
    return df, feature_names


# ============================================================
# 模型加载与预测
# ============================================================

def load_model():
    """加载模型和相关组件"""
    print("\n" + "=" * 60)
    print("加载XGBoost模型")
    print("=" * 60)
    
    # 加载模型
    model_path = f"{MODEL_DIR}/xgboost_{TIMESTAMP}.joblib"
    model = joblib.load(model_path)
    print(f"  ✓ 模型: {model_path}")
    
    # 加载特征名称
    feature_path = f"{MODEL_DIR}/feature_names_{TIMESTAMP}.joblib"
    feature_names = joblib.load(feature_path)
    print(f"  ✓ 特征: {len(feature_names)} 个")
    
    # 加载标准化器
    scaler_path = f"{MODEL_DIR}/scaler_{TIMESTAMP}.joblib"
    scaler = joblib.load(scaler_path)
    print(f"  ✓ 标准化器")
    
    # 加载元数据
    metadata_path = f"{MODEL_DIR}/metadata_{TIMESTAMP}.joblib"
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        threshold = metadata.get('threshold', 0.5)
        print(f"  ✓ 阈值: {threshold:.2f}")
    else:
        threshold = 0.5
        print(f"  ⚠ 使用默认阈值: {threshold}")
    
    return model, feature_names, scaler, threshold


def predict_2024(model, feature_names, scaler, threshold, data_2024):
    """对2024年数据进行预测"""
    print("\n" + "=" * 60)
    print("2024年数据预测")
    print("=" * 60)
    
    # 准备特征
    X = data_2024[feature_names].copy()
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测概率
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    # 使用阈值预测
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob


# ============================================================
# 评估结果
# ============================================================

def evaluate_results(y_true, y_pred, y_prob):
    """评估预测结果"""
    print("\n" + "=" * 60)
    print("预测准确率评估")
    print("=" * 60)
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"\n【基本指标】")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数:             {f1:.4f}")
    print(f"  AUC:                {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n【混淆矩阵】")
    print(f"  预测\\实际    非ST    ST")
    print(f"  非ST        {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  ST          {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # 详细分类报告
    print(f"\n【详细分类报告】")
    print(classification_report(y_true, y_pred, target_names=['非ST', 'ST']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    
    # 1. 加载2024年数据
    data_2024 = load_and_process_2024_data()
    
    # 2. 计算特征
    data_2024, feature_names = calculate_features(data_2024)
    
    # 3. 加载模型
    model, model_feature_names, scaler, threshold = load_model()
    
    # 4. 预测
    y_true = data_2024['is_st'].values
    y_pred, y_prob = predict_2024(model, model_feature_names, scaler, threshold, data_2024)
    
    # 5. 评估
    results = evaluate_results(y_true, y_pred, y_prob)
    
    # 6. 保存结果
    print("\n" + "=" * 60)
    print("保存预测结果")
    print("=" * 60)
    
    output_df = data_2024[['stock_code', 'is_st']].copy()
    output_df['predicted'] = y_pred
    output_df['probability'] = y_prob
    output_df['risk_level'] = output_df['predicted'].map({0: '低风险', 1: '高风险'})
    
    output_df.to_csv('prediction_2024_results.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 预测结果已保存: prediction_2024_results.csv")
    
    # 显示预测为ST的公司
    st_predicted = output_df[output_df['predicted'] == 1]
    if len(st_predicted) > 0:
        print(f"\n【预测为ST的公司】共 {len(st_predicted)} 家")
        print(st_predicted[['stock_code', 'probability', 'risk_level']].head(20).to_string(index=False))
    
    # 显示实际ST但未预测出的公司（漏报）
    false_negatives = output_df[(output_df['is_st'] == 1) & (output_df['predicted'] == 0)]
    if len(false_negatives) > 0:
        print(f"\n【漏报的ST公司】共 {len(false_negatives)} 家")
        print(false_negatives[['stock_code', 'probability']].head(20).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()