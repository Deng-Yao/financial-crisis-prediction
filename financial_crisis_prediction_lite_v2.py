# -*- coding: utf-8 -*-
"""
中国上市公司财务危机预测模型（精简版V2）
- 去掉卡尔曼滤波模型
- 使用时序交叉验证（TimeSeriesSplit）
- 验证模型过拟合
- 加强XGBoost正则化
- 调整预测阈值处理数据不平衡
- 保存训练好的模型文件
- 移除LDA和神经网络（在不平衡数据上效果差）

数据表（共5个）：
1. STK_LISTEDCOINFOANL.csv - 上市公司基本信息表
2. FS_Combas.csv - 资产负债表
3. FS_Comins.csv - 利润表
4. FS_Comscfd.csv - 现金流量表（直接法）
5. SPT_Trdchg.csv - 特殊处理变动文件

时间范围：2003-2023年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CSMAR字段映射
# ============================================================

CSMAR_FIELD_MAPPING = {
    'STK_LISTEDCOINFOANL': {
        'stock_code': 'Symbol',
        'company_name': 'ShortName',
        'report_date': 'EndDate',
        'industry_code': 'IndustryCodeD',
        'industry_name': 'IndustryNameD',
        'listing_date': 'LISTINGDATE',
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
        'sales_cash_received': 'C001001000',
        'operating_cash_inflow': 'C001100000',
        'operating_cash_outflow': 'C001200000',
        'operating_cash_flow': 'C001000000',
        'investing_cash_flow': 'C002000000',
        'financing_cash_flow': 'C003000000',
        'cash_increase': 'C005000000',
    },
    'SPT_Trdchg': {
        'stock_code': 'Stkcd',
        'name_before': 'Stknmebc',
        'name_after': 'Stknmeac',
        'change_type': 'Chgtype',
        'announcement_date': 'Annoudt',
        'execute_date': 'Execudt',
        'change_reason': 'Chgrsdis',
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
# 第一部分：数据加载
# ============================================================

class CSMARDataLoader:
    """CSMAR数据加载器"""
    
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.company_info = None
        self.balance_sheet = None
        self.income_statement = None
        self.cash_flow = None
        self.st_data = None
        self.field_mapping = CSMAR_FIELD_MAPPING
    
    def load_all_data(self):
        """加载所有CSMAR数据"""
        print("=" * 60)
        print("开始加载CSMAR数据...")
        print("=" * 60)
        
        self.load_company_info()
        self.load_balance_sheet()
        self.load_income_statement()
        self.load_cash_flow()
        self.load_st_data()
        
        print("=" * 60)
        print("数据加载完成")
        print("=" * 60)
        
        return {
            'company_info': self.company_info,
            'balance_sheet': self.balance_sheet,
            'income_statement': self.income_statement,
            'cash_flow': self.cash_flow,
            'st_data': self.st_data,
        }
    
    def load_company_info(self):
        filepath = f"{self.data_path}/STK_LISTEDCOINFOANL.csv"
        try:
            self.company_info = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 公司基本信息表加载成功，共 {len(self.company_info)} 条记录")
            return self.company_info
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return None
    
    def load_balance_sheet(self):
        filepath = f"{self.data_path}/FS_Combas.csv"
        try:
            self.balance_sheet = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 资产负债表加载成功，共 {len(self.balance_sheet)} 条记录")
            return self.balance_sheet
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return None
    
    def load_income_statement(self):
        filepath = f"{self.data_path}/FS_Comins.csv"
        try:
            self.income_statement = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 利润表加载成功，共 {len(self.income_statement)} 条记录")
            return self.income_statement
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return None
    
    def load_cash_flow(self):
        filepath = f"{self.data_path}/FS_Comscfd.csv"
        try:
            self.cash_flow = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 现金流量表加载成功，共 {len(self.cash_flow)} 条记录")
            return self.cash_flow
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return None
    
    def load_st_data(self):
        filepath = f"{self.data_path}/SPT_Trdchg.csv"
        try:
            self.st_data = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 特殊处理变动文件加载成功，共 {len(self.st_data)} 条记录")
            
            if 'Chgtype' in self.st_data.columns:
                print(f"  变动类型分布:")
                type_counts = self.st_data['Chgtype'].value_counts().head(10)
                for ctype, count in type_counts.items():
                    desc = ST_CHANGE_TYPES.get(ctype, '未知')
                    print(f"    {ctype}: {count} 条 ({desc})")
            
            return self.st_data
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return None


# ============================================================
# 第二部分：数据预处理
# ============================================================

class CSMARDataProcessor:
    """CSMAR数据处理器"""
    
    def __init__(self, data_dict):
        self.data = data_dict
        self.processed_data = None
        self.field_mapping = CSMAR_FIELD_MAPPING
    
    def standardize_columns(self, df, table_name):
        df_std = df.copy()
        mapping = self.field_mapping.get(table_name, {})
        rename_dict = {}
        for std_name, orig_name in mapping.items():
            if orig_name in df_std.columns:
                rename_dict[orig_name] = std_name
        df_std.rename(columns=rename_dict, inplace=True)
        return df_std
    
    def merge_financial_data(self, start_year=2003, end_year=2023):
        """合并三大报表数据"""
        print("合并三大报表数据...")
        
        if self.data['balance_sheet'] is None or self.data['income_statement'] is None:
            print("  错误：缺少必需的财务数据")
            return None
        
        balance_df = self.standardize_columns(self.data['balance_sheet'], 'FS_Combas')
        income_df = self.standardize_columns(self.data['income_statement'], 'FS_Comins')
        
        merge_keys = ['stock_code', 'report_date']
        
        merged_df = pd.merge(balance_df, income_df, on=merge_keys, how='inner', suffixes=('_bs', '_is'))
        print(f"  资产负债表与利润表合并完成")
        
        if self.data['cash_flow'] is not None:
            cashflow_df = self.standardize_columns(self.data['cash_flow'], 'FS_Comscfd')
            if all(key in cashflow_df.columns for key in merge_keys):
                merged_df = pd.merge(merged_df, cashflow_df, on=merge_keys, how='inner', suffixes=('', '_cf'))
                print(f"  现金流量表合并完成")
        
        if 'report_date' in merged_df.columns:
            merged_df['report_date'] = pd.to_datetime(merged_df['report_date'], errors='coerce')
            merged_df['year'] = merged_df['report_date'].dt.year
            merged_df['month'] = merged_df['report_date'].dt.month
            merged_df = merged_df[(merged_df['year'] >= start_year) & (merged_df['year'] <= end_year)]
        
        if 'report_type' in merged_df.columns:
            merged_df = merged_df[merged_df['report_type'] == 'A']
        
        # 只保留年报数据（12月）
        if 'month' in merged_df.columns:
            merged_df = merged_df[merged_df['month'] == 12]
            print(f"  筛选年报数据（12月）")
        
        # 去重：每只股票每年只保留一条记录
        merged_df = merged_df.drop_duplicates(subset=['stock_code', 'year'], keep='first')
        
        print(f"  最终合并数据: {len(merged_df)} 条记录")
        return merged_df
    
    def create_st_labels(self, start_year=2003, end_year=2023):
        """创建ST标记标签"""
        print("创建ST标记标签...")
        
        st_df = self.data.get('st_data')
        
        if st_df is None or len(st_df) == 0:
            print("  警告：无ST数据，将创建模拟标签")
            return self._create_simulated_labels(start_year, end_year)
        
        st_std = self.standardize_columns(st_df, 'SPT_Trdchg')
        labels = []
        
        if 'stock_code' in st_std.columns and 'announcement_date' in st_std.columns:
            st_std['announcement_date'] = pd.to_datetime(st_std['announcement_date'], errors='coerce')
            st_std['year'] = st_std['announcement_date'].dt.year
            st_filtered = st_std[(st_std['year'] >= start_year) & (st_std['year'] <= end_year)]
            
            for _, row in st_filtered.iterrows():
                stock_code = row['stock_code']
                year = row['year']
                change_type = str(row.get('change_type', ''))
                
                if len(change_type) >= 2:
                    after_status = change_type[1]
                    is_st = 1 if after_status in ['B', 'D', 'S'] else 0
                else:
                    is_st = 1 if change_type in ['B', 'D', 'S'] else 0
                
                labels.append({'stock_code': stock_code, 'year': year, 'is_st': is_st})
        
        if len(labels) == 0:
            return self._create_simulated_labels(start_year, end_year)
        
        labels_df = pd.DataFrame(labels)
        labels_df = labels_df.groupby(['stock_code', 'year'])['is_st'].max().reset_index()
        
        print(f"  ST标签创建完成，共 {len(labels_df)} 条记录")
        print(f"  ST公司占比: {labels_df['is_st'].mean()*100:.2f}%")
        
        return labels_df
    
    def _create_simulated_labels(self, start_year, end_year):
        np.random.seed(42)
        company_df = self.data.get('company_info')
        if company_df is not None and 'Symbol' in company_df.columns:
            companies = company_df['Symbol'].unique()[:500]
        else:
            companies = [f'{i:06d}' for i in range(1, 501)]
        
        labels = []
        for company in companies:
            for year in range(start_year, end_year + 1):
                is_st = np.random.random() < 0.05
                labels.append({'stock_code': company, 'year': year, 'is_st': int(is_st)})
        
        return pd.DataFrame(labels)
    
    def merge_all_data(self, start_year=2003, end_year=2023):
        """合并所有数据"""
        print("\n开始合并所有数据...")
        
        financial_data = self.merge_financial_data(start_year, end_year)
        if financial_data is None:
            return None
        
        st_labels = self.create_st_labels(start_year, end_year)
        
        if st_labels is not None and len(st_labels) > 0:
            merge_keys = ['stock_code', 'year']
            merged_data = pd.merge(financial_data, st_labels, on=merge_keys, how='left')
            merged_data['is_st'].fillna(0, inplace=True)
            merged_data['is_st'] = merged_data['is_st'].astype(int)
            print(f"  数据合并完成，共 {len(merged_data)} 条记录")
        else:
            merged_data = financial_data
            merged_data['is_st'] = 0
        
        self.processed_data = merged_data
        return merged_data


# ============================================================
# 第三部分：特征工程
# ============================================================

class CSMARFeatureEngineer:
    """CSMAR特征工程器"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        self.scaler = None
        
    def calculate_financial_ratios(self):
        """计算财务比率指标"""
        print("计算财务比率指标...")
        
        # 盈利能力指标
        if 'net_profit' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['ROA'] = self.df['net_profit'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('ROA')
        
        if 'net_profit' in self.df.columns and 'total_equity' in self.df.columns:
            self.df['ROE'] = self.df['net_profit'] / self.df['total_equity'].replace(0, np.nan)
            self.feature_names.append('ROE')
        
        if 'operating_profit' in self.df.columns and 'revenue' in self.df.columns:
            self.df['operating_margin'] = self.df['operating_profit'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('operating_margin')
        
        if 'total_profit' in self.df.columns and 'revenue' in self.df.columns:
            self.df['profit_margin'] = self.df['total_profit'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('profit_margin')
        
        if 'net_profit' in self.df.columns and 'revenue' in self.df.columns:
            self.df['net_margin'] = self.df['net_profit'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('net_margin')
        
        # 偿债能力指标
        if 'total_liabilities' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['debt_ratio'] = self.df['total_liabilities'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('debt_ratio')
        
        if 'current_assets' in self.df.columns and 'current_liabilities' in self.df.columns:
            self.df['current_ratio'] = self.df['current_assets'] / self.df['current_liabilities'].replace(0, np.nan)
            self.feature_names.append('current_ratio')
        
        if 'total_liabilities' in self.df.columns and 'total_equity' in self.df.columns:
            self.df['debt_to_equity'] = self.df['total_liabilities'] / self.df['total_equity'].replace(0, np.nan)
            self.feature_names.append('debt_to_equity')
        
        if 'short_term_debt' in self.df.columns and 'total_liabilities' in self.df.columns:
            self.df['short_debt_ratio'] = self.df['short_term_debt'] / self.df['total_liabilities'].replace(0, np.nan)
            self.feature_names.append('short_debt_ratio')
        
        if 'long_term_debt' in self.df.columns and 'total_liabilities' in self.df.columns:
            self.df['long_debt_ratio'] = self.df['long_term_debt'] / self.df['total_liabilities'].replace(0, np.nan)
            self.feature_names.append('long_debt_ratio')
        
        # 运营能力指标
        if 'revenue' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['asset_turnover'] = self.df['revenue'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('asset_turnover')
        
        if 'revenue' in self.df.columns and 'accounts_receivable' in self.df.columns:
            self.df['receivables_turnover'] = self.df['revenue'] / self.df['accounts_receivable'].replace(0, np.nan)
            self.feature_names.append('receivables_turnover')
        
        if 'operating_cost' in self.df.columns and 'accounts_payable' in self.df.columns:
            self.df['payables_turnover'] = self.df['operating_cost'] / self.df['accounts_payable'].replace(0, np.nan)
            self.feature_names.append('payables_turnover')
        
        # 成长能力指标
        if 'revenue' in self.df.columns:
            self.df['revenue_growth'] = self.df.groupby('stock_code')['revenue'].pct_change()
            self.feature_names.append('revenue_growth')
        
        if 'net_profit' in self.df.columns:
            self.df['profit_growth'] = self.df.groupby('stock_code')['net_profit'].pct_change()
            self.feature_names.append('profit_growth')
        
        if 'total_assets' in self.df.columns:
            self.df['asset_growth'] = self.df.groupby('stock_code')['total_assets'].pct_change()
            self.feature_names.append('asset_growth')
        
        # 现金流指标
        if 'operating_cash_flow' in self.df.columns and 'revenue' in self.df.columns:
            self.df['cash_flow_ratio'] = self.df['operating_cash_flow'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('cash_flow_ratio')
        
        if 'operating_cash_flow' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['cash_flow_to_assets'] = self.df['operating_cash_flow'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('cash_flow_to_assets')
        
        if 'operating_cash_flow' in self.df.columns and 'total_liabilities' in self.df.columns:
            self.df['cash_flow_to_debt'] = self.df['operating_cash_flow'] / self.df['total_liabilities'].replace(0, np.nan)
            self.feature_names.append('cash_flow_to_debt')
        
        if 'operating_cash_flow' in self.df.columns and 'net_profit' in self.df.columns:
            self.df['cash_flow_to_profit'] = self.df['operating_cash_flow'] / self.df['net_profit'].replace(0, np.nan)
            self.feature_names.append('cash_flow_to_profit')
        
        # 费用率指标
        if 'selling_expense' in self.df.columns and 'revenue' in self.df.columns:
            self.df['selling_expense_ratio'] = self.df['selling_expense'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('selling_expense_ratio')
        
        if 'admin_expense' in self.df.columns and 'revenue' in self.df.columns:
            self.df['admin_expense_ratio'] = self.df['admin_expense'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('admin_expense_ratio')
        
        if 'financial_expense' in self.df.columns and 'revenue' in self.df.columns:
            self.df['financial_expense_ratio'] = self.df['financial_expense'] / self.df['revenue'].replace(0, np.nan)
            self.feature_names.append('financial_expense_ratio')
        
        # 资产结构指标
        if 'current_assets' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['current_asset_ratio'] = self.df['current_assets'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('current_asset_ratio')
        
        if 'fixed_assets' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['fixed_asset_ratio'] = self.df['fixed_assets'] / self.df['total_assets'].replace(0, np.nan)
            self.feature_names.append('fixed_asset_ratio')
        
        print(f"  计算了 {len(self.feature_names)} 个财务指标")
        return self.df
    
    def handle_missing_values(self, method='median'):
        """处理缺失值"""
        print(f"处理缺失值（方法：{method}）...")
        feature_cols = [col for col in self.feature_names if col in self.df.columns]
        
        if method == 'median':
            self.df[feature_cols] = self.df[feature_cols].fillna(self.df[feature_cols].median())
        elif method == 'mean':
            self.df[feature_cols] = self.df[feature_cols].fillna(self.df[feature_cols].mean())
        elif method == 'drop':
            self.df.dropna(subset=feature_cols, inplace=True)
        
        self.df[feature_cols] = self.df[feature_cols].replace([np.inf, -np.inf], np.nan)
        self.df[feature_cols] = self.df[feature_cols].fillna(0)
        
        print(f"  缺失值处理完成")
        return self.df
    
    def normalize_features(self, method='standard'):
        """特征标准化"""
        print(f"特征标准化（方法：{method}）...")
        feature_cols = [col for col in self.feature_names if col in self.df.columns]
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        
        print(f"  标准化完成")
        return self.df
    
    def get_feature_matrix(self, target_col='is_st'):
        """获取特征矩阵和目标变量"""
        feature_cols = [col for col in self.feature_names if col in self.df.columns]
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy() if target_col in self.df.columns else None
        
        print(f"特征矩阵形状: {X.shape}")
        if y is not None:
            print(f"目标变量分布:")
            print(f"  非ST公司: {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
            print(f"  ST公司: {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
        
        return X, y


# ============================================================
# 第四部分：时序交叉验证与模型训练
# ============================================================

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class TimeSeriesModelTrainer:
    """时序交叉验证模型训练器"""
    
    def __init__(self, X, y, n_splits=5):
        """
        初始化时序模型训练器
        
        Parameters:
        -----------
        X : DataFrame
            特征矩阵
        y : Series
            目标变量
        n_splits : int
            时序交叉验证折数
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.models = {}
        self.cv_results = {}
        self.final_results = {}
        
        # 按时间排序（假设数据已按时间排序）
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def train_logistic_regression(self):
        """训练逻辑回归模型（带时序交叉验证）"""
        print("\n训练逻辑回归模型（时序交叉验证）...")
        
        model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight='balanced',  # 处理类别不平衡
            C=0.1,  # 正则化
            penalty='l2'
        )
        
        # 时序交叉验证
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]
            X_val_fold = self.X.iloc[val_idx]
            y_val_fold = self.y.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_prob))
        
        self.cv_results['logistic'] = {k: np.mean(v) for k, v in cv_scores.items()}
        self.cv_results['logistic_std'] = {k: np.std(v) for k, v in cv_scores.items()}
        
        print(f"  时序CV AUC: {self.cv_results['logistic']['auc']:.4f} (+/- {self.cv_results['logistic_std']['auc']:.4f})")
        
        # 用全部数据训练最终模型
        model.fit(self.X, self.y)
        self.models['logistic'] = model
        
        return model
    
    def train_random_forest(self, n_estimators=200, max_depth=10):
        """训练随机森林模型（带时序交叉验证）"""
        print("\n训练随机森林模型（时序交叉验证）...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # 时序交叉验证
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]
            X_val_fold = self.X.iloc[val_idx]
            y_val_fold = self.y.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_prob))
        
        self.cv_results['random_forest'] = {k: np.mean(v) for k, v in cv_scores.items()}
        self.cv_results['random_forest_std'] = {k: np.std(v) for k, v in cv_scores.items()}
        
        print(f"  时序CV AUC: {self.cv_results['random_forest']['auc']:.4f} (+/- {self.cv_results['random_forest_std']['auc']:.4f})")
        
        # 用全部数据训练最终模型
        model.fit(self.X, self.y)
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, n_estimators=200, max_depth=6, learning_rate=0.05):
        """训练XGBoost模型（加强正则化，带时序交叉验证）"""
        print("\n训练XGBoost模型（加强正则化，时序交叉验证）...")
        
        # 计算正负样本比例
        pos_count = (self.y == 1).sum()
        neg_count = (self.y == 0).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=5,  # 增加正则化
            gamma=0.1,  # 增加正则化
            subsample=0.8,  # 防止过拟合
            colsample_bytree=0.8,  # 防止过拟合
            reg_alpha=0.1,  # L1正则化
            reg_lambda=1.0,  # L2正则化
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='aucpr',  # 使用AUCPR更适合不平衡数据
            early_stopping_rounds=20
        )
        
        # 时序交叉验证
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]
            X_val_fold = self.X.iloc[val_idx]
            y_val_fold = self.y.iloc[val_idx]
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_pred = model.predict(X_val_fold)
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_prob))
        
        self.cv_results['xgboost'] = {k: np.mean(v) for k, v in cv_scores.items()}
        self.cv_results['xgboost_std'] = {k: np.std(v) for k, v in cv_scores.items()}
        
        print(f"  时序CV AUC: {self.cv_results['xgboost']['auc']:.4f} (+/- {self.cv_results['xgboost_std']['auc']:.4f})")
        
        # 用全部数据训练最终模型（带early stopping）
        # 分割出验证集用于early stopping
        split_idx = int(len(self.X) * 0.8)
        X_train_final = self.X.iloc[:split_idx]
        y_train_final = self.y.iloc[:split_idx]
        X_val_final = self.X.iloc[split_idx:]
        y_val_final = self.y.iloc[split_idx:]
        
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        
        return model
    
    def find_optimal_threshold(self, model_name='xgboost'):
        """寻找最优预测阈值"""
        print(f"\n寻找{model_name}最优预测阈值...")
        
        model = self.models.get(model_name)
        if model is None:
            print(f"  错误：模型 {model_name} 不存在")
            return 0.5
        
        # 获取预测概率
        y_prob = model.predict_proba(self.X)[:, 1]
        
        # 计算不同阈值下的指标
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            results.append({
                'threshold': threshold,
                'precision': precision_score(self.y, y_pred, zero_division=0),
                'recall': recall_score(self.y, y_pred, zero_division=0),
                'f1': f1_score(self.y, y_pred, zero_division=0),
            })
        
        results_df = pd.DataFrame(results)
        
        # 找到F1最大的阈值
        best_idx = results_df['f1'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f1 = results_df.loc[best_idx, 'f1']
        
        print(f"  最优阈值: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
        return best_threshold, results_df
    
    def evaluate_overfitting(self, X_test, y_test, model_name='xgboost'):
        """验证模型是否过拟合"""
        print(f"\n验证{model_name}模型过拟合情况...")
        
        model = self.models.get(model_name)
        if model is None:
            print(f"  错误：模型 {model_name} 不存在")
            return None
        
        # 训练集表现
        y_train_pred = model.predict(self.X)
        y_train_prob = model.predict_proba(self.X)[:, 1]
        train_auc = roc_auc_score(self.y, y_train_prob)
        
        # 测试集表现
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_prob)
        
        # 计算过拟合程度
        overfit_ratio = (train_auc - test_auc) / train_auc * 100
        
        print(f"  训练集AUC: {train_auc:.4f}")
        print(f"  测试集AUC: {test_auc:.4f}")
        print(f"  过拟合程度: {overfit_ratio:.2f}%")
        
        if overfit_ratio > 10:
            print(f"  ⚠️ 警告：模型可能存在过拟合（差异>10%）")
        elif overfit_ratio > 5:
            print(f"  ⚡ 注意：模型存在轻微过拟合（差异5-10%）")
        else:
            print(f"  ✓ 模型过拟合程度可接受（差异<5%）")
        
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfit_ratio': overfit_ratio
        }
    
    def get_cv_comparison_table(self):
        """获取交叉验证比较表"""
        print("\n" + "=" * 60)
        print("时序交叉验证结果比较")
        print("=" * 60)
        
        # 构建比较表
        comparison = []
        for model_name in ['logistic', 'random_forest', 'xgboost']:
            if model_name in self.cv_results:
                row = {'模型': model_name}
                row.update(self.cv_results[model_name])
                comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.round(4)
        print(comparison_df.to_string(index=False))
        
        return comparison_df


# ============================================================
# 第五部分：模型评估与可视化
# ============================================================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name, threshold=0.5):
        """评估单个模型"""
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        self.results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob),
            'ap': average_precision_score(y_test, y_prob)  # Average Precision
        }
        
        return self.results[model_name]
    
    def create_comparison_table(self):
        """创建模型比较表"""
        print("\n" + "=" * 60)
        print("测试集模型性能比较")
        print("=" * 60)
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        print(comparison_df.to_string())
        
        return comparison_df
    
    def plot_roc_curves(self, models_dict, X_test, y_test, threshold=0.5):
        """绘制ROC曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC曲线
        ax1 = axes[0]
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        ax1.set_xlabel('假正率 (FPR)', fontsize=12)
        ax1.set_ylabel('真正率 (TPR)', fontsize=12)
        ax1.set_title('ROC曲线', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # PR曲线
        ax2 = axes[1]
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                ax2.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
        
        ax2.set_xlabel('召回率 (Recall)', fontsize=12)
        ax2.set_ylabel('精确率 (Precision)', fontsize=12)
        ax2.set_title('Precision-Recall曲线', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  ROC和PR曲线已保存: roc_pr_curves.png")
    
    def plot_feature_importance(self, model, feature_names, top_n=20, model_name='model'):
        """绘制特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[-top_n:]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('特征重要性')
            plt.title(f'Top {top_n} 重要特征 ({model_name})')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  特征重要性图已保存: feature_importance_{model_name}.png")
    
    def plot_overfitting_analysis(self, cv_results, test_results):
        """绘制过拟合分析图"""
        models = list(cv_results.keys())
        models = [m for m in models if not m.endswith('_std')]
        
        cv_aucs = [cv_results[m]['auc'] for m in models]
        test_aucs = [test_results[m]['auc'] for m in models if m in test_results]
        
        if len(cv_aucs) != len(test_aucs):
            print("  警告：CV和测试结果数量不匹配")
            return
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cv_aucs, width, label='时序CV AUC', color='steelblue')
        bars2 = ax.bar(x + width/2, test_aucs, width, label='测试集 AUC', color='coral')
        
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('过拟合分析：时序CV vs 测试集', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  过拟合分析图已保存: overfitting_analysis.png")


# ============================================================
# 第六部分：模型保存与加载
# ============================================================

class ModelSaver:
    """模型保存器"""
    
    def __init__(self, save_dir='./trained_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model, model_name, feature_names, scaler, threshold, metadata=None):
        """保存模型及相关组件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(self.save_dir, f'{model_name}_{timestamp}.joblib')
        joblib.dump(model, model_path)
        print(f"  ✓ 模型已保存: {model_path}")
        
        # 保存特征名称
        feature_path = os.path.join(self.save_dir, f'feature_names_{timestamp}.joblib')
        joblib.dump(feature_names, feature_path)
        print(f"  ✓ 特征名称已保存: {feature_path}")
        
        # 保存标准化器
        scaler_path = os.path.join(self.save_dir, f'scaler_{timestamp}.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"  ✓ 标准化器已保存: {scaler_path}")
        
        # 保存阈值和元数据
        metadata_path = os.path.join(self.save_dir, f'metadata_{timestamp}.joblib')
        metadata_save = {
            'model_name': model_name,
            'threshold': threshold,
            'feature_names': feature_names,
            'timestamp': timestamp,
        }
        if metadata:
            metadata_save.update(metadata)
        joblib.dump(metadata_save, metadata_path)
        print(f"  ✓ 元数据已保存: {metadata_path}")
        
        return {
            'model_path': model_path,
            'feature_path': feature_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }


# ============================================================
# 第七部分：预测器（用于未来预测）
# ============================================================

class FinancialCrisisPredictor:
    """财务危机预测器"""
    
    def __init__(self, model_dir='./trained_models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        self.metadata = None
    
    def load_latest_model(self, model_name='xgboost'):
        """加载最新的模型"""
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith(model_name)]
        if not model_files:
            print(f"  错误：未找到 {model_name} 模型文件")
            return False
        
        # 按时间排序
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        timestamp = latest_model.split('_')[-1].replace('.joblib', '')
        
        # 加载模型
        model_path = os.path.join(self.model_dir, latest_model)
        self.model = joblib.load(model_path)
        print(f"  ✓ 加载模型: {model_path}")
        
        # 加载特征名称
        feature_path = os.path.join(self.model_dir, f'feature_names_{timestamp}.joblib')
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)
            print(f"  ✓ 加载特征名称: {len(self.feature_names)} 个特征")
        
        # 加载标准化器
        scaler_path = os.path.join(self.model_dir, f'scaler_{timestamp}.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ 加载标准化器")
        
        # 加载元数据
        metadata_path = os.path.join(self.model_dir, f'metadata_{timestamp}.joblib')
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            self.threshold = self.metadata.get('threshold', 0.5)
            print(f"  ✓ 加载元数据（阈值: {self.threshold:.2f}）")
        
        return True
    
    def predict(self, data):
        """
        预测财务危机风险
        
        Parameters:
        -----------
        data : DataFrame or dict
            包含财务指标的数据
            
        Returns:
        --------
        dict : 预测结果
        """
        if self.model is None:
            print("  错误：模型未加载，请先调用 load_latest_model()")
            return None
        
        # 转换为DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 确保特征顺序一致
        if self.feature_names:
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                print(f"  警告：缺少特征 {missing_features}")
                # 用0填充缺失特征
                for feat in missing_features:
                    data[feat] = 0
            data = data[self.feature_names]
        
        # 标准化
        if self.scaler:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data
        
        # 预测
        y_prob = self.model.predict_proba(data_scaled)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # 构建结果
        results = []
        for i in range(len(data)):
            result = {
                'risk_probability': float(y_prob[i]),
                'risk_score': float(y_prob[i]) * 100,
                'prediction': int(y_pred[i]),
                'risk_level': '高风险' if y_pred[i] == 1 else '低风险',
                'threshold_used': self.threshold
            }
            results.append(result)
        
        return results


# ============================================================
# 第八部分：主程序
# ============================================================

def main(data_path='./data', start_year=2003, end_year=2023, save_models=True):
    """
    主程序
    
    Parameters:
    -----------
    data_path : str
        CSMAR数据文件路径
    start_year : int
        开始年份
    end_year : int
        结束年份
    save_models : bool
        是否保存模型
    """
    print("=" * 60)
    print("中国上市公司财务危机预测系统（精简版V2）")
    print("特点：时序交叉验证 + 过拟合检测 + 阈值优化 + 模型保存")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n1. 数据加载")
    loader = CSMARDataLoader(data_path=data_path)
    data_dict = loader.load_all_data()
    
    # 2. 数据预处理
    print("\n2. 数据预处理")
    processor = CSMARDataProcessor(data_dict)
    merged_data = processor.merge_all_data(start_year, end_year)
    
    if merged_data is None:
        print("错误：数据处理失败")
        return None
    
    # 3. 特征工程
    print("\n3. 特征工程")
    feature_engineer = CSMARFeatureEngineer(merged_data)
    featured_data = feature_engineer.calculate_financial_ratios()
    featured_data = feature_engineer.handle_missing_values(method='median')
    featured_data = feature_engineer.normalize_features(method='standard')
    
    X, y = feature_engineer.get_feature_matrix(target_col='is_st')
    
    if y is None:
        print("错误：缺少目标变量")
        return None
    
    # 4. 时序数据划分（按时间顺序）
    print("\n4. 时序数据划分")
    # 按时间排序
    if 'year' in featured_data.columns:
        sort_idx = featured_data.sort_values('year').index
        X = X.loc[sort_idx]
        y = y.loc[sort_idx]
    
    # 按80/20划分（前80%训练，后20%测试）
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集ST占比: {y_train.mean()*100:.2f}%")
    print(f"测试集ST占比: {y_test.mean()*100:.2f}%")
    
    # 5. 模型训练（时序交叉验证）
    print("\n5. 模型训练（时序交叉验证）")
    trainer = TimeSeriesModelTrainer(X_train, y_train, n_splits=5)
    
    # 训练模型
    trainer.train_logistic_regression()
    trainer.train_random_forest(n_estimators=200, max_depth=10)
    trainer.train_xgboost(n_estimators=200, max_depth=6, learning_rate=0.05)
    
    # 获取交叉验证比较表
    cv_comparison = trainer.get_cv_comparison_table()
    
    # 6. 寻找最优阈值
    print("\n6. 寻找最优预测阈值")
    best_threshold, threshold_df = trainer.find_optimal_threshold('xgboost')
    
    # 7. 过拟合验证
    print("\n7. 过拟合验证")
    overfit_results = {}
    for model_name in ['logistic', 'random_forest', 'xgboost']:
        overfit_results[model_name] = trainer.evaluate_overfitting(
            X_test, y_test, model_name
        )
    
    # 8. 测试集评估
    print("\n8. 测试集评估")
    evaluator = ModelEvaluator()
    
    for model_name, model in trainer.models.items():
        evaluator.evaluate_model(model, X_test, y_test, model_name, threshold=best_threshold)
    
    test_comparison = evaluator.create_comparison_table()
    
    # 9. 可视化
    print("\n9. 可视化")
    evaluator.plot_roc_curves(trainer.models, X_test, y_test, threshold=best_threshold)
    
    if 'xgboost' in trainer.models:
        evaluator.plot_feature_importance(
            trainer.models['xgboost'],
            feature_engineer.feature_names,
            model_name='xgboost'
        )
    
    evaluator.plot_overfitting_analysis(trainer.cv_results, evaluator.results)
    
    # 10. 保存模型
    if save_models:
        print("\n10. 保存模型")
        saver = ModelSaver(save_dir='./trained_models')
        
        # 保存XGBoost模型（表现最好）
        if 'xgboost' in trainer.models:
            saver.save_model(
                trainer.models['xgboost'],
                'xgboost',
                feature_engineer.feature_names,
                feature_engineer.scaler,
                best_threshold,
                metadata={
                    'cv_auc': trainer.cv_results.get('xgboost', {}).get('auc'),
                    'test_auc': evaluator.results.get('xgboost', {}).get('auc'),
                }
            )
        
        # 保存随机森林模型
        if 'random_forest' in trainer.models:
            saver.save_model(
                trainer.models['random_forest'],
                'random_forest',
                feature_engineer.feature_names,
                feature_engineer.scaler,
                best_threshold,
                metadata={
                    'cv_auc': trainer.cv_results.get('random_forest', {}).get('auc'),
                    'test_auc': evaluator.results.get('random_forest', {}).get('auc'),
                }
            )
    
    # 11. 保存结果
    print("\n11. 保存结果")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cv_comparison.to_csv(f'cv_comparison_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"  交叉验证结果已保存: cv_comparison_{timestamp}.csv")
    
    test_comparison.to_csv(f'test_comparison_{timestamp}.csv', encoding='utf-8-sig')
    print(f"  测试集结果已保存: test_comparison_{timestamp}.csv")
    
    threshold_df.to_csv(f'threshold_analysis_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"  阈值分析已保存: threshold_analysis_{timestamp}.csv")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 打印使用说明
    print("\n" + "=" * 60)
    print("模型使用说明")
    print("=" * 60)
    print("""
1. 模型已保存在 ./trained_models/ 目录

2. 使用预测器进行预测：

   from financial_crisis_prediction_lite_v2 import FinancialCrisisPredictor
   
   # 创建预测器
   predictor = FinancialCrisisPredictor('./trained_models')
   
   # 加载最新模型
   predictor.load_latest_model('xgboost')
   
   # 准备数据（需要包含所有特征）
   company_data = {
       'ROA': 0.05,
       'ROE': 0.10,
       'debt_ratio': 0.45,
       # ... 其他特征
   }
   
   # 预测
   results = predictor.predict(company_data)
   print(results)
""")
    
    return cv_comparison


if __name__ == '__main__':
    main()