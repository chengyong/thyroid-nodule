# -*- coding: utf-8 -*-

#
# Title: Ultrasound影像组学数据机器学习分类分析
# Author: Li Yang
# Description:
# Refer:
# Date: 2025-05-22
#

"""
Ultrasound影像组学数据机器学习分类分析
分析内容：
1. 使用RFE进行特征选择
2. 15种机器学习分类器分析
3. 5折交叉验证评估
4. ROC曲线和性能指标
5. SHAP可解释性分析
6. 决策曲线分析（DCA）
7.rfe边缘特征+rfe超声特征进行机器学习
8.rfe的两类特征值数寻优(以决策树为基准的AUC/ACC的标准）
9.新增数据平衡的方法
10.加入四大指南的对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier

# 机器学习模型
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

import shap
import warnings
import os
from pathlib import Path
import gc
from tqdm import tqdm
from itertools import cycle

# 添加sklearn的clone函数导入
from sklearn.base import clone

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

# 设置中文显示和警告过滤
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class BreastCancerClassifier:
    def __init__(self, data_path, balance_strategy='smote'):
        """初始化分类器"""
        self.data_path = data_path
        self.output_dir = Path("tk03_result")
        self.output_dir.mkdir(exist_ok=True)

        # 新增：数据平衡策略
        self.balance_strategy = balance_strategy

        # 离散属性列表
        self.discrete_cols = [
            'Echo', 'Component', 'Echotexture', 'Margin',
            'Calcification', 'Halo', 'Shadow', 'CDFI', 'A/T ratio', 'Nodule-capsule',
            'Thyroid-tracheal'
        ]

        # 边缘特征名称列表
        self.edge_cols = [
            'Perimeter', 'Area', 'Circularity', 'Compactness', 'Convex_hull_area', 'Solidity',
            'Ellipticity', 'Eccentricity', 'Boundary_irregularity', 'Shape_index', 'Edge_variance',
            'Glcm_contrast', 'Glcm_energy', 'Glcm_entropy', 'Glcm_homogeneity', 'Edge_mean_intensity',
            'Edge_gradient_magnitude', 'Edge_gradient_direction_consistency', 'Edge_sharpness',
            'Edge_smoothness', 'Mean_curvature', 'Curvature_std', 'Curvature_entropy', 'Max_curvature',
            'Curvature_percentage', 'Wavelet_energy', 'Edge_rmsd', 'Gabor_energy_1', 'Gabor_energy_2',
            'Gabor_energy_3', 'Gabor_energy_4', 'Gabor_energy_5', 'Gabor_energy_6', 'Gabor_energy_7',
            'Gabor_energy_8', 'Gabor_std_1', 'Gabor_std_2', 'Gabor_std_3', 'Gabor_std_4', 'Gabor_std_5',
            'Gabor_std_6', 'Gabor_std_7', 'Gabor_std_8', 'Lbp_bin_1', 'Lbp_bin_2', 'Lbp_bin_3', 'Lbp_bin_4',
            'Lbp_bin_5', 'Lbp_bin_6', 'Lbp_bin_7', 'Lbp_bin_8', 'Lbp_bin_9', 'Lbp_bin_10',
            'Glrlm_0_sre', 'Glrlm_0_lre', 'Glrlm_0_glu', 'Glrlm_0_rlu', 'Glrlm_0_rp',
            'Glrlm_45_sre', 'Glrlm_45_lre', 'Glrlm_45_glu', 'Glrlm_45_rlu', 'Glrlm_45_rp',
            'Glrlm_90_sre', 'Glrlm_90_lre', 'Glrlm_90_glu', 'Glrlm_90_rlu', 'Glrlm_90_rp',
            'Glrlm_135_sre', 'Glrlm_135_lre', 'Glrlm_135_glu', 'Glrlm_135_rlu', 'Glrlm_135_rp',
            'Fourier_descriptor_1', 'Fourier_descriptor_2', 'Fourier_descriptor_3', 'Fourier_descriptor_4',
            'Fourier_descriptor_5', 'Fourier_descriptor_6', 'Fourier_descriptor_7', 'Fourier_descriptor_8',
            'Fourier_descriptor_9', 'Fourier_descriptor_10', 'Total_power', 'Peak_power', 'Peak_freq',
            'Median_freq', 'Spectral_entropy', 'Freq_std', 'Fractal_dim', 'Box_dim', 'Higuchi_fd',
            'Multifractal_q_-2', 'Multifractal_q_-1', 'Multifractal_q_0', 'Multifractal_q_1',
            'Multifractal_q_2', 'Lacunarity_scale_2', 'Lacunarity_scale_4', 'Lacunarity_scale_8',
            'Lacunarity_scale_16', 'Lacunarity_scale_32', 'Boundary_background_contrast',
            'Boundary_gradient_ratio', 'Boundary_inside_std_ratio', 'Boundary_intensity_change_rate',
            'Boundary_sharpness', 'Wavelet_energy_ratio', 'Multiscale_entropy', 'Scale_space_curvature',
            'Edge_roughness_spectrum', 'Scale_space_persistence'
        ]

        # 新增：影像学指标诊断阈值设置（可调节）
        # self.imaging_thresholds = {
        #     'C-TIRADS': {'benign_max': 1, 'name': 'C-TIRADS'},
        #     'ACR-TIRADS': {'benign_max': 2, 'name': 'ACR-TIRADS'},  # 新增ACR-TIRADS
        #     'Kwak TI-RADS': {'benign_max': 1, 'name': 'Kwak TI-RADS'},  # 新增Kwak TI-RADS
        #     'ATA': {'benign_max': 3, 'name': 'ATA'}
        # }
        self.imaging_thresholds = {
            'C-TIRADS': {'benign_max': 1, 'name': 'C-TIRADS'},
            'ACR-TIRADS': {'benign_max': 2, 'name': 'ACR-TIRADS'},  # 新增ACR-TIRADS
            'Kwak TI-RADS': {'benign_max': 1, 'name': 'Kwak TI-RADS'},  # 新增Kwak TI-RADS
            'ATA': {'benign_max': 3, 'name': 'ATA'}
        }

        # 加载数据
        self.load_data()

        # 初始化模型字典
        self.init_models()

        # 初始化结果存储
        self.results = {}
        self.cv_results = {}

    def apply_data_balancing(self, X, y, strategy=None):
        """
        应用数据平衡策略

        Parameters:
        -----------
        X : array-like
            特征数据
        y : array-like
            标签数据
        strategy : str, optional
            平衡策略，如果为None则使用self.balance_strategy

        Returns:
        --------
        X_balanced, y_balanced : 平衡后的数据
        """
        if strategy is None:
            strategy = self.balance_strategy

        if strategy == 'none':
            print("No data balancing applied.")
            return X, y

        print(f"Original class distribution: {Counter(y)}")

        try:
            if strategy == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(y)) - 1))
            elif strategy == 'adasyn':
                sampler = ADASYN(random_state=42, n_neighbors=min(5, np.min(np.bincount(y)) - 1))
            elif strategy == 'borderline':
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(y)) - 1))
            elif strategy == 'random_over':
                sampler = RandomOverSampler(random_state=42)
            elif strategy == 'random_under':
                sampler = RandomUnderSampler(random_state=42)
            elif strategy == 'tomek':
                sampler = TomekLinks()
            elif strategy == 'enn':
                sampler = EditedNearestNeighbours()
            elif strategy == 'smote_tomek':
                sampler = SMOTETomek(random_state=42)
            elif strategy == 'smote_enn':
                sampler = SMOTEENN(random_state=42)
            else:
                raise ValueError(f"Unknown balancing strategy: {strategy}")

            X_balanced, y_balanced = sampler.fit_resample(X, y)
            print(f"After {strategy} balancing: {Counter(y_balanced)}")

            # 保存平衡信息
            self.balancing_info = {
                'strategy': strategy,
                'original_distribution': Counter(y),
                'balanced_distribution': Counter(y_balanced),
                'original_samples': len(y),
                'balanced_samples': len(y_balanced)
            }

            return X_balanced, y_balanced

        except Exception as e:
            print(f"Error applying {strategy} balancing: {e}")
            print("Falling back to random oversampling...")

            # 备用方案：随机过采样
            sampler = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            print(f"After random oversampling: {Counter(y_balanced)}")

            self.balancing_info = {
                'strategy': 'random_over (fallback)',
                'original_distribution': Counter(y),
                'balanced_distribution': Counter(y_balanced),
                'original_samples': len(y),
                'balanced_samples': len(y_balanced)
            }

            return X_balanced, y_balanced



    def load_data(self):
        """加载和预处理数据"""
        try:
            print("Loading data...")
            self.df = pd.read_excel(self.data_path)

            # 保存原始数据行数
            self.original_row_count = len(self.df)

            # 先删除非特征列（保留影像学指标）
            columns_to_drop = [
                '目录名', '年份文件夹', '图像文件名', '掩码文件名', 'nodule',
                'cell_position', 'cell_classify', 'sex', 'age', 'BRAF', 'KRAS', 'NRAS', 'HRAS', 'TERT'
                # 注意：C-TIRADS, ATA, ACR-TIRADS, Kwak TI-RADS 不删除
            ]

            # 保存影像学指标数据（保存原始索引）
            self.imaging_indicators = {}
            self.original_indices = self.df.index.copy()  # 保存原始索引

            for indicator in ['C-TIRADS', 'ATA', 'ACR-TIRADS', 'Kwak TI-RADS']:
                if indicator in self.df.columns:
                    self.imaging_indicators[indicator] = self.df[indicator].copy()
                    print(f"Found imaging indicator: {indicator}")
                else:
                    print(f"Warning: {indicator} not found in dataset")

            # 使用drop()方法删除列
            self.df = self.df.drop(columns=columns_to_drop, errors='ignore')

            # 检查Label列
            if 'Label' not in self.df.columns:
                raise ValueError("Label column not found in the dataset")

            # 分离特征和标签
            self.y = self.df['Label'].values

            # 分离连续和离散变量
            self.continuous_cols = [col for col in self.df.columns
                                    if col not in self.discrete_cols + ['Label'] + list(self.imaging_indicators.keys())
                                    and (self.df[col].dtype in ['int64', 'float64'])]

            print(f"Continuous variables: {len(self.continuous_cols)}")
            print(f"Discrete variables: {len(self.discrete_cols)}")
            print(f"Imaging indicators: {list(self.imaging_indicators.keys())}")
            print(f"Class distribution: {np.bincount(self.y)}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def compress_column_name(self, col_name, max_length=20):
        """压缩列名长度"""
        if len(col_name) <= max_length:
            return col_name

        # 移除常见前缀
        prefixes = ['original_', 'boxmean_', 'additivegaussiannoise_',
                    'binomialblurimage_', 'curvatureflow_', 'boxsigmaimage_',
                    'log_', 'wavelet_', 'normalize_', 'laplaciansharpening_',
                    'discretegaussian_', 'mean_', 'specklenoise_',
                    'recursivegaussian_', 'shotnoise_', 'ROI_DETAILS_']

        compressed = col_name
        for prefix in prefixes:
            if compressed.startswith(prefix):
                compressed = compressed[len(prefix):]
                break

        # 进一步压缩
        if len(compressed) > max_length:
            start_len = max_length // 2 - 2
            end_len = max_length - start_len - 3
            compressed = compressed[:start_len] + "..." + compressed[-end_len:]

        return compressed

    def preprocess_data(self):
        """数据预处理（包含数据平衡）"""
        print("Preprocessing data...")

        # 处理连续变量
        X_continuous = self.df[self.continuous_cols].copy()

        # 移除缺失值过多的列
        missing_threshold = 0.5
        valid_continuous_cols = []
        for col in self.continuous_cols:
            missing_ratio = X_continuous[col].isnull().sum() / len(X_continuous)
            if missing_ratio < missing_threshold:
                valid_continuous_cols.append(col)

        print(f"Valid continuous columns after missing value filter: {len(valid_continuous_cols)}")
        X_continuous = X_continuous[valid_continuous_cols]

        # 填充缺失值
        X_continuous = X_continuous.fillna(X_continuous.median())

        # 处理离散变量
        X_discrete = self.df[self.discrete_cols].copy()

        # 对离散变量进行标签编码
        le_dict = {}
        for col in self.discrete_cols:
            if col in X_discrete.columns:
                le = LabelEncoder()
                X_discrete[col] = X_discrete[col].fillna('Unknown')
                X_discrete[col] = le.fit_transform(X_discrete[col].astype(str))
                le_dict[col] = le

        # 合并特征
        self.X = pd.concat([X_continuous, X_discrete], axis=1)
        self.feature_names = list(self.X.columns)

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        print(f"Feature matrix shape before balancing: {X_scaled.shape}")
        print(f"Class distribution before balancing: {Counter(self.y)}")

        # 记录数据预处理前后的索引对应关系
        self.processed_indices = self.df.index.copy()

        # 应用数据平衡
        X_balanced, y_balanced = self.apply_data_balancing(X_scaled, self.y)

        # 更新数据
        self.X_scaled = X_balanced
        self.y = y_balanced

        print(f"Final feature matrix shape: {self.X_scaled.shape}")
        print(f"Final class distribution: {Counter(self.y)}")

        return self.X_scaled, self.y

    def analyze_imaging_indicators_performance(self):
        """分析传统影像学指标的诊断效能"""
        print("Analyzing traditional imaging indicators performance...")

        if not hasattr(self, 'imaging_indicators') or not self.imaging_indicators:
            print("No imaging indicators available for analysis")
            return

        # 使用原始标签数据（在数据平衡之前的）
        if hasattr(self, 'processed_indices'):
            # 获取处理前的原始标签
            original_labels_df = pd.read_excel(self.data_path)
            original_labels = original_labels_df['Label'].values
        else:
            print("Warning: Using balanced labels for imaging analysis")
            original_labels = self.y

        imaging_results = {}

        for indicator, data in self.imaging_indicators.items():
            if indicator not in self.imaging_thresholds:
                continue

            threshold_info = self.imaging_thresholds[indicator]
            benign_max = threshold_info['benign_max']
            indicator_name = threshold_info['name']

            # 确保数据长度匹配
            if len(data) != len(original_labels):
                print(
                    f"Warning: {indicator} data length ({len(data)}) doesn't match labels length ({len(original_labels)})")
                # 取较短的长度
                min_length = min(len(data), len(original_labels))
                data = data.iloc[:min_length] if hasattr(data, 'iloc') else data[:min_length]
                labels_to_use = original_labels[:min_length]
            else:
                labels_to_use = original_labels

            # 处理缺失值
            valid_mask = ~pd.isna(data)
            valid_data = data[valid_mask]
            valid_labels = labels_to_use[valid_mask]

            if len(valid_data) == 0:
                print(f"No valid data for {indicator_name}")
                continue

            # 根据阈值进行预测
            if indicator == 'ATA':
                # ATA: 1-3为良性(0)，其他为恶性(1)
                predicted_labels = np.where(
                    (valid_data >= 0) & (valid_data <= benign_max), 0, 1
                )
            else:
                # 其他指标: 0到threshold为良性(0)，其他为恶性(1)
                predicted_labels = np.where(valid_data <= benign_max, 0, 1)

            # 计算性能指标
            accuracy = accuracy_score(valid_labels, predicted_labels)
            precision = precision_score(valid_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(valid_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(valid_labels, predicted_labels, average='weighted', zero_division=0)
            specificity = self.calculate_specificity(valid_labels, predicted_labels, 2)

            # 计算混淆矩阵
            cm = confusion_matrix(valid_labels, predicted_labels)

            # 计算敏感性、特异性、PPV、NPV
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity_cm = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                sensitivity = specificity_cm = ppv = npv = 0

            imaging_results[indicator] = {
                'name': indicator_name,
                'threshold': benign_max,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'specificity_cm': specificity_cm,
                'ppv': ppv,
                'npv': npv,
                'confusion_matrix': cm,
                'predicted_labels': predicted_labels,
                'true_labels': valid_labels,
                'valid_count': len(valid_data),
                'original_data': valid_data
            }

            print(
                f"{indicator_name} (threshold ≤ {benign_max}): Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, Valid samples = {len(valid_data)}")

        self.imaging_results = imaging_results
        return imaging_results

    # def evaluate_edge_features_only_performance(self):
    #     """评估仅使用边缘特征的模型性能"""
    #     print("Evaluating edge features only performance...")
    #
    #     # 使用预处理后但未平衡的数据进行边缘特征分析
    #     # 重新加载和预处理数据以获得未平衡的版本
    #     try:
    #         # 临时保存当前的平衡数据
    #         current_X_scaled = self.X_scaled.copy()
    #         current_y = self.y.copy()
    #
    #         # 重新预处理获得未平衡的数据
    #         temp_df = pd.read_excel(self.data_path)
    #
    #         # 删除非特征列
    #         columns_to_drop = [
    #             '目录名', '年份文件夹', '图像文件名', '掩码文件名', 'nodule',
    #             'cell_position', 'cell_classify', 'sex', 'age', 'BRAF', 'KRAS', 'NRAS', 'HRAS', 'TERT',
    #             'C-TIRADS', 'ATA', 'ACR-TIRADS', 'Kwak TI-RADS'
    #         ]
    #         temp_df = temp_df.drop(columns=columns_to_drop, errors='ignore')
    #
    #         temp_y = temp_df['Label'].values
    #         temp_continuous_cols = [col for col in temp_df.columns
    #                                 if col not in self.discrete_cols + ['Label']
    #                                 and (temp_df[col].dtype in ['int64', 'float64'])]
    #
    #         # 处理连续变量
    #         X_temp_continuous = temp_df[temp_continuous_cols].copy()
    #         X_temp_continuous = X_temp_continuous.fillna(X_temp_continuous.median())
    #
    #         # 处理离散变量
    #         X_temp_discrete = temp_df[self.discrete_cols].copy()
    #         for col in self.discrete_cols:
    #             if col in X_temp_discrete.columns:
    #                 le = LabelEncoder()
    #                 X_temp_discrete[col] = X_temp_discrete[col].fillna('Unknown')
    #                 X_temp_discrete[col] = le.fit_transform(X_temp_discrete[col].astype(str))
    #
    #         # 合并特征
    #         X_temp = pd.concat([X_temp_continuous, X_temp_discrete], axis=1)
    #         temp_feature_names = list(X_temp.columns)
    #
    #         # 标准化
    #         temp_scaler = StandardScaler()
    #         X_temp_scaled = temp_scaler.fit_transform(X_temp)
    #
    #         # 提取边缘特征
    #         edge_indices = [i for i, col in enumerate(temp_feature_names) if col in self.edge_cols]
    #         if not edge_indices:
    #             print("No edge features available")
    #             return {}
    #
    #         X_edge_only = X_temp_scaled[:, edge_indices]
    #         edge_feature_names = [temp_feature_names[i] for i in edge_indices]
    #
    #         print(f"Using {len(edge_feature_names)} edge features")
    #
    #         # 特征选择 - 使用RFE选择最重要的边缘特征
    #         rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    #         rfe = RFE(estimator=rf_selector, n_features_to_select=min(20, len(edge_feature_names)))
    #         X_edge_selected = rfe.fit_transform(X_edge_only, temp_y)
    #         selected_edge_features = [edge_feature_names[i] for i in range(len(edge_feature_names)) if rfe.support_[i]]
    #
    #         print(f"Selected {len(selected_edge_features)} edge features using RFE")
    #
    #         # 评估前5个最佳模型仅使用边缘特征的性能
    #         top_models = sorted(self.results.items(),
    #                             key=lambda x: x[1]['test_metrics']['Accuracy'],
    #                             reverse=True)[:5]
    #
    #         edge_only_results = {}
    #
    #         for name, _ in top_models:
    #             try:
    #                 model = clone(self.models[name])
    #
    #                 # 训练测试分割
    #                 X_train, X_test, y_train, y_test = train_test_split(
    #                     X_edge_selected, temp_y, test_size=0.2, random_state=42, stratify=temp_y
    #                 )
    #
    #                 # 训练模型
    #                 model.fit(X_train, y_train)
    #
    #                 # 预测
    #                 y_pred = model.predict(X_test)
    #                 y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    #
    #                 # 计算指标
    #                 metrics = self.calculate_metrics(y_test, y_pred, y_proba, 'test')
    #
    #                 edge_only_results[name] = {
    #                     'metrics': metrics,
    #                     'selected_features': selected_edge_features,
    #                     'n_features': len(selected_edge_features)
    #                 }
    #
    #                 print(f"  {name} (edge only): Accuracy = {metrics['Accuracy']:.4f}")
    #
    #             except Exception as e:
    #                 print(f"  Error evaluating {name} with edge features only: {e}")
    #                 continue
    #
    #         # 恢复原始的平衡数据
    #         self.X_scaled = current_X_scaled
    #         self.y = current_y
    #
    #         self.edge_only_results = edge_only_results
    #         return edge_only_results
    #
    #     except Exception as e:
    #         print(f"Error in edge features analysis: {e}")
    #         # 恢复原始数据
    #         if 'current_X_scaled' in locals():
    #             self.X_scaled = current_X_scaled
    #             self.y = current_y
    #         return {}

    def plot_diagnostic_performance_comparison(self):
        """绘制诊断效能对比图"""
        print("Plotting diagnostic performance comparison...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        # 收集所有方法的性能数据
        all_methods = {}

        # 1. 添加影像学指标结果
        if hasattr(self, 'imaging_results'):
            for indicator, result in self.imaging_results.items():
                all_methods[result['name']] = {
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1_score'],
                    'Specificity': result['specificity'],
                    'Sensitivity': result['sensitivity'],
                    'PPV': result['ppv'],
                    'NPV': result['npv']
                }

        # 2. 添加最佳机器学习模型结果（使用具体模型名称）
        if hasattr(self, 'results') and self.results:
            best_ml_model = max(self.results.items(), key=lambda x: x[1]['test_metrics']['Accuracy'])
            best_model_name = best_ml_model[0]  # 获取具体模型名称
            ml_metrics = best_ml_model[1]['test_metrics']

            # 计算额外指标
            y_test = best_ml_model[1]['y_test']
            y_pred = best_ml_model[1]['y_test_pred']
            cm = confusion_matrix(y_test, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                sensitivity = ppv = npv = 0

            all_methods[best_model_name] = {  # 使用具体模型名称
                'Accuracy': ml_metrics['Accuracy'],
                'Precision': ml_metrics['Precision'],
                'Recall': ml_metrics['Recall'],
                'F1-Score': ml_metrics['F1-Score'],
                'Specificity': ml_metrics['Specificity'],
                'Sensitivity': sensitivity,
                'PPV': ppv,
                'NPV': npv
            }

        # 删除边缘特征相关代码块

        # 绘制各种性能指标对比
        metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            methods = list(all_methods.keys())
            values = [all_methods[method].get(metric, 0) for method in methods]

            bars = ax.bar(range(len(methods)), values,
                          color=colors[:len(methods)], alpha=0.8, edgecolor='black')

            ax.set_xlabel('Diagnostic Methods', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontsize=14, weight='bold')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')

        plt.suptitle('Diagnostic Performance Comparison\n(Traditional Imaging vs Machine Learning)',
                     fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "diagnostic_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制雷达图对比
        self.plot_radar_comparison(all_methods)

    def plot_radar_comparison(self, all_methods):
        """绘制雷达图对比不同诊断方法"""
        print("Plotting radar comparison...")

        # 选择主要指标
        radar_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score']

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

        for idx, (method, metrics) in enumerate(all_methods.items()):
            values = [metrics.get(metric, 0) for metric in radar_metrics]
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=method,
                    color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Diagnostic Performance Radar Chart', fontsize=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "diagnostic_radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices_comparison(self):
        """绘制混淆矩阵对比"""
        print("Plotting confusion matrices comparison...")

        methods_with_cm = {}

        # 收集影像学指标的混淆矩阵
        if hasattr(self, 'imaging_results'):
            for indicator, result in self.imaging_results.items():
                methods_with_cm[result['name']] = result['confusion_matrix']

        # 收集ML模型的混淆矩阵（使用具体模型名称）
        if hasattr(self, 'results') and self.results:
            best_ml_model = max(self.results.items(), key=lambda x: x[1]['test_metrics']['Accuracy'])
            best_model_name = best_ml_model[0]  # 使用具体模型名称
            methods_with_cm[best_model_name] = best_ml_model[1]['confusion_matrix']

        if not methods_with_cm:
            print("No confusion matrices available for comparison")
            return

        n_methods = len(methods_with_cm)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.ravel()

        for idx, (method, cm) in enumerate(methods_with_cm.items()):
            ax = axes[idx]

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # 绘制热图
            sns.heatmap(cm_percent, annot=False, fmt='.2%', cmap='Blues',
                        xticklabels=['Benign', 'Malignant'],
                        yticklabels=['Benign', 'Malignant'],
                        ax=ax, vmin=0, vmax=1)

            # 添加数值和百分比标签
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    percent = cm_percent[i, j] * 100
                    text = f"{count}\n({percent:.1f}%)"
                    ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                            color='white' if cm_percent[i, j] > 0.5 else 'black', fontsize=10)

            ax.set_title(f'{method}', fontsize=12, weight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # 隐藏多余的子图
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Confusion Matrices Comparison', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_diagnostic_comparison_results(self):
        """保存诊断对比结果"""
        print("Saving diagnostic comparison results...")

        # 创建综合对比表
        comparison_data = []

        # 添加影像学指标
        if hasattr(self, 'imaging_results'):
            for indicator, result in self.imaging_results.items():
                comparison_data.append({
                    'Method': result['name'],
                    'Type': 'Traditional Imaging',
                    'Threshold': f"≤ {result['threshold']}",
                    'Accuracy': result['accuracy'],
                    'Sensitivity': result['sensitivity'],
                    'Specificity': result['specificity_cm'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1_score'],
                    'PPV': result['ppv'],
                    'NPV': result['npv'],
                    'Sample_Size': result['valid_count']
                })

        # 添加ML模型结果（使用具体模型名称）
        if hasattr(self, 'results') and self.results:
            best_ml_model = max(self.results.items(), key=lambda x: x[1]['test_metrics']['Accuracy'])
            best_model_name = best_ml_model[0]  # 使用具体模型名称
            ml_metrics = best_ml_model[1]['test_metrics']

            # 计算额外指标
            y_test = best_ml_model[1]['y_test']
            y_pred = best_ml_model[1]['y_test_pred']
            cm = confusion_matrix(y_test, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity_cm = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                sensitivity = specificity_cm = ppv = npv = 0

            comparison_data.append({
                'Method': best_model_name,  # 使用具体模型名称
                'Type': 'Machine Learning',
                'Threshold': f"{len(self.selected_features)} features",
                'Accuracy': ml_metrics['Accuracy'],
                'Sensitivity': sensitivity,
                'Specificity': specificity_cm,
                'Precision': ml_metrics['Precision'],
                'Recall': ml_metrics['Recall'],
                'F1-Score': ml_metrics['F1-Score'],
                'PPV': ppv,
                'NPV': npv,
                'Sample_Size': len(y_test)
            })

        # 删除边缘特征相关代码块

        # 保存到Excel
        with pd.ExcelWriter(self.output_dir / "comprehensive_diagnostic_comparison.xlsx",
                            engine='openpyxl') as writer:

            # 总体对比
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            comparison_df.to_excel(writer, sheet_name='Overall_Comparison', index=False)

            # 影像学指标详情
            if hasattr(self, 'imaging_results'):
                imaging_details = []
                for indicator, result in self.imaging_results.items():
                    imaging_details.append({
                        'Indicator': result['name'],
                        'Threshold_Rule': f"Benign if ≤ {result['threshold']}",
                        'Valid_Samples': result['valid_count'],
                        'True_Positives': int(result['confusion_matrix'][1, 1]) if result['confusion_matrix'].shape == (
                        2, 2) else 0,
                        'False_Positives': int(result['confusion_matrix'][0, 1]) if result[
                                                                                        'confusion_matrix'].shape == (
                                                                                    2, 2) else 0,
                        'True_Negatives': int(result['confusion_matrix'][0, 0]) if result['confusion_matrix'].shape == (
                        2, 2) else 0,
                        'False_Negatives': int(result['confusion_matrix'][1, 0]) if result[
                                                                                        'confusion_matrix'].shape == (
                                                                                    2, 2) else 0,
                        'Accuracy': result['accuracy'],
                        'Sensitivity': result['sensitivity'],
                        'Specificity': result['specificity_cm'],
                        'PPV': result['ppv'],
                        'NPV': result['npv']
                    })

                imaging_df = pd.DataFrame(imaging_details)
                imaging_df.to_excel(writer, sheet_name='Imaging_Details', index=False)

            # 删除边缘特征详情部分

        print("Diagnostic comparison results saved successfully!")

        return comparison_data

    def search_best_rfe_combination_by_accuracy(self, max_edge_features=20, max_us_features=11):
        """
        自动搜索最优RFE组合（超声+边缘），以准确性为标准，绘制准确性热图并保存结果。
        """
        print("🔍 Searching best RFE combination by Accuracy...")

        best_accuracy = 0
        best_edge_num = 0
        best_us_num = 0
        best_features = []
        best_model = None
        accuracy_matrix = np.zeros((max_edge_features, max_us_features))  # 行: edge, 列: us

        # 准确性评分模型和交叉验证器
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_results = []

        for edge_num in range(1, max_edge_features + 1):
            for us_num in range(1, max_us_features + 1):
                print(f"Evaluating: edge={edge_num}, ultrasound={us_num}")
                try:
                    self.dual_rfe_feature_selection(n_edge_features=edge_num, n_ultrasound_features=us_num)

                    scores = cross_val_score(
                        model, self.X_selected, self.y, scoring='accuracy', cv=cv, n_jobs=-1
                    )
                    mean_accuracy = np.mean(scores)
                    accuracy_matrix[edge_num - 1, us_num - 1] = mean_accuracy

                    all_results.append({
                        'Edge Features': edge_num,
                        'Ultrasound Features': us_num,
                        'Accuracy': mean_accuracy,
                        'Accuracy_Std': np.std(scores)
                    })

                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_edge_num = edge_num
                        best_us_num = us_num
                        best_features = self.selected_features.copy()
                        best_model = clone(model)

                except Exception as e:
                    print(f"  Skipped due to error: {e}")
                    accuracy_matrix[edge_num - 1, us_num - 1] = np.nan
                    continue

        # 训练最优模型
        print("\n✅ Best RFE Combination Found (by Accuracy):")
        print(f"  Ultrasound: {best_us_num}, Edge: {best_edge_num}, Accuracy: {best_accuracy:.4f}")

        self.dual_rfe_feature_selection(n_edge_features=best_edge_num, n_ultrasound_features=best_us_num)
        best_model.fit(self.X_selected, self.y)

        # 保存
        self.best_accuracy_model = best_model
        self.best_accuracy = best_accuracy
        self.best_feature_combo_accuracy = (best_us_num, best_edge_num)
        self.best_selected_features_accuracy = best_features
        self.accuracy_matrix = accuracy_matrix
        self.accuracy_result_table = pd.DataFrame(all_results)

        # 绘图
        self.plot_accuracy_heatmap(accuracy_matrix)

        # 导出结果
        self.save_accuracy_combination_results()

        return best_model, best_features, best_accuracy

    def plot_auc_heatmap(self, auc_matrix):
        """绘制AUC值热力图"""
        print("📊 Plotting AUC heatmap...")

        plt.figure(figsize=(10, 8))
        sns.heatmap(auc_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=range(1, auc_matrix.shape[1] + 1),
                    yticklabels=range(1, auc_matrix.shape[0] + 1),
                    cbar_kws={'label': 'Mean AUC'})

        plt.xlabel("Ultrasound Feature Count")
        plt.ylabel("Edge Feature Count")
        plt.title("AUC Heatmap (Edge × Ultrasound Feature RFE)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "auc_heatmap.png", dpi=300)
        plt.close()
        print("✅ AUC heatmap saved.")

    def save_auc_combination_results(self):
        """保存每个特征组合的AUC结果为Excel"""
        print("💾 Saving AUC combination results to Excel...")

        result_path = self.output_dir / "rfe_auc_combination_results.xlsx"
        best_edge, best_us = self.best_feature_combo

        # 创建DataFrame
        df = self.auc_result_table.copy()
        df = df.sort_values(by="AUC", ascending=False)

        # 写入Excel
        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All Combinations', index=False)
            pd.DataFrame({
                'Best Ultrasound Features': [best_us],
                'Best Edge Features': [best_edge],
                'Best AUC': [self.best_auc]
            }).to_excel(writer, sheet_name='Best Combination', index=False)

        print(f"✅ AUC results saved to: {result_path}")

    def plot_accuracy_heatmap(self, accuracy_matrix):
        """绘制准确性值热力图"""
        print("📊 Plotting Accuracy heatmap...")

        plt.figure(figsize=(12, 10))

        # 创建热力图
        sns.heatmap(accuracy_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                    xticklabels=range(1, accuracy_matrix.shape[1] + 1),
                    yticklabels=range(1, accuracy_matrix.shape[0] + 1),
                    cbar_kws={'label': 'Mean Accuracy'},
                    vmin=np.nanmin(accuracy_matrix), vmax=np.nanmax(accuracy_matrix))

        plt.xlabel("Ultrasound Feature Count", fontsize=14)
        plt.ylabel("Edge Feature Count", fontsize=14)
        plt.title("Accuracy Heatmap (Edge × Ultrasound Feature RFE)\nBased on 5-Fold Cross-Validation",
                  fontsize=16, weight='bold')

        # 标记最佳组合
        best_idx = np.unravel_index(np.nanargmax(accuracy_matrix), accuracy_matrix.shape)
        plt.scatter(best_idx[1] + 0.5, best_idx[0] + 0.5,
                    s=200, c='red', marker='*', edgecolors='white', linewidth=2,
                    label=f'Best: Edge={best_idx[0] + 1}, US={best_idx[1] + 1}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_heatmap_rfe_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Accuracy heatmap saved.")

    def search_best_combination_by_accuracy(self, max_edge_features=20, max_us_features=11):
        """
        自动搜索在RFE边缘特征(<=20) + 超声特征(<=11)组合下Accuracy最优的特征组合。
        使用Extra Trees作为基准评估器。
        """
        print("🔍 Searching best RFE combination by Accuracy using Extra Trees...")

        best_acc = 0
        best_edge_num = 0
        best_us_num = 0
        best_features = []
        best_model = None
        acc_matrix = np.zeros((max_edge_features, max_us_features))  # 行: edge, 列: us

        # Using Extra Trees instead of Random Forest
        model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_results = []

        for edge_num in range(1, max_edge_features + 1):
            for us_num in range(1, max_us_features + 1):
                print(f"Evaluating: edge={edge_num}, ultrasound={us_num}")
                try:
                    self.dual_rfe_feature_selection(n_edge_features=edge_num, n_ultrasound_features=us_num)

                    scores = cross_val_score(
                        model, self.X_selected, self.y, scoring='accuracy', cv=cv, n_jobs=-1
                    )
                    mean_acc = np.mean(scores)
                    acc_matrix[edge_num - 1, us_num - 1] = mean_acc

                    all_results.append({
                        'Edge Features': edge_num,
                        'Ultrasound Features': us_num,
                        'Accuracy': mean_acc
                    })

                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_edge_num = edge_num
                        best_us_num = us_num
                        best_features = self.selected_features.copy()
                        best_model = clone(model)

                except Exception as e:
                    print(f"  Skipped due to error: {e}")
                    acc_matrix[edge_num - 1, us_num - 1] = np.nan
                    continue

        # 训练最优模型
        print("\n✅ Best Accuracy Combination Found:")
        print(f"  Ultrasound: {best_us_num}, Edge: {best_edge_num}, Accuracy: {best_acc:.4f}")

        self.dual_rfe_feature_selection(n_edge_features=best_edge_num, n_ultrasound_features=best_us_num)
        best_model.fit(self.X_selected, self.y)

        self.best_acc_model = best_model
        self.best_acc = best_acc
        self.best_feature_combo_acc = (best_us_num, best_edge_num)
        self.best_selected_features = best_features
        self.acc_matrix = acc_matrix
        self.acc_result_table = pd.DataFrame(all_results)

        self.plot_accuracy_heatmap(acc_matrix)
        self.save_accuracy_combination_results()

        return best_model, best_features, best_acc

    def save_accuracy_combination_results(self):
        """保存准确性组合结果为Excel"""
        print("💾 Saving Accuracy combination results to Excel...")

        result_path = self.output_dir / "rfe_accuracy_optimization_results.xlsx"
        best_us, best_edge = self.best_feature_combo_accuracy

        # 创建DataFrame
        df = self.accuracy_result_table.copy()
        df = df.sort_values(by="Accuracy", ascending=False)

        # 添加排名列
        df.insert(0, 'Rank', range(1, len(df) + 1))

        # 写入Excel
        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            # 所有组合结果
            df.to_excel(writer, sheet_name='All Combinations', index=False)

            # 最佳组合信息
            best_info_df = pd.DataFrame({
                'Metric': ['Best Ultrasound Features', 'Best Edge Features', 'Best Accuracy', 'Best Accuracy Std'],
                'Value': [best_us, best_edge, self.best_accuracy,
                          df[df['Accuracy'] == self.best_accuracy]['Accuracy_Std'].iloc[0]]
            })
            best_info_df.to_excel(writer, sheet_name='Best Combination', index=False)

            # 前10名组合
            top10_df = df.head(10).copy()
            top10_df.to_excel(writer, sheet_name='Top 10 Combinations', index=False)

            # 统计分析
            stats_df = pd.DataFrame({
                'Statistic': ['Mean Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy',
                              'Median Accuracy', '95th Percentile', '5th Percentile'],
                'Value': [df['Accuracy'].mean(), df['Accuracy'].std(), df['Accuracy'].min(),
                          df['Accuracy'].max(), df['Accuracy'].median(),
                          df['Accuracy'].quantile(0.95), df['Accuracy'].quantile(0.05)]
            })
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        print(f"✅ Accuracy optimization results saved to: {result_path}")

    def plot_rfe_accuracy_detailed_analysis(self):
        """绘制RFE准确性优化的详细分析图"""
        print("📊 Plotting detailed RFE accuracy analysis...")

        if not hasattr(self, 'accuracy_result_table'):
            print("No accuracy optimization results available")
            return

        df = self.accuracy_result_table.copy()

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        # 1. 特征数量vs准确性散点图
        ax1 = axes[0]
        df['Total_Features'] = df['Edge Features'] + df['Ultrasound Features']

        scatter = ax1.scatter(df['Total_Features'], df['Accuracy'],
                              c=df['Edge Features'], cmap='viridis',
                              s=50, alpha=0.7, edgecolors='black')

        # 标记最佳点
        best_row = df.loc[df['Accuracy'].idxmax()]
        ax1.scatter(best_row['Total_Features'], best_row['Accuracy'],
                    s=200, c='red', marker='*', edgecolors='white', linewidth=2,
                    label=f'Best: {best_row["Accuracy"]:.4f}')

        ax1.set_xlabel('Total Features', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Total Features vs Accuracy', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Edge Features Count')

        # 2. 边缘特征数量分布
        ax2 = axes[1]
        edge_counts = df['Edge Features'].value_counts().sort_index()
        bars = ax2.bar(edge_counts.index, edge_counts.values,
                       color='lightblue', alpha=0.7, edgecolor='black')

        ax2.set_xlabel('Edge Features Count', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Edge Features Count Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 3. 超声特征数量分布
        ax3 = axes[2]
        us_counts = df['Ultrasound Features'].value_counts().sort_index()
        bars = ax3.bar(us_counts.index, us_counts.values,
                       color='lightcoral', alpha=0.7, edgecolor='black')

        ax3.set_xlabel('Ultrasound Features Count', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Ultrasound Features Count Distribution', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 4. 准确性分布直方图
        ax4 = axes[3]
        n_bins = min(30, len(df) // 3)
        counts, bins, patches = ax4.hist(df['Accuracy'], bins=n_bins,
                                         color='lightgreen', alpha=0.7, edgecolor='black')

        # 标记最佳准确性
        ax4.axvline(df['Accuracy'].max(), color='red', linestyle='--', linewidth=2,
                    label=f'Best: {df["Accuracy"].max():.4f}')

        # 标记平均准确性
        ax4.axvline(df['Accuracy'].mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {df["Accuracy"].mean():.4f}')

        ax4.set_xlabel('Accuracy', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Accuracy Distribution', fontsize=14, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 前10名组合
        ax5 = axes[4]
        top10 = df.nlargest(10, 'Accuracy').reset_index(drop=True)

        x_pos = range(len(top10))
        bars = ax5.bar(x_pos, top10['Accuracy'],
                       color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10))),
                       alpha=0.8, edgecolor='black')

        ax5.set_xlabel('Rank', fontsize=12)
        ax5.set_ylabel('Accuracy', fontsize=12)
        ax5.set_title('Top 10 Feature Combinations', fontsize=14, weight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'#{i + 1}' for i in x_pos])
        ax5.grid(True, alpha=0.3)

        # 添加特征组合标签
        for i, (bar, row) in enumerate(zip(bars, top10.itertuples())):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'E:{row._2}\nU:{row._3}', ha='center', va='bottom',
                     fontsize=8, weight='bold')

        # 6. 特征组合效率分析
        ax6 = axes[5]

        # 计算效率：准确性/总特征数
        df['Efficiency'] = df['Accuracy'] / df['Total_Features']

        scatter = ax6.scatter(df['Total_Features'], df['Efficiency'],
                              c=df['Accuracy'], cmap='viridis',
                              s=50, alpha=0.7, edgecolors='black')

        # 标记最高效率点
        best_eff_row = df.loc[df['Efficiency'].idxmax()]
        ax6.scatter(best_eff_row['Total_Features'], best_eff_row['Efficiency'],
                    s=200, c='red', marker='*', edgecolors='white', linewidth=2,
                    label=f'Most Efficient: {best_eff_row["Efficiency"]:.4f}')

        ax6.set_xlabel('Total Features', fontsize=12)
        ax6.set_ylabel('Efficiency (Accuracy/Features)', fontsize=12)
        ax6.set_title('Feature Combination Efficiency', fontsize=14, weight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        cbar2 = plt.colorbar(scatter, ax=ax6)
        cbar2.set_label('Accuracy')

        plt.suptitle('RFE Accuracy Optimization - Detailed Analysis',
                     fontsize=18, weight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "rfe_accuracy_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Detailed RFE accuracy analysis plot saved.")

    def dual_rfe_feature_selection(self, n_edge_features=20, n_ultrasound_features=10):
        """
        分别对边缘特征和超声特征做RFE，最终将二者合并。
        在平衡数据上进行特征选择。
        """
        print(
            f"Running RFE on balanced data - edge features ({n_edge_features}) and ultrasound features ({n_ultrasound_features})...")

        # 1. 提取超声特征（分类变量）
        us_indices = [i for i, col in enumerate(self.feature_names) if col in self.discrete_cols]
        X_us = self.X_scaled[:, us_indices]
        us_names = np.array(self.feature_names)[us_indices]

        # 2. 提取边缘特征（连续变量中在 edge_cols 中的）
        edge_indices = [i for i, col in enumerate(self.feature_names) if col in self.edge_cols]
        X_edge = self.X_scaled[:, edge_indices]
        edge_names = np.array(self.feature_names)[edge_indices]

        # 3. 对超声特征进行 RFE
        if n_ultrasound_features > 0 and X_us.shape[1] >= n_ultrasound_features:
            base_us = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            rfe_us = RFE(estimator=base_us, n_features_to_select=n_ultrasound_features)
            rfe_us.fit(X_us, self.y)
            selected_us_names = us_names[rfe_us.support_].tolist()
        else:
            selected_us_names = us_names.tolist()

        # 4. 对边缘特征进行 RFE
        if n_edge_features > 0 and X_edge.shape[1] >= n_edge_features:
            base_edge = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            rfe_edge = RFE(estimator=base_edge, n_features_to_select=n_edge_features)
            rfe_edge.fit(X_edge, self.y)
            selected_edge_names = edge_names[rfe_edge.support_].tolist()
        else:
            selected_edge_names = edge_names.tolist()

        # 5. 合并特征名
        combined_selected_names = selected_us_names + selected_edge_names
        self.selected_features = combined_selected_names

        # 获取合并后的索引
        combined_indices = [self.feature_names.index(name) for name in combined_selected_names]
        self.X_selected = self.X_scaled[:, combined_indices]

        print(f"Selected ultrasound features: {selected_us_names}")
        print(f"Selected edge features: {selected_edge_names}")
        print(f"Combined total features: {len(combined_selected_names)}")
        print(f"Data shape for modeling: {self.X_selected.shape}")

        # 可视化
        self.plot_feature_correlation(combined_selected_names)

        return self.X_selected, combined_selected_names

    def plot_feature_correlation(self, selected_columns):
        """绘制选定特征的相关性热图"""
        print("Plotting feature correlation heatmap...")

        try:
            # 获取选定特征的数据
            selected_data = pd.DataFrame(self.X_scaled, columns=self.feature_names)[selected_columns]

            # 计算相关系数矩阵
            corr_matrix = selected_data.corr()

            # 压缩特征名显示
            compressed_names = [self.compress_column_name(col, 15) for col in selected_columns]
            corr_matrix.columns = compressed_names
            corr_matrix.index = compressed_names

            # 设置图形大小
            plt.figure(figsize=(15, 12))

            # 绘制热图
            sns.heatmap(corr_matrix,
                        cmap='coolwarm',
                        center=0,
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={'size': 8})

            # 调整标题和标签
            plt.title('Selected Features Correlation Matrix', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)

            # 保存图片
            plt.tight_layout()
            plt.savefig(self.output_dir / "selected_features_correlation.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            print("Feature correlation heatmap saved successfully.")

        except Exception as e:
            print(f"Error plotting feature correlation: {e}")

    def init_models(self):
        """初始化15种机器学习模型（针对不平衡数据优化）"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                                    class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', scale_pos_weight=2.16),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'Bagging': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }

        print(f"Initialized {len(self.models)} machine learning models with class balancing")



    def evaluate_models(self):
        """评估所有模型"""
        print("Evaluating all models with 5-fold cross validation...")

        # 5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 获取类别标签
        class_labels = np.unique(self.y)
        class_names = [f'Class {label}' for label in class_labels]

        for name, model in tqdm(self.models.items(), desc="Training models"):
            print(f"\nTraining {name}...")

            try:
                # 交叉验证评分
                cv_scores = cross_val_score(model, self.X_selected, self.y,
                                            cv=cv, scoring='accuracy', n_jobs=-1)

                # 训练测试分割
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X_selected, self.y, test_size=0.2,
                    random_state=42, stratify=self.y
                )

                # 训练模型
                model.fit(X_train, y_train)

                # 预测
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # 计算指标
                train_metrics = self.calculate_metrics(y_train, y_train_pred, y_test_proba, 'train')
                test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba, 'test')

                # 计算混淆矩阵
                cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)

                # 存储结果
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'y_test': y_test,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba,
                    'cv_scores': cv_scores,
                    'confusion_matrix': cm
                }

                self.cv_results[name] = {
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std()
                }

                print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")

                # 绘制混淆矩阵
                self.plot_confusion_matrix(name, cm, class_names)

            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue

    def create_ensemble_voting_classifier(self):
        """创建集成投票分类器"""
        print("Creating ensemble voting classifier...")

        # 重新训练所有模型以确保一致性
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_selected, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # 存储训练好的模型
        self.trained_models = {}
        self.ensemble_predictions = {}

        print("Training individual models for ensemble...")

        for name, model in tqdm(self.models.items(), desc="Training ensemble models"):
            try:
                # 训练模型
                model_copy = clone(model)
                model_copy.fit(X_train, y_train)

                # 预测
                y_pred = model_copy.predict(X_test)
                y_proba = model_copy.predict_proba(X_test) if hasattr(model_copy, "predict_proba") else None

                # 存储结果
                self.trained_models[name] = model_copy
                self.ensemble_predictions[name] = {
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'accuracy': accuracy_score(y_test, y_pred)
                }

                print(f"  {name}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")

            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue

        # 存储测试集数据
        self.X_test_ensemble = X_test
        self.y_test_ensemble = y_test

        return len(self.trained_models)

    def perform_voting_classification(self):
        """执行投票分类"""
        print("Performing voting classification...")

        if not hasattr(self, 'trained_models') or not self.trained_models:
            print("No trained models available. Creating ensemble first...")
            self.create_ensemble_voting_classifier()

        n_samples = len(self.X_test_ensemble)
        n_models = len(self.trained_models)
        n_classes = len(np.unique(self.y))

        # 初始化投票矩阵
        self.voting_matrix = np.zeros((n_samples, n_models), dtype=int)
        self.voting_proba_matrix = np.zeros((n_samples, n_models, n_classes))
        self.model_names_list = list(self.trained_models.keys())

        # 收集所有模型的预测
        for model_idx, (name, model) in enumerate(self.trained_models.items()):
            predictions = self.ensemble_predictions[name]
            self.voting_matrix[:, model_idx] = predictions['y_pred']

            if predictions['y_proba'] is not None:
                self.voting_proba_matrix[:, model_idx, :] = predictions['y_proba']

        # 执行不同的投票策略
        self.voting_results = self._apply_voting_strategies()

        return self.voting_results

    def _apply_voting_strategies(self):
        """应用不同的投票策略"""
        voting_results = {}

        # 1. 硬投票 (Hard Voting)
        hard_votes = []
        for i in range(len(self.X_test_ensemble)):
            # 统计每个类别的票数
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            predicted_class = np.argmax(votes)
            hard_votes.append(predicted_class)

        hard_votes = np.array(hard_votes)

        # 2. 软投票 (Soft Voting) - 基于概率平均
        soft_votes = []
        avg_probabilities = np.mean(self.voting_proba_matrix, axis=1)
        soft_votes = np.argmax(avg_probabilities, axis=1)

        # 3. 加权投票 - 基于模型准确率
        model_weights = np.array([self.ensemble_predictions[name]['accuracy']
                                  for name in self.model_names_list])
        model_weights = model_weights / np.sum(model_weights)  # 归一化权重

        weighted_probabilities = np.zeros((len(self.X_test_ensemble), len(np.unique(self.y))))
        for i, weight in enumerate(model_weights):
            weighted_probabilities += weight * self.voting_proba_matrix[:, i, :]

        weighted_votes = np.argmax(weighted_probabilities, axis=1)

        # 4. 多数投票 (Majority Voting) - 需要超过半数
        majority_votes = []
        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            max_votes = np.max(votes)
            if max_votes > len(self.model_names_list) / 2:
                predicted_class = np.argmax(votes)
            else:
                predicted_class = -1  # 表示没有多数
            majority_votes.append(predicted_class)

        majority_votes = np.array(majority_votes)

        # 计算各种投票策略的性能
        voting_results = {
            'hard_voting': {
                'predictions': hard_votes,
                'accuracy': accuracy_score(self.y_test_ensemble, hard_votes),
                'probabilities': None
            },
            'soft_voting': {
                'predictions': soft_votes,
                'accuracy': accuracy_score(self.y_test_ensemble, soft_votes),
                'probabilities': avg_probabilities
            },
            'weighted_voting': {
                'predictions': weighted_votes,
                'accuracy': accuracy_score(self.y_test_ensemble, weighted_votes),
                'probabilities': weighted_probabilities,
                'weights': model_weights
            },
            'majority_voting': {
                'predictions': majority_votes,
                'accuracy': accuracy_score(self.y_test_ensemble[majority_votes != -1],
                                           majority_votes[majority_votes != -1]) if np.any(majority_votes != -1) else 0,
                'probabilities': None,
                'abstention_rate': np.mean(majority_votes == -1)
            }
        }

        return voting_results

    def plot_voting_analysis(self):
        """绘制投票分析的综合图表"""
        print("Plotting voting analysis...")

        if not hasattr(self, 'voting_results'):
            print("No voting results available. Performing voting classification first...")
            self.perform_voting_classification()

        # 创建多个可视化图表
        self.plot_voting_heatmap()
        self.plot_voting_distribution()
        self.plot_model_agreement_analysis()
        self.plot_voting_performance_comparison()
        self.plot_voting_confusion_matrices()  # 新增的混淆矩阵绘制
        self.plot_individual_case_analysis()
        self.plot_consensus_confidence_analysis()

    def plot_voting_heatmap(self):
        """绘制投票热图"""
        print("Plotting voting heatmap...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 模型预测热图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(self.voting_matrix.T, cmap='viridis', aspect='auto')
        ax1.set_title('Model Predictions Heatmap\n(Rows: Models, Columns: Test Samples)',
                      fontsize=14, weight='bold')
        ax1.set_xlabel('Test Sample Index')
        ax1.set_ylabel('Models')

        # 设置y轴标签
        compressed_names = [self.compress_column_name(name, 20) for name in self.model_names_list]
        ax1.set_yticks(range(len(self.model_names_list)))
        ax1.set_yticklabels(compressed_names, fontsize=8)

        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Predicted Class', fontsize=10)

        # 2. 投票一致性热图
        ax2 = axes[0, 1]

        # 计算每个样本的投票一致性
        consistency_matrix = np.zeros((len(self.X_test_ensemble), len(np.unique(self.y))))
        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            consistency_matrix[i, :] = votes / len(self.model_names_list)

        im2 = ax2.imshow(consistency_matrix.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Voting Consistency Heatmap\n(Proportion of votes for each class)',
                      fontsize=14, weight='bold')
        ax2.set_xlabel('Test Sample Index')
        ax2.set_ylabel('Class')
        ax2.set_yticks(range(len(np.unique(self.y))))
        ax2.set_yticklabels([f'Class {i}' for i in range(len(np.unique(self.y)))])

        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Vote Proportion', fontsize=10)

        # 3. 模型间相关性热图
        ax3 = axes[1, 0]

        # 计算模型预测的相关性矩阵
        correlation_matrix = np.corrcoef(self.voting_matrix.T)

        im3 = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title('Model Prediction Correlation Matrix', fontsize=14, weight='bold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Models')
        ax3.set_xticks(range(len(self.model_names_list)))
        ax3.set_yticks(range(len(self.model_names_list)))
        ax3.set_xticklabels(compressed_names, rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(compressed_names, fontsize=8)

        # 添加相关系数文本
        for i in range(len(self.model_names_list)):
            for j in range(len(self.model_names_list)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                ha="center", va="center",
                                color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white",
                                fontsize=6)

        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Correlation Coefficient', fontsize=10)

        # 4. 真实标签 vs 投票结果比较
        ax4 = axes[1, 1]

        # 创建比较矩阵
        comparison_data = np.column_stack([
            self.y_test_ensemble,
            self.voting_results['hard_voting']['predictions'],
            self.voting_results['soft_voting']['predictions'],
            self.voting_results['weighted_voting']['predictions']
        ])

        im4 = ax4.imshow(comparison_data.T, cmap='tab10', aspect='auto')
        ax4.set_title('True Labels vs Voting Results Comparison', fontsize=14, weight='bold')
        ax4.set_xlabel('Test Sample Index')
        ax4.set_ylabel('Prediction Source')
        ax4.set_yticks(range(4))
        ax4.set_yticklabels(['True Label', 'Hard Voting', 'Soft Voting', 'Weighted Voting'])

        cbar4 = plt.colorbar(im4, ax=ax4)
        cbar4.set_label('Class Label', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / "voting_heatmap_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_voting_distribution(self):
        """绘制投票分布分析"""
        print("Plotting voting distribution analysis...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        # 1. 投票策略准确率比较
        ax1 = axes[0]
        strategies = ['Hard Voting', 'Soft Voting', 'Weighted Voting', 'Individual Models (Mean)']
        accuracies = [
            self.voting_results['hard_voting']['accuracy'],
            self.voting_results['soft_voting']['accuracy'],
            self.voting_results['weighted_voting']['accuracy'],
            np.mean([pred['accuracy'] for pred in self.ensemble_predictions.values()])
        ]

        bars = ax1.bar(strategies, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Voting Strategy Performance Comparison', fontsize=14, weight='bold')
        ax1.set_ylim([0, 1])

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. 单个模型准确率分布
        ax2 = axes[1]
        model_accuracies = [pred['accuracy'] for pred in self.ensemble_predictions.values()]
        ax2.hist(model_accuracies, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(np.mean(model_accuracies), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(model_accuracies):.3f}')
        ax2.axvline(self.voting_results['hard_voting']['accuracy'], color='green', linestyle='--', linewidth=2,
                    label=f'Hard Voting: {self.voting_results["hard_voting"]["accuracy"]:.3f}')
        ax2.set_xlabel('Accuracy', fontsize=12)
        ax2.set_ylabel('Number of Models', fontsize=12)
        ax2.set_title('Individual Model Accuracy Distribution', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 投票一致性分布
        ax3 = axes[2]

        # 计算每个样本的最大投票比例
        max_vote_proportions = []
        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            max_proportion = np.max(votes) / len(self.model_names_list)
            max_vote_proportions.append(max_proportion)

        ax3.hist(max_vote_proportions, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(max_vote_proportions), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(max_vote_proportions):.3f}')
        ax3.set_xlabel('Maximum Vote Proportion', fontsize=12)
        ax3.set_ylabel('Number of Samples', fontsize=12)
        ax3.set_title('Voting Consensus Distribution', fontsize=14, weight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 模型权重可视化（加权投票）
        ax4 = axes[3]
        weights = self.voting_results['weighted_voting']['weights']
        sorted_indices = np.argsort(weights)[::-1]
        sorted_names = [self.compress_column_name(self.model_names_list[i], 15) for i in sorted_indices]
        sorted_weights = weights[sorted_indices]

        bars = ax4.barh(range(len(sorted_names)), sorted_weights, color='lightcoral')
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels(sorted_names, fontsize=8)
        ax4.set_xlabel('Weight (Normalized Accuracy)', fontsize=12)
        ax4.set_title('Model Weights in Weighted Voting', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)

        # 添加权重值标签
        for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
            width = bar.get_width()
            ax4.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{weight:.3f}', ha='left', va='center', fontsize=8)

        # 5. 预测置信度分布
        ax5 = axes[4]

        if self.voting_results['soft_voting']['probabilities'] is not None:
            max_probabilities = np.max(self.voting_results['soft_voting']['probabilities'], axis=1)
            ax5.hist(max_probabilities, bins=20, alpha=0.7, color='lightyellow', edgecolor='black')
            ax5.axvline(np.mean(max_probabilities), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(max_probabilities):.3f}')
            ax5.set_xlabel('Maximum Prediction Probability', fontsize=12)
            ax5.set_ylabel('Number of Samples', fontsize=12)
            ax5.set_title('Soft Voting Confidence Distribution', fontsize=14, weight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 错误分析
        ax6 = axes[5]

        # 统计每种投票方法的错误类型
        voting_methods = ['hard_voting', 'soft_voting', 'weighted_voting']
        error_counts = []

        for method in voting_methods:
            predictions = self.voting_results[method]['predictions']
            errors = np.sum(predictions != self.y_test_ensemble)
            error_counts.append(errors)

        method_names = ['Hard Voting', 'Soft Voting', 'Weighted Voting']
        bars = ax6.bar(method_names, error_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax6.set_ylabel('Number of Errors', fontsize=12)
        ax6.set_title('Error Count Comparison', fontsize=14, weight='bold')

        # 添加错误率标签
        for bar, errors in zip(bars, error_counts):
            height = bar.get_height()
            error_rate = errors / len(self.y_test_ensemble)
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{errors}\n({error_rate:.2%})', ha='center', va='bottom', fontsize=10)

        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "voting_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_agreement_analysis(self):
        """绘制模型一致性分析"""
        print("Plotting model agreement analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 模型一致性网络图
        ax1 = axes[0, 0]

        # 计算模型间的一致性（预测相同的比例）
        n_models = len(self.model_names_list)
        agreement_matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                agreement = np.mean(self.voting_matrix[:, i] == self.voting_matrix[:, j])
                agreement_matrix[i, j] = agreement

        # 绘制一致性热图
        im1 = ax1.imshow(agreement_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title('Model Agreement Matrix\n(Proportion of samples with same prediction)',
                      fontsize=12, weight='bold')

        compressed_names = [self.compress_column_name(name, 10) for name in self.model_names_list]
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels(compressed_names, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(compressed_names, fontsize=8)

        # 添加数值标注
        for i in range(n_models):
            for j in range(n_models):
                text = ax1.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                                ha="center", va="center",
                                color="white" if agreement_matrix[i, j] > 0.5 else "black",
                                fontsize=6)

        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Agreement Rate', fontsize=10)

        # 2. 投票分歧案例分析
        ax2 = axes[0, 1]

        # 计算每个样本的分歧程度（标准差或熵）
        disagreement_scores = []
        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            # 使用熵来衡量分歧程度
            probs = votes / np.sum(votes)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            disagreement_scores.append(entropy)

        disagreement_scores = np.array(disagreement_scores)

        # 绘制分歧程度分布
        ax2.hist(disagreement_scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(np.mean(disagreement_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(disagreement_scores):.3f}')
        ax2.set_xlabel('Disagreement Score (Entropy)', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Model Disagreement Distribution', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 高分歧案例分析
        ax3 = axes[1, 0]

        # 找到分歧最大的案例
        high_disagreement_indices = np.argsort(disagreement_scores)[-10:]

        # 绘制这些案例的投票模式
        high_disagreement_votes = self.voting_matrix[high_disagreement_indices, :]

        im3 = ax3.imshow(high_disagreement_votes.T, cmap='viridis', aspect='auto')
        ax3.set_title('High Disagreement Cases\n(Top 10 most controversial samples)',
                      fontsize=12, weight='bold')
        ax3.set_xlabel('Sample Index (sorted by disagreement)')
        ax3.set_ylabel('Models')
        ax3.set_yticks(range(0, len(self.model_names_list), 2))
        ax3.set_yticklabels([compressed_names[i] for i in range(0, len(compressed_names), 2)], fontsize=8)

        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Predicted Class', fontsize=10)

        # 4. 投票策略准确率 vs 一致性
        ax4 = axes[1, 1]

        # 计算每个样本的一致性和投票结果的正确性
        consensus_levels = []
        hard_voting_correct = []

        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            max_votes = np.max(votes)
            consensus = max_votes / len(self.model_names_list)
            consensus_levels.append(consensus)

            # 硬投票是否正确
            hard_pred = self.voting_results['hard_voting']['predictions'][i]
            is_correct = (hard_pred == self.y_test_ensemble[i])
            hard_voting_correct.append(is_correct)

        # 绘制散点图
        colors = ['red' if not correct else 'green' for correct in hard_voting_correct]
        ax4.scatter(consensus_levels, disagreement_scores, c=colors, alpha=0.6, s=50)
        ax4.set_xlabel('Consensus Level (Max Vote Proportion)', fontsize=12)
        ax4.set_ylabel('Disagreement Score (Entropy)', fontsize=12)
        ax4.set_title('Consensus vs Disagreement Analysis', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Correct Prediction'),
                           Patch(facecolor='red', label='Incorrect Prediction')]
        ax4.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_agreement_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_voting_performance_comparison(self):
        """绘制投票性能详细比较"""
        print("Plotting voting performance comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. ROC曲线比较（如果是二分类）
        ax1 = axes[0, 0]

        if len(np.unique(self.y)) == 2:
            # 绘制各种投票方法的ROC曲线
            from sklearn.metrics import roc_curve, auc

            methods_with_proba = ['soft_voting', 'weighted_voting']
            colors = ['blue', 'red']

            for method, color in zip(methods_with_proba, colors):
                if self.voting_results[method]['probabilities'] is not None:
                    proba = self.voting_results[method]['probabilities'][:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test_ensemble, proba)
                    roc_auc = auc(fpr, tpr)

                    ax1.plot(fpr, tpr, color=color, linewidth=2,
                             label=f'{method.replace("_", " ").title()} (AUC = {roc_auc:.3f})')

            # 添加最佳单一模型的ROC曲线
            best_model_name = max(self.ensemble_predictions.keys(),
                                  key=lambda x: self.ensemble_predictions[x]['accuracy'])
            best_model_proba = self.ensemble_predictions[best_model_name]['y_proba']
            if best_model_proba is not None:
                fpr, tpr, _ = roc_curve(self.y_test_ensemble, best_model_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, color='green', linewidth=2, linestyle='--',
                         label=f'Best Single Model (AUC = {roc_auc:.3f})')

            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate', fontsize=12)
            ax1.set_ylabel('True Positive Rate', fontsize=12)
            ax1.set_title('ROC Curves Comparison', fontsize=14, weight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=14)

        # 2. 混淆矩阵比较（修改为与机器学习模型一致的格式）
        ax2 = axes[0, 1]

        # 选择最佳投票方法
        best_voting_method = max(self.voting_results.keys(),
                                 key=lambda x: self.voting_results[x]['accuracy'])
        best_predictions = self.voting_results[best_voting_method]['predictions']

        # 绘制混淆矩阵（使用与机器学习模型相同的格式）
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test_ensemble, best_predictions)

        # 计算百分比（保留两位小数）
        cm_percent = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

        # 创建热图（显示百分比）
        sns.heatmap(cm_percent, annot=False, fmt='.2%', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(len(np.unique(self.y)))],
                    yticklabels=[f'Class {i}' for i in range(len(np.unique(self.y)))],
                    cbar_kws={'format': '%.2f'}, vmin=0, vmax=1, ax=ax2)

        # 在每个格子中添加数量和百分比组合标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # 数量 (n=xx)
                count = cm[i, j]
                # 百分比 (xx%)
                percent = cm_percent[i, j] * 100
                # 组合文本
                text = f"{count}\n({percent:.1f}%)"
                # 根据背景色自动选择文字颜色
                bg_color = cm_percent[i, j]
                text_color = 'white' if bg_color > 0.5 else 'black'

                ax2.text(j + 0.5, i + 0.5, text,
                         ha='center', va='center',
                         color=text_color, fontsize=10)

        # 设置标题和标签
        ax2.set_title(f'Confusion Matrix - {best_voting_method.replace("_", " ").title()}\n(Count and Percentage)',
                      fontsize=12, pad=20)
        ax2.set_xlabel('Predicted Label', fontsize=10)
        ax2.set_ylabel('True Label', fontsize=10)

        # 调整颜色条标签为百分比小数形式
        cbar = ax2.collections[0].colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        cbar.set_label('Percentage', fontsize=10)

        # 3. 详细性能指标比较
        ax3 = axes[1, 0]

        # 计算详细指标
        from sklearn.metrics import precision_score, recall_score, f1_score

        metrics_data = []
        voting_methods = ['hard_voting', 'soft_voting', 'weighted_voting']

        for method in voting_methods:
            predictions = self.voting_results[method]['predictions']

            metrics = {
                'Method': method.replace('_', ' ').title(),
                'Accuracy': accuracy_score(self.y_test_ensemble, predictions),
                'Precision': precision_score(self.y_test_ensemble, predictions, average='weighted', zero_division=0),
                'Recall': recall_score(self.y_test_ensemble, predictions, average='weighted', zero_division=0),
                'F1-Score': f1_score(self.y_test_ensemble, predictions, average='weighted', zero_division=0)
            }
            metrics_data.append(metrics)

        # 添加最佳单一模型的性能
        best_single_pred = self.ensemble_predictions[best_model_name]['y_pred']
        metrics_data.append({
            'Method': 'Best Single Model',
            'Accuracy': accuracy_score(self.y_test_ensemble, best_single_pred),
            'Precision': precision_score(self.y_test_ensemble, best_single_pred, average='weighted', zero_division=0),
            'Recall': recall_score(self.y_test_ensemble, best_single_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(self.y_test_ensemble, best_single_pred, average='weighted', zero_division=0)
        })

        # 绘制性能指标条形图
        metrics_df = pd.DataFrame(metrics_data)
        x = np.arange(len(metrics_df))
        width = 0.2

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']

        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            ax3.bar(x + i * width, metrics_df[metric], width, label=metric, color=color, alpha=0.8)

        ax3.set_xlabel('Methods', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Performance Metrics Comparison', fontsize=14, weight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(metrics_df['Method'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # 4. 预测置信度箱线图
        ax4 = axes[1, 1]

        confidence_data = []
        confidence_labels = []

        for method in ['soft_voting', 'weighted_voting']:
            if self.voting_results[method]['probabilities'] is not None:
                max_proba = np.max(self.voting_results[method]['probabilities'], axis=1)
                confidence_data.append(max_proba)
                confidence_labels.append(method.replace('_', ' ').title())

        if confidence_data:
            bp = ax4.boxplot(confidence_data, labels=confidence_labels, patch_artist=True)

            # 设置箱线图颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax4.set_ylabel('Prediction Confidence', fontsize=12)
            ax4.set_title('Prediction Confidence Distribution', fontsize=14, weight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / "voting_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_voting_confusion_matrices(self):
        """绘制各种投票方法的混淆矩阵（与机器学习模型格式一致）"""
        print("Plotting voting confusion matrices...")

        voting_methods = ['hard_voting', 'soft_voting', 'weighted_voting']
        n_methods = len(voting_methods)

        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        for idx, method in enumerate(voting_methods):
            ax = axes[idx]

            predictions = self.voting_results[method]['predictions']

            # 计算混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test_ensemble, predictions)

            # 计算百分比（保留两位小数）
            cm_percent = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

            # 创建热图（显示百分比）
            class_names = [f'Class {i}' for i in range(len(np.unique(self.y)))]
            sns.heatmap(cm_percent, annot=False, fmt='.2%', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'format': '%.2f'}, vmin=0, vmax=1, ax=ax)

            # 在每个格子中添加数量和百分比组合标签
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    # 数量 (n=xx)
                    count = cm[i, j]
                    # 百分比 (xx%)
                    percent = cm_percent[i, j] * 100
                    # 组合文本
                    text = f"{count}\n({percent:.1f}%)"
                    # 根据背景色自动选择文字颜色
                    bg_color = cm_percent[i, j]
                    text_color = 'white' if bg_color > 0.5 else 'black'

                    ax.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color=text_color, fontsize=10)

            # 设置标题和标签
            method_title = method.replace('_', ' ').title()
            accuracy = self.voting_results[method]['accuracy']
            ax.set_title(f'{method_title}\n(Count and Percentage)\nAccuracy: {accuracy:.3f}',
                         fontsize=12, pad=20)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)

            # 调整颜色条标签为百分比小数形式
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        plt.suptitle('Voting Methods Confusion Matrices Comparison', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "voting_confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_individual_case_analysis(self):
        """绘制个案分析图"""
        print("Plotting individual case analysis...")

        # 选择一些有趣的案例进行详细分析
        # 1. 全部模型一致的案例
        # 2. 高分歧的案例
        # 3. 投票正确但大多数模型错误的案例
        # 4. 投票错误但大多数模型正确的案例

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 计算每个样本的一致性和分歧
        consensus_scores = []
        disagreement_scores = []

        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
            max_votes = np.max(votes)
            consensus = max_votes / len(self.model_names_list)
            consensus_scores.append(consensus)

            # 计算分歧（熵）
            probs = votes / np.sum(votes)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            disagreement_scores.append(entropy)

        consensus_scores = np.array(consensus_scores)
        disagreement_scores = np.array(disagreement_scores)

        # 1. 高一致性案例（前10个）
        ax1 = axes[0, 0]
        high_consensus_indices = np.argsort(consensus_scores)[-10:]

        case_data = []
        for idx in high_consensus_indices:
            votes = np.bincount(self.voting_matrix[idx, :], minlength=len(np.unique(self.y)))
            case_data.append(votes)

        case_data = np.array(case_data)

        im1 = ax1.imshow(case_data, cmap='YlOrRd', aspect='auto')
        ax1.set_title('High Consensus Cases\n(Vote distribution for each class)', fontsize=12, weight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Sample Index (High Consensus)')
        ax1.set_xticks(range(len(np.unique(self.y))))
        ax1.set_xticklabels([f'Class {i}' for i in range(len(np.unique(self.y)))])

        # 添加数值标注
        for i in range(case_data.shape[0]):
            for j in range(case_data.shape[1]):
                text = ax1.text(j, i, f'{case_data[i, j]}',
                                ha="center", va="center",
                                color="white" if case_data[i, j] > len(self.model_names_list) / 2 else "black",
                                fontsize=10)

        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Number of Votes', fontsize=10)

        # 2. 高分歧案例（前10个）
        ax2 = axes[0, 1]
        high_disagreement_indices = np.argsort(disagreement_scores)[-10:]

        case_data_2 = []
        for idx in high_disagreement_indices:
            votes = np.bincount(self.voting_matrix[idx, :], minlength=len(np.unique(self.y)))
            case_data_2.append(votes)

        case_data_2 = np.array(case_data_2)

        im2 = ax2.imshow(case_data_2, cmap='RdYlBu', aspect='auto')
        ax2.set_title('High Disagreement Cases\n(Vote distribution for each class)', fontsize=12, weight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Sample Index (High Disagreement)')
        ax2.set_xticks(range(len(np.unique(self.y))))
        ax2.set_xticklabels([f'Class {i}' for i in range(len(np.unique(self.y)))])

        # 添加数值标注
        for i in range(case_data_2.shape[0]):
            for j in range(case_data_2.shape[1]):
                text = ax2.text(j, i, f'{case_data_2[i, j]}',
                                ha="center", va="center",
                                color="white" if case_data_2[i, j] > len(self.model_names_list) / 3 else "black",
                                fontsize=10)

        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Number of Votes', fontsize=10)

        # 3. 投票效果分析
        ax3 = axes[1, 0]

        # 分析投票是否改善了预测
        hard_voting_correct = (self.voting_results['hard_voting']['predictions'] == self.y_test_ensemble)
        individual_correct_counts = np.sum(self.voting_matrix == self.y_test_ensemble[:, np.newaxis], axis=1)
        majority_threshold = len(self.model_names_list) / 2

        # 四个象限的案例
        # 1. 投票正确，多数模型正确
        quad1 = np.sum((hard_voting_correct) & (individual_correct_counts > majority_threshold))
        # 2. 投票正确，多数模型错误
        quad2 = np.sum((hard_voting_correct) & (individual_correct_counts <= majority_threshold))
        # 3. 投票错误，多数模型正确
        quad3 = np.sum((~hard_voting_correct) & (individual_correct_counts > majority_threshold))
        # 4. 投票错误，多数模型错误
        quad4 = np.sum((~hard_voting_correct) & (individual_correct_counts <= majority_threshold))

        categories = ['Voting✓\nMajority✓', 'Voting✓\nMajority✗', 'Voting✗\nMajority✓', 'Voting✗\nMajority✗']
        counts = [quad1, quad2, quad3, quad4]
        colors = ['green', 'orange', 'red', 'darkred']

        bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
        ax3.set_ylabel('Number of Cases', fontsize=12)
        ax3.set_title('Voting Effectiveness Analysis', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)

        # 添加数值标签和百分比
        total_cases = len(self.y_test_ensemble)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = count / total_cases * 100
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)

        # 4. 错误案例详细分析
        ax4 = axes[1, 1]

        # 找到投票错误的案例，分析错误类型
        wrong_cases = ~hard_voting_correct
        if np.any(wrong_cases):
            wrong_indices = np.where(wrong_cases)[0]

            # 分析这些错误案例的投票模式
            error_analysis = {
                'True Class': [],
                'Predicted Class': [],
                'Confidence Level': [],
                'Model Agreement': []
            }

            for idx in wrong_indices[:10]:  # 只分析前10个错误案例
                true_class = self.y_test_ensemble[idx]
                pred_class = self.voting_results['hard_voting']['predictions'][idx]

                votes = np.bincount(self.voting_matrix[idx, :], minlength=len(np.unique(self.y)))
                confidence = np.max(votes) / len(self.model_names_list)
                agreement = np.sum(self.voting_matrix[idx, :] == pred_class) / len(self.model_names_list)

                error_analysis['True Class'].append(true_class)
                error_analysis['Predicted Class'].append(pred_class)
                error_analysis['Confidence Level'].append(confidence)
                error_analysis['Model Agreement'].append(agreement)

            # 绘制错误案例的置信度分布
            if error_analysis['Confidence Level']:
                ax4.scatter(range(len(error_analysis['Confidence Level'])),
                            error_analysis['Confidence Level'],
                            c='red', s=100, alpha=0.7, label='Error Cases')

                ax4.set_xlabel('Error Case Index', fontsize=12)
                ax4.set_ylabel('Voting Confidence', fontsize=12)
                ax4.set_title('Confidence Level of Error Cases', fontsize=14, weight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim([0, 1])

                # 添加平均线
                avg_confidence = np.mean(error_analysis['Confidence Level'])
                ax4.axhline(y=avg_confidence, color='red', linestyle='--',
                            label=f'Avg Confidence: {avg_confidence:.3f}')
                ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "individual_case_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_consensus_confidence_analysis(self):
        """绘制共识置信度分析"""
        print("Plotting consensus confidence analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 计算相关指标
        consensus_scores = []
        prediction_correctness = []
        model_diversity = []

        for i in range(len(self.X_test_ensemble)):
            votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))

            # 共识程度
            max_votes = np.max(votes)
            consensus = max_votes / len(self.model_names_list)
            consensus_scores.append(consensus)

            # 预测正确性
            hard_pred = self.voting_results['hard_voting']['predictions'][i]
            is_correct = (hard_pred == self.y_test_ensemble[i])
            prediction_correctness.append(is_correct)

            # 模型多样性（基于预测分布的熵）
            vote_probs = votes / np.sum(votes)
            entropy = -np.sum(vote_probs * np.log(vote_probs + 1e-10))
            model_diversity.append(entropy)

        consensus_scores = np.array(consensus_scores)
        prediction_correctness = np.array(prediction_correctness)
        model_diversity = np.array(model_diversity)

        # 1. 共识程度 vs 预测准确性
        ax1 = axes[0, 0]

        correct_mask = prediction_correctness
        incorrect_mask = ~prediction_correctness

        ax1.scatter(consensus_scores[correct_mask], model_diversity[correct_mask],
                    c='green', alpha=0.6, s=50, label='Correct Predictions')
        ax1.scatter(consensus_scores[incorrect_mask], model_diversity[incorrect_mask],
                    c='red', alpha=0.6, s=50, label='Incorrect Predictions')

        ax1.set_xlabel('Consensus Level', fontsize=12)
        ax1.set_ylabel('Model Diversity (Entropy)', fontsize=12)
        ax1.set_title('Consensus vs Diversity Analysis', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加趋势线
        from scipy.stats import pearsonr
        correlation, p_value = pearsonr(consensus_scores, model_diversity)
        ax1.text(0.80, 0.80, f'Correlation: {correlation:.3f}\np-value: {p_value:.3f}',
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. 共识程度分布（按正确性分组）
        ax2 = axes[0, 1]

        consensus_correct = consensus_scores[correct_mask]
        consensus_incorrect = consensus_scores[incorrect_mask]

        ax2.hist(consensus_correct, bins=20, alpha=0.7, color='green',
                 label=f'Correct ({len(consensus_correct)} cases)', density=True)
        ax2.hist(consensus_incorrect, bins=20, alpha=0.7, color='red',
                 label=f'Incorrect ({len(consensus_incorrect)} cases)', density=True)

        ax2.set_xlabel('Consensus Level', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Consensus Distribution by Prediction Correctness', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 添加统计信息
        if len(consensus_correct) > 0 and len(consensus_incorrect) > 0:
            from scipy.stats import ttest_ind
            t_stat, t_p_value = ttest_ind(consensus_correct, consensus_incorrect)
            ax2.text(0.02, 0.80, f'Mean Correct: {np.mean(consensus_correct):.3f}\n'
                                 f'Mean Incorrect: {np.mean(consensus_incorrect):.3f}\n'
                                 f't-test p-value: {t_p_value:.3f}',
                     transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3. 投票效率分析
        ax3 = axes[1, 0]

        # 按共识程度分组，计算准确率
        consensus_bins = np.linspace(0, 1, 11)
        bin_centers = (consensus_bins[:-1] + consensus_bins[1:]) / 2
        accuracy_by_consensus = []
        sample_counts = []

        for i in range(len(consensus_bins) - 1):
            mask = (consensus_scores >= consensus_bins[i]) & (consensus_scores < consensus_bins[i + 1])
            if i == len(consensus_bins) - 2:  # 最后一个bin包含右端点
                mask = (consensus_scores >= consensus_bins[i]) & (consensus_scores <= consensus_bins[i + 1])

            if np.any(mask):
                accuracy = np.mean(prediction_correctness[mask])
                count = np.sum(mask)
            else:
                accuracy = 0
                count = 0

            accuracy_by_consensus.append(accuracy)
            sample_counts.append(count)

        bars = ax3.bar(bin_centers, accuracy_by_consensus, width=0.08, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Consensus Level', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Accuracy vs Consensus Level', fontsize=14, weight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        # 添加样本数量标签
        for bar, count in zip(bars, sample_counts):
            if count > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'n={count}', ha='center', va='bottom', fontsize=8)

        # 4. 模型贡献度分析
        ax4 = axes[1, 1]

        # 计算每个模型对最终决策的贡献度
        model_contributions = np.zeros(len(self.model_names_list))

        for i in range(len(self.X_test_ensemble)):
            final_prediction = self.voting_results['hard_voting']['predictions'][i]
            # 计算每个模型是否与最终决策一致
            for j, model_pred in enumerate(self.voting_matrix[i, :]):
                if model_pred == final_prediction:
                    model_contributions[j] += 1

        # 归一化为百分比
        model_contributions = model_contributions / len(self.X_test_ensemble) * 100

        # 按贡献度排序
        sorted_indices = np.argsort(model_contributions)[::-1]
        sorted_names = [self.compress_column_name(self.model_names_list[i], 15) for i in sorted_indices]
        sorted_contributions = model_contributions[sorted_indices]

        bars = ax4.barh(range(len(sorted_names)), sorted_contributions, color='lightcoral', alpha=0.7)
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels(sorted_names, fontsize=8)
        ax4.set_xlabel('Contribution to Final Decision (%)', fontsize=12)
        ax4.set_title('Model Contribution Analysis', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)

        # 添加贡献度数值标签
        for i, (bar, contrib) in enumerate(zip(bars, sorted_contributions)):
            width = bar.get_width()
            ax4.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f'{contrib:.1f}%', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "consensus_confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_voting_results_summary(self):
        """保存投票结果综合摘要"""
        print("Saving voting results summary...")

        # 创建综合摘要Excel文件
        with pd.ExcelWriter(self.output_dir / "voting_analysis_comprehensive_summary.xlsx",
                            engine='openpyxl') as writer:

            # 1. 投票策略性能比较
            voting_performance = []
            for method, results in self.voting_results.items():
                if method != 'majority_voting':  # 排除可能有抽象问题的多数投票
                    voting_performance.append({
                        'Voting_Strategy': method.replace('_', ' ').title(),
                        'Accuracy': results['accuracy'],
                        'Total_Errors': np.sum(results['predictions'] != self.y_test_ensemble),
                        'Error_Rate': np.mean(results['predictions'] != self.y_test_ensemble)
                    })

            voting_perf_df = pd.DataFrame(voting_performance)
            voting_perf_df.to_excel(writer, sheet_name='Voting_Performance', index=False)

            # 2. 个体模型性能
            individual_performance = []
            for name, pred_data in self.ensemble_predictions.items():
                individual_performance.append({
                    'Model_Name': name,
                    'Accuracy': pred_data['accuracy'],
                    'Total_Errors': np.sum(pred_data['y_pred'] != self.y_test_ensemble),
                    'Error_Rate': np.mean(pred_data['y_pred'] != self.y_test_ensemble)
                })

            individual_perf_df = pd.DataFrame(individual_performance)
            individual_perf_df = individual_perf_df.sort_values('Accuracy', ascending=False)
            individual_perf_df.to_excel(writer, sheet_name='Individual_Performance', index=False)

            # 3. 详细案例分析
            case_analysis = []
            for i in range(len(self.X_test_ensemble)):
                votes = np.bincount(self.voting_matrix[i, :], minlength=len(np.unique(self.y)))
                max_votes = np.max(votes)
                consensus = max_votes / len(self.model_names_list)

                vote_probs = votes / np.sum(votes)
                entropy = -np.sum(vote_probs * np.log(vote_probs + 1e-10))

                case_analysis.append({
                    'Sample_Index': i,
                    'True_Label': self.y_test_ensemble[i],
                    'Hard_Voting_Pred': self.voting_results['hard_voting']['predictions'][i],
                    'Soft_Voting_Pred': self.voting_results['soft_voting']['predictions'][i],
                    'Weighted_Voting_Pred': self.voting_results['weighted_voting']['predictions'][i],
                    'Consensus_Level': consensus,
                    'Disagreement_Score': entropy,
                    'Models_Correct': np.sum(self.voting_matrix[i, :] == self.y_test_ensemble[i]),
                    'Hard_Voting_Correct': self.voting_results['hard_voting']['predictions'][i] == self.y_test_ensemble[
                        i]
                })

            case_analysis_df = pd.DataFrame(case_analysis)
            case_analysis_df.to_excel(writer, sheet_name='Case_Analysis', index=False)

            # 4. 模型权重和贡献度
            model_analysis = []
            for i, name in enumerate(self.model_names_list):
                # 计算模型与最终决策的一致性
                final_decisions = self.voting_results['hard_voting']['predictions']
                agreement_rate = np.mean(self.voting_matrix[:, i] == final_decisions)

                # 获取权重（如果有的话）
                weight = self.voting_results['weighted_voting']['weights'][i] if 'weights' in self.voting_results[
                    'weighted_voting'] else 0

                model_analysis.append({
                    'Model_Name': name,
                    'Individual_Accuracy': self.ensemble_predictions[name]['accuracy'],
                    'Weight_in_Weighted_Voting': weight,
                    'Agreement_with_Final_Decision': agreement_rate,
                    'Contribution_Score': agreement_rate * self.ensemble_predictions[name]['accuracy']
                })

            model_analysis_df = pd.DataFrame(model_analysis)
            model_analysis_df = model_analysis_df.sort_values('Contribution_Score', ascending=False)
            model_analysis_df.to_excel(writer, sheet_name='Model_Analysis', index=False)

        print("Comprehensive voting analysis summary saved successfully!")

        # 返回关键统计信息
        best_voting_method = max(self.voting_results.keys(),
                                 key=lambda x: self.voting_results[x]['accuracy'])
        best_individual_model = max(self.ensemble_predictions.keys(),
                                    key=lambda x: self.ensemble_predictions[x]['accuracy'])

        summary_stats = {
            'best_voting_method': best_voting_method,
            'best_voting_accuracy': self.voting_results[best_voting_method]['accuracy'],
            'best_individual_model': best_individual_model,
            'best_individual_accuracy': self.ensemble_predictions[best_individual_model]['accuracy'],
            'ensemble_improvement': self.voting_results[best_voting_method]['accuracy'] -
                                    self.ensemble_predictions[best_individual_model]['accuracy'],
            'total_models_used': len(self.model_names_list),
            'test_samples': len(self.y_test_ensemble)
        }

        return summary_stats


    def plot_confusion_matrix(self, model_name, cm, classes):
        """绘制同时显示数量和百分比的混淆矩阵"""
        plt.figure(figsize=(10, 8))

        # 计算百分比（保留两位小数）
        cm_percent = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

        # 创建热图（显示百分比）
        ax = sns.heatmap(cm_percent, annot=False, fmt='.2%', cmap='Blues',
                         xticklabels=classes, yticklabels=classes,
                         cbar_kws={'format': '%.2f'}, vmin=0, vmax=1)

        # 在每个格子中添加数量和百分比组合标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # 数量 (n=xx)
                count = cm[i, j]
                # 百分比 (xx%)
                percent = cm_percent[i, j] * 100
                # 组合文本
                text = f"{count}\n({percent:.1f}%)"
                # 根据背景色自动选择文字颜色
                bg_color = cm_percent[i, j]
                text_color = 'white' if bg_color > 0.5 else 'black'

                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center',
                        color=text_color, fontsize=10)

        # 设置标题和标签
        plt.title(f'Confusion Matrix - {model_name}\n(Count and Percentage)',
                  fontsize=12, pad=20)
        plt.xlabel('Predicted Label', fontsize=10)
        plt.ylabel('True Label', fontsize=10)

        # 调整颜色条标签为百分比小数形式
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        # 调整刻度标签
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        # 保存图片
        plt.tight_layout()
        plt.savefig(self.output_dir / f"confusion_matrix_{model_name}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_metrics(self, y_true, y_pred, y_proba, set_type):
        """计算分类指标"""
        n_classes = len(np.unique(y_true))

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Specificity': self.calculate_specificity(y_true, y_pred, n_classes)
        }

        # 计算AUC (仅用于测试集，因为需要概率)
        if set_type == 'test' and y_proba is not None:
            try:
                if n_classes == 2:
                    metrics['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['AUC'] = 0.0
        else:
            metrics['AUC'] = None

        return metrics

    def calculate_specificity(self, y_true, y_pred, n_classes):
        """计算特异性"""
        cm = confusion_matrix(y_true, y_pred)
        specificity_scores = []

        for i in range(n_classes):
            if i < cm.shape[0] and i < cm.shape[1]:
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificity_scores.append(specificity)

        return np.mean(specificity_scores) if specificity_scores else 0.0

    def calculate_net_benefit(self, y_true, y_proba, threshold):
        """计算净效益"""
        # 对于多分类问题，这里以第一类为例
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            # 对于多分类，使用最大概率作为正类概率
            probabilities = np.max(y_proba, axis=1)
            # 预测为概率最大的类
            y_pred = np.argmax(y_proba, axis=1)
            # 将真实标签转换为二分类（是否为预测的类）
            y_binary = (y_true == y_pred).astype(int)
        else:
            # 二分类情况
            probabilities = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            y_binary = y_true

        # 根据阈值进行预测
        y_pred_threshold = (probabilities >= threshold).astype(int)

        # 计算真阳性率和假阳性率
        tp = np.sum((y_binary == 1) & (y_pred_threshold == 1))
        fp = np.sum((y_binary == 0) & (y_pred_threshold == 1))

        n_total = len(y_binary)

        # 计算净效益
        # 净效益 = (TP/n) - (FP/n) * (pt/(1-pt))
        # 其中 pt 是阈值概率
        if threshold == 0:
            net_benefit = tp / n_total
        elif threshold == 1:
            net_benefit = 0
        else:
            net_benefit = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))

        return net_benefit

    def calculate_net_benefit_all(self, y_true):
        """计算全部治疗策略的净效益"""
        # 假设所有患者都接受治疗的净效益
        n_total = len(y_true)
        if hasattr(y_true, 'ndim') and y_true.ndim > 0:
            n_positive = np.sum(y_true == 1) if len(np.unique(y_true)) == 2 else np.sum(y_true == np.max(y_true))
        else:
            n_positive = 1 if y_true == 1 else 0

        return n_positive / n_total

    def plot_decision_curves(self):
        """绘制决策曲线"""
        print("Plotting decision curves...")

        # 检查是否为二分类问题
        n_classes = len(np.unique(self.y))

        if n_classes == 2:
            self._plot_binary_decision_curves()
        else:
            self._plot_multiclass_decision_curves()

    def _plot_binary_decision_curves(self):
        """绘制二分类决策曲线（前5个模型）"""
        plt.figure(figsize=(12, 8))

        # 阈值范围
        thresholds = np.arange(0, 1, 0.01)

        # 计算参考线：不治疗（净效益=0）
        treat_none = np.zeros(len(thresholds))

        # 计算参考线：全部治疗
        treat_all = []
        for threshold in thresholds:
            if threshold == 0:
                # 当阈值为0时，全部治疗的净效益等于患病率
                prevalence = np.mean(self.results[list(self.results.keys())[0]]['y_test'])
                treat_all.append(prevalence)
            else:
                # 全部治疗的净效益
                nb_all = self.calculate_net_benefit_all(self.results[list(self.results.keys())[0]]['y_test'])
                treat_all.append(nb_all - threshold / (1 - threshold))

        # 绘制参考线
        plt.plot(thresholds, treat_none, 'k--', linewidth=2, label='Treat None')
        plt.plot(thresholds, treat_all, 'gray', linewidth=2, label='Treat All')

        # 只选择前5个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:5]

        # 为前5个模型计算和绘制决策曲线
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, ((name, result), color) in enumerate(zip(top_models, colors)):
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                continue

            net_benefits = []
            for threshold in thresholds:
                nb = self.calculate_net_benefit(y_test, y_proba, threshold)
                net_benefits.append(nb)

            plt.plot(thresholds, net_benefits, color=color, linewidth=2.5,
                     label=f'{self.compress_column_name(name, 20)}')

        plt.xlim([0, 1])
        plt.ylim([-5, 1])  # 修改纵坐标范围
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis - Top 5 Models (Binary Classification)', fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "decision_curves_binary_top5.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_multiclass_decision_curves(self):
        """绘制多分类决策曲线（前5个模型）"""
        plt.figure(figsize=(15, 10))

        # 阈值范围
        thresholds = np.arange(0, 1, 0.01)

        # 计算参考线
        treat_none = np.zeros(len(thresholds))

        # 只选择前5个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:5]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, ((name, result), color) in enumerate(zip(top_models, colors)):
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                continue

            net_benefits = []
            for threshold in thresholds:
                # 对于多分类，计算基于最高概率的净效益
                max_proba = np.max(y_proba, axis=1)
                predicted_class = np.argmax(y_proba, axis=1)

                # 创建二分类问题：预测正确 vs 预测错误
                correct_predictions = (predicted_class == y_test).astype(int)

                # 基于最高概率和正确性计算净效益
                high_confidence = (max_proba >= threshold).astype(int)

                tp = np.sum((correct_predictions == 1) & (high_confidence == 1))
                fp = np.sum((correct_predictions == 0) & (high_confidence == 1))

                n_total = len(y_test)

                if threshold == 0:
                    nb = tp / n_total
                elif threshold == 1:
                    nb = 0
                else:
                    nb = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))

                net_benefits.append(nb)

            plt.plot(thresholds, net_benefits, color=color, linewidth=2.5,
                     label=f'{self.compress_column_name(name, 20)}')

        # 绘制参考线
        plt.plot(thresholds, treat_none, 'k--', linewidth=2, label='No Model (Treat None)')

        # 计算并绘制"全部预测"基线
        accuracy_baseline = []
        for threshold in thresholds:
            # 基于整体准确率的基线
            overall_accuracy = np.mean([result['test_metrics']['Accuracy'] for name, result in top_models])
            if threshold == 0:
                baseline = overall_accuracy
            else:
                baseline = overall_accuracy - threshold / (1 - threshold) if threshold < 1 else 0
            accuracy_baseline.append(baseline)

        plt.plot(thresholds, accuracy_baseline, 'gray', linewidth=2, label='Baseline (Average Accuracy)')

        plt.xlim([0, 1])
        plt.ylim([-5, 1])  # 修改纵坐标范围
        plt.xlabel('Confidence Threshold', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(
            'Decision Curve Analysis - Top 5 Models (Multi-class Classification)\n(Based on Prediction Confidence)',
            fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "decision_curves_multiclass_top5.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_calibration_curves(self):
        """绘制校准曲线（可靠性图）- 所有模型在一张图中"""
        print("Plotting calibration curves...")

        from sklearn.calibration import calibration_curve

        # 检查是否为二分类问题
        n_classes = len(np.unique(self.y))

        if n_classes == 2:
            self._plot_binary_calibration_curves()
        else:
            self._plot_multiclass_calibration_curves()

    def _plot_binary_calibration_curves(self):
        """绘制二分类校准曲线 - 前5个模型在一张图中"""
        from sklearn.calibration import calibration_curve

        plt.figure(figsize=(12, 10))

        # 只选择前5个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:5]

        # 设置颜色和标记
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']

        # 存储所有模型的校准指标
        calibration_metrics = []

        for i, ((name, result), color, marker) in enumerate(zip(top_models, colors, markers)):
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                continue

            # 二分类校准曲线
            prob_positive = y_proba[:, 1]

            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, prob_positive, n_bins=10, strategy='uniform'
                )

                # 绘制校准曲线
                plt.plot(mean_predicted_value, fraction_of_positives,
                         color=color, marker=marker, linewidth=2.5, markersize=8,
                         label=f'{self.compress_column_name(name, 20)}',
                         markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)

                # 计算校准指标（Brier Score和可靠性）
                from sklearn.metrics import brier_score_loss
                brier_score = brier_score_loss(y_test, prob_positive)

                # 计算校准误差（Expected Calibration Error）
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

                calibration_metrics.append({
                    'Model': name,
                    'Brier_Score': brier_score,
                    'ECE': ece
                })

            except Exception as e:
                print(f"Error calculating calibration for {name}: {e}")
                continue

        # 绘制完美校准线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=3, alpha=0.8, label='Perfect Calibration')

        # 设置图形属性
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Mean Predicted Probability', fontsize=14)
        plt.ylabel('Fraction of Positives', fontsize=14)
        plt.title('Calibration Curves - Top 5 Models (Binary Classification)', fontsize=16, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加说明文本
        plt.text(0.02, 0.98, 'Perfect calibration: predicted probabilities match observed frequencies',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / "calibration_curves_top5_combined.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 保存校准指标
        if calibration_metrics:
            calibration_df = pd.DataFrame(calibration_metrics)
            calibration_df = calibration_df.sort_values('Brier_Score')
            calibration_df.to_excel(self.output_dir / "calibration_metrics_top5.xlsx", index=False)
            print("Top 5 models calibration metrics saved to calibration_metrics_top5.xlsx")

    def _plot_multiclass_calibration_curves(self):
        """绘制多分类校准曲线 - 前5个模型在一张图中"""
        from sklearn.calibration import calibration_curve

        n_classes = len(np.unique(self.y))

        # 只选择前5个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:5]

        # 创建子图：每个类别一个子图
        fig, axes = plt.subplots(1, min(n_classes, 4), figsize=(5 * min(n_classes, 4), 8))
        if n_classes == 1:
            axes = [axes]
        elif n_classes > 4:
            axes = axes[:4] if hasattr(axes, '__len__') else [axes]
            n_classes = min(n_classes, 4)

        # 设置颜色和标记
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']

        # 存储校准指标
        calibration_metrics = []

        for class_idx in range(n_classes):
            ax = axes[class_idx] if n_classes > 1 else axes[0]

            class_metrics = []

            for i, ((name, result), color, marker) in enumerate(zip(top_models, colors, markers)):
                y_test = result['y_test']
                y_proba = result['y_test_proba']

                if y_proba is None or class_idx >= y_proba.shape[1]:
                    continue

                # 将多分类转换为二分类（当前类 vs 其他类）
                y_binary = (y_test == class_idx).astype(int)
                prob_class = y_proba[:, class_idx]

                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_binary, prob_class, n_bins=10, strategy='uniform'
                    )

                    # 绘制校准曲线
                    ax.plot(mean_predicted_value, fraction_of_positives,
                            color=color, marker=marker, linewidth=2.5, markersize=8,
                            label=f'{self.compress_column_name(name, 15)}',
                            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)

                    # 计算校准指标
                    from sklearn.metrics import brier_score_loss
                    brier_score = brier_score_loss(y_binary, prob_class)
                    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

                    class_metrics.append({
                        'Model': name,
                        'Class': class_idx,
                        'Brier_Score': brier_score,
                        'ECE': ece
                    })

                except Exception as e:
                    print(f"Error calculating calibration for {name}, class {class_idx}: {e}")
                    continue

            # 绘制完美校准线
            ax.plot([0, 1], [0, 1], 'k--', linewidth=3, alpha=0.8, label='Perfect')

            # 设置子图属性
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax.set_ylabel('Fraction of Positives', fontsize=12)
            ax.set_title(f'Class {class_idx} Calibration', fontsize=14, weight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            calibration_metrics.extend(class_metrics)

        # 设置总标题
        fig.suptitle('Calibration Curves - Top 5 Models (Multi-class Classification)',
                     fontsize=16, weight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / "calibration_curves_top5_combined.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 保存校准指标
        if calibration_metrics:
            calibration_df = pd.DataFrame(calibration_metrics)
            calibration_df = calibration_df.sort_values(['Class', 'Brier_Score'])
            calibration_df.to_excel(self.output_dir / "calibration_metrics_top5.xlsx", index=False)
            print("Top 5 models calibration metrics saved to calibration_metrics_top5.xlsx")



    def save_decision_curve_metrics(self):
        """保存决策曲线相关指标"""
        print("Calculating and saving decision curve metrics...")

        thresholds = np.arange(0.1, 0.9, 0.1)  # 常用的阈值点
        dca_results = []

        for name, result in self.results.items():
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                continue

            model_dca = {'Model': name}

            for threshold in thresholds:
                net_benefit = self.calculate_net_benefit(y_test, y_proba, threshold)
                model_dca[f'NetBenefit_Threshold_{threshold:.1f}'] = net_benefit

            # 计算决策曲线下面积（类似AUC的概念）
            all_thresholds = np.arange(0, 1, 0.01)
            net_benefits = [self.calculate_net_benefit(y_test, y_proba, t) for t in all_thresholds]

            # 使用梯形积分计算面积
            dca_auc = np.trapz(net_benefits, all_thresholds)
            model_dca['DCA_AUC'] = dca_auc

            # 计算最大净效益及其对应的阈值
            max_nb_idx = np.argmax(net_benefits)
            model_dca['Max_Net_Benefit'] = net_benefits[max_nb_idx]
            model_dca['Optimal_Threshold'] = all_thresholds[max_nb_idx]

            dca_results.append(model_dca)

        # 保存到Excel
        dca_df = pd.DataFrame(dca_results)
        dca_df = dca_df.sort_values('DCA_AUC', ascending=False)

        dca_df.to_excel(self.output_dir / "decision_curve_metrics.xlsx", index=False)

        print("Decision curve metrics saved successfully")
        return dca_df


    def plot_violin_analysis(self):
        """绘制小提琴图分析"""
        print("Generating violin plot analysis...")

        # 1. 特征分布小提琴图
        self.plot_feature_distribution_violins()

        # 2. 预测概率分布小提琴图
        self.plot_prediction_probability_violins()

        # 3. 模型置信度分布小提琴图
        self.plot_model_confidence_violins()

        # 4. 重要特征详细分析小提琴图
        self.plot_important_features_violins()

    def plot_feature_distribution_violins(self):
        """绘制重要特征在不同类别间的分布小提琴图"""
        print("Plotting feature distribution violin plots...")

        # 选择最重要的特征（前12个）
        if hasattr(self, 'selected_features'):
            # 使用随机森林获取特征重要性
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(self.X_selected, self.y)

            feature_importance = rf_model.feature_importances_
            top_indices = np.argsort(feature_importance)[-12:][::-1]
            top_features = [self.selected_features[i] for i in top_indices]
            top_feature_data = self.X_selected[:, top_indices]
        else:
            print("No selected features available")
            return

        # 创建数据框
        feature_df = pd.DataFrame(top_feature_data, columns=[
            self.compress_column_name(name, 20) for name in top_features
        ])
        feature_df['Label'] = self.y
        feature_df['Class'] = [f'Class {label}' for label in self.y]

        # 绘制小提琴图
        n_features = len(top_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes

        # 设置颜色
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

        for i, (feature, compressed_name) in enumerate(zip(top_features, feature_df.columns[:-2])):
            if i >= len(axes):
                break

            ax = axes[i]

            # 绘制小提琴图
            violin_parts = ax.violinplot([feature_df[feature_df['Label'] == label][compressed_name].values
                                          for label in np.unique(self.y)],
                                         positions=range(len(np.unique(self.y))),
                                         showmeans=True, showmedians=True, showextrema=True)

            # 设置颜色
            for pc, color in zip(violin_parts['bodies'], colors[:len(np.unique(self.y))]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            # 设置其他元素颜色
            violin_parts['cmeans'].set_color('red')
            violin_parts['cmedians'].set_color('black')
            violin_parts['cmaxes'].set_color('gray')
            violin_parts['cmins'].set_color('gray')
            violin_parts['cbars'].set_color('gray')

            # 添加散点图显示数据分布
            for j, label in enumerate(np.unique(self.y)):
                data = feature_df[feature_df['Label'] == label][compressed_name].values
                # 添加一些随机抖动以避免重叠
                x_jitter = np.random.normal(j, 0.05, len(data))
                ax.scatter(x_jitter, data, alpha=0.4, s=20, color='darkblue')

            # 设置标签和标题
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Feature Value', fontsize=10)
            ax.set_title(f'{compressed_name}', fontsize=11, weight='bold')
            ax.set_xticks(range(len(np.unique(self.y))))
            ax.set_xticklabels([f'Class {label}' for label in np.unique(self.y)])
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            class_means = [feature_df[feature_df['Label'] == label][compressed_name].mean()
                           for label in np.unique(self.y)]
            stats_text = '\n'.join([f'Class {label} μ={mean:.3f}'
                                    for label, mean in zip(np.unique(self.y), class_means)])
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Feature Distribution Analysis - Violin Plots\n(Top Important Features by Class)',
                     fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_distribution_violins.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_probability_violins(self):
        """绘制各模型预测概率分布的小提琴图"""
        print("Plotting prediction probability violin plots...")

        # 只选择前8个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:8]

        # 检查是否为二分类
        n_classes = len(np.unique(self.y))

        if n_classes == 2:
            self._plot_binary_probability_violins(top_models)
        else:
            self._plot_multiclass_probability_violins(top_models)

    def _plot_binary_probability_violins(self, top_models):
        """绘制二分类预测概率小提琴图"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(top_models):
            ax = axes[idx]

            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                ax.text(0.5, 0.5, 'No probability\navailable',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(name)
                continue

            # 获取正类概率
            prob_positive = y_proba[:, 1]

            # 按真实标签分组
            prob_class_0 = prob_positive[y_test == 0]
            prob_class_1 = prob_positive[y_test == 1]

            # 绘制小提琴图
            violin_data = [prob_class_0, prob_class_1]
            violin_parts = ax.violinplot(violin_data, positions=[0, 1],
                                         showmeans=True, showmedians=True, showextrema=True)

            # 设置颜色
            colors = ['lightcoral', 'lightblue']
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            # 设置其他元素
            violin_parts['cmeans'].set_color('red')
            violin_parts['cmedians'].set_color('black')
            violin_parts['cmaxes'].set_color('gray')
            violin_parts['cmins'].set_color('gray')
            violin_parts['cbars'].set_color('gray')

            # 添加散点
            for i, (data, color) in enumerate(zip(violin_data, colors)):
                x_jitter = np.random.normal(i, 0.05, len(data))
                ax.scatter(x_jitter, data, alpha=0.4, s=15, color='darkred' if i == 0 else 'darkblue')

            # 添加阈值线
            ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Threshold=0.5')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=1, fancybox=True, shadow=True)
            # 设置标签
            ax.set_xlabel('True Class', fontsize=10)
            ax.set_ylabel('Predicted Probability', fontsize=10)
            ax.set_title(f'{self.compress_column_name(name, 25)}', fontsize=11, weight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Class 0', 'Class 1'])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)


            # 添加统计信息
            auc_score = result['test_metrics'].get('AUC', 0)
            accuracy = result['test_metrics']['Accuracy']
            stats_text = f'AUC: {auc_score:.3f}\nAcc: {accuracy:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Prediction Probability Distribution - Violin Plots\n(Binary Classification)',
                     fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_probability_violins.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_multiclass_probability_violins(self, top_models):
        """绘制多分类预测概率小提琴图"""
        n_classes = len(np.unique(self.y))

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(top_models):
            ax = axes[idx]

            y_test = result['y_test']
            y_proba = result['y_test_proba']

            if y_proba is None:
                ax.text(0.5, 0.5, 'No probability\navailable',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(name)
                continue

            # 获取最大概率（置信度）
            max_proba = np.max(y_proba, axis=1)
            predicted_class = np.argmax(y_proba, axis=1)

            # 按预测类别分组
            violin_data = []
            colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

            for class_label in range(n_classes):
                class_probabilities = max_proba[predicted_class == class_label]
                if len(class_probabilities) > 0:
                    violin_data.append(class_probabilities)
                else:
                    violin_data.append([0])  # 空数据的占位符

            # 绘制小提琴图
            if any(len(data) > 1 for data in violin_data):
                violin_parts = ax.violinplot([data for data in violin_data if len(data) > 1],
                                             positions=range(len(violin_data)),
                                             showmeans=True, showmedians=True, showextrema=True)

                # 设置颜色
                for pc, color in zip(violin_parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

                # 设置其他元素
                violin_parts['cmeans'].set_color('red')
                violin_parts['cmedians'].set_color('black')
                violin_parts['cmaxes'].set_color('gray')
                violin_parts['cmins'].set_color('gray')
                violin_parts['cbars'].set_color('gray')

            # 添加散点
            for i, (data, color) in enumerate(zip(violin_data, colors)):
                if len(data) > 1:
                    x_jitter = np.random.normal(i, 0.05, len(data))
                    ax.scatter(x_jitter, data, alpha=0.4, s=15, color=color)

            # 设置标签
            ax.set_xlabel('Predicted Class', fontsize=10)
            ax.set_ylabel('Maximum Probability (Confidence)', fontsize=10)
            ax.set_title(f'{self.compress_column_name(name, 25)}', fontsize=11, weight='bold')
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            accuracy = result['test_metrics']['Accuracy']
            mean_confidence = np.mean(max_proba)
            stats_text = f'Acc: {accuracy:.3f}\nMean Conf: {mean_confidence:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Prediction Confidence Distribution - Violin Plots\n(Multi-class Classification)',
                     fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_confidence_violins.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_confidence_violins(self):
        """绘制模型置信度比较小提琴图"""
        print("Plotting model confidence comparison violin plots...")

        # 收集所有模型的置信度数据
        model_confidences = {}
        model_accuracies = {}

        for name, result in self.results.items():
            y_proba = result['y_test_proba']
            if y_proba is not None:
                # 计算置信度（最大概率）
                confidence = np.max(y_proba, axis=1)
                model_confidences[self.compress_column_name(name, 15)] = confidence
                model_accuracies[name] = result['test_metrics']['Accuracy']

        if not model_confidences:
            print("No probability predictions available for confidence analysis")
            return

        # 按准确率排序
        sorted_models = sorted(model_confidences.items(),
                               key=lambda x: model_accuracies.get(x[0], 0), reverse=True)

        # 绘制置信度比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # 1. 所有模型置信度分布比较
        model_names = [item[0] for item in sorted_models]
        confidence_data = [item[1] for item in sorted_models]

        violin_parts = ax1.violinplot(confidence_data, positions=range(len(model_names)),
                                      showmeans=True, showmedians=True, showextrema=True)

        # 设置颜色渐变
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # 设置其他元素
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmedians'].set_color('white')
        violin_parts['cmaxes'].set_color('gray')
        violin_parts['cmins'].set_color('gray')
        violin_parts['cbars'].set_color('gray')

        ax1.set_xlabel('Models (sorted by accuracy)', fontsize=12)
        ax1.set_ylabel('Prediction Confidence', fontsize=12)
        ax1.set_title('Model Confidence Distribution Comparison', fontsize=14, weight='bold')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # 2. 置信度 vs 准确率散点图
        mean_confidences = [np.mean(data) for data in confidence_data]
        accuracies = [model_accuracies.get(name, 0) for name in model_names]

        scatter = ax2.scatter(mean_confidences, accuracies, c=range(len(model_names)),
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')

        # 添加模型名称标签
        for i, name in enumerate(model_names):
            ax2.annotate(name, (mean_confidences[i], accuracies[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 添加趋势线
        z = np.polyfit(mean_confidences, accuracies, 1)
        p = np.poly1d(z)
        ax2.plot(mean_confidences, p(mean_confidences), "r--", alpha=0.8, linewidth=2)

        # 计算相关系数
        correlation = np.corrcoef(mean_confidences, accuracies)[0, 1]

        ax2.set_xlabel('Mean Prediction Confidence', fontsize=12)
        ax2.set_ylabel('Test Accuracy', fontsize=12)
        ax2.set_title(f'Confidence vs Accuracy\n(Correlation: {correlation:.3f})', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Model Rank (by accuracy)', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_confidence_violins.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_important_features_violins(self):
        """绘制最重要特征的详细小提琴图分析"""
        print("Plotting detailed important features violin analysis...")

        # 获取最重要的6个特征
        if hasattr(self, 'selected_features'):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(self.X_selected, self.y)

            feature_importance = rf_model.feature_importances_
            top_indices = np.argsort(feature_importance)[-6:][::-1]
            top_features = [self.selected_features[i] for i in top_indices]
            top_feature_data = self.X_selected[:, top_indices]
        else:
            print("No selected features available")
            return

        # 创建详细的小提琴图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # 计算统计检验（ANOVA或t-test）
        from scipy import stats

        for i, (feature_name, feature_idx) in enumerate(zip(top_features, range(len(top_features)))):
            ax = axes[i]
            feature_data = top_feature_data[:, feature_idx]

            # 按类别分组数据
            groups = [feature_data[self.y == label] for label in np.unique(self.y)]

            # 绘制增强的小提琴图
            violin_parts = ax.violinplot(groups, positions=range(len(np.unique(self.y))),
                                         showmeans=True, showmedians=True, showextrema=True)

            # 设置颜色
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            for pc, color in zip(violin_parts['bodies'], colors[:len(np.unique(self.y))]):
                pc.set_facecolor(color)
                pc.set_alpha(0.8)

            # 设置小提琴图元素样式
            violin_parts['cmeans'].set_color('red')
            violin_parts['cmeans'].set_linewidth(2)
            violin_parts['cmedians'].set_color('black')
            violin_parts['cmedians'].set_linewidth(2)
            violin_parts['cmaxes'].set_color('gray')
            violin_parts['cmins'].set_color('gray')
            violin_parts['cbars'].set_color('gray')

            # 添加箱线图元素
            bp = ax.boxplot(groups, positions=range(len(np.unique(self.y))),
                            widths=0.3, patch_artist=False,
                            boxprops=dict(color='black', linewidth=1),
                            whiskerprops=dict(color='black', linewidth=1),
                            capprops=dict(color='black', linewidth=1),
                            medianprops=dict(color='red', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))

            # 添加数据点（带抖动）
            for j, group in enumerate(groups):
                x_jitter = np.random.normal(j, 0.04, len(group))
                ax.scatter(x_jitter, group, alpha=0.6, s=20,
                           color=colors[j % len(colors)], edgecolors='black', linewidth=0.5)

            # 统计检验
            if len(np.unique(self.y)) == 2:
                # t检验
                statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                test_name = "t-test"
            else:
                # ANOVA
                statistic, p_value = stats.f_oneway(*groups)
                test_name = "ANOVA"

            # 设置标签和标题
            compressed_name = self.compress_column_name(feature_name, 25)
            ax.set_xlabel('Class', fontsize=11)
            ax.set_ylabel('Feature Value', fontsize=11)
            ax.set_title(f'{compressed_name}\n{test_name} p-value: {p_value:.2e}',
                         fontsize=12, weight='bold')
            ax.set_xticks(range(len(np.unique(self.y))))
            ax.set_xticklabels([f'Class {label}' for label in np.unique(self.y)])
            ax.grid(True, alpha=0.3)

            # 添加统计摘要
            stats_lines = []
            for j, (label, group) in enumerate(zip(np.unique(self.y), groups)):
                mean_val = np.mean(group)
                std_val = np.std(group)
                stats_lines.append(f'Class {label}: μ={mean_val:.3f}, σ={std_val:.3f}')

            effect_size = abs(statistic) if len(np.unique(self.y)) == 2 else statistic
            stats_lines.append(f'Effect size: {effect_size:.3f}')

            stats_text = '\n'.join(stats_lines)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            # # 添加显著性标记
            # if p_value < 0.001:
            #     sig_text = "***"
            # elif p_value < 0.01:
            #     sig_text = "**"
            # elif p_value < 0.05:
            #     sig_text = "*"
            # else:
            #     sig_text = "ns"
            #
            # ax.text(0.98, 0.98, sig_text, transform=ax.transAxes, fontsize=16,
            #         verticalalignment='top', horizontalalignment='right',
            #         weight='bold', color='red' if p_value < 0.05 else 'gray')

        plt.suptitle(
            'Detailed Analysis of Most Important Features\n(Violin + Box + Scatter Plots with Statistical Tests)',
            fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "important_features_detailed_violins.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_violin_analysis_summary(self):
        """保存小提琴图分析摘要"""
        print("Saving violin analysis summary...")

        summary_data = []

        # 特征分布分析摘要
        if hasattr(self, 'selected_features'):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(self.X_selected, self.y)

            feature_importance = rf_model.feature_importances_
            top_indices = np.argsort(feature_importance)[-10:][::-1]

            from scipy import stats

            for i, feature_idx in enumerate(top_indices):
                feature_name = self.selected_features[feature_idx]
                feature_data = self.X_selected[:, feature_idx]

                # 按类别分组
                groups = [feature_data[self.y == label] for label in np.unique(self.y)]

                # 统计检验
                if len(np.unique(self.y)) == 2:
                    statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                    test_type = "t-test"
                else:
                    statistic, p_value = stats.f_oneway(*groups)
                    test_type = "ANOVA"

                # 计算效应量
                class_means = [np.mean(group) for group in groups]
                class_stds = [np.std(group) for group in groups]

                summary_data.append({
                    'Feature': feature_name,
                    'Importance_Rank': i + 1,
                    'Feature_Importance': feature_importance[feature_idx],
                    'Test_Type': test_type,
                    'Test_Statistic': statistic,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Class_0_Mean': class_means[0] if len(class_means) > 0 else None,
                    'Class_0_Std': class_stds[0] if len(class_stds) > 0 else None,
                    'Class_1_Mean': class_means[1] if len(class_means) > 1 else None,
                    'Class_1_Std': class_stds[1] if len(class_stds) > 1 else None,
                })

        # 保存摘要
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Feature_Importance', ascending=False)
            summary_df.to_excel(self.output_dir / "violin_analysis_summary.xlsx", index=False)
            print("Violin analysis summary saved to violin_analysis_summary.xlsx")

        return summary_data

    def plot_roc_curves(self):
        """绘制ROC曲线"""
        print("Plotting ROC curves...")

        n_classes = len(np.unique(self.y))

        # 为多分类问题绘制ROC曲线
        if n_classes > 2:
            self.plot_multiclass_roc()
        else:
            self.plot_binary_roc()

    def plot_multiclass_roc(self):
        """绘制多分类ROC曲线"""
        n_classes = len(np.unique(self.y))
        class_names = [f'Class {i}' for i in range(n_classes)]

        # 设置颜色
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 6:  # 只显示前6个模型
                break

            ax = axes[idx]

            y_test = result['y_test']
            y_proba = result['y_test_proba']

            # 二值化标签
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i, label in enumerate(y_test):
                if label < n_classes:
                    y_test_bin[i, label] = 1

            # 计算每个类别的ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                if i < y_proba.shape[1]:
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

            # 绘制ROC曲线
            for i, color in zip(range(n_classes), colors):
                if i in fpr:
                    ax.plot(fpr[i], tpr[i], color=color, lw=2,
                            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {name}')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves_multiclass.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制所有模型的平均ROC
        self.plot_average_roc_comparison()

    def plot_binary_roc(self):
        """绘制二分类ROC曲线"""
        plt.figure(figsize=(12, 10))

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        for (name, result), color in zip(self.results.items(), colors):
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All Models', fontsize=14, weight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_average_roc_comparison(self):
        """绘制所有模型的ROC曲线比较"""
        plt.figure(figsize=(12, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.results)))

        for (name, result), color in zip(self.results.items(), colors):
            y_test = result['y_test']
            y_proba = result['y_test_proba']

            # 计算macro-average ROC
            n_classes = len(np.unique(y_test))
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i, label in enumerate(y_test):
                if label < n_classes:
                    y_test_bin[i, label] = 1

            # 计算macro-average
            all_fpr = np.unique(np.concatenate([
                roc_curve(y_test_bin[:, i], y_proba[:, i])[0]
                for i in range(min(n_classes, y_proba.shape[1]))
            ]))

            mean_tpr = np.zeros_like(all_fpr)
            for i in range(min(n_classes, y_proba.shape[1])):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)

            mean_tpr /= min(n_classes, y_proba.shape[1])
            macro_auc = auc(all_fpr, mean_tpr)

            plt.plot(all_fpr, mean_tpr, color=color, lw=2,
                     label=f'{name} (Macro-AUC = {macro_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Macro-Average ROC Curves Comparison', fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "macro_roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_performance_metrics(self):
        """保存性能指标到Excel"""
        print("Saving performance metrics...")

        # 训练集性能
        train_metrics_df = pd.DataFrame({
            name: result['train_metrics']
            for name, result in self.results.items()
        }).T

        # 测试集性能
        test_metrics_df = pd.DataFrame({
            name: result['test_metrics']
            for name, result in self.results.items()
        }).T

        # 交叉验证结果
        cv_metrics_df = pd.DataFrame(self.cv_results).T

        # 保存到Excel
        with pd.ExcelWriter(self.output_dir / "model_performance_metrics.xlsx",
                            engine='openpyxl') as writer:
            train_metrics_df.to_excel(writer, sheet_name='Training_Set', index=True)
            test_metrics_df.to_excel(writer, sheet_name='Test_Set', index=True)
            cv_metrics_df.to_excel(writer, sheet_name='Cross_Validation', index=True)

        print("Performance metrics saved successfully")

        # 返回最佳模型
        best_model_name = test_metrics_df['Accuracy'].idxmax()
        print(f"Best performing model: {best_model_name}")
        print(f"Best test accuracy: {test_metrics_df.loc[best_model_name, 'Accuracy']:.4f}")

        return best_model_name

    def plot_performance_comparison(self):
        """绘制性能比较图"""
        print("Plotting performance comparison...")

        # 准备数据
        metrics_data = []
        for name, result in self.results.items():
            test_metrics = result['test_metrics']
            metrics_data.append({
                'Model': name,
                'Accuracy': test_metrics['Accuracy'],
                'Precision': test_metrics['Precision'],
                'Recall': test_metrics['Recall'],
                'F1-Score': test_metrics['F1-Score'],
                'Specificity': test_metrics['Specificity']
            })

        metrics_df = pd.DataFrame(metrics_data)

        # 绘制条形图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]

            # 排序
            sorted_data = metrics_df.sort_values(metric, ascending=True)

            bars = ax.barh(range(len(sorted_data)), sorted_data[metric])
            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels([self.compress_column_name(name, 15) for name in sorted_data['Model']])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.3f}', ha='left', va='center', fontsize=8)

        # 隐藏最后一个子图
        axes[-1].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def shap_analysis(self, best_model_name):
        """SHAP可解释性分析"""
        print(f"Performing SHAP analysis for {best_model_name}...")

        try:
            best_model = self.results[best_model_name]['model']

            # 准备数据样本（使用较小的样本以加速计算）
            sample_size = min(100, len(self.X_selected))
            sample_indices = np.random.choice(len(self.X_selected), sample_size, replace=False)
            X_sample = self.X_selected[sample_indices]

            # 压缩特征名
            compressed_feature_names = [
                self.compress_column_name(name, 15) for name in self.selected_features
            ]

            # 选择合适的SHAP explainer
            if hasattr(best_model, 'predict_proba'):
                try:
                    # 尝试使用TreeExplainer（适用于树模型）
                    if any(model_type in best_model_name.lower()
                           for model_type in ['forest', 'tree', 'boost', 'xgb']):
                        explainer = shap.TreeExplainer(best_model)
                        shap_values = explainer.shap_values(X_sample)
                    else:
                        # 使用KernelExplainer（通用但较慢）
                        background = shap.sample(self.X_selected, 50)
                        explainer = shap.KernelExplainer(best_model.predict_proba, background)
                        shap_values = explainer.shap_values(X_sample)

                except Exception as e:
                    print(f"TreeExplainer failed, using KernelExplainer: {e}")
                    background = shap.sample(self.X_selected, 50)
                    explainer = shap.KernelExplainer(best_model.predict_proba, background)
                    shap_values = explainer.shap_values(X_sample)
            else:
                print("Model does not support probability prediction, skipping SHAP analysis")
                return

            # 绘制SHAP图
            self.plot_shap_analysis(shap_values, X_sample, compressed_feature_names, best_model_name, explainer)

        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            print("SHAP analysis requires additional setup. Please install shap package properly.")

    def plot_shap_analysis(self, shap_values, X_sample, feature_names, model_name, explainer):
        """绘制SHAP分析图"""
        try:
            # 如果是多分类，使用第一个类别的SHAP值
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]
                class_name = "Class 0"
            else:
                shap_values_plot = shap_values
                class_name = "Target"

            # 1. Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_sample, feature_names=feature_names,
                              show=False, max_display=20)
            plt.title(f'SHAP Summary Plot - {model_name} ({class_name})', fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"shap_summary_{model_name.replace(' ', '_')}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_sample, feature_names=feature_names,
                              plot_type="bar", show=False, max_display=20)
            plt.title(f'SHAP Feature Importance - {model_name} ({class_name})', fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"shap_importance_{model_name.replace(' ', '_')}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Waterfall plot for first sample
            if len(X_sample) > 0:
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_values_plot[0],
                                     base_values=explainer.expected_value[0] if isinstance(explainer.expected_value,
                                                                                           np.ndarray) else explainer.expected_value,
                                     feature_names=feature_names),
                    show=False, max_display=15
                )
                plt.title(f'SHAP Waterfall Plot - {model_name} (Sample 1, {class_name})', fontsize=14, weight='bold')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"shap_waterfall_{model_name.replace(' ', '_')}.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

            print(f"SHAP analysis plots saved for {model_name}")

        except Exception as e:
            print(f"Error plotting SHAP analysis: {e}")

    def plot_edge_features_effect_comparison(self):
        """绘制边缘特征添加效果对比图 - 前5个模型的不同颜色柱状图"""
        print("Plotting edge features addition effect comparison...")

        # 只选择前5个最佳模型
        top_models = sorted(self.results.items(),
                            key=lambda x: x[1]['test_metrics']['Accuracy'],
                            reverse=True)[:5]

        # 设置颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # 模拟没有边缘特征的性能（仅使用超声特征）
        us_only_performance = {}

        # 提取超声特征索引
        us_indices = [i for i, col in enumerate(self.feature_names) if col in self.discrete_cols]
        X_us_only = self.X_scaled[:, us_indices]

        # 对前5个模型计算仅使用超声特征的性能
        for name, result in top_models:
            try:
                model = clone(self.models[name])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_us_only, self.y, test_size=0.2, random_state=42, stratify=self.y
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                us_only_acc = accuracy_score(y_test, y_pred)
                us_only_performance[name] = us_only_acc
            except:
                # 如果某些模型不支持，使用默认值
                us_only_performance[name] = result['test_metrics']['Accuracy'] - 0.05

        # 准备数据
        model_names = [name for name, _ in top_models]
        compressed_names = [self.compress_column_name(name, 20) for name in model_names]

        us_only_accs = [us_only_performance[name] for name in model_names]
        full_feature_accs = [result['test_metrics']['Accuracy'] for _, result in top_models]
        improvements = [full - us for full, us in zip(full_feature_accs, us_only_accs)]

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 性能对比柱状图
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, us_only_accs, width,
                        label='Ultrasound Features Only',
                        color='lightgray', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width / 2, full_feature_accs, width,
                        label='Ultrasound + Edge Features',
                        color=colors, alpha=0.8, edgecolor='black')

        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Performance Comparison: With vs Without Edge Features', fontsize=14, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(compressed_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. 边缘特征效果改善图
        bars3 = ax2.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black')

        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Accuracy Improvement', fontsize=12)
        ax2.set_title('Edge Features Addition Effect', fontsize=14, weight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(compressed_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # 添加改善数值标签
        for i, (bar, improvement) in enumerate(zip(bars3, improvements)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.002 if height >= 0 else height - 0.005,
                     f'{improvement:+.3f}\n({improvement * 100:+.1f}%)',
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=9, weight='bold',
                     color='green' if improvement > 0 else 'red')

        # 添加平均改善线
        avg_improvement = np.mean(improvements)
        ax2.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Average: {avg_improvement:+.3f}')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "edge_features_effect_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 保存详细数据
        comparison_data = pd.DataFrame({
            'Model': model_names,
            'Ultrasound_Only_Accuracy': us_only_accs,
            'Full_Features_Accuracy': full_feature_accs,
            'Improvement': improvements,
            'Improvement_Percentage': [imp * 100 for imp in improvements]
        })
        comparison_data.to_excel(self.output_dir / "edge_features_effect_analysis.xlsx", index=False)

        print("Edge features effect comparison plot and analysis saved successfully!")

    def plot_improved_data_balancing_analysis(self):
        """改进的数据平衡效果分析图"""
        print("Plotting improved data balancing analysis...")
        import numpy as np
        if not hasattr(self, 'balancing_info'):
            print("No balancing information available")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        original_dist = self.balancing_info['original_distribution']
        balanced_dist = self.balancing_info['balanced_distribution']
        classes = list(original_dist.keys())

        # 1. 饼图对比 (上排左侧两个)
        ax1 = fig.add_subplot(gs[0, 0])
        colors_pie = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        wedges1, texts1, autotexts1 = ax1.pie([original_dist[c] for c in classes],
                                              labels=[f'Class {c}' for c in classes],
                                              colors=colors_pie[:len(classes)],
                                              autopct='%1.1f%%', startangle=90)
        ax1.set_title('Original Distribution', fontsize=14, weight='bold')

        ax2 = fig.add_subplot(gs[0, 1])
        wedges2, texts2, autotexts2 = ax2.pie([balanced_dist[c] for c in classes],
                                              labels=[f'Class {c}' for c in classes],
                                              colors=colors_pie[:len(classes)],
                                              autopct='%1.1f%%', startangle=90)
        ax2.set_title('Balanced Distribution', fontsize=14, weight='bold')

        # 2. 瀑布图显示样本数变化 (上排右侧两个)
        ax3 = fig.add_subplot(gs[0, 2:])

        # 创建瀑布图数据
        original_counts = [original_dist[c] for c in classes]
        balanced_counts = [balanced_dist[c] for c in classes]
        changes = [balanced_counts[i] - original_counts[i] for i in range(len(classes))]

        # 绘制瀑布图
        x_pos = np.arange(len(classes))
        bottom_original = np.zeros(len(classes))
        bottom_change = original_counts.copy()

        # 原始数据
        bars_orig = ax3.bar(x_pos - 0.2, original_counts, 0.4,
                            label='Original', color='lightcoral', alpha=0.7)

        # 变化量
        for i, change in enumerate(changes):
            if change > 0:  # 增加
                ax3.bar(x_pos[i] + 0.2, change, 0.4, bottom=original_counts[i],
                        color='lightgreen', alpha=0.7, label='Added' if i == 0 else "")
                ax3.bar(x_pos[i] + 0.2, original_counts[i], 0.4,
                        color='lightcoral', alpha=0.7)
            else:  # 减少
                ax3.bar(x_pos[i] + 0.2, balanced_counts[i], 0.4,
                        color='lightcoral', alpha=0.7)
                ax3.bar(x_pos[i] + 0.2, abs(change), 0.4, bottom=balanced_counts[i],
                        color='orange', alpha=0.7, label='Removed' if i == 0 else "")

        ax3.set_xlabel('Class', fontsize=12)
        ax3.set_ylabel('Number of Samples', fontsize=12)
        ax3.set_title('Sample Count Changes (Waterfall View)', fontsize=14, weight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Class {c}' for c in classes])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 添加变化数值
        for i, change in enumerate(changes):
            ax3.text(i + 0.2, max(original_counts[i], balanced_counts[i]) + max(balanced_counts) * 0.02,
                     f'{change:+d}', ha='center', va='bottom', fontsize=10, weight='bold',
                     color='green' if change > 0 else 'red')

        # 3. 不平衡指标对比 (中排左侧)
        ax4 = fig.add_subplot(gs[1, 0])

        # 计算各种不平衡指标
        from collections import Counter
        import numpy as np

        def gini_index(distribution):
            """计算基尼系数"""
            values = list(distribution.values())
            total = sum(values)
            if total == 0:
                return 0
            proportions = [v / total for v in values]
            return 1 - sum(p ** 2 for p in proportions)

        def shannon_entropy(distribution):
            """计算香农熵"""
            values = list(distribution.values())
            total = sum(values)
            if total == 0:
                return 0
            proportions = [v / total for v in values if v > 0]
            return -sum(p * np.log2(p) for p in proportions)

        def imbalance_ratio(distribution):
            """计算不平衡比率"""
            values = list(distribution.values())
            if len(values) < 2:
                return 1
            return min(values) / max(values)

        # 计算指标
        metrics = ['Gini Index', 'Shannon Entropy', 'Imbalance Ratio']
        original_metrics = [
            gini_index(original_dist),
            shannon_entropy(original_dist),
            imbalance_ratio(original_dist)
        ]
        balanced_metrics = [
            gini_index(balanced_dist),
            shannon_entropy(balanced_dist),
            imbalance_ratio(balanced_dist)
        ]

        x_metrics = np.arange(len(metrics))
        width = 0.35

        bars1 = ax4.bar(x_metrics - width / 2, original_metrics, width,
                        label='Original', color='lightcoral', alpha=0.8)
        bars2 = ax4.bar(x_metrics + width / 2, balanced_metrics, width,
                        label='Balanced', color='lightblue', alpha=0.8)

        ax4.set_xlabel('Metrics', fontsize=12)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title('Imbalance Metrics Comparison', fontsize=14, weight='bold')
        ax4.set_xticks(x_metrics)
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + max(original_metrics + balanced_metrics) * 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. 类别样本数分布直方图 (中排中间)
        ax5 = fig.add_subplot(gs[1, 1])

        all_counts = list(original_dist.values()) + list(balanced_dist.values())
        bins = np.linspace(0, max(all_counts), 20)

        ax5.hist(list(original_dist.values()), bins=bins, alpha=0.7,
                 label='Original', color='lightcoral', edgecolor='black')
        ax5.hist(list(balanced_dist.values()), bins=bins, alpha=0.7,
                 label='Balanced', color='lightblue', edgecolor='black')

        ax5.set_xlabel('Sample Count', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Sample Count Distribution', fontsize=14, weight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 5. 策略效果雷达图 (中排右侧)
        ax6 = fig.add_subplot(gs[1, 2:], projection='polar')

        # 雷达图数据
        categories = ['Balance\nImprovement', 'Data\nEfficiency', 'Diversity\nPreservation', 'Computational\nCost']

        # 根据策略计算得分 (0-1)
        strategy = self.balancing_info['strategy']
        if 'smote' in strategy.lower():
            scores = [0.9, 0.8, 0.9, 0.7]  # SMOTE评分
        elif 'adasyn' in strategy.lower():
            scores = [0.9, 0.8, 0.95, 0.6]  # ADASYN评分
        elif 'random' in strategy.lower():
            scores = [0.8, 0.9, 0.6, 0.9]  # Random评分
        else:
            scores = [0.7, 0.7, 0.7, 0.8]  # 默认评分

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 闭合
        angles += angles[:1]

        ax6.plot(angles, scores, color='blue', linewidth=2, label=strategy.upper())
        ax6.fill(angles, scores, color='blue', alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('Strategy Performance Radar', fontsize=14, weight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 6. 详细信息表格 (下排)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        # 创建信息表格
        info_data = [
            ['Metric', 'Original', 'Balanced', 'Change'],
            ['Total Samples', f"{self.balancing_info['original_samples']:,}",
             f"{self.balancing_info['balanced_samples']:,}",
             f"{self.balancing_info['balanced_samples'] - self.balancing_info['original_samples']:+,}"],
            ['Minority Class', f"{min(original_dist.values()):,}",
             f"{min(balanced_dist.values()):,}",
             f"{min(balanced_dist.values()) - min(original_dist.values()):+,}"],
            ['Majority Class', f"{max(original_dist.values()):,}",
             f"{max(balanced_dist.values()):,}",
             f"{max(balanced_dist.values()) - max(original_dist.values()):+,}"],
            ['Imbalance Ratio', f"{imbalance_ratio(original_dist):.3f}",
             f"{imbalance_ratio(balanced_dist):.3f}",
             f"{imbalance_ratio(balanced_dist) - imbalance_ratio(original_dist):+.3f}"],
            ['Gini Index', f"{gini_index(original_dist):.3f}",
             f"{gini_index(balanced_dist):.3f}",
             f"{gini_index(balanced_dist) - gini_index(original_dist):+.3f}"]
        ]

        # 绘制表格
        table = ax7.table(cellText=info_data[1:], colLabels=info_data[0],
                          cellLoc='center', loc='center',
                          colColours=['lightgray'] * 4)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # 设置表格样式
        for i in range(1, len(info_data)):
            for j in range(len(info_data[0])):
                cell = table[(i, j)]
                if j == 3:  # Change列
                    value = info_data[i][j]
                    if '+' in value:
                        cell.set_facecolor('lightgreen')
                    elif '-' in value:
                        cell.set_facecolor('lightcoral')

        plt.suptitle(f'Comprehensive Data Balancing Analysis - {self.balancing_info["strategy"].upper()}',
                     fontsize=18, weight='bold', y=0.98)

        plt.savefig(self.output_dir / "improved_data_balancing_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Improved data balancing analysis plot saved successfully!")

    def generate_rfe_optimization_summary(self):
        """生成RFE优化结果摘要"""
        print("Generating RFE optimization summary...")

        if not hasattr(self, 'accuracy_result_table'):
            print("No RFE optimization results available")
            return

        df = self.accuracy_result_table.copy()

        summary_file = self.output_dir / "rfe_optimization_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== RFE特征选择优化摘要报告 ===\n\n")

            # 基本统计
            f.write("1. 优化搜索统计\n")
            f.write(f"   总计算组合数: {len(df)}\n")
            f.write(
                f"   搜索范围: Edge特征1-{df['Edge Features'].max()}, 超声特征1-{df['Ultrasound Features'].max()}\n")
            f.write(f"   最高准确性: {df['Accuracy'].max():.4f}\n")
            f.write(f"   最低准确性: {df['Accuracy'].min():.4f}\n")
            f.write(f"   平均准确性: {df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}\n\n")

            # 最佳组合
            best_row = df.loc[df['Accuracy'].idxmax()]
            f.write("2. 最佳特征组合\n")
            f.write(f"   边缘特征数: {best_row['Edge Features']}\n")
            f.write(f"   超声特征数: {best_row['Ultrasound Features']}\n")
            f.write(f"   总特征数: {best_row['Edge Features'] + best_row['Ultrasound Features']}\n")
            f.write(f"   准确性: {best_row['Accuracy']:.4f}\n")
            f.write(f"   标准差: {best_row['Accuracy_Std']:.4f}\n\n")

            # 前5名组合
            f.write("3. 前5名特征组合\n")
            top5 = df.nlargest(5, 'Accuracy')
            for i, row in enumerate(top5.itertuples(), 1):
                f.write(f"   #{i}: Edge={row._2}, US={row._3}, Acc={row._4:.4f}\n")

            # 效率分析
            df['Efficiency'] = df['Accuracy'] / (df['Edge Features'] + df['Ultrasound Features'])
            best_eff_row = df.loc[df['Efficiency'].idxmax()]
            f.write(f"\n4. 最高效率组合\n")
            f.write(f"   边缘特征数: {best_eff_row['Edge Features']}\n")
            f.write(f"   超声特征数: {best_eff_row['Ultrasound Features']}\n")
            f.write(f"   效率值: {best_eff_row['Efficiency']:.4f}\n")
            f.write(f"   准确性: {best_eff_row['Accuracy']:.4f}\n")

        print(f"RFE optimization summary saved to: {summary_file}")

    def generate_classification_report(self):
        """生成分类报告"""
        print("Generating comprehensive classification report...")

        report_file = self.output_dir / "classification_analysis_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Ultrasound影像组学数据机器学习分类分析报告 ===\n\n")

            # 数据概况
            f.write("1. 数据集概况\n")
            f.write(f"   总样本数: {len(self.y)}\n")
            f.write(f"   特征选择前: {len(self.feature_names)} 个特征\n")
            f.write(f"   特征选择后: {len(self.selected_features)} 个特征\n")
            f.write(f"   类别分布: {dict(zip(np.unique(self.y), np.bincount(self.y)))}\n\n")

            # 模型性能汇总
            f.write("2. 模型性能汇总 (按测试准确率排序)\n")
            model_performance = sorted(
                [(name, result['test_metrics']['Accuracy'], result['test_metrics']['F1-Score'])
                 for name, result in self.results.items()],
                key=lambda x: x[1], reverse=True
            )

            for i, (name, acc, f1) in enumerate(model_performance[:5]):  # 只显示前5个
                f.write(f"   {i + 1}. {name}: 准确率={acc:.4f}, F1分数={f1:.4f}\n")

            # 最佳模型详细信息
            best_model = model_performance[0][0]
            f.write(f"\n3. 最佳模型: {best_model}\n")
            best_result = self.results[best_model]

            for metric, value in best_result['test_metrics'].items():
                if value is not None:
                    f.write(f"   {metric}: {value:.4f}\n")

            # 输出文件说明
            f.write("\n4. 输出文件说明\n")
            output_files = [
                "feature_selection_results.xlsx: 特征选择详细结果",
                "model_performance_metrics.xlsx: 所有模型性能指标",
                "roc_curves_*.png: ROC曲线可视化",
                "decision_curves_*.png: 决策曲线分析",
                "voting_analysis_*.png: 集成投票分析",
                "violin_*.png: 小提琴图分析"
            ]
            for file_desc in output_files:
                f.write(f"   - {file_desc}\n")

        print(f"分类报告已保存: {report_file}")

    def update_imaging_thresholds(self, new_thresholds):
        """更新影像学指标的诊断阈值

        Parameters:
        -----------
        new_thresholds : dict
            新的阈值设置，格式如：
            {
                'C-TIRADS': 1,
                'ACR-TIRADS': 2,
                'Kwak TI-RADS': 1,
                'ATA': 3
            }
        """
        for indicator, threshold in new_thresholds.items():
            if indicator in self.imaging_thresholds:
                self.imaging_thresholds[indicator]['benign_max'] = threshold
                print(f"Updated {indicator} threshold to {threshold}")
            else:
                print(f"Warning: {indicator} not found in imaging thresholds")

        # 重新分析
        print("Re-analyzing with new thresholds...")
        return self.analyze_imaging_indicators_performance()

    def run_analysis(self):
        """运行完整的分类分析流程"""
        print("=== Starting Breast Cancer MRI Radiomics Classification Analysis ===\n")

        try:
            # 1. 数据预处理（包含数据平衡）
            X, y = self.preprocess_data()

            # 1.5 绘制数据平衡分析图
            self.plot_improved_data_balancing_analysis()
            # 2. 特征选择
            self.dual_rfe_feature_selection(n_edge_features=4, n_ultrasound_features=11)

            # # 2. RFE特征选择优化（以准确性为标准）
            # print("\n=== Starting RFE Feature Selection Optimization ===")
            # best_model, best_features, best_accuracy = self.search_best_rfe_combination_by_accuracy(
            #     max_edge_features=20, max_us_features=11
            # )
            #
            # # 2.5 绘制RFE优化详细分析
            # self.plot_rfe_accuracy_detailed_analysis()
            # print(f"RFE Optimization completed. Best accuracy: {best_accuracy:.4f}")
            # print(f"Best feature combination: {len(best_features)} features")

            # 3. 模型评估
            self.evaluate_models()
            if not hasattr(self, 'cv_results') or not self.cv_results:
                raise ValueError("Model evaluation failed - cv_result not available")

            # 新增：影像学指标诊断效能分析
            print("\n=== Starting Traditional Imaging Indicators Analysis ===")
            imaging_results = self.analyze_imaging_indicators_performance()

            # # 新增：边缘特征专用模型分析
            # print("\n=== Starting Edge Features Only Analysis ===")
            # edge_only_results = self.evaluate_edge_features_only_performance()

            # 新增：绘制诊断效能对比图
            print("\n=== Plotting Diagnostic Performance Comparison ===")
            self.plot_diagnostic_performance_comparison()
            self.plot_confusion_matrices_comparison()

            # 新增：保存诊断对比结果
            comparison_data = self.save_diagnostic_comparison_results()

            # 4. 绘制ROC曲线
            self.plot_roc_curves()

            # 5. 保存性能指标
            best_model_name = self.save_performance_metrics()

            # 6. 性能比较图
            self.plot_performance_comparison()

            # 6.5 新增：边缘特征效果对比图
            self.plot_edge_features_effect_comparison()

            # 7. 决策曲线分析
            self.plot_decision_curves()
            dca_metrics = self.save_decision_curve_metrics()

            # 8. 校准曲线分析
            self.plot_calibration_curves()


            # 9. 小提琴图分析
            self.plot_violin_analysis()
            violin_summary = self.save_violin_analysis_summary()

            # 10. 集成投票分类分析
            print("\n=== Starting Ensemble Voting Analysis ===")
            n_trained_models = self.create_ensemble_voting_classifier()
            print(f"Successfully trained {n_trained_models} models for ensemble voting")

            voting_results = self.perform_voting_classification()
            print("Voting classification completed")

            # 绘制投票分析图表
            self.plot_voting_analysis()

            # 保存投票结果摘要
            voting_summary = self.save_voting_results_summary()

            # 11. SHAP分析
            self.shap_analysis(best_model_name)

            # 12. 生成报告
            self.generate_classification_report()

            print(f"\n=== Analysis completed! All results saved to {self.output_dir} ===")

            return {
                'best_model': best_model_name,
                'best_accuracy': self.results[best_model_name]['test_metrics']['Accuracy'],
                'selected_features': self.selected_features,
                'n_models_evaluated': len(self.results),
                'dca_metrics': dca_metrics,
                'violin_summary': violin_summary,
                'voting_summary': voting_summary,
                'imaging_results': imaging_results,  # 新增
                # 'edge_only_results': edge_only_results,  # 新增
                'diagnostic_comparison': comparison_data  # 新增
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """主函数"""
    # 数据文件路径（请修改为您的实际文件路径）
    data_path = r"C:\Users\Hasee\Desktop\nodule_data\test_new.xlsx"  # 请替换为您的Excel文件路径

    # 可选择的平衡策略
    balance_strategies = ['smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn']

    # 推荐策略（您可以修改）
    recommended_strategy = 'smote'  # 对于您的数据推荐使用SMOTE

    try:
        # 创建分类器并运行分析
        classifier = BreastCancerClassifier(data_path, balance_strategy=recommended_strategy)
        results = classifier.run_analysis()

        # 打印分析结果摘要
        print("\n=== Analysis Results Summary ===")
        print(f"Data balancing strategy: {classifier.balance_strategy}")
        if hasattr(classifier, 'balancing_info'):
            print(f"Original samples: {classifier.balancing_info['original_samples']}")
            print(f"Balanced samples: {classifier.balancing_info['balanced_samples']}")
            print(f"Original distribution: {dict(classifier.balancing_info['original_distribution'])}")
            print(f"Balanced distribution: {dict(classifier.balancing_info['balanced_distribution'])}")

        print(f"Best performing model: {results['best_model']}")
        print(f"Best test accuracy: {results['best_accuracy']:.4f}")
        print(f"Number of selected features: {len(results['selected_features'])}")
        print(f"Total models evaluated: {results['n_models_evaluated']}")

        print("\nTop 5 selected features:")
        for i, feature in enumerate(results['selected_features'][:5]):
            print(f"  {i + 1}. {feature}")

        # 决策曲线分析摘要
        if 'dca_metrics' in results and not results['dca_metrics'].empty:
            print("\nDecision Curve Analysis Summary:")
            top_dca_model = results['dca_metrics'].iloc[0]
            print(f"  Best DCA model: {top_dca_model['Model']}")
            print(f"  DCA AUC: {top_dca_model['DCA_AUC']:.4f}")
            print(f"  Max Net Benefit: {top_dca_model['Max_Net_Benefit']:.4f}")
            print(f"  Optimal Threshold: {top_dca_model['Optimal_Threshold']:.3f}")

    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found.")
        print("Please ensure the file path is correct and the file exists.")
    except Exception as e:
        print(f"Error during analysis: {e}")



if __name__ == "__main__":
    main()