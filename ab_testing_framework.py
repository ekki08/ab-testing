# ============================================== A/B TESTING FRAMEWORK FOR MULTINOMIAL REGRESSION ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score)
import mlflow
import warnings
from scipy import stats
import time
from datetime import datetime
import json
import os
warnings.filterwarnings('ignore')

class ABTestingFramework:
    """
    Framework A/B Testing untuk model Multinomial Regression
    """
    
    def __init__(self, data_path="data/winequality-white.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.results = {}
        self.experiments = {}
        
        # Load dan prepare data
        self.load_data()
        self.prepare_data()
        
    def load_data(self):
        """Load dataset wine quality"""
        print("Loading wine quality dataset...")
        self.df = pd.read_csv(self.data_path, sep=";")
        
        # Create quality groups
        def simplify_quality(q):
            if q <= 4:  # low
                return 0
            if q <= 6:  # medium
                return 1 
            else:  # high
                return 2
        
        self.df["qualityg"] = self.df["quality"].apply(simplify_quality)
        
        # Prepare features dan target
        self.X = self.df.drop(columns=["quality", "qualityg"])
        self.y = self.df["qualityg"]
        
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Target distribution: {np.bincount(self.y)}")
        
    def prepare_data(self):
        """Prepare data untuk A/B testing"""
        # Split data dengan stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def create_model_variant(self, variant_name, **params):
        """
        Membuat variant model dengan parameter yang berbeda
        
        Parameters:
        - variant_name: Nama variant (A, B, C, dll)
        - **params: Parameter untuk model (C, solver, max_iter, dll)
        """
        default_params = {
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': self.random_state
        }
        
        # Update dengan parameter yang diberikan
        model_params = {**default_params, **params}
        
        # Buat pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('multinomial_lr', LogisticRegression(**model_params))
        ])
        
        self.experiments[variant_name] = {
            'model': model,
            'params': model_params,
            'created_at': datetime.now().isoformat()
        }
        
        print(f"Model variant '{variant_name}' created with params: {model_params}")
        return model
    
    def train_model(self, variant_name):
        """Train model variant"""
        if variant_name not in self.experiments:
            raise ValueError(f"Variant '{variant_name}' not found. Create it first.")
        
        model = self.experiments[variant_name]['model']
        
        print(f"Training model variant '{variant_name}'...")
        start_time = time.time()
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Store results
        self.results[variant_name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        print(f"Model '{variant_name}' trained successfully!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    # ============================== NEW: EXTERNAL DATA EVALUATION HELPERS ==============================
    def _assert_trained_variant(self, variant_name):
        if variant_name not in self.experiments:
            raise ValueError(f"Variant '{variant_name}' not found. Create it first.")
        model = self.experiments[variant_name]['model']
        # Quick fit-check: sklearn pipelines have 'named_steps'; assume trained if scaler has mean_
        if not hasattr(model.named_steps['scaler'], 'mean_'):
            raise ValueError(f"Variant '{variant_name}' is not trained. Call train_model('{variant_name}') first.")
        return model

    def _align_features_or_raise(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure external dataframe columns match training features exactly (order and names)."""
        train_cols = list(self.X_train.columns)
        incoming_cols = list(X_df.columns)
        missing = [c for c in train_cols if c not in incoming_cols]
        extra = [c for c in incoming_cols if c not in train_cols]
        if missing or extra:
            raise ValueError(
                "Feature mismatch with training schema. "
                f"Missing: {missing}; Extra: {extra}. Ensure columns exactly match training features: {train_cols}"
            )
        # Reorder to training order
        return X_df[train_cols]

    def evaluate_on_dataframe(self, variant_name: str, X_df: pd.DataFrame, y_true: pd.Series):
        """
        Evaluasi variant terlatih pada dataframe eksternal (skema fitur harus sama).
        Mengembalikan dict berisi predictions, probabilities, dan metrics.
        """
        model = self._assert_trained_variant(variant_name)
        X_aligned = self._align_features_or_raise(X_df)
        y_pred = model.predict(X_aligned)
        y_pred_proba = model.predict_proba(X_aligned)
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }

    def evaluate_on_csv(self, variant_name: str, csv_path: str, sep: str = ';', target_col: str = 'quality', target_grouping: bool = True):
        """
        Evaluasi variant pada file CSV eksternal. Jika target_col='quality' ada dan target_grouping=True,
        maka akan diubah ke 'qualityg' dengan mapping yang sama seperti training.
        Jika 'qualityg' sudah ada, kolom itu yang dipakai sebagai target.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df_ext = pd.read_csv(csv_path, sep=sep)

        # Derive target
        if 'qualityg' in df_ext.columns:
            y_ext = df_ext['qualityg']
        elif target_col in df_ext.columns and target_grouping:
            def simplify_quality(q):
                if q <= 4:
                    return 0
                if q <= 6:
                    return 1
                return 2
            y_ext = df_ext[target_col].apply(simplify_quality)
        else:
            raise ValueError("Target not found. Provide 'qualityg' or enable target_grouping with 'quality' present.")

        # Build X by dropping known target columns if present
        drop_cols = [c for c in ['quality', 'qualityg'] if c in df_ext.columns]
        X_ext = df_ext.drop(columns=drop_cols)

        return self.evaluate_on_dataframe(variant_name, X_ext, y_ext)

    def run_ab_test_on_dataset(self, variant_a_name: str, variant_b_name: str, X_df: pd.DataFrame, y_true: pd.Series, confidence_level: float = 0.95):
        """
        Jalankan A/B test pada dataset eksternal (tanpa retrain), menggunakan dua variant yang SUDAH dilatih.
        """
        model_a = self._assert_trained_variant(variant_a_name)
        model_b = self._assert_trained_variant(variant_b_name)

        X_aligned = self._align_features_or_raise(X_df)
        # Predictions
        y_pred_a = model_a.predict(X_aligned)
        y_proba_a = model_a.predict_proba(X_aligned)
        y_pred_b = model_b.predict(X_aligned)
        y_proba_b = model_b.predict_proba(X_aligned)

        # Metrics
        metrics_a = self.calculate_metrics(y_true, y_pred_a, y_proba_a)
        metrics_b = self.calculate_metrics(y_true, y_pred_b, y_proba_b)

        comparison_results = {}
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            improvement = ((value_b - value_a) / value_a) * 100 if value_a != 0 else np.inf
            significance = self.bootstrap_significance_test(
                y_pred_a, y_pred_b, y_true, metric, confidence_level
            )
            comparison_results[metric] = {
                'variant_a': value_a,
                'variant_b': value_b,
                'improvement': improvement,
                'significant': significance['significant'],
                'p_value': significance['p_value'],
                'confidence_interval': significance['confidence_interval']
            }

        self.print_ab_test_results(comparison_results, variant_a_name, variant_b_name)
        return comparison_results

    def run_ab_test_on_csv(self, variant_a_name: str, variant_b_name: str, csv_path: str, sep: str = ';', target_col: str = 'quality', target_grouping: bool = True, confidence_level: float = 0.95):
        """Shortcut untuk menjalankan A/B test pada satu file CSV eksternal."""
        df_ext = pd.read_csv(csv_path, sep=sep)
        if 'qualityg' in df_ext.columns:
            y_ext = df_ext['qualityg']
        elif target_col in df_ext.columns and target_grouping:
            def simplify_quality(q):
                if q <= 4:
                    return 0
                if q <= 6:
                    return 1
                return 2
            y_ext = df_ext[target_col].apply(simplify_quality)
        else:
            raise ValueError("Target not found. Provide 'qualityg' or enable target_grouping with 'quality' present.")
        X_ext = df_ext.drop(columns=[c for c in ['quality', 'qualityg'] if c in df_ext.columns])
        return self.run_ab_test_on_dataset(variant_a_name, variant_b_name, X_ext, y_ext, confidence_level)

    def run_ab_test_on_multiple_datasets(self, variant_a_name: str, variant_b_name: str, data_paths: list, sep: str = ';', target_col: str = 'quality', target_grouping: bool = True, confidence_level: float = 0.95, save_json_prefix: str = None):
        """
        Loop A/B test untuk banyak dataset CSV eksternal tanpa retrain.
        Mengembalikan dict hasil per dataset.
        """
        if not data_paths:
            raise ValueError("data_paths is empty")

        aggregated = {}
        for path in data_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            print(f"\n==== Running A/B test on dataset: {base} ====")
            result = self.run_ab_test_on_csv(
                variant_a_name,
                variant_b_name,
                csv_path=path,
                sep=sep,
                target_col=target_col,
                target_grouping=target_grouping,
                confidence_level=confidence_level
            )
            aggregated[base] = result
            if save_json_prefix:
                out_path = f"{save_json_prefix}_{base}.json"
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved A/B results to {out_path}")

        return aggregated

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Class-wise metrics
        class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(['Low', 'Medium', 'High']):
            metrics[f'precision_{class_name.lower()}'] = class_precision[i]
            metrics[f'recall_{class_name.lower()}'] = class_recall[i]
            metrics[f'f1_{class_name.lower()}'] = class_f1[i]
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.experiments.get(list(self.experiments.keys())[0])['model'], 
            self.X, self.y, cv=5, scoring='f1_weighted'
        )
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        # Prediction confidence
        metrics['avg_confidence'] = np.mean(np.max(y_pred_proba, axis=1))
        metrics['confidence_std'] = np.std(np.max(y_pred_proba, axis=1))
        
        return metrics
    
    def run_ab_test(self, variant_a_name, variant_b_name, confidence_level=0.95):
        """
        Menjalankan A/B test antara dua variant
        
        Parameters:
        - variant_a_name: Nama variant A (control)
        - variant_b_name: Nama variant B (treatment)
        - confidence_level: Level kepercayaan untuk statistical test
        """
        if variant_a_name not in self.results or variant_b_name not in self.results:
            raise ValueError("Both variants must be trained first. Use train_model() method.")
        
        print(f"\n{'='*60}")
        print(f"A/B TESTING: {variant_a_name} vs {variant_b_name}")
        print(f"{'='*60}")
        
        # Get metrics
        metrics_a = self.results[variant_a_name]['metrics']
        metrics_b = self.results[variant_b_name]['metrics']
        
        # Compare metrics
        comparison_results = {}
        
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            if metric in metrics_a and metric in metrics_b:
                value_a = metrics_a[metric]
                value_b = metrics_b[metric]
                
                # Calculate improvement
                improvement = ((value_b - value_a) / value_a) * 100
                
                # Statistical significance test (bootstrap)
                significance = self.bootstrap_significance_test(
                    self.results[variant_a_name]['predictions'],
                    self.results[variant_b_name]['predictions'],
                    self.y_test,
                    metric,
                    confidence_level
                )
                
                comparison_results[metric] = {
                    'variant_a': value_a,
                    'variant_b': value_b,
                    'improvement': improvement,
                    'significant': significance['significant'],
                    'p_value': significance['p_value'],
                    'confidence_interval': significance['confidence_interval']
                }
        
        # Print results
        self.print_ab_test_results(comparison_results, variant_a_name, variant_b_name)
        
        return comparison_results
    
    def bootstrap_significance_test(self, pred_a, pred_b, y_true, metric, confidence_level=0.95, n_bootstrap=1000):
        """
        Bootstrap test untuk statistical significance
        """
        n_samples = len(y_true)
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_bootstrap = y_true.iloc[indices]
            pred_a_bootstrap = pred_a[indices]
            pred_b_bootstrap = pred_b[indices]
            
            # Calculate metric difference
            if metric == 'accuracy':
                metric_a = accuracy_score(y_bootstrap, pred_a_bootstrap)
                metric_b = accuracy_score(y_bootstrap, pred_b_bootstrap)
            elif metric == 'f1_weighted':
                metric_a = f1_score(y_bootstrap, pred_a_bootstrap, average='weighted', zero_division=0)
                metric_b = f1_score(y_bootstrap, pred_b_bootstrap, average='weighted', zero_division=0)
            elif metric == 'precision_weighted':
                metric_a = precision_score(y_bootstrap, pred_a_bootstrap, average='weighted', zero_division=0)
                metric_b = precision_score(y_bootstrap, pred_b_bootstrap, average='weighted', zero_division=0)
            elif metric == 'recall_weighted':
                metric_a = recall_score(y_bootstrap, pred_b_bootstrap, average='weighted', zero_division=0)
                metric_b = recall_score(y_bootstrap, pred_b_bootstrap, average='weighted', zero_division=0)
            
            bootstrap_diffs.append(metric_b - metric_a)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_interval = np.percentile(bootstrap_diffs, [lower_percentile, upper_percentile])
        
        # Check significance (0 not in confidence interval)
        significant = not (confidence_interval[0] <= 0 <= confidence_interval[1])
        
        # Calculate p-value (proportion of bootstrap samples with opposite sign)
        p_value = np.mean(np.array(bootstrap_diffs) <= 0) if np.mean(bootstrap_diffs) > 0 else np.mean(np.array(bootstrap_diffs) >= 0)
        p_value = min(p_value, 1 - p_value) * 2  # Two-tailed test
        
        return {
            'significant': significant,
            'p_value': p_value,
            'confidence_interval': confidence_interval.tolist()
        }
    
    def print_ab_test_results(self, comparison_results, variant_a_name, variant_b_name):
        """Print hasil A/B test dengan format yang rapi"""
        print(f"\n📊 A/B TEST RESULTS: {variant_a_name} (Control) vs {variant_b_name} (Treatment)")
        print("="*80)
        
        for metric, results in comparison_results.items():
            print(f"\n🔍 {metric.upper().replace('_', ' ')}:")
            print(f"   {variant_a_name}: {results['variant_a']:.4f}")
            print(f"   {variant_b_name}: {results['variant_b']:.4f}")
            print(f"   Improvement: {results['improvement']:+.2f}%")
            
            if results['significant']:
                print(f"   ✅ Statistically Significant (p < 0.05)")
                print(f"   📈 {variant_b_name} is significantly better!")
            else:
                print(f"   ❌ Not Statistically Significant (p >= 0.05)")
                print(f"   🤷 No clear winner")
            
            print(f"   P-value: {results['p_value']:.4f}")
            print(f"   Confidence Interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    
    def create_multiple_variants(self):
        """Membuat beberapa variant model untuk testing"""
        print("\n🔧 Creating multiple model variants for A/B testing...")
        
        # Variant A: Baseline model
        self.create_model_variant("A", C=1.0, solver='lbfgs', max_iter=1000)
        
        # Variant B: Higher regularization
        self.create_model_variant("B", C=0.1, solver='lbfgs', max_iter=1000)
        
        # Variant C: Lower regularization
        self.create_model_variant("C", C=10.0, solver='lbfgs', max_iter=1000)
        
        # Variant D: Different solver
        self.create_model_variant("D", C=1.0, solver='newton-cg', max_iter=1000)
        
        # Variant E: More iterations
        self.create_model_variant("E", C=1.0, solver='lbfgs', max_iter=2000)
        
        print(f"✅ Created {len(self.experiments)} model variants")
    
    def run_comprehensive_ab_test(self):
        """Menjalankan comprehensive A/B testing untuk semua variant"""
        print("\n🚀 Running comprehensive A/B testing...")
        
        # Create variants
        self.create_multiple_variants()
        
        # Train all variants
        for variant_name in self.experiments.keys():
            self.train_model(variant_name)
        
        # Run pairwise comparisons
        variants = list(self.experiments.keys())
        all_comparisons = {}
        
        for i in range(len(variants)):
            for j in range(i+1, len(variants)):
                variant_a = variants[i]
                variant_b = variants[j]
                
                print(f"\n🔄 Comparing {variant_a} vs {variant_b}...")
                comparison = self.run_ab_test(variant_a, variant_b)
                all_comparisons[f"{variant_a}_vs_{variant_b}"] = comparison
        
        return all_comparisons
    
    def visualize_ab_test_results(self, comparison_results, variant_a_name, variant_b_name):
        """Visualisasi hasil A/B test"""
        metrics = list(comparison_results.keys())
        improvements = [comparison_results[m]['improvement'] for m in metrics]
        significant = [comparison_results[m]['significant'] for m in metrics]
        
        # Create color map
        colors = ['green' if sig else 'red' for sig in significant]
        
        plt.figure(figsize=(15, 10))
        
        # Bar plot of improvements
        plt.subplot(2, 2, 1)
        bars = plt.bar(metrics, improvements, color=colors, alpha=0.7)
        plt.title(f'A/B Test Results: {variant_a_name} vs {variant_b_name}')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add max and min labels for improvements
        max_improvement = max(improvements)
        min_improvement = min(improvements)
        max_idx = improvements.index(max_improvement)
        min_idx = improvements.index(min_improvement)
        
        # Label max improvement
        plt.text(max_idx, max_improvement + 0.5, f'MAX: {max_improvement:.2f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
        
        # Label min improvement
        plt.text(min_idx, min_improvement - 0.5, f'MIN: {min_improvement:.2f}%', 
                ha='center', va='top', fontsize=10, fontweight='bold', color='darkred')
        
        # Add significance indicators
        for i, (bar, sig) in enumerate(zip(bars, significant)):
            if sig:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        '★', ha='center', va='bottom', fontsize=12, color='green')
        
        # Metric comparison
        plt.subplot(2, 2, 2)
        x = np.arange(len(metrics))
        width = 0.35
        
        values_a = [comparison_results[m]['variant_a'] for m in metrics]
        values_b = [comparison_results[m]['variant_b'] for m in metrics]
        
        plt.bar(x - width/2, values_a, width, label=variant_a_name, alpha=0.7)
        plt.bar(x + width/2, values_b, width, label=variant_b_name, alpha=0.7)
        
        # Add max and min labels for metric comparison
        all_values = values_a + values_b
        max_value = max(all_values)
        min_value = min(all_values)
        
        # Find which bar has max and min values
        max_bar_idx = all_values.index(max_value)
        min_bar_idx = all_values.index(min_value)
        
        if max_bar_idx < len(values_a):  # Max is in variant A
            plt.text(max_bar_idx - width/2, max_value + 0.01, f'MAX: {max_value:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')
        else:  # Max is in variant B
            plt.text(max_bar_idx - len(values_a) + width/2, max_value + 0.01, f'MAX: {max_value:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')
        
        if min_bar_idx < len(values_a):  # Min is in variant A
            plt.text(min_bar_idx - width/2, min_value - 0.01, f'MIN: {min_value:.3f}', 
                    ha='center', va='top', fontsize=9, fontweight='bold', color='darkred')
        else:  # Min is in variant B
            plt.text(min_bar_idx - len(values_a) + width/2, min_value - 0.01, f'MIN: {min_value:.3f}', 
                    ha='center', va='top', fontsize=9, fontweight='bold', color='darkred')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Metric Comparison')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        
        # P-values
        plt.subplot(2, 2, 3)
        p_values = [comparison_results[m]['p_value'] for m in metrics]
        plt.bar(metrics, p_values, color=['red' if p < 0.05 else 'blue' for p in p_values], alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        
        # Add max and min labels for p-values
        max_p = max(p_values)
        min_p = min(p_values)
        max_p_idx = p_values.index(max_p)
        min_p_idx = p_values.index(min_p)
        
        plt.text(max_p_idx, max_p + 0.01, f'MAX: {max_p:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
        plt.text(min_p_idx, min_p - 0.01, f'MIN: {min_p:.4f}', 
                ha='center', va='top', fontsize=9, fontweight='bold', color='darkblue')
        
        plt.title('P-values')
        plt.ylabel('P-value')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Confidence intervals
        plt.subplot(2, 2, 4)
        ci_lower = [comparison_results[m]['confidence_interval'][0] for m in metrics]
        ci_upper = [comparison_results[m]['confidence_interval'][1] for m in metrics]
        
        # Calculate confidence interval widths
        ci_widths = [abs(ci_upper[i] - ci_lower[i]) for i in range(len(metrics))]
        
        plt.errorbar(metrics, improvements, yerr=[np.abs(np.array(ci_lower)), np.array(ci_upper)], 
                    fmt='o', capsize=5, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add max and min labels for confidence intervals
        max_ci_width = max(ci_widths)
        min_ci_width = min(ci_widths)
        max_ci_idx = ci_widths.index(max_ci_width)
        min_ci_idx = ci_widths.index(min_ci_width)
        
        plt.text(max_ci_idx, improvements[max_ci_idx] + 0.5, f'MAX CI: {max_ci_width:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkorange')
        plt.text(min_ci_idx, improvements[min_ci_idx] - 0.5, f'MIN CI: {min_ci_width:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold', color='darkgreen')
        
        plt.title('Improvement with Confidence Intervals')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_comprehensive_results(self, comparison_results, variant_a_name, variant_b_name):
        """Visualisasi komprehensif dengan label max dan min yang lebih detail"""
        metrics = list(comparison_results.keys())
        improvements = [comparison_results[m]['improvement'] for m in metrics]
        significant = [comparison_results[m]['significant'] for m in metrics]
        p_values = [comparison_results[m]['p_value'] for m in metrics]
        
        # Create figure with larger size
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive A/B Test Analysis: {variant_a_name} vs {variant_b_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Improvement Bar Chart with Max/Min Labels
        ax1 = axes[0, 0]
        colors = ['green' if sig else 'red' for sig in significant]
        bars = ax1.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Improvement Analysis', fontweight='bold')
        ax1.set_ylabel('Improvement (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add max/min labels with better positioning
        max_improvement = max(improvements)
        min_improvement = min(improvements)
        max_idx = improvements.index(max_improvement)
        min_idx = improvements.index(min_improvement)
        
        # Max label
        ax1.annotate(f'MAX: {max_improvement:.2f}%', 
                    xy=(max_idx, max_improvement), xytext=(max_idx, max_improvement + 1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
        
        # Min label
        ax1.annotate(f'MIN: {min_improvement:.2f}%', 
                    xy=(min_idx, min_improvement), xytext=(min_idx, min_improvement - 1),
                    ha='center', va='top', fontsize=11, fontweight='bold', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
        
        # Add significance stars
        for i, (bar, sig) in enumerate(zip(bars, significant)):
            if sig:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                        '★', ha='center', va='bottom', fontsize=14, color='gold')
        
        # 2. Metric Comparison with Max/Min
        ax2 = axes[0, 1]
        x = np.arange(len(metrics))
        width = 0.35
        
        values_a = [comparison_results[m]['variant_a'] for m in metrics]
        values_b = [comparison_results[m]['variant_b'] for m in metrics]
        
        bars_a = ax2.bar(x - width/2, values_a, width, label=variant_a_name, alpha=0.7, color='skyblue')
        bars_b = ax2.bar(x + width/2, values_b, width, label=variant_b_name, alpha=0.7, color='lightcoral')
        
        # Find overall max and min
        all_values = values_a + values_b
        max_value = max(all_values)
        min_value = min(all_values)
        
        # Add max/min annotations
        ax2.annotate(f'OVERALL MAX: {max_value:.3f}', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=10, fontweight='bold', color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        ax2.annotate(f'OVERALL MIN: {min_value:.3f}', 
                    xy=(0.5, 0.05), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Metric Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        
        # 3. P-values with Max/Min
        ax3 = axes[0, 2]
        bars_p = ax3.bar(metrics, p_values, color=['red' if p < 0.05 else 'blue' for p in p_values], 
                        alpha=0.7, edgecolor='black', linewidth=1)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='p=0.05', linewidth=2)
        
        # Add max/min labels for p-values
        max_p = max(p_values)
        min_p = min(p_values)
        max_p_idx = p_values.index(max_p)
        min_p_idx = p_values.index(min_p)
        
        ax3.annotate(f'MAX P: {max_p:.4f}', 
                    xy=(max_p_idx, max_p), xytext=(max_p_idx, max_p + 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
        
        ax3.annotate(f'MIN P: {min_p:.4f}', 
                    xy=(min_p_idx, min_p), xytext=(min_p_idx, min_p - 0.02),
                    ha='center', va='top', fontsize=10, fontweight='bold', color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
        
        ax3.set_title('P-values Analysis', fontweight='bold')
        ax3.set_ylabel('P-value')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 4. Confidence Intervals with Max/Min
        ax4 = axes[1, 0]
        ci_lower = [comparison_results[m]['confidence_interval'][0] for m in metrics]
        ci_upper = [comparison_results[m]['confidence_interval'][1] for m in metrics]
        ci_widths = [abs(ci_upper[i] - ci_lower[i]) for i in range(len(metrics))]
        
        ax4.errorbar(metrics, improvements, yerr=[np.abs(np.array(ci_lower)), np.array(ci_upper)], 
                    fmt='o', capsize=5, alpha=0.7, markersize=8, linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add max/min CI width labels
        max_ci_width = max(ci_widths)
        min_ci_width = min(ci_widths)
        max_ci_idx = ci_widths.index(max_ci_width)
        min_ci_idx = ci_widths.index(min_ci_width)
        
        ax4.annotate(f'MAX CI WIDTH: {max_ci_width:.3f}', 
                    xy=(max_ci_idx, improvements[max_ci_idx]), 
                    xytext=(max_ci_idx, improvements[max_ci_idx] + 1.5),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkorange',
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
        
        ax4.annotate(f'MIN CI WIDTH: {min_ci_width:.3f}', 
                    xy=(min_ci_idx, improvements[min_ci_idx]), 
                    xytext=(min_ci_idx, improvements[min_ci_idx] - 1.5),
                    ha='center', va='top', fontsize=10, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
        
        ax4.set_title('Confidence Intervals', fontweight='bold')
        ax4.set_ylabel('Improvement (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Significance Summary
        ax5 = axes[1, 1]
        sig_count = sum(significant)
        non_sig_count = len(significant) - sig_count
        
        labels = ['Significant', 'Not Significant']
        sizes = [sig_count, non_sig_count]
        colors_pie = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90, explode=(0.05, 0.05))
        
        # Add max/min labels for pie chart
        if sig_count > 0:
            ax5.annotate(f'MAX: {sig_count} Significant', 
                        xy=(0.5, 0.8), xycoords='axes fraction',
                        ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        if non_sig_count > 0:
            ax5.annotate(f'MIN: {non_sig_count} Non-Significant', 
                        xy=(0.5, 0.2), xycoords='axes fraction',
                        ha='center', va='center', fontsize=11, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        
        ax5.set_title('Significance Summary', fontweight='bold')
        
        # 6. Performance Heatmap
        ax6 = axes[1, 2]
        performance_matrix = np.array([
            [comparison_results[m]['variant_a'] for m in metrics],
            [comparison_results[m]['variant_b'] for m in metrics]
        ])
        
        im = ax6.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(metrics, rotation=45)
        ax6.set_yticks([0, 1])
        ax6.set_yticklabels([variant_a_name, variant_b_name])
        
        # Add value annotations
        for i in range(2):
            for j in range(len(metrics)):
                value = performance_matrix[i, j]
                ax6.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='black')
        
        # Add max/min labels for heatmap
        max_perf = np.max(performance_matrix)
        min_perf = np.min(performance_matrix)
        max_pos = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
        min_pos = np.unravel_index(np.argmin(performance_matrix), performance_matrix.shape)
        
        ax6.annotate(f'MAX: {max_perf:.3f}', 
                    xy=(max_pos[1], max_pos[0]), xytext=(max_pos[1], max_pos[0] - 0.3),
                    ha='center', va='top', fontsize=10, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
        
        ax6.annotate(f'MIN: {min_perf:.3f}', 
                    xy=(min_pos[1], min_pos[0]), xytext=(min_pos[1], min_pos[0] + 0.3),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
        
        ax6.set_title('Performance Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax6, label='Score')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename="ab_test_results.json"):
        """Save hasil A/B testing ke file JSON"""
        results_to_save = {
            'experiments': self.experiments,
            'results': self.results,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'dataset_shape': self.df.shape,
                'random_state': self.random_state
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for variant in results_to_save['results']:
            results_to_save['results'][variant]['predictions'] = results_to_save['results'][variant]['predictions'].tolist()
            results_to_save['results'][variant]['probabilities'] = results_to_save['results'][variant]['probabilities'].tolist()
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"✅ Results saved to {filename}")
    
    def generate_report(self, comparison_results, variant_a_name, variant_b_name):
        """Generate comprehensive A/B test report"""
        print(f"\n📋 A/B TEST REPORT")
        print("="*60)
        print(f"Control Variant: {variant_a_name}")
        print(f"Treatment Variant: {variant_b_name}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        print(f"\n📊 SUMMARY:")
        significant_metrics = [m for m, r in comparison_results.items() if r['significant']]
        if significant_metrics:
            print(f"✅ {len(significant_metrics)} metrics show significant improvement:")
            for metric in significant_metrics:
                improvement = comparison_results[metric]['improvement']
                print(f"   - {metric}: {improvement:+.2f}% improvement")
        else:
            print("❌ No metrics show significant improvement")
        
        print(f"\n🎯 RECOMMENDATION:")
        if significant_metrics:
            best_metric = max(significant_metrics, key=lambda m: comparison_results[m]['improvement'])
            best_improvement = comparison_results[best_metric]['improvement']
            print(f"   Deploy {variant_b_name} - shows {best_improvement:+.2f}% improvement in {best_metric}")
        else:
            print(f"   Keep {variant_a_name} - no significant improvement found")

# ============================================== EXAMPLE USAGE ==============================================

def run_example_ab_test():
    """Contoh penggunaan framework A/B testing"""
    print("🍷 WINE QUALITY A/B TESTING FRAMEWORK")
    print("="*60)
    
    # Initialize framework
    ab_framework = ABTestingFramework()
    
    # Create model variants
    print("\n1️⃣ Creating model variants...")
    ab_framework.create_model_variant("Baseline", C=1.0, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("High_Regularization", C=0.1, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("Low_Regularization", C=10.0, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("Newton_Solver", C=1.0, solver='newton-cg', max_iter=1000)
    
    # Train all variants
    print("\n2️⃣ Training all variants...")
    for variant in ["Baseline", "High_Regularization", "Low_Regularization", "Newton_Solver"]:
        ab_framework.train_model(variant)
    
    # Run A/B tests
    print("\n3️⃣ Running A/B tests...")
    comparison = ab_framework.run_ab_test("Baseline", "High_Regularization")
    
    # Visualize results
    print("\n4️⃣ Visualizing results...")
    ab_framework.visualize_ab_test_results(comparison, "Baseline", "High_Regularization")
    
    # Generate report
    print("\n5️⃣ Generating report...")
    ab_framework.generate_report(comparison, "Baseline", "High_Regularization")
    
    # Save results
    ab_framework.save_results("wine_quality_ab_test_results.json")
    
    print("\n✅ A/B testing completed successfully!")
    return ab_framework

if __name__ == "__main__":
    # Run example A/B test
    framework = run_example_ab_test()
