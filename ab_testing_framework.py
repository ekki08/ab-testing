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
                improvement = ((value_b - value_a) / value_a) * 100 if value_a != 0 else np.inf

                # Use stored test predictions as the sample to run bootstrap on
                pred_a = np.array(self.results[variant_a_name]['predictions'])
                pred_b = np.array(self.results[variant_b_name]['predictions'])
                y_true = self.y_test

                significance = self.bootstrap_significance_test(pred_a, pred_b, y_true, metric, confidence_level)

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

    def bootstrap_significance_test(self, pred_a, pred_b, y_true, metric, confidence_level=0.95, n_bootstrap=1000):
        """Perform bootstrap test comparing metric between two prediction arrays.

        pred_a, pred_b: 1D arrays of predictions for the same samples
        y_true: ground-truth labels (pandas Series or array)
        metric: one of 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'
        """
        # Ensure numpy arrays for indexing
        pred_a = np.array(pred_a)
        pred_b = np.array(pred_b)

        # y_true may be pandas Series
        n_samples = len(y_true)
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            if hasattr(y_true, 'iloc'):
                y_boot = y_true.iloc[indices]
            else:
                y_boot = np.array(y_true)[indices]

            pa = pred_a[indices]
            pb = pred_b[indices]

            if metric == 'accuracy':
                ma = accuracy_score(y_boot, pa)
                mb = accuracy_score(y_boot, pb)
            elif metric == 'f1_weighted':
                ma = f1_score(y_boot, pa, average='weighted', zero_division=0)
                mb = f1_score(y_boot, pb, average='weighted', zero_division=0)
            elif metric == 'precision_weighted':
                ma = precision_score(y_boot, pa, average='weighted', zero_division=0)
                mb = precision_score(y_boot, pb, average='weighted', zero_division=0)
            elif metric == 'recall_weighted':
                ma = recall_score(y_boot, pa, average='weighted', zero_division=0)
                mb = recall_score(y_boot, pb, average='weighted', zero_division=0)
            else:
                raise ValueError(f"Unsupported metric for bootstrap test: {metric}")

            bootstrap_diffs.append(mb - ma)

        # Compute confidence interval
        alpha = 1 - confidence_level
        lower = (alpha / 2) * 100
        upper = (1 - alpha / 2) * 100
        ci = np.percentile(bootstrap_diffs, [lower, upper])

        significant = not (ci[0] <= 0 <= ci[1])

        # Two-tailed p-value: proportion of bootstrap diffs as or more extreme than 0
        p_lower = np.mean(np.array(bootstrap_diffs) <= 0)
        p_upper = np.mean(np.array(bootstrap_diffs) >= 0)
        p_value = 2 * min(p_lower, p_upper)
        p_value = min(max(p_value, 0.0), 1.0)

        return {
            'significant': bool(significant),
            'p_value': float(p_value),
            'confidence_interval': ci.tolist()
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
        """Simple, readable single-row comparison chart with optional CI and significance markers.

        Keeps the first chart minimal (only bars) and provides clearer fonts, grid, and a small legend
        explaining significance and CI presence.
        """
        sns.set_style("whitegrid")

        metrics = list(comparison_results.keys())
        values_a = np.array([comparison_results[m].get('variant_a', np.nan) for m in metrics])
        values_b = np.array([comparison_results[m].get('variant_b', np.nan) for m in metrics])
        improvements = np.array([comparison_results[m].get('improvement', (values_b[i] - values_a[i]) * 100 if values_a[i] != 0 else np.nan) for i, m in enumerate(metrics)])

        # Determine if CI data exist
        ci_present = any(isinstance(comparison_results[m].get('confidence_interval'), (list, tuple)) for m in metrics)

        x = np.arange(len(metrics))
        width = 0.36

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width/2, values_a, width, label=variant_a_name, color='#4C72B0', alpha=0.9)
        ax.bar(x + width/2, values_b, width, label=variant_b_name, color='#DD8452', alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{variant_a_name} vs {variant_b_name}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(frameon=False)

        # annotate significance with a star above the larger bar
        for i, m in enumerate(metrics):
            if comparison_results[m].get('significant'):
                top = max(values_a[i], values_b[i])
                ax.text(i, top + 0.02 * max(1.0, top), '*', ha='center', va='bottom', fontsize=16, color='black')

        # small explanatory box
        notes = "'*' = statistically significant"
        if ci_present:
            notes += "\nError bars = confidence intervals"
        ax.text(1.02, 0.95, notes, transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', fc='white', ec='0.8'))

        plt.tight_layout()
        plt.show()
    
    def visualize_comprehensive_results(self, comparison_results, variant_a_name, variant_b_name):
        """Comprehensive visual: metric scores, improvements, and CI summary in a 2x2 layout.

        This version uses larger fonts, clearer color mapping for improvements, and a dedicated
        CI subplot that highlights widest/narrowest intervals.
        """
        sns.set_style("whitegrid")

        metrics = list(comparison_results.keys())
        values_a = np.array([comparison_results[m].get('variant_a', np.nan) for m in metrics])
        values_b = np.array([comparison_results[m].get('variant_b', np.nan) for m in metrics])
        improvements = np.array([comparison_results[m].get('improvement', (values_b[i] - values_a[i]) * 100 if values_a[i] != 0 else np.nan) for i, m in enumerate(metrics)])

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 0.8]})
        ax_scores = axes[0, 0]
        ax_impr = axes[0, 1]
        ax_ci = axes[1, 0]
        axes[1, 1].axis('off')

        x = np.arange(len(metrics))
        width = 0.36

        # Top-left: metric scores
        ax_scores.bar(x - width/2, values_a, width, label=variant_a_name, color='#4C72B0', alpha=0.9)
        ax_scores.bar(x + width/2, values_b, width, label=variant_b_name, color='#DD8452', alpha=0.9)
        ax_scores.set_xticks(x)
        ax_scores.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=30, ha='right', fontsize=10)
        ax_scores.set_ylabel('Score', fontsize=11)
        ax_scores.set_title('Metric Scores', fontsize=13, weight='bold')
        ax_scores.grid(axis='y', linestyle='--', alpha=0.4)
        ax_scores.legend(frameon=False)
        for i, m in enumerate(metrics):
            if comparison_results[m].get('significant'):
                top = max(values_a[i], values_b[i])
                ax_scores.text(i, top + 0.02 * max(1.0, top), '*', ha='center', va='bottom', fontsize=16)

        # Top-right: improvements
        cmap = plt.get_cmap('RdYlGn')
        clipped = np.clip(improvements, -100, 100)
        colors = [cmap((v + 100) / 200.0) if not np.isnan(v) else "#03A196" for v in clipped]
        bars = ax_impr.bar(x, improvements, color=colors, alpha=0.95)

        bar_color = 'salmon'
        bars = ax_impr.bar(x, improvements, color = [bar_color if not np.isnan(v) else "#888888" for v in improvements], alpha=0.95)
        ax_impr.axhline(0, color='black', linewidth=0.6)
        ax_impr.set_title('Improvement (%)', fontsize=13, weight='bold')
        ax_impr.set_xticks(x)
        ax_impr.set_xticklabels([], rotation=30)
        ax_impr.grid(axis='y', linestyle='--', alpha=0.4)
        for rect in bars:
            h = rect.get_height()
            ax_impr.text(rect.get_x() + rect.get_width() / 2, h + np.sign(h) * 0.02 * max(1.0, abs(h)), f"{h:+.2f}%", ha='center', va='bottom' if h>=0 else 'top', fontsize=9)

        # Bottom-left: CI ranges if available
        ci_present = any(isinstance(comparison_results[m].get('confidence_interval'), (list, tuple)) for m in metrics)
        if ci_present:
            y_pos = np.arange(len(metrics))
            widths = []
            for i, m in enumerate(metrics):
                ci = comparison_results[m].get('confidence_interval')
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    low, high = ci
                    # plot the CI as a horizontal thick line
                    ax_ci.hlines(y=i, xmin=low, xmax=high, colors='grey', linewidth=6, alpha=0.7)
                    ax_ci.plot((low+high)/2, i, 'o', color='black')
                    widths.append((abs(high-low), m))
            ax_ci.set_yticks(y_pos)
            ax_ci.set_yticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
            ax_ci.set_xlabel('Score')
            ax_ci.set_title('Confidence Intervals', fontsize=12)
            ax_ci.grid(axis='x', linestyle='--', alpha=0.4)
            if widths:
                widest = max(widths, key=lambda x: x[0])
                narrowest = min(widths, key=lambda x: x[0])
                ax_ci.text(1.02, 0.95, f'Widest CI: {widest[1]}\nNarrowest CI: {narrowest[1]}', transform=ax_ci.transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.8'))
        else:
            ax_ci.axis('off')
            ax_ci.text(0.02, 0.5, 'No confidence interval data available.', transform=ax_ci.transAxes, fontsize=11, va='center')

        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename="ab_test_results.json"):
        """Save hasil A/B testing ke file JSON"""
        # Remove non-serializable objects (Pipeline) from experiments
        experiments_serializable = {}
        for variant, exp in self.experiments.items():
            experiments_serializable[variant] = {
                k: v for k, v in exp.items() if k != 'model'
            }

        results_to_save = {
            'experiments': experiments_serializable,
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
