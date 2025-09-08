# ============================================== A/B TESTING EXAMPLES FOR YOUR MODEL ==============================================

from ab_testing_framework import ABTestingFramework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def example_1_basic_ab_test():
    """
    Contoh 1: A/B Testing Dasar - Regularization Parameter
    """
    print("🍷 CONTOH 1: A/B Testing Regularization Parameter")
    print("="*60)
    
    # Initialize framework
    ab_framework = ABTestingFramework()
    
    # Create two variants dengan regularization yang berbeda
    ab_framework.create_model_variant("Control", C=1.0, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("Treatment", C=0.1, solver='lbfgs', max_iter=1000)
    
    # Train both models
    print("\n🔄 Training models...")
    ab_framework.train_model("Control")
    ab_framework.train_model("Treatment")
    
    # Run A/B test
    print("\n📊 Running A/B test...")
    results = ab_framework.run_ab_test("Control", "Treatment")
    
    # Visualize results
    ab_framework.visualize_ab_test_results(results, "Control", "Treatment")
    
    # Visualize comprehensive results with max/min labels
    print("\n🔍 Showing comprehensive visualization with max/min labels...")
    ab_framework.visualize_comprehensive_results(results, "Control", "Treatment")
    
    # Generate report
    ab_framework.generate_report(results, "Control", "Treatment")
    
    return ab_framework, results

def example_2_solver_comparison():
    """
    Contoh 2: A/B Testing Solver Algorithms
    """
    print("\n🍷 CONTOH 2: A/B Testing Solver Algorithms")
    print("="*60)
    
    ab_framework = ABTestingFramework()
    
    # Test different solvers
    ab_framework.create_model_variant("LBFGS", C=1.0, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("Newton-CG", C=1.0, solver='newton-cg', max_iter=1000)
    
    # Train models
    print("\n🔄 Training models...")
    ab_framework.train_model("LBFGS")
    ab_framework.train_model("Newton-CG")
    
    # Run A/B test
    print("\n📊 Running A/B test...")
    results = ab_framework.run_ab_test("LBFGS", "Newton-CG")
    
    # Visualize results
    ab_framework.visualize_ab_test_results(results, "LBFGS", "Newton-CG")
    
    # Visualize comprehensive results with max/min labels
    print("\n🔍 Showing comprehensive visualization with max/min labels...")
    ab_framework.visualize_comprehensive_results(results, "LBFGS", "Newton-CG")
    
    return ab_framework, results

def example_3_comprehensive_testing():
    """
    Contoh 3: Comprehensive A/B Testing dengan Multiple Variants
    """
    print("\n🍷 CONTOH 3: Comprehensive A/B Testing")
    print("="*60)
    
    ab_framework = ABTestingFramework()
    
    # Create multiple variants
    variants = {
        "Baseline": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        "High_Reg": {"C": 0.1, "solver": "lbfgs", "max_iter": 1000},
        "Low_Reg": {"C": 10.0, "solver": "lbfgs", "max_iter": 1000},
        "Newton": {"C": 1.0, "solver": "newton-cg", "max_iter": 1000},
        "More_Iter": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}
    }
    
    # Create and train all variants
    print("\n🔧 Creating and training variants...")
    for name, params in variants.items():
        ab_framework.create_model_variant(name, **params)
        ab_framework.train_model(name)
    
    # Run comprehensive comparisons
    print("\n📊 Running comprehensive comparisons...")
    all_results = {}
    
    # Compare Baseline vs each other variant
    for variant_name in variants.keys():
        if variant_name != "Baseline":
            print(f"\n🔄 Comparing Baseline vs {variant_name}...")
            results = ab_framework.run_ab_test("Baseline", variant_name)
            all_results[f"Baseline_vs_{variant_name}"] = results
    
    # Find best performing variant
    print("\n🏆 FINDING BEST PERFORMING VARIANT")
    print("="*40)
    
    best_variant = "Baseline"
    best_f1 = ab_framework.results["Baseline"]["metrics"]["f1_weighted"]
    
    for variant_name in variants.keys():
        if variant_name != "Baseline":
            f1_score = ab_framework.results[variant_name]["metrics"]["f1_weighted"]
            print(f"{variant_name}: F1 = {f1_score:.4f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_variant = variant_name
    
    print(f"\n🎯 Best performing variant: {best_variant} (F1 = {best_f1:.4f})")
    
    return ab_framework, all_results

def example_4_production_ready_ab_test():
    """
    Contoh 4: Production-Ready A/B Testing dengan MLflow Integration
    """
    print("\n🍷 CONTOH 4: Production-Ready A/B Testing")
    print("="*60)
    
    import mlflow
    
    ab_framework = ABTestingFramework()
    
    # Create production variants
    ab_framework.create_model_variant("Production_Current", C=1.0, solver='lbfgs', max_iter=1000)
    ab_framework.create_model_variant("Production_Candidate", C=0.5, solver='lbfgs', max_iter=1000)
    
    # Train models
    print("\n🔄 Training production models...")
    ab_framework.train_model("Production_Current")
    ab_framework.train_model("Production_Candidate")
    
    # Run A/B test
    print("\n📊 Running production A/B test...")
    results = ab_framework.run_ab_test("Production_Current", "Production_Candidate")
    
    # Log to MLflow
    with mlflow.start_run(run_name="Production_AB_Test"):
        # Log parameters
        mlflow.log_param("test_type", "production_ab_test")
        mlflow.log_param("control_variant", "Production_Current")
        mlflow.log_param("treatment_variant", "Production_Candidate")
        
        # Log metrics
        for metric, result in results.items():
            mlflow.log_metric(f"{metric}_control", result['variant_a'])
            mlflow.log_metric(f"{metric}_treatment", result['variant_b'])
            mlflow.log_metric(f"{metric}_improvement", result['improvement'])
            mlflow.log_metric(f"{metric}_significant", int(result['significant']))
            mlflow.log_metric(f"{metric}_p_value", result['p_value'])
        
        # Log models
        mlflow.sklearn.log_model(
            ab_framework.experiments["Production_Current"]["model"],
            "production_current_model"
        )
        mlflow.sklearn.log_model(
            ab_framework.experiments["Production_Candidate"]["model"],
            "production_candidate_model"
        )
        
        print("✅ Production A/B test logged to MLflow!")
    
    # Generate deployment recommendation
    print("\n🚀 DEPLOYMENT RECOMMENDATION")
    print("="*40)
    
    significant_improvements = [m for m, r in results.items() if r['significant'] and r['improvement'] > 0]
    
    if significant_improvements:
        best_metric = max(significant_improvements, key=lambda m: results[m]['improvement'])
        improvement = results[best_metric]['improvement']
        print(f"✅ DEPLOY Production_Candidate")
        print(f"   Reason: {improvement:+.2f}% improvement in {best_metric}")
        print(f"   Statistical significance: p < 0.05")
    else:
        print("❌ KEEP Production_Current")
        print("   Reason: No significant improvement found")
    
    return ab_framework, results

def example_5_custom_ab_test_scenario():
    """
    Contoh 5: Custom A/B Testing Scenario - Feature Engineering
    """
    print("\n🍷 CONTOH 5: Custom A/B Testing - Feature Engineering")
    print("="*60)
    
    # Load data
    df = pd.read_csv("data/winequality-red.csv", sep=";")
    
    # Create quality groups
    def simplify_quality(q):
        if q <= 4: return 0
        if q <= 6: return 1 
        else: return 2
    
    df["qualityg"] = df["quality"].apply(simplify_quality)
    
    # Create two different feature sets
    # Variant A: Original features
    X_a = df.drop(columns=["quality", "qualityg"])
    
    # Variant B: Original features + engineered features
    X_b = df.drop(columns=["quality", "qualityg"]).copy()
    X_b['alcohol_ph_ratio'] = X_b['alcohol'] / (X_b['pH'] + 1e-8)
    X_b['sulphates_chlorides_ratio'] = X_b['sulphates'] / (X_b['chlorides'] + 1e-8)
    X_b['total_acidity'] = X_b['fixed acidity'] + X_b['volatile acidity'] + X_b['citric acid']
    
    y = df["qualityg"]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train_a, X_test_a, y_train, y_test = train_test_split(
        X_a, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_b, X_test_b, _, _ = train_test_split(
        X_b, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train models
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    model_a = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))
    ])
    
    model_b = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))
    ])
    
    print("🔄 Training models with different feature sets...")
    model_a.fit(X_train_a, y_train)
    model_b.fit(X_train_b, y_train)
    
    # Make predictions
    y_pred_a = model_a.predict(X_test_a)
    y_pred_b = model_b.predict(X_test_b)
    y_pred_proba_a = model_a.predict_proba(X_test_a)
    y_pred_proba_b = model_b.predict_proba(X_test_b)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics_a = {
        'accuracy': accuracy_score(y_test, y_pred_a),
        'f1_weighted': f1_score(y_test, y_pred_a, average='weighted'),
        'precision_weighted': precision_score(y_test, y_pred_a, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred_a, average='weighted')
    }
    
    metrics_b = {
        'accuracy': accuracy_score(y_test, y_pred_b),
        'f1_weighted': f1_score(y_test, y_pred_b, average='weighted'),
        'precision_weighted': precision_score(y_test, y_pred_b, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred_b, average='weighted')
    }
    
    # Compare results
    print("\n📊 FEATURE ENGINEERING A/B TEST RESULTS")
    print("="*50)
    
    for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
        value_a = metrics_a[metric]
        value_b = metrics_b[metric]
        improvement = ((value_b - value_a) / value_a) * 100
        
        print(f"\n{metric.upper()}:")
        print(f"   Original Features: {value_a:.4f}")
        print(f"   Engineered Features: {value_b:.4f}")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"   ✅ Engineered features perform better!")
        else:
            print(f"   ❌ Original features perform better")
    
    # Visualize comparison
    metrics = list(metrics_a.keys())
    values_a = [metrics_a[m] for m in metrics]
    values_b = [metrics_b[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, values_a, width, label='Original Features', alpha=0.7)
    plt.bar(x + width/2, values_b, width, label='Engineered Features', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Feature Engineering A/B Test Results')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model_a, model_b, metrics_a, metrics_b

def run_all_examples():
    """
    Menjalankan semua contoh A/B testing
    """
    print("🍷 COMPREHENSIVE A/B TESTING EXAMPLES")
    print("="*60)
    
    # Run all examples
    print("\n1️⃣ Running Basic A/B Test...")
    framework1, results1 = example_1_basic_ab_test()
    
    print("\n2️⃣ Running Solver Comparison...")
    framework2, results2 = example_2_solver_comparison()
    
    print("\n3️⃣ Running Comprehensive Testing...")
    framework3, results3 = example_3_comprehensive_testing()
    
    print("\n4️⃣ Running Production-Ready Test...")
    framework4, results4 = example_4_production_ready_ab_test()
    
    print("\n5️⃣ Running Feature Engineering Test...")
    model_a, model_b, metrics_a, metrics_b = example_5_custom_ab_test_scenario()
    
    print("\n✅ All A/B testing examples completed!")
    
    # Save all results
    framework1.save_results("example_1_results.json")
    framework2.save_results("example_2_results.json")
    framework3.save_results("example_3_results.json")
    framework4.save_results("example_4_results.json")
    
    return {
        'framework1': framework1, 'results1': results1,
        'framework2': framework2, 'results2': results2,
        'framework3': framework3, 'results3': results3,
        'framework4': framework4, 'results4': results4,
        'model_a': model_a, 'model_b': model_b,
        'metrics_a': metrics_a, 'metrics_b': metrics_b
    }

if __name__ == "__main__":
    # Run specific example
    # framework, results = example_1_basic_ab_test()
    
    # Or run all examples
    all_results = run_all_examples()

