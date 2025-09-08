#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🍷 Simple A/B Testing Script untuk Model Multinomial Regression
Jalankan script ini untuk melakukan A/B testing dengan model Anda
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ab_testing_framework import ABTestingFramework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Main function untuk menjalankan A/B testing
    """
    print("🍷 WINE QUALITY A/B TESTING")
    print("="*50)
    
    try:
        # Initialize framework
        print("1️⃣ Initializing A/B Testing Framework...")
        ab_framework = ABTestingFramework()
        
        # Create model variants
        print("\n2️⃣ Creating Model Variants...")
        
        # Variant A: Baseline (model original Anda)
        ab_framework.create_model_variant("Baseline", C=1.0, solver='lbfgs', max_iter=1000)
        
        # Variant B: Higher regularization (lebih konservatif)
        ab_framework.create_model_variant("Conservative", C=0.1, solver='lbfgs', max_iter=1000)
        
        # Variant C: Lower regularization (lebih fleksibel)
        ab_framework.create_model_variant("Flexible", C=10.0, solver='lbfgs', max_iter=1000)
        
        # Variant D: Different solver
        ab_framework.create_model_variant("Newton", C=1.0, solver='newton-cg', max_iter=1000)
        
        # Train all models
        print("\n3️⃣ Training All Models...")
        variants = ["Baseline", "Conservative", "Flexible", "Newton"]
        
        for variant in variants:
            print(f"   Training {variant}...")
            ab_framework.train_model(variant)
        
        # Run A/B tests
        print("\n4️⃣ Running A/B Tests...")
        
        # Test 1: Baseline vs Conservative
        print("\n🔄 Test 1: Baseline vs Conservative")
        results_1 = ab_framework.run_ab_test("Baseline", "Conservative")
        
        # Test 2: Baseline vs Flexible
        print("\n🔄 Test 2: Baseline vs Flexible")
        results_2 = ab_framework.run_ab_test("Baseline", "Flexible")
        
        # Test 3: Baseline vs Newton
        print("\n🔄 Test 3: Baseline vs Newton")
        results_3 = ab_framework.run_ab_test("Baseline", "Newton")
        
        # Visualize results
        print("\n5️⃣ Visualizing Results...")
        
        # Visualize Test 1
        ab_framework.visualize_ab_test_results(results_1, "Baseline", "Conservative")
        
        # Visualize Test 2
        ab_framework.visualize_ab_test_results(results_2, "Baseline", "Flexible")
        
        # Visualize Test 3
        ab_framework.visualize_ab_test_results(results_3, "Baseline", "Newton")
        
        # Generate comprehensive report
        print("\n6️⃣ Generating Comprehensive Report...")
        
        # Find best performing variant
        print("\n🏆 PERFORMANCE SUMMARY")
        print("="*40)
        
        best_variant = "Baseline"
        best_f1 = ab_framework.results["Baseline"]["metrics"]["f1_weighted"]
        
        for variant in variants:
            f1_score = ab_framework.results[variant]["metrics"]["f1_weighted"]
            accuracy = ab_framework.results[variant]["metrics"]["accuracy"]
            print(f"{variant:12}: F1 = {f1_score:.4f}, Accuracy = {accuracy:.4f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_variant = variant
        
        print(f"\n🎯 Best performing variant: {best_variant} (F1 = {best_f1:.4f})")
        
        # Statistical significance summary
        print("\n📊 STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*45)
        
        all_tests = {
            "Baseline vs Conservative": results_1,
            "Baseline vs Flexible": results_2,
            "Baseline vs Newton": results_3
        }
        
        for test_name, results in all_tests.items():
            significant_metrics = [m for m, r in results.items() if r['significant']]
            if significant_metrics:
                print(f"✅ {test_name}: {len(significant_metrics)} metrics significant")
                for metric in significant_metrics:
                    improvement = results[metric]['improvement']
                    print(f"   - {metric}: {improvement:+.2f}% improvement")
            else:
                print(f"❌ {test_name}: No significant improvement")
        
        # Save results
        print("\n7️⃣ Saving Results...")
        ab_framework.save_results("wine_quality_ab_test_results.json")
        
        # Final recommendation
        print("\n🚀 FINAL RECOMMENDATION")
        print("="*30)
        
        # Check if any variant significantly outperforms baseline
        significant_improvements = []
        
        for test_name, results in all_tests.items():
            for metric, result in results.items():
                if result['significant'] and result['improvement'] > 0:
                    significant_improvements.append({
                        'test': test_name,
                        'metric': metric,
                        'improvement': result['improvement'],
                        'variant': test_name.split(' vs ')[1]
                    })
        
        if significant_improvements:
            # Find best improvement
            best_improvement = max(significant_improvements, key=lambda x: x['improvement'])
            print(f"✅ DEPLOY: {best_improvement['variant']}")
            print(f"   Reason: {best_improvement['improvement']:+.2f}% improvement in {best_improvement['metric']}")
            print(f"   Test: {best_improvement['test']}")
        else:
            print("❌ KEEP: Baseline model")
            print("   Reason: No variant shows significant improvement")
        
        print("\n✅ A/B Testing completed successfully!")
        print(f"📁 Results saved to: wine_quality_ab_test_results.json")
        
        return ab_framework, all_tests
        
    except Exception as e:
        print(f"❌ Error during A/B testing: {str(e)}")
        print("Please check your data and model configuration.")
        return None, None

def quick_test():
    """
    Quick test untuk memverifikasi framework berfungsi
    """
    print("🧪 QUICK TEST - Verifying Framework")
    print("="*40)
    
    try:
        # Initialize framework
        ab_framework = ABTestingFramework()
        
        # Create simple variants
        ab_framework.create_model_variant("A", C=1.0, solver='lbfgs', max_iter=1000)
        ab_framework.create_model_variant("B", C=0.5, solver='lbfgs', max_iter=1000)
        
        # Train models
        ab_framework.train_model("A")
        ab_framework.train_model("B")
        
        # Run test
        results = ab_framework.run_ab_test("A", "B")
        
        print("✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if user wants to run quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_test()
        sys.exit(0 if success else 1)
    
    # Run full A/B testing
    framework, results = main()
    
    if framework is not None:
        print("\n🎉 A/B Testing completed! Check the results above.")
    else:
        print("\n💥 A/B Testing failed. Please check the error messages above.")
        sys.exit(1)
