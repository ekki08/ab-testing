# 🍷 A/B Testing Framework untuk Model Multinomial Regression

Framework komprehensif untuk melakukan A/B testing pada model multinomial regression wine quality Anda.

## 📋 Daftar Isi

1. [Overview](#overview)
2. [Fitur Utama](#fitur-utama)
3. [Instalasi](#instalasi)
4. [Penggunaan Dasar](#penggunaan-dasar)
5. [Contoh Penggunaan](#contoh-penggunaan)
6. [Metrik yang Diukur](#metrik-yang-diukur)
7. [Statistical Significance](#statistical-significance)
8. [Visualisasi](#visualisasi)
9. [MLflow Integration](#mlflow-integration)
10. [Best Practices](#best-practices)

## 🎯 Overview

Framework ini dirancang khusus untuk melakukan A/B testing pada model multinomial regression wine quality. Anda dapat membandingkan berbagai variant model dengan parameter yang berbeda dan menentukan mana yang memberikan performa terbaik secara statistik.

## ✨ Fitur Utama

- **Multiple Model Variants**: Buat dan bandingkan berbagai variant model
- **Statistical Significance Testing**: Bootstrap test untuk memastikan perbedaan signifikan
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, dan class-wise metrics
- **Visualization**: Grafik dan plot untuk analisis hasil
- **MLflow Integration**: Logging dan tracking eksperimen
- **Production Ready**: Framework siap untuk deployment
- **Custom Scenarios**: Support untuk feature engineering dan preprocessing yang berbeda

## 🚀 Instalasi

```bash
# Pastikan semua dependencies terinstall
pip install pandas numpy matplotlib seaborn scikit-learn mlflow scipy

# Clone atau download file framework
# ab_testing_framework.py
# ab_testing_examples.py
```

## 📖 Penggunaan Dasar

### 1. Import Framework

```python
from ab_testing_framework import ABTestingFramework
```

### 2. Initialize Framework

```python
# Initialize dengan dataset wine quality
ab_framework = ABTestingFramework(data_path="data/winequality-red.csv")
```

### 3. Create Model Variants

```python
# Variant A: Baseline model
ab_framework.create_model_variant("Baseline", C=1.0, solver='lbfgs', max_iter=1000)

# Variant B: Higher regularization
ab_framework.create_model_variant("High_Reg", C=0.1, solver='lbfgs', max_iter=1000)

# Variant C: Different solver
ab_framework.create_model_variant("Newton", C=1.0, solver='newton-cg', max_iter=1000)
```

### 4. Train Models

```python
# Train semua variant
ab_framework.train_model("Baseline")
ab_framework.train_model("High_Reg")
ab_framework.train_model("Newton")
```

### 5. Run A/B Test

```python
# Bandingkan Baseline vs High_Reg
results = ab_framework.run_ab_test("Baseline", "High_Reg")
```

### 6. Visualize Results

```python
# Visualisasi hasil A/B test
ab_framework.visualize_ab_test_results(results, "Baseline", "High_Reg")
```

### 7. Generate Report

```python
# Generate laporan komprehensif
ab_framework.generate_report(results, "Baseline", "High_Reg")
```

## 🎯 Contoh Penggunaan

### Contoh 1: A/B Testing Regularization

```python
from ab_testing_examples import example_1_basic_ab_test

# Test regularization parameter yang berbeda
framework, results = example_1_basic_ab_test()
```

### Contoh 2: Solver Comparison

```python
from ab_testing_examples import example_2_solver_comparison

# Bandingkan solver algorithms
framework, results = example_2_solver_comparison()
```

### Contoh 3: Comprehensive Testing

```python
from ab_testing_examples import example_3_comprehensive_testing

# Test multiple variants sekaligus
framework, results = example_3_comprehensive_testing()
```

### Contoh 4: Production-Ready Testing

```python
from ab_testing_examples import example_4_production_ready_ab_test

# A/B testing dengan MLflow integration
framework, results = example_4_production_ready_ab_test()
```

### Contoh 5: Feature Engineering

```python
from ab_testing_examples import example_5_custom_ab_test_scenario

# Test feature engineering yang berbeda
model_a, model_b, metrics_a, metrics_b = example_5_custom_ab_test_scenario()
```

## 📊 Metrik yang Diukur

### Basic Metrics
- **Accuracy**: Proporsi prediksi yang benar
- **Precision (Weighted)**: Rata-rata precision untuk semua kelas
- **Recall (Weighted)**: Rata-rata recall untuk semua kelas
- **F1-Score (Weighted)**: Harmonic mean dari precision dan recall

### Class-wise Metrics
- **Precision per Class**: Precision untuk setiap kelas (Low, Medium, High)
- **Recall per Class**: Recall untuk setiap kelas
- **F1-Score per Class**: F1-score untuk setiap kelas

### Additional Metrics
- **Cross-validation Score**: Robust evaluation dengan CV
- **Training Time**: Waktu training model
- **Prediction Confidence**: Rata-rata confidence dari prediksi

## 🔬 Statistical Significance

Framework menggunakan **Bootstrap Test** untuk menentukan statistical significance:

### Bootstrap Test
- **Sample Size**: 1000 bootstrap samples (default)
- **Confidence Level**: 95% (default)
- **P-value**: Two-tailed test
- **Significance**: p < 0.05

### Interpretation
- ✅ **Significant**: Variant B secara statistik lebih baik dari Variant A
- ❌ **Not Significant**: Tidak ada perbedaan signifikan antara kedua variant

## 📈 Visualisasi

Framework menyediakan berbagai visualisasi:

### 1. Improvement Bar Chart
- Menampilkan persentase improvement untuk setiap metrik
- Indikator significance dengan bintang (★)

### 2. Metric Comparison
- Side-by-side comparison antara dua variant
- Bar chart untuk setiap metrik

### 3. P-values Chart
- Visualisasi p-values untuk setiap metrik
- Threshold line pada p=0.05

### 4. Confidence Intervals
- Error bars menunjukkan confidence intervals
- Indikator significance

## 🔄 MLflow Integration

Framework terintegrasi dengan MLflow untuk experiment tracking:

### Logging Parameters
```python
mlflow.log_param("test_type", "production_ab_test")
mlflow.log_param("control_variant", "Production_Current")
mlflow.log_param("treatment_variant", "Production_Candidate")
```

### Logging Metrics
```python
mlflow.log_metric("accuracy_control", result['variant_a'])
mlflow.log_metric("accuracy_treatment", result['variant_b'])
mlflow.log_metric("accuracy_improvement", result['improvement'])
mlflow.log_metric("accuracy_significant", int(result['significant']))
```

### Model Logging
```python
mlflow.sklearn.log_model(
    ab_framework.experiments["Production_Current"]["model"],
    "production_current_model"
)
```

## 🎯 Best Practices

### 1. Experimental Design
- **Control Group**: Gunakan model baseline sebagai control
- **Treatment Group**: Variant yang ingin diuji
- **Random State**: Gunakan random state yang konsisten
- **Stratification**: Pastikan split data stratified

### 2. Statistical Testing
- **Sample Size**: Minimal 1000 bootstrap samples
- **Confidence Level**: 95% untuk production, 90% untuk exploration
- **Multiple Testing**: Pertimbangkan Bonferroni correction untuk multiple comparisons

### 3. Metric Selection
- **Primary Metric**: Fokus pada satu primary metric (biasanya F1-score)
- **Secondary Metrics**: Monitor metrics lain untuk context
- **Business Metrics**: Pertimbangkan business impact

### 4. Production Deployment
- **Gradual Rollout**: Deploy secara bertahap
- **Monitoring**: Monitor performa di production
- **Rollback Plan**: Siapkan plan untuk rollback jika diperlukan

## 📁 File Structure

```
├── ab_testing_framework.py      # Main framework
├── ab_testing_examples.py       # Contoh penggunaan
├── multinomial_regression_enhanced.py  # Model original Anda
├── README_AB_Testing.md         # Dokumentasi ini
├── data/
│   └── winequality-red.csv      # Dataset wine quality
└── results/
    ├── example_1_results.json   # Hasil A/B test
    ├── example_2_results.json
    └── ...
```

## 🚀 Quick Start

```python
# Run semua contoh sekaligus
from ab_testing_examples import run_all_examples
all_results = run_all_examples()

# Atau run contoh spesifik
from ab_testing_examples import example_1_basic_ab_test
framework, results = example_1_basic_ab_test()
```

## 📞 Support

Jika Anda memiliki pertanyaan atau masalah:

1. **Check Documentation**: Baca dokumentasi ini dengan teliti
2. **Run Examples**: Coba jalankan contoh-contoh yang disediakan
3. **Debug Mode**: Gunakan print statements untuk debugging
4. **Error Handling**: Framework memiliki error handling yang komprehensif

## 🔄 Version History

- **v1.0**: Initial release dengan basic A/B testing
- **v1.1**: Added MLflow integration
- **v1.2**: Enhanced visualization dan reporting
- **v1.3**: Added feature engineering examples

---

**Happy A/B Testing! 🍷📊**
