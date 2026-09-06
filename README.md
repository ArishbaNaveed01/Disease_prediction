DiseasePrediction

Predict the possibility of disease in a patient based on structured medical data, using classification algorithms.

*Dataset*
*Breast Cancer Wisconsin (Diagnostic) Dataset* 569 patient records, 30 numeric diagnostic
features (radius, texture, perimeter, area, smoothness, concavity, symmetry, etc.) derived from
digitized images of breast masses. 

The dataset ships with `scikit-learn` (`sklearn.datasets.load_breast_cancer`), so the project runs
end-to-end with no external downloads — it's one of the standard UCI ML Repository medical datasets
referenced in the task brief.

All four models achieve >96% accuracy and >0.98 ROC-AUC, with Logistic Regression scoring
highest on ROC-AUC and SVM/Random Forest tying on accuracy/F1. Full metrics, confusion matrices,
ROC curves, and feature-importance charts are saved to `outputs/`.


DiseasePrediction
disease_prediction.py     
requirements.txt
README.md
__outputs/
    dataset.csv
    model_comparison.csv
    results_summary.txt
    01_class_distribution.png
    02_correlation_heatmap.png
    03_model_comparison_bar.png
    04_roc_curves.png
    05_feature_importance.png
              



