# Zaawansowany skrypt trenowania modelu predykcji ryzyka cukrzycy
# Ulepszenia:
# - SMOTE oversampling
# - class_weight w RF
# - próg decyzyjny tuning
# - porównanie RandomForest vs XGBoost
# - kalibracja prawdopodobieństw
# - analiza ważności cech za pomocą SHAP

# 1. Instalacja zależności
# pip install ucimlrepo scikit-learn pandas joblib tqdm tqdm-joblib matplotlib imbalanced-learn xgboost shap

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, f1_score
)
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# 2. Pobranie danych
dataset = fetch_ucirepo(id=891)
X = dataset.data.features
y = dataset.data.targets.values.ravel()

# 3. Definicja cech
cat_features = ['HighBP','HighChol','CholCheck','Smoker','Stroke',
                'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
                'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
num_features = ['BMI','GenHlth','MentHlth','PhysHlth','Age','Education','Income']

# 4. Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessing
to_num = StandardScaler()
to_cat = OneHotEncoder(drop='if_binary', dtype=int)
preprocessor = ColumnTransformer([
    ('num', to_num, num_features),
    ('cat', to_cat, cat_features)
])

# 6a. Pipeline RandomForest + SMOTE
pipe_rf = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
])

# 6b. Pipeline XGBoost (z oversampling)
pipe_xgb = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# 6c. Dodanie różnych technik oversamplingu dla lepszego balansu klas
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

# 7. GridSearchCV: parametry dla obu modeli z rozszerzonymi opcjami
grid_params = [
    {
        'clf': [pipe_rf.named_steps['clf']],
        'smote': [SMOTE(random_state=42), BorderlineSMOTE(random_state=42), ADASYN(random_state=42)],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', None]
    },
    {
        'clf': [pipe_xgb.named_steps['clf']],
        'smote': [SMOTE(random_state=42), SMOTETomek(random_state=42), SMOTEENN(random_state=42)],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 6, 9],
        'clf__learning_rate': [0.1, 0.05, 0.01],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'clf__gamma': [0, 0.1]
    }
]

# Zastosowanie stratyfikowanej walidacji krzyżowej dla lepszej oceny
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

# Bardziej zaawansowana walidacja krzyżowa
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
tasks = sum(len(list(ParameterGrid(params))) for params in grid_params) * 5 * 2
print(f"Łącznie trenowań: {tasks}")

# 8. Wspólna pipeline (placeholder dla 'clf')
pipeline = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier())  # nadpisywane w GridSearchCV
])

# Dodanie różnych metryk oceny dla lepszego wyboru modelu
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, matthews_corrcoef, average_precision_score

# Niestandardowa metryka F-beta dla wyższej wagi precyzji
f2_scorer = make_scorer(fbeta_score, beta=2)
mcc_scorer = make_scorer(matthews_corrcoef)
ap_scorer = make_scorer(average_precision_score)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=cv,
    scoring={
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'f2': f2_scorer,
        'mcc': mcc_scorer,
        'avg_prec': ap_scorer
    },
    refit='mcc',  # Matthews Correlation Coefficient jest dobrą metryką dla zbiorów niezbalansowanych
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

with tqdm_joblib(tqdm(desc="GridSearchCV", total=tasks, unit="job")):
    grid.fit(X_train, y_train)

# 9. Pobranie najlepszego modelu
del pipeline
best_model = grid.best_estimator_
print("\nBest params:", grid.best_params_)
print(f"Najlepszy wynik {grid.refit}: {grid.best_score_:.4f}")

# Analiza wyników GridSearch
cv_results = pd.DataFrame(grid.cv_results_)
print("\nWyniki dla najlepszego modelu:")
metrics = ['mean_test_f1', 'mean_test_roc_auc', 'mean_test_mcc', 'mean_test_avg_prec', 'mean_test_f2']
best_idx = grid.best_index_
for metric in metrics:
    print(f"{metric}: {cv_results.loc[best_idx, metric]:.4f}")

# 10. Kalibracja prawdopodobieństw - testowanie różnych metod kalibracji
from sklearn.calibration import calibration_curve
import numpy as np

# Testowanie różnych metod kalibracji
calibration_methods = ['sigmoid', 'isotonic']
calibrated_models = {}

for method in calibration_methods:
    print(f"\nKalibracja metodą {method}:")
    calibrated = CalibratedClassifierCV(best_model, cv='prefit', method=method)
    calibrated.fit(X_train, y_train)
    calibrated_models[method] = calibrated
    
    # Ocena kalibracji
    y_proba = calibrated.predict_proba(X_test)[:,1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label=f'Kalibracja {method}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Idealna kalibracja')
    plt.xlabel('Przewidywane prawdopodobieństwo')
    plt.ylabel('Rzeczywista częstość')
    plt.title(f'Wykres kalibracji - metoda {method}')
    plt.legend()
    plt.grid()
    plt.show()

# Wybór najlepszej metody kalibracji (na podstawie Brier score)
from sklearn.metrics import brier_score_loss

brier_scores = {}
for method, model in calibrated_models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    brier_scores[method] = brier_score_loss(y_test, y_proba)
    
best_calibration = min(brier_scores, key=brier_scores.get)
print(f"\nNajlepsza metoda kalibracji: {best_calibration} (Brier score: {brier_scores[best_calibration]:.4f})")
calibrated = calibrated_models[best_calibration]

# 11. Zaawansowany tuning progu decyzyjnego
y_proba = calibrated.predict_proba(X_test)[:,1]

# Testowanie różnych progów decyzyjnych
thresholds = np.linspace(0.05, 0.95, 19)
results = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    
    # Obliczanie różnych metryk
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Sensitivity (true positive rate / recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'f1': f1,
        'mcc': mcc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'balanced_acc': (specificity + sensitivity) / 2
    })

results_df = pd.DataFrame(results)
print("\nWyniki dla różnych progów decyzyjnych:")
print(results_df)

# Wizualizacja metryk w funkcji progu
plt.figure(figsize=(12, 8))
for metric in ['accuracy', 'f1', 'mcc', 'specificity', 'sensitivity', 'balanced_acc']:
    plt.plot(results_df['threshold'], results_df[metric], marker='o', label=metric)
plt.xlabel('Próg decyzyjny')
plt.ylabel('Wartość metryki')
plt.title('Metryki w funkcji progu decyzyjnego')
plt.legend()
plt.grid()
plt.show()

# Wybór optymalnego progu - maksymalizacja balanced accuracy
best_threshold_idx = results_df['balanced_acc'].idxmax()
best_threshold = results_df.loc[best_threshold_idx, 'threshold']
print(f"\nOptymalny próg decyzyjny: {best_threshold:.3f}")
print(f"Balanced Accuracy: {results_df.loc[best_threshold_idx, 'balanced_acc']:.4f}")

# Finalna ocena z optymalnym progiem
y_pred = (y_proba >= best_threshold).astype(int)

print("\nOcena z optymalnym progiem:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Precision-Recall curve
dp, dr, thr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(dr, dp)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.grid()
plt.show()

# 12. Rozszerzona analiza SHAP explainability
# Transformacja treningowa i testowa
X_train_trans = best_model.named_steps['preproc'].transform(X_train)
X_test_trans = best_model.named_steps['preproc'].transform(X_test)
model_for_shap = best_model.named_steps['clf']
explainer = shap.TreeExplainer(model_for_shap)

# Pobierz nazwy cech po transformacji
feature_names = (
    num_features +
    list(best_model.named_steps['preproc']
         .named_transformers_['cat']
         .get_feature_names_out(cat_features))
)

# Analiza SHAP dla danych treningowych i testowych
shap_values_train = explainer.shap_values(X_train_trans)
shap_values_test = explainer.shap_values(X_test_trans)

# Funkcja do generowania różnych wykresów SHAP
def generate_shap_plots(shap_values, X, title_suffix=''):
    # 1. Summary plot - ogólny przegląd ważności cech
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        plot_size=(12, 10),
        title=f'SHAP Summary Plot {title_suffix}'
    )
    plt.tight_layout()
    plt.show()
    
    # 2. Bar plot - ranking ważności cech
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        plot_type='bar',
        title=f'SHAP Feature Importance {title_suffix}'
    )
    plt.tight_layout()
    plt.show()
    
    # 3. Wykres zależności dla najważniejszych cech
    top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]  # 5 najważniejszych cech
    for i in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            i, shap_values, X,
            feature_names=feature_names,
            title=f'SHAP Dependence Plot for {feature_names[i]} {title_suffix}'
        )
        plt.tight_layout()
        plt.show()

# Generowanie wykresów dla danych treningowych
print("\n\nAnalizowanie ważności cech dla danych treningowych:")
generate_shap_plots(shap_values_train, X_train_trans, 'na danych treningowych')

# Generowanie wykresów dla danych testowych
print("\n\nAnalizowanie ważności cech dla danych testowych:")
generate_shap_plots(shap_values_test, X_test_trans, 'na danych testowych')

# 13. Analiza krzywej ROC
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, marker='.', label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()

# 14. Zaawansowana analiza błędów
errors = y_test != y_pred
X_errors = X_test[errors]
y_errors = y_test[errors]
y_proba_errors = y_proba[errors]

print("\nAnaliza błędów klasyfikacji:")
print(f"Liczba błędnie sklasyfikowanych przypadków: {errors.sum()} z {len(y_test)}")

# Jeśli mamy jakieś błędy, analizujemy ich charakterystykę
if errors.sum() > 0:
    fp_idx = (y_test == 0) & (y_pred == 1)  # False Positives
    fn_idx = (y_test == 1) & (y_pred == 0)  # False Negatives
    
    print(f"False Positives: {fp_idx.sum()}")
    print(f"False Negatives: {fn_idx.sum()}")
    
    if fp_idx.sum() > 0:
        print("\nCharakterystyka False Positives (błędnie zdiagnozowani jako chorzy):")
        fp_data = X_test[fp_idx]
        for feature in num_features:
            avg_value = fp_data[feature].mean()
            all_avg = X_test[feature].mean()
            print(f"{feature}: {avg_value:.3f} (vs {all_avg:.3f} w całym zbiorze)")
    
    if fn_idx.sum() > 0:
        print("\nCharakterystyka False Negatives (błędnie zdiagnozowani jako zdrowi):")
        fn_data = X_test[fn_idx]
        for feature in num_features:
            avg_value = fn_data[feature].mean()
            all_avg = X_test[feature].mean()
            print(f"{feature}: {avg_value:.3f} (vs {all_avg:.3f} w całym zbiorze)")

# 15. Zapisz model wraz z dodatkowymi informacjami
to_save = {
    'model': calibrated, 
    'preprocessor': preprocessor,
    'best_threshold': best_threshold,
    'feature_names': feature_names,
    'cat_features': cat_features,
    'num_features': num_features,
    'grid_search_results': grid.cv_results_,
    'model_type': type(best_model.named_steps['clf']).__name__,
    'calibration_method': best_calibration
}
joblib.dump(to_save, 'diabetes_advanced_model.pkl')

print("\nModel został zapisany jako diabetes_advanced_model.pkl")
print("Zakończono optymalizację modelu predykcji cukrzycy.")

# Podsumowanie ulepszeń
print("\nWprowadzone ulepszenia:")
print("1. Rozszerzone parametry wyszukiwania w GridSearchCV")
print("2. Zaawansowane techniki oversamplingu (SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek, SMOTEENN)")
print("3. Stratyfikowana walidacja krzyżowa z powtórzeniami")
print("4. Wykorzystanie wielu metryk oceny (F1, F2, MCC, ROC AUC, Average Precision)")
print("5. Optymalizacja metod kalibracji prawdopodobieństw")
print("6. Zaawansowany tuning progu decyzyjnego")
print("7. Rozszerzona analiza ważności cech z wykorzystaniem SHAP")
print("8. Dodatkowe analizy błędów klasyfikacji")

