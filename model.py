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

# 7. GridSearchCV: parametry dla obu modeli
grid_params = [
    {
        'clf': [pipe_rf.named_steps['clf']],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20],
        'clf__min_samples_split': [2, 5]
    },
    {
        'clf': [pipe_xgb.named_steps['clf']],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.1, 0.01]
    }
]

# Skrócenie cv dla przyspieszenia
cv = 3
tasks = sum(len(list(ParameterGrid(params))) for params in grid_params) * cv
print(f"Łącznie trenowań: {tasks}")

# 8. Wspólna pipeline (placeholder dla 'clf')
pipeline = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier())  # nadpisywane w GridSearchCV
])

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    refit=True
)

with tqdm_joblib(tqdm(desc="GridSearchCV", total=tasks, unit="job")):
    grid.fit(X_train, y_train)

# 9. Pobranie najlepszego modelu
del pipeline
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# 10. Kalibracja prawdopodobieństw
calibrated = CalibratedClassifierCV(best_model, cv='prefit')
calibrated.fit(X_train, y_train)

# 11. Ocena
y_proba = calibrated.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.2).astype(int)
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

# 12. SHAP explainability
# Transformacja treningowa
data_trans = best_model.named_steps['preproc'].transform(X_train)
model_for_shap = best_model.named_steps['clf']
explainer = shap.TreeExplainer(model_for_shap)
shap_values = explainer.shap_values(data_trans)

# summary plot
def plot_shap():
    shap.summary_plot(
        shap_values, data_trans,
        feature_names=(
            num_features +
            list(best_model.named_steps['preproc']
                 .named_transformers_['cat']
                 .get_feature_names_out(cat_features))
        )
    )

plot_shap()

# 13. Zapis modelu
to_save = {'model': calibrated, 'preprocessor': preprocessor}
joblib.dump(to_save, 'diabetes_advanced_model.pkl')

