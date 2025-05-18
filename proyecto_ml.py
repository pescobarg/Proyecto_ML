# Sección de imports
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score
)
from joblib import dump
from collections import Counter

# 1. Carga y limpieza
ds = pd.read_csv("https://www.datos.gov.co/resource/cm2t-qreq.csv")
ds = ds.drop(columns=['instanceid', 'no_h', 'no_v'])
ds = ds.drop(columns=[c for c in ds.columns if ds[c].isnull().all()])

# 2. Imputación
num_cols = ds.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = ds.select_dtypes(include=['object', 'category']).columns.tolist()

ds[num_cols] = KNNImputer(n_neighbors=5).fit_transform(ds[num_cols])

for c in cat_cols:
    ds[c] = ds[c].fillna(ds[c].mode()[0])

# 3. One-Hot Encoding
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe = pd.DataFrame(enc.fit_transform(ds[cat_cols]),
                   columns=enc.get_feature_names_out(cat_cols),
                   index=ds.index)

ds = pd.concat([ds.drop(columns=cat_cols), ohe], axis=1)

# 4. Binarizar target
ds['p28'] = ds['p28'].map({1: 1, 2: 0})

# 5. Eliminar columnas innecesarias
ds = ds.drop(columns=[c for c in ds.columns if c.startswith('tmp_')])

# 6. Separar X e y
X = ds.drop(columns=['p28'] + [c for c in ds.columns if c.startswith('tmp_dep')])
y = ds['p28']

# 7. Eliminar columnas constantes
constant_filter = VarianceThreshold(threshold=0.0)
X = pd.DataFrame(constant_filter.fit_transform(X),
                 columns=X.columns[constant_filter.get_support()],
                 index=X.index)

# Selección de variables por importancia RandomForest
rf_fs = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_fs.fit(X, y)
importances = rf_fs.feature_importances_
indices = np.argsort(importances)[::-1]

# Seleccionar top 100 variables más importantes
top_k = 100
selected_columns = X.columns[indices[:top_k]]
X = X[selected_columns]

# 8. Split con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train antes SMOTE reforzado:", Counter(y_train))

# 9. Pipeline SMOTE reforzado (SMOTE + Tomek Links) + UnderSampling
resample_pipeline = ImbPipeline(steps=[
    ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('tomek', TomekLinks(sampling_strategy='auto'))
])

X_res, y_res = resample_pipeline.fit_resample(X_train, y_train)
print("Train después SMOTE reforzado + UnderSampling:", Counter(y_res))

# 10. Escalado
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)
X_test = scaler.transform(X_test)

# 11. Modelos mejor configurados
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear', random_state=42),
    "SVM": SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale', probability=True, random_state=42)
}


# 12. Entrenamiento y evaluación
for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"Acc={accuracy_score(y_test, y_pred):.3f}, "
          f"BalAcc={balanced_accuracy_score(y_test, y_pred):.3f}")



# Guardar modelos
dump(models["RandomForest"], "modelos/modelo_rf.joblib")
dump(models["SVM"], "modelos/modelo_svm.joblib")
dump(models["GradientBoosting"], "modelos/modelo_gb.joblib")
dump(models["LogisticRegression"], "modelos/modelo_lr.joblib")

# Guardar scaler y columnas
dump(scaler, "modelos/scaler.joblib")
dump(selected_columns, "modelos/columns.joblib")