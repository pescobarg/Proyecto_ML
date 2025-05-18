# Sección de imports
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from collections import Counter
from joblib import dump


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

# Selección por importancia (RandomForest)
from sklearn.ensemble import RandomForestClassifier
rf_fs = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_fs.fit(X, y)
importances = rf_fs.feature_importances_
indices = np.argsort(importances)[::-1]

top_k = 100
selected_columns = X.columns[indices[:top_k]]
X = X[selected_columns]

# 8. Split con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. PCA para reducción dimensional (conservando 95% varianza)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA redujo a {X_train_pca.shape[1]} dimensiones.")

# Función para mapear clusters a clases según la clase mayoritaria en entrenamiento
def map_clusters_to_labels(clusters, y_true):
    labels = np.zeros_like(clusters)
    for cluster in np.unique(clusters):
        mask = (clusters == cluster)
        majority_class = y_true.iloc[mask].mode()[0]
        labels[mask] = majority_class
    return labels

# 11. KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters_train = kmeans.fit_predict(X_train_pca)
labels_train = map_clusters_to_labels(clusters_train, y_train)

clusters_test = kmeans.predict(X_test_pca)
# Asumimos clusters similares en test según asignación train
labels_test = np.zeros_like(clusters_test)
for cluster in np.unique(clusters_test):
    mask = (clusters_test == cluster)
    majority_class = y_train.iloc[clusters_train == cluster].mode()[0]
    labels_test[mask] = majority_class

print("\n=== KMeans Clustering (no supervisado) ===")
print(confusion_matrix(y_test, labels_test))
print(classification_report(y_test, labels_test, digits=4, zero_division=0))
print(f"Acc={accuracy_score(y_test, labels_test):.3f}, BalAcc={balanced_accuracy_score(y_test, labels_test):.3f}")

# 12. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, random_state=42)
clusters_train = gmm.fit_predict(X_train_pca)
labels_train = map_clusters_to_labels(clusters_train, y_train)

clusters_test = gmm.predict(X_test_pca)
labels_test = np.zeros_like(clusters_test)
for cluster in np.unique(clusters_test):
    mask = (clusters_test == cluster)
    majority_class = y_train.iloc[clusters_train == cluster].mode()[0]
    labels_test[mask] = majority_class

print("\n=== GMM Clustering (no supervisado) ===")
print(confusion_matrix(y_test, labels_test))
print(classification_report(y_test, labels_test, digits=4, zero_division=0))
print(f"Acc={accuracy_score(y_test, labels_test):.3f}, BalAcc={balanced_accuracy_score(y_test, labels_test):.3f}")

# 13. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_train = dbscan.fit_predict(X_train_pca)
# DBSCAN puede asignar -1 para ruido, filtramos ese caso asignando clase minoritaria (o -1)
labels_train = np.zeros_like(clusters_train)
for cluster in np.unique(clusters_train):
    mask = (clusters_train == cluster)
    if cluster == -1:
        # Asignar la clase minoritaria como etiqueta para ruido
        labels_train[mask] = y_train.mode()[0]
    else:
        majority_class = y_train.iloc[mask].mode()[0]
        labels_train[mask] = majority_class

clusters_test = dbscan.fit_predict(X_test_pca)
labels_test = np.zeros_like(clusters_test)
for cluster in np.unique(clusters_test):
    mask = (clusters_test == cluster)
    if cluster == -1:
        labels_test[mask] = y_train.mode()[0]
    else:
        # Usamos la mayoría de la clase entrenada para ese cluster (o la moda general)
        mask_train = (clusters_train == cluster)
        if np.any(mask_train):
            majority_class = y_train.iloc[mask_train].mode()[0]
        else:
            majority_class = y_train.mode()[0]
        labels_test[mask] = majority_class

print("\n=== DBSCAN Clustering (no supervisado) ===")
print(confusion_matrix(y_test, labels_test))
print(classification_report(y_test, labels_test, digits=4, zero_division=0))
print(f"Acc={accuracy_score(y_test, labels_test):.3f}, BalAcc={balanced_accuracy_score(y_test, labels_test):.3f}")


# Después de entrenar tus modelos
dump(dbscan, "modelos/modelo_DB.joblib")
dump(gmm, "modelos/modelo_GMM.joblib")
dump(kmeans, "modelos/modelo_KM.joblib")

# Guardar el scaler
dump(scaler, "modelos/scaler_NS.joblib")

# Guardar las columnas seleccionadas (es una lista o Index de pandas)
dump(selected_columns, "modelos/columns_NS.joblib")

# Guardar PCA
dump(pca, "modelos/pca_NS.joblib")