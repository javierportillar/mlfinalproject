# inspect_model.py
from joblib import load  # o pickle, si lo guardaste con pickle
import pprint

MODEL_PATH = "model_admit.pkl"  # ajusta si está en otra carpeta
m = load(MODEL_PATH)

print("✅ Modelo cargado desde:", MODEL_PATH)
print("n_features_in_  ➜", m.n_features_in_)

# Si guardaste un Pipeline, busca el paso final (estimador) así:
try:
    # Para RandomForest dentro de un Pipeline llamado "clf"
    estimator = m.named_steps.get("clf", m)
except AttributeError:
    estimator = m  # no es Pipeline, es el estimador directo

# Nombres de columnas (si los guardaste)
try:
    print("\nfeature_names_in_ ➜")
    pprint.pprint(estimator.feature_names_in_.tolist())
except AttributeError:
    print("⚠️  El modelo no guardó los nombres, solo la cantidad.")

# Extras útiles
print("\nTipo de objeto:", type(m))
