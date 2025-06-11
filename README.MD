# BotDetector – Proyecto Final de *Machine Learning*

Aplicación web desarrollada con **Flask** que clasifica perfiles de redes sociales como **bot** o **humano**.  
Carga un modelo previamente entrenado (`model_admit.pkl`) y expone una interfaz gráfica amigable y una API REST para predicciones en tiempo real. :contentReference[oaicite:0]{index=0}

---

## 🧩  Características principales

| Función | Descripción |
|---------|-------------|
| **Interfaz Web** | Formulario dinámico que solicita los atributos del perfil y muestra resultados con barras de probabilidad y gráficos de apoyo. :contentReference[oaicite:1]{index=1} |
| **API `/predict`** | Recibe datos vía `POST` y devuelve un JSON con `human_prob` y `bot_prob`. :contentReference[oaicite:2]{index=2} |
| **Modelo ML** | Modelo serializado con `pickle` (ej. RandomForest/GradientBoosting). Incluye script `inspect_model.py` para explorar características, tipo de estimador y metadatos. :contentReference[oaicite:3]{index=3} |
| **UI moderna** | Estilos CSS con paleta accesible y diseño *responsive* (grid). :contentReference[oaicite:4]{index=4} |

### Variables utilizadas por el modelo

avg_keywords, can_send_message, is_verified, subscribers_count,
can_post_on_wall, can_invite_to_group, posting_frequency_days,
avg_comments, has_mobile, reposts_ratio

:contentReference[oaicite:5]{index=5}

---

## Instalación rápida

```bash
# 1. Clona el repositorio
git clone https://github.com/javierportillar/mlfinalproject.git
cd mlfinalproject

# 2. Crea y activa un entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt
# ó, manualmente:
pip install flask scikit-learn numpy joblib

### Ejecución local
bash
Copiar
export FLASK_APP=app.py          # Windows: set FLASK_APP=app.py
flask run
Abre http://127.0.0.1:5000/ en tu navegador para acceder a la interfaz.

Para probar la API directamente:

bash
Copiar
curl -X POST http://127.0.0.1:5000/predict \
     -d "avg_keywords=3" \
     -d "can_send_message=1" \
     ... (resto de variables)
🛠️ Entrenamiento del modelo (opcional)
Prepara tu dataset en data/.

Entrena tu modelo en un notebook o script y guarda con:

python
Copiar
import pickle
with open("model_admit.pkl", "wb") as f:
    pickle.dump(model, f)
Verifica el artefacto con:

bash
Copiar
python inspect_model.py
🤝 Contribuciones
¡Las pull requests son bienvenidas!
Por favor abre un issue para discutir cambios mayores antes de hacer push.


<p align="center"> Hecho por Ali Valentina Mera Giraldo, Javier Portilla Rosero, Jhonathan Leandro Clavijo Troches </p> ``` ::contentReference[oaicite:6]{index=6}

