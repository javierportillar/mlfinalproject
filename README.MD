# BotDetector – Proyecto Final de *Machine Learning*

Aplicación web desarrollada con **Flask** que clasifica perfiles de redes sociales como **bot** o **humano**.  
Carga un modelo previamente entrenado (`model_admit.pkl`) este modelo se encuentra en el siguiente link [Link Colab](https://colab.research.google.com/drive/1ut2pMnxQV1ILBj91ydiIBFoLJ3avo5Rm) y expone una interfaz gráfica amigable y una API REST para predicciones en tiempo real.
---

## 🧩  Características principales

| Función | Descripción |
|---------|-------------|
| **Interfaz Web** | Formulario dinámico que solicita los atributos del perfil y muestra resultados con barras de probabilidad y gráficos de apoyo.|
| **API `/predict`** | Recibe datos vía `POST` y devuelve un JSON con `human_prob` y `bot_prob`.|
| **Modelo ML** | Modelo serializado con `pickle` (ej. RandomForest/GradientBoosting). Incluye script `inspect_model.py` para explorar características, tipo de estimador y metadatos.|
| **UI moderna** | Estilos CSS con paleta accesible y diseño *responsive* (grid).|

### Variables utilizadas por el modelo

avg_keywords, can_send_message, is_verified, subscribers_count,
can_post_on_wall, can_invite_to_group, posting_frequency_days,
avg_comments, has_mobile, reposts_ratio

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
```
### Ejecución local

```bash

export FLASK_APP=app.py          # Windows: set FLASK_APP=app.py
flask run
Abre http://127.0.0.1:5000/ en tu navegador para acceder a la interfaz.

```


Hecho por Ali Valentina Mera Giraldo, Javier Portilla Rosero, Jhonathan Leandro Clavijo Troches

