from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle, os

app = Flask(__name__)

# ---------- carga del modelo ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_admit.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- 7 características reales ----------
FEATURE_ORDER = [
    "avg_keywords",
    "can_send_message",
    "is_verified",
    "subscribers_count",
    "can_post_on_wall",
    "can_invite_to_group",
    "posting_frequency_days",
    "avg_comments",
    "has_mobile",
    "reposts_ratio",
]


# ---------- rutas ----------
@app.route("/")
def home():
    # pasamos la lista al template para generar dinámicamente el formulario
    return render_template("index.html", features=FEATURE_ORDER)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recibe los 7 valores del formulario, los pasa al modelo
    y devuelve JSON con las probabilidades.
    """
    try:
        # extraer los valores en el orden correcto
        vals = [float(request.form[feat]) for feat in FEATURE_ORDER]
        X = np.array([vals])

        if hasattr(model, "predict_proba"):
            bot_p = float(model.predict_proba(X)[0][1])  # clase 1 = bot
            human_p = 1.0 - bot_p
            out = {
                "human_prob": round(human_p * 100, 1),
                "bot_prob": round(bot_p * 100, 1),
            }
        else:  # por si tuvieras un modelo sin proba
            pred = float(model.predict(X)[0])
            out = {"prediction": pred}

        return jsonify(out)

    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)
