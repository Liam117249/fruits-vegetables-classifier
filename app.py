import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# ── Model & class configuration ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "fruits_veg_model.keras")

CLASS_NAMES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum",
    "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant",
    "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce",
    "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple",
    "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn",
    "sweetpotato", "tomato", "turnip", "watermelon"
]

# Emoji map for a delightful UI touch
CLASS_EMOJI = {
    "apple": "🍎", "banana": "🍌", "beetroot": "🫚", "bell pepper": "🫑",
    "cabbage": "🥬", "capsicum": "🫑", "carrot": "🥕", "cauliflower": "🥦",
    "chilli pepper": "🌶️", "corn": "🌽", "cucumber": "🥒", "eggplant": "🍆",
    "garlic": "🧄", "ginger": "🫚", "grapes": "🍇", "jalepeno": "🌶️",
    "kiwi": "🥝", "lemon": "🍋", "lettuce": "🥬", "mango": "🥭",
    "onion": "🧅", "orange": "🍊", "paprika": "🫑", "pear": "🍐",
    "peas": "🫛", "pineapple": "🍍", "pomegranate": "🍎", "potato": "🥔",
    "raddish": "🌱", "soy beans": "🫘", "spinach": "🥬", "sweetcorn": "🌽",
    "sweetpotato": "🍠", "tomato": "🍅", "turnip": "🥕", "watermelon": "🍉"
}

IMG_SIZE = (224, 224)

# Load model once at startup
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_bytes):
    """Load image bytes → (1, 224, 224, 3) float32 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)   # model has Rescaling layer inside
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()

        # Build base64 preview to send back
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = file.content_type or "image/jpeg"
        image_data_url = f"data:{mime};base64,{b64}"

        # Run inference
        net = load_model()
        tensor = preprocess_image(image_bytes)
        logits = net.predict(tensor, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()

        top5_idx = np.argsort(probs)[::-1][:5]
        results = [
            {
                "label": CLASS_NAMES[i],
                "emoji": CLASS_EMOJI.get(CLASS_NAMES[i], "🌿"),
                "confidence": float(round(probs[i] * 100, 2))
            }
            for i in top5_idx
        ]

        return jsonify({
            "prediction": results[0]["label"],
            "emoji": results[0]["emoji"],
            "confidence": results[0]["confidence"],
            "top5": results,
            "image": image_data_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
