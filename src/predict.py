import joblib
import tensorflow as tf
import os

def load_models(model_dir="models"):
    models = {}

    models["flood"] = joblib.load(os.path.join(model_dir, "flood_model.joblib"))
    models["storm"] = joblib.load(os.path.join(model_dir, "storm_model.joblib"))
    models["rain"] = joblib.load(os.path.join(model_dir, "rain_model.joblib"))
    models["landslide"] = joblib.load(os.path.join(model_dir, "landslide_model.joblib"))

    models["dl"] = tf.keras.models.load_model(
        os.path.join(model_dir, "dl_model.keras"),
        compile=False
    )

    print("All models loaded successfully ✅")
    return models
