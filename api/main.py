from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "Penguins_RF"
MODEL_VERSION = "1"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Penguins Classification API",
    description="""
API para predecir la especie de un pingüino usando un modelo
Random Forest registrado en MLflow.

### Valores para la inferencia

- **island:** Biscoe, Dream o Torgersen
- **bill_length_mm:** longitud del pico en mm
- **bill_depth_mm:** profundidad del pico en mm
- **flipper_length_mm:** longitud de la aleta en mm
- **body_mass_g:** peso en gramos
- **sex:** male o female

Ejemplo:
`Torgersen, 39.1, 18.7, 181, 3750, male`

""",
    version="1.0.0"
)


# ============================================================
# CONFIGURAR MLFLOW
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# CARGAR MODELO DESDE MLFLOW
# ============================================================

model = None


@app.on_event("startup")
def load_model():

    global model

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    print("======================================")
    print("CARGANDO MODELO DESDE MLFLOW")
    print("======================================")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Versión: {MODEL_VERSION}")
    print(f"Model URI: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("Modelo cargado correctamente")
    print("======================================")


# ============================================================
# MODELO DE ENTRADA
# ============================================================

class PenguinInput(BaseModel):

    island: str
    bill_length_mm: float | None = None
    bill_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Penguins Classification API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "mlflow": MLFLOW_TRACKING_URI
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "error",
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post(
    "/predict",
    summary="Realizar inferencia",
    description="""
Ingrese los datos del pingüino para predecir su especie.

Valores:
- island: Biscoe, Dream o Torgersen
- bill_length_mm: longitud del pico en mm
- bill_depth_mm: profundidad del pico en mm
- flipper_length_mm: longitud de la aleta en mm
- body_mass_g: peso en gramos
- sex: male o female

Ejemplo:
Torgersen, 39.1, 18.7, 181, 3750, male
"""
)
def predict(data: PenguinInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está cargado"
        )

    try:

        input_data = pd.DataFrame([{
            "bill_length_mm": data.bill_length_mm,
            "bill_depth_mm": data.bill_depth_mm,
            "flipper_length_mm": data.flipper_length_mm,
            "body_mass_g": data.body_mass_g,
            "island": data.island,
            "sex": data.sex
        }])

        prediction = model.predict(input_data)

        return {
            "prediction": str(prediction[0]),
            "model": MODEL_NAME,
            "version": MODEL_VERSION
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )