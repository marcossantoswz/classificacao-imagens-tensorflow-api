import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI(title="API de classificação de imagens")

MODEL_PATH = "model.keras"
IMAGE_SIZE = (200, 200)
CLASS_NAMES = ['animal', 'hydrant', 'stop']

# Carrega o modelo apenas uma vez para não travar a API
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lê os bytes do arquivo enviado
    contents = await file.read()

    # Converte bytes para imagem e garante o modo RGB
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Redimensiona para o tamanho do treino
    img = img.resize(IMAGE_SIZE)

    # Converte para array numérico (Equivalente ao image_to_array)
    img_array = np.array(img).astype('float32')

    # Adiciona dimensão de batch [1, 200, 200, 3]
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    highest_prob_index = np.argmax(predictions)

    resultado = {
        "filename": file.filename,
        "prediction": CLASS_NAMES[np.argmax(predictions)],
        "confidence": float(predictions[highest_prob_index]),
    }

    return resultado

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)