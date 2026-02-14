import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

MODEL_PATH = "model.keras"  # caminho do modelo treinado
IMAGE_SIZE = (200, 200)     # tamanho da imagem (o mesmo do treino)
CLASS_NAMES = ['animal', 'hydrant', 'stop']

def image_classifier(model, photo):
    img = image.load_img(photo, target_size=IMAGE_SIZE) # carrega a imagem
    img_array = image.img_to_array(img) # converte a imagem para array numérico

    img_array = np.expand_dims(img_array, axis=0) # adiciona dimensão de batch (modelo espera [batch, h, w, c])
    #img_array = img_array / 255.0 # normaliza (igual ao treino)
    
    predictions = model.predict(img_array) # faz a previsão
    predictions = predictions[0] # muda lista para 1D pois o batch é igual a 1

    return predictions

def dir_photos(model, dir_path):
    result = []
    for image_path in dir_path.iterdir():
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            predictions = image_classifier(model, image_path)
            linha ={ "imagem":image_path.name }
            for classe, prob in zip(CLASS_NAMES, predictions):
                linha[classe] = prob
            
            ######corrigir aqui
            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)
            print(f"confidence= {confidence}: \n")
            if confidence >= 0.50:
                linha["classe_predita"] = CLASS_NAMES[predicted_index]
            elif confidence > 0.35 and confidence < 0.50:
                linha["classe_predita"] = f"~{CLASS_NAMES[predicted_index]}"
            else:
                linha["classe_predita"] = "UNCLASSIFIED"
                ################

            result.append(linha)
    
    return result

def one_photo(model, photo):
    result = []

    predictions = image_classifier(model, photo)
    linha ={ "imagem":photo }
    for classe, prob in zip(CLASS_NAMES, predictions):
        linha[classe] = prob
                
    linha["classe_predita"] = CLASS_NAMES[np.argmax(predictions)]

    result.append(linha)

    return result

def main():
    image_path = None
    dir_path = None
    result = []
    model = tf.keras.models.load_model(MODEL_PATH) # carrega o modelo treinado

    num = int(input("Digite 1 para escolher uma foto ou 2 para escolher um diretorio com varias fotos: "))

    if num==1:
        image = input("Digite o nome da imagem (ex: image.jpg): ")
        result = one_photo(model, image)
        indice_max = result.index(max(result))
        valor_max = max(result)
        print(indice_max)
        coluna_class_max = CLASS_NAMES[indice_max]
        if valor_max >= 0.80:
            print(f"Essa imagem pertence a classe: {coluna_class_max} com confiança de {valor_max}")
        elif valor_max > 0.60 and valor_max < 0.80:
            print(f"Não tenho muita certeza se essa imagem é da classe: {coluna_class_max}")
        else:
            print(f"Não foi possivel classificar essa imagem, treine o modelo com mais amostras!")

    else:
        dir_path = Path(input("Digite o caminho do diretorio onde estão as fotos: "))
        result = dir_photos(model, dir_path)
        print("\nSegue abaixo o resultado da classificação das fotos:\n\n")
        df = pd.DataFrame(result)
        print(df)
    

if __name__ == "__main__":
    main()