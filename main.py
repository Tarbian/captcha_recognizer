import io
import cv2
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()
model = load_model('../data/captcha_recognizer.h5')
info = {0: '2', 4: '6', 13: 'm', 9: 'd', 3: '5', 14: 'n', 1: '3', 12: 'g', 6: '8', 2: '4', 10: 'e', 18: 'y', 11: 'f', 16: 'w', 15: 'p', 5: '7', 8: 'c', 17: 'x', 7: 'b'}

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    x = [image[10:50, 30:50], image[10:50, 50:70], image[10:50, 70:90], image[10:50, 90:110], image[10:50, 110:130]]
    
    X_pred = []
    for i in range(5):
        X_pred.append(img_to_array(Image.fromarray(x[i])))
    
    X_pred = np.array(X_pred)
    X_pred /= 255.0
    
    return X_pred

@app.post("/predict")
async def predict_captcha(image: UploadFile):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(pil_image)
    y_pred = model.predict(processed_image)
    y_pred = np.argmax(y_pred, axis=1)
    
    prediction = ""
    for res in y_pred:
        prediction += info[res]
    #print(prediction)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
