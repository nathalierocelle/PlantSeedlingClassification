import numpy as np
import cv2 as cv

def preprocess_data(img):
    img = cv.resize(cv.imread(img), (120, 120))
    img = np.asarray(img)
    img = img.astype('float32') / 255
    return img.reshape(-1, 120, 120, 3)

def prediction(model,image,labels):
    pred_class = model.predict(image)[0]
    pred_class_np = np.argmax(pred_class, axis=-1) 
    label = labels[pred_class_np]
    score = max(pred_class)*100
    return label, score   
