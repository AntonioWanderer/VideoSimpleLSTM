from tensorflow.keras.models import load_model
import Config
import numpy
import cv2

import Preprocessing


def get_model():
    model = load_model(filepath=Config.modelname)
    return model

def generate(full_tensor, model, name):
    sample = full_tensor[-Config.batch_size:,:,:,:]
    prepared = numpy.expand_dims(sample,axis=0)
    result = model.predict(prepared)
    img = result[0,:,:,:]
    cv2.imwrite(filename=f"{name}.jpg", img=img)
    return img

def serial(full_tensor, model):
    generation = []
    for i in range(Config.depth):
        img = generate(full_tensor=full_tensor, model=model, name=str(i))
        generation.append(img)
        new = numpy.expand_dims(img, axis=0)
        # full_tensor = numpy.concatenate([full_tensor,new], axis=0)
    full_tensor = Preprocessing.normalization(full_tensor, direct=False)
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    out = cv2.VideoWriter('output.avi', fourcc=fourcc, fps=2.0, frameSize=full_tensor.shape[1:3])
    for i in range(full_tensor.shape[0]):
        out.write(full_tensor[i,:,:,:])
    out.release()
    return full_tensor