from tensorflow.keras.models import load_model
import Config
import numpy
import cv2
import imageio

import Preprocessing


def get_model():
    model = load_model(filepath=Config.modelname)
    return model

def generate(full_tensor, model, name):
    sample = full_tensor[-Config.batch_size:,:,:,:]
    prepared = numpy.expand_dims(sample,axis=0)
    result = model.predict(prepared)
    img = result[0,:,:,:]
    cv2.imwrite(filename=f"results/{name}.jpg", img=Preprocessing.normalization(img,direct=False))
    return img

def videoSave(name, tensor):
    tensor = numpy.uint8(tensor)
    imageio.mimwrite(name, tensor, fps=20)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter(name, fourcc=fourcc, fps=2.0, frameSize=tensor.shape[1:3])
    # for i in range(tensor.shape[0]):
    #     frame = numpy.uint8(tensor[i, :, :, :])
    #     out.write(frame)
    # out.release()

def serial(full_tensor, model):
    generation = []
    for i in range(Config.depth):
        img = generate(full_tensor=full_tensor, model=model, name=str(i))
        generation.append(img)
        new = numpy.expand_dims(img, axis=0)
        full_tensor = numpy.concatenate([full_tensor,new], axis=0)
    full_tensor = Preprocessing.normalization(full_tensor, direct=False)#.astype(dtype="float16")
    print(full_tensor.shape)
    videoSave('finally/output.avi', full_tensor)

    return full_tensor