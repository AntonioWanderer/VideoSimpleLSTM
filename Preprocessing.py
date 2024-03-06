import cv2
import Config
import numpy
def loadVideo():
    frames = []
    cap = cv2.VideoCapture(Config.filename)
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            h,w,d = frame.shape
            new = cv2.resize(src=frame,dsize=(w//4,h//4))
            frames.append(new)
    return numpy.array(frames)

def normalization(tensor, direct = True):
    if direct:
        return tensor/255
    else:
        return tensor*255
def to_batch(video_tensor):
    N = video_tensor.shape[0]
    x = []
    y = []
    for i in range(N-Config.batch_size-1):
        batch = video_tensor[i:i+Config.batch_size,:,:,:]
        x.append(batch)
        y.append(video_tensor[i+Config.batch_size+1])
    return numpy.array(x), numpy.array(y)

if __name__ == "__main__":
    video = loadVideo()
    x,y = to_batch(video)
    print(x.shape, y.shape)
