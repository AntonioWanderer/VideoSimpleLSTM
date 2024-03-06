import Preprocessing, Config, Model, Prediction

if __name__ == "__main__":
    video = Preprocessing.loadVideo()
    video = Preprocessing.normalization(video)
    model = Prediction.get_model()
    print(model.summary())
    #Prediction.generate(full_tensor=video,model=model)
    Prediction.serial(full_tensor=video,model=model)