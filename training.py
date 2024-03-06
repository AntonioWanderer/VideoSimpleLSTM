import Preprocessing, Config, Model, Prediction

if __name__ == "__main__":
    video = Preprocessing.loadVideo()
    x, y = Preprocessing.to_batch(video)
    x,y = Preprocessing.normalization(x), Preprocessing.normalization(y)
    model = Model.get_model(x.shape[1:])
    print(x.shape, y.shape)
    print(model.summary())
    model.fit(x,y,epochs=10)
    model.save(filepath=Config.modelname)