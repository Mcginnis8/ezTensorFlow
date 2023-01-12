from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class EasyTensorflow():
    def createModel(numInputs, X_train, y_train, layers=2, outputType='relu', modelName = "tfmodel"):
        pow2 = 1
        while numInputs < pow2:
            pow2 = pow2 * 2
        model = Sequential()
        model.add(Dense(units=pow2, activation='relu', input_dim=len(X_train.columns)))
        temp = pow2
        for i in range(0,layers-1):
            temp = temp/2
            model.add(Dense(units=temp, activation='relu'))
        model.add(Dense(units=1, activation=outputType))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        model.fit(X_train, y_train, epochs=256, batch_size=8)
        model.save(modelName)
        return model