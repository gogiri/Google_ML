from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

class AlexNet:
    
    @staticmethod
    def build(input_shape=(224,224,3), activation='relu', class_num=1000):
        model = Sequential()
        # 첫 번째 합성곱 층 
        model.add(Conv2D(96, 11, strides=(4,4), 
                         input_shape=input_shape, activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=2))
        model.add(BatchNormalization())

        # 두 번째 합성곱 층
        model.add(Conv2D(256, 5, activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=2))
        model.add(BatchNormalization())

        # 세 번째 합성곱 층
        model.add(Conv2D(384, 3, activation=activation, padding='same'))

        # 네 번째 합성곱 층
        model.add(Conv2D(384, 3, activation=activation, padding='same'))

        # 다섯 번째 합성곱 층
        model.add(Conv2D(256, 3, activation=activation, padding='same'))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=2))

        model.add(Flatten())
        # 여섯 번째 Fully connected layer(은닉층)
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.4))

        # 일곱 번째 Fully connected layer(은닉층)
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.4))

        # 출력층
        model.add(Dense(class_num, activation='softmax'))

        return model
