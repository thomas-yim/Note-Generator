from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential

x = [[[1],[1]],[[2],[2]],[[3],[3]],[[4],[4]]]
y = [1,2,3,4]
model = Sequential()
model.add(Conv2D(32, (1,1), activation='relu', input_shape=(len(x),len(x[0]),1)))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
model.fit(x,y)