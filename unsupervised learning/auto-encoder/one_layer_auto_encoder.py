import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt

loss = 'categorical_crossentropy'  #'mean_squared_error'
model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=784))
model.add(Dense(units=784, activation='softmax', name='Dense1'))
model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])

training_data, testing_data = mnist.load_data()
data_train = training_data[0]
label_train = training_data[1]
data_test = testing_data[0]
label_test = testing_data[1]

# num = 10
# plt.figure(figsize=(10, 6))
# plt.imshow(data_train[num], cmap='gray')
# plt.title(str(label_train[num]))
# plt.show()

init_training_data = data_train / 255.
print(init_training_data.shape)
denoise_training_data = []
for img in init_training_data:
    denoise_training_data.append(img + 0.05*np.random.random((28, 28)))
denoise_training_data = np.array(denoise_training_data)
print(denoise_training_data.shape)

denoise_training_data = denoise_training_data.reshape((60000, 784))
y = init_training_data.reshape(60000, 784)
model.fit(x=denoise_training_data, y=y, batch_size=100, epochs=7)

input_img = data_test[0].reshape((1, 784)) / 255.
output_img = model.predict(x=input_img).reshape((28, 28))

plt.figure(figsize=(10, 6))
plt.subplot(121), plt.imshow(input_img.reshape((28, 28)), cmap='gray')
plt.subplot(122), plt.imshow(output_img, cmap='gray')
plt.show()
