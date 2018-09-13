# listing 2.1
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Raw data:')
print(train_images, train_labels)
# listing 2.2
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
print(network)
print(network.layers)
print(network.get_layer(index=0))
print(network.get_layer(index=1))
# listing 2.3
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# listing 2.4
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print('Reshaped data:')
print(train_images, train_labels)
#listing 2.5
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Now let's apply the learned
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
