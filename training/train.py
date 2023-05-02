# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Array of images, size = (n,x,y)
# test_images = np.loadtxt(open("testImages.csv", "rb"), delimiter=",", skiprows=1)
# print('test images loaded')
 test_images=np.load('testImages.npy')
print(test_images.shape)
train_labels = np.loadtxt('trainLabels.txt', dtype=int)
train_images = np.load('trainImages.npy')
test_labels = np.zeros((2631,1))
n,x,y=test_images.shape
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(x,y)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3) # Three classes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

p = probability_model.predict(test_images)
np.savetxt('predictions.txt', p, fmt='%d')
for row in range(2631):
    if p[row,0]>p[row,1] and p[row,0]>p[row,2]:
        test_labels[row]=0
    elif p[row,1]>p[row,0] and p[row,1]>p[row,2]:
        test_labels[row]=1
    else:
        test_labels[row]=2
np.savetxt('maxPred.txt', test_labels, fmt='%d')



     