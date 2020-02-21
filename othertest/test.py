import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt


celsius_q    = np.array([2, 3, 4, 5, 6, 7, 8,9,10],  dtype=float)
fahrenheit_a = np.array([50, 70, 90, 110, 130, 150, 170,190,210],  dtype=float)
# celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
# fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#units=1 — This specifies the number of neurons in the layer
#input_shape=[1] — This specifies that the input to this layer is a single value.

l = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.9))
history = model.fit(celsius_q, fahrenheit_a, epochs=2000, verbose=False)
# print("Finished training the model")
#use Matplotlib to visualize this (you could use another tool).
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
model.save("firstmodel.h5")
print(model.predict([11]))
print("These are the layer variables: {}".format(l.get_weights()))