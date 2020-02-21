import tensorflow as tf
model = tf.keras.models.load_model("firstmodel.h5")
print(model.predict([10]))