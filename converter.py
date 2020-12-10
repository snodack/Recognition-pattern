import tensorflow as tf
model = tf.keras.models.load_model('model3')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open ("model.tflite", "wb") .write(tfmodel)