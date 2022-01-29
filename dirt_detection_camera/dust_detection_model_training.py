import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

image_size = (256, 256)
batch_size = 32


def model_training(epochs=2):
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		"../dataset2",
		validation_split=0.2,
		subset="training",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		"../dataset2",
		validation_split=0.2,
		subset="validation",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)

	data_augmentation = keras.Sequential(
		[
			#         layers.RandomFlip("horizontal"),
			#         layers.RandomRotation(0.1),
		]
	)

	def make_model(input_shape, num_classes):
		inputs = keras.Input(shape=input_shape)
		# Image augmentation block
		x = data_augmentation(inputs)

		# Entry block
		x = layers.Rescaling(1.0 / 255)(x)
		x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation("relu")(x)

		x = layers.Conv2D(64, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation("relu")(x)

		previous_block_activation = x  # Set aside residual

		for size in [128, 256, 512, 728]:
			x = layers.Activation("relu")(x)
			x = layers.SeparableConv2D(size, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.Activation("relu")(x)
			x = layers.SeparableConv2D(size, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

			# Project residual
			residual = layers.Conv2D(size, 1, strides=2, padding="same")(
				previous_block_activation
			)
			x = layers.add([x, residual])  # Add back residual
			previous_block_activation = x  # Set aside next residual

		x = layers.SeparableConv2D(1024, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation("relu")(x)

		x = layers.GlobalAveragePooling2D()(x)
		if num_classes == 2:
			activation = "sigmoid"
			units = 1
		else:
			activation = "softmax"
			units = num_classes

		x = layers.Dropout(0.5)(x)
		outputs = layers.Dense(units, activation=activation)(x)
		return keras.Model(inputs, outputs)

	# model = make_model(input_shape=image_size + (3,), num_classes=2)
	# keras.utils.plot_model(model, show_shapes=True)

	model = Sequential()
	model.add(ResNet50(include_top=False, pooling='max'))
	model.add(Dense(1, activation='sigmoid'))
	# ResNet-50 model is already trained, should not be trained
	model.layers[0].trainable = True

	callbacks = [
		keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
	]
	model.compile(
		optimizer=keras.optimizers.Adam(1e-3),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)
	model.fit(
		train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
	)
	return model

model = model_training(2)
print(model)

#
# from keras.models import load_model
# model1 = load_model('save_at_10.h5')
#
# # In[75]:
#
#
# import os
# for i in os.listdir("C:\MyScripts\ML\BITS\Sem4\CameraQuality\Practice\dataset2\Positive"):
#     img = keras.preprocessing.image.load_img(
#         r"C:\MyScripts\ML\BITS\Sem4\CameraQuality\Practice\dataset2\Positive\{}".format(i), target_size=image_size
#     )
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#
#     predictions = model2.predict(img_array)
#     score = predictions[0]
#     if score > 0.5:
#         print("Positive")
#     else:
#         print("Nagative")
#
# import os
# for i in os.listdir(r"C:\MyScripts\ML\BITS\Sem4\CameraQuality\Practice\dataset2\Negative"):
#     img = keras.preprocessing.image.load_img(
#         r"C:\MyScripts\ML\BITS\Sem4\CameraQuality\Practice\dataset2\Negative\{}".format(i), target_size=image_size
#     )
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#
#     predictions = model1.predict(img_array)
# #     print(predictions)
#     score = predictions[0]
#     if score > 0.5:
#         print("Positive")
#     else:
#         print("Nagative")
# #     print(score)
# #     print(
# #         "This image is %.2f percent Dirt and %.2f percent Positive."
# #         % (100 * (1 - score), 100 * score)
# #     )
#
#
#
# model2.summary()
#
#
#
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# import onnx
# import os
#
# onnx_model_name = 'dirt-resnet50.onnx'
#
# model10 = load_model(r"C:\MyScripts\ML\BITS\Sem4\CameraQuality\Practice\save_at_10.h5")
# tf.saved_model.save(model10, "tmp_model1")
# # onnx_model = keras2onnx.convert_keras(model10, model10.name)
# # onnx.save_model(onnx_model, onnx_model_name)
