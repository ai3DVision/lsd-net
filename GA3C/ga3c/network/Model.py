from keras.layers import Flatten, Dense, Input, Convolution2D
from keras.layers import LSTM,Reshape
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from GA3C.ga3c.Config import Config
import keras.backend as K

def create_p_and_v_models(model_name, num_actions, img_height, img_width, img_channels):
	K.set_learning_phase(Config.TRAIN_MODELS)

	if 'nbv' in model_name:
		p_model, v_model = create_nbv_models(num_actions, img_height, img_width, img_channels)
	elif 'atari' in model_name:
		p_model, v_model = create_atari_models(num_actions, img_height, img_width, img_channels)
	elif 'cartpole' in model_name:
		p_model, v_model = create_cartpole_models(num_actions, img_height, img_width, img_channels)
	else:
		raise('Model does not exist.')

	print(p_model.summary())
	print(v_model.summary())

	return p_model, v_model

def create_atari_models(num_actions, img_height, img_width, img_channels):
	inputs = Input(shape=(img_height, img_width, img_channels))
	conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(inputs)
	conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
	conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
	flatten = Flatten()(conv3)
	fc = Dense(512, activation='relu')(flatten)
	action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(fc)
	state_value = Dense(name="v", output_dim=1, activation='linear')(fc)

	p_model = Model(input=inputs, output=action_probs)
	v_model = Model(input=inputs, output=state_value)

	return p_model, v_model

def create_cartpole_models(num_actions, img_height, img_width, img_channels):
	inputs = Input(shape=(img_height, img_width, img_channels))
	flatten = Flatten()(inputs)
	fc1 = Dense(32, activation="relu")(flatten)
	fc2 = Dense(32, activation="relu")(fc1)
	fc3 = Dense(32, activation="relu")(fc2)
	action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(fc3)
	state_value = Dense(name="v", output_dim=1, activation='linear')(fc3)

	p_model = Model(input=inputs, output=action_probs)
	v_model = Model(input=inputs, output=state_value)

	return p_model, v_model

def create_nbv_models(num_actions, img_height, img_width, img_channels):
	assert(img_channels == 3)
	assert(img_height >= 197 and img_width >= 197)

	inputs = Input(shape=(img_height, img_width, img_channels))
	resnet50 = ResNet50(include_top=False, weights='imagenet')(inputs)
	flatten = Flatten()(resnet50)
	fc = Dense(256)(flatten)
	reshape = Reshape((1,256))(fc)
	lstm = LSTM(12)(reshape)
	action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(lstm)
	state_value = Dense(name="v", output_dim=1, activation='linear')(lstm)

	p_model = Model(input=inputs, output=action_probs)
	v_model = Model(input=inputs, output=state_value)

	return p_model, v_model
