from keras.models import Sequential, Model
from keras.layers import LSTM, Input, Reshape, Activation
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

from keras import optimizers
import librosa
import numpy
import librosa.display
import pandas
import pydub
import matplotlib.pyplot as plt
from matplotlib import interactive
from scipy import spatial
interactive(True)
from sklearn import decomposition, preprocessing
from sklearn.preprocessing import normalize
from numpy import array

adam = optimizers.adam(lr=0.0001, clipnorm=1.)



filename = 'Adele - Someone Like You-hLQl3WQQoQ0.wav'
filename_2 = 'Adele - Rolling in the Deep-rYEDA3JcQqw.wav'

y, sr = librosa.load(filename, offset=15, duration=60)
y_2, sr_2 = librosa.load(filename_2, offset=15, duration=60)
MFCC_features = librosa.feature.mfcc(y=y, sr=sr)
MFCC_features_2 = librosa.feature.mfcc(y=y_2, sr=sr_2)

MFCC_features_normalized = normalize(MFCC_features, axis=1).flatten()
MFCC_features_normalized_2 = normalize(MFCC_features_2, axis=1).flatten()

MFCC_input = numpy.array([MFCC_features_normalized])
# flat_MFCC = numpy.array([flat_MFCC]).T
# d = {'index': x, 'values': flat_MFCC}
# input = pandas.DataFrame(d)
encoding_dimension = 10336
x_train = numpy.random.random((1000, 20))
print(x_train.shape)
print(MFCC_input.shape)
# input_spectrogram = Input(shape=(51680,))
#
# encoded = Dense(encoding_dimension, activation='relu')(input_spectrogram)
#
# decoded = Dense(51680, activation='sigmoid')(encoded)
#
# autoencoder = Sequential(input_spectrogram, decoded)
# encoder = Model(input_spectrogram, encoded)
#
# encoded_input = (Input(shape=(encoding_dimension,)))
# decoder_layer = autoencoder.layers[-1]
# decoder = Sequential(encoded_input, decoder_layer(encoded_input))
# print(MFCC_input.shape)
autoencoder = Sequential()
autoencoder.add(Dense(encoding_dimension, input_dim=51680))
autoencoder.add(Activation('relu'))
autoencoder.add(Dense(5183, activation='relu', input_dim=10336, name='encoder'))
autoencoder.add(Dense(10336, activation='relu', input_dim=5138))
autoencoder.add(Dense(51680, activation='relu', input_dim=10336))
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.fit(x=MFCC_input, y=MFCC_input, epochs=1)

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
decoded_audio = autoencoder.predict(MFCC_input)

encoded_audio = encoder.predict(MFCC_input)
print(encoded_audio)



