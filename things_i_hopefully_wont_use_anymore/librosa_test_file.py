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
# from keras.models import Sequential, Model
# from keras.layers import LSTM, Input, RepeatVector, Bidirectional
# import keras.backend as K
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.utils import plot_model

# from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
# adam = optimizers.adam(lr=0.0001, clipnorm=1.)

filename = '/Users/m_vys/Music/Karel Gott - C\'est la vie.mp3'
# filename_3 = 'Black Sabbath - Die Young (lyrics)-CJgHn7MeAwc.wav'
# filename_2 = 'Adele - Rolling in the Deep-rYEDA3JcQqw.wav'
y, sr = librosa.load(filename)
# y_2, sr_2 = librosa.load(filename_2, offset=15, duration=5)
# y_3, sr_3 = librosa.load(filename_3, offset=15, duration=60)
# plt.figure(figsize=(10, 4))
#
# MFCC = librosa.feature.melspectrogram(y=y, sr=sr)
# MFCC_2 = librosa.feature.melspectrogram(y=y_2, sr=sr_2)
# MFCC_3 = librosa.feature.melspectrogram(y=y_3, sr=sr_3)
#
# MFCC_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=320)
# MFCC_2_features = librosa.feature.mfcc(y=y_2, sr=sr_2)
# MFCC_3_features = librosa.feature.mfcc(y=y_3, sr=sr_3)
# MFCC_features_normalized = normalize(MFCC_features, axis=0)
# MFCC_features_normalized = MFCC_features_normalized.reshape(1, len(MFCC_features_normalized[0]), len(MFCC_features))
spectrogram = numpy.abs(librosa.core.stft(y, n_fft=4410, hop_length=812))
# spectrogram_2 = numpy.abs(librosa.core.stft(y_2, n_fft=4410, hop_length=812))
# spectrogram_3 = numpy.abs(librosa.core.stft(y_3))
#
# mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
# mel_spectrogram2 = librosa.feature.melspectrogram(y=y_2, sr=sr_2, n_mels=320, n_fft=4410, hop_length=812)
#
# normalized_input = normalize(mel_spectrogram, axis=1).reshape(len(mel_spectrogram[0]), len(mel_spectrogram))
# normalized_input2 = normalize(mel_spectrogram2, axis=1).reshape(len(mel_spectrogram2[0]), len(mel_spectrogram2))
#
# double_input = numpy.array([normalized_input, normalized_input2])
# vector = MFCC.flatten()
# vector_2 = MFCC_2.flatten()
# vector_3 = MFCC_3.flatten()


# define model
# model = Sequential()
# model.add(LSTM(160, activation='sigmoid', return_sequences=True, input_shape=(136, 320)))
# model.add(LSTM(80, activation='sigmoid', return_sequences=True, input_shape=(136, 160)))
# model.add(Bidirectional(LSTM(160, activation='tanh', return_sequences=True)))
# model.compile(optimizer=adam, loss='mse')

# model.summary()
# model.fit(double_input, double_input, epochs=1)
# plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
# # encoder = K.function([model.layers[0].input], [model.layers[1].output])
# yhat = model.predict(normalized_input.reshape(1, len(mel_spectrogram[0]), len(mel_spectrogram)))
# encoded = encoder.predict(normalized_input.reshape(1, len(mel_spectrogram[0]), len(mel_spectrogram)))
# print(encoded[0, :, 0])
# print(encoded.shape)
# print(yhat.shape)
# print(yhat[0, :, 0])

librosa.display.specshow(librosa.power_to_db(spectrogram),
                         y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()
plt.show()

# librosa.display.specshow(librosa.power_to_db(spectrogram_2),
#                          y_axis='log', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Power spectrogram 2' )
# plt.tight_layout()
# plt.show()
#
# librosa.display.specshow(librosa.power_to_db(spectrogram_3),
#                          y_axis='log', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Power spectrogram 3')
# plt.tight_layout()
# plt.show()

# df = pandas.DataFrame(data=[vector, vector_2, vector_3])
#
# scaled_df = pandas.DataFrame(preprocessing.scale(df), columns=df.columns)
# pca = decomposition.PCA()
# df2 = pca.fit_transform(scaled_df)
# print(pca.explained_variance_ratio_)
# print(pandas.DataFrame(pca.components_, columns=scaled_df.columns, index=['PC-1', 'PC-2', 'PC-3']))
#
#
# print(df[0])

# dist_12 = spatial.distance.cosine(vector, vector_2)
# dist_23 = spatial.distance.cosine(vector_2, vector_3)
# dist_13 = spatial.distance.cosine(vector, vector_3)
#
# dist_PCA_12 = spatial.distance.cosine(df[0], df[1])
# dist_PCA_13 = spatial.distance.cosine(df[0], df[2])
# dist_PCA_23 = spatial.distance.cosine(df[1], df[2])

# librosa.display.specshow(librosa.power_to_db(normalized_input),
#                          y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
# librosa.display.specshow(librosa.power_to_db(yhat[0]),
#                          y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('MFCC spectrogram 2')
# plt.tight_layout()
# plt.show()
#
# plt.show()
# print(y)