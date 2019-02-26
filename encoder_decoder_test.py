from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU, Bidirectional, Embedding
from keras import optimizers
import keras
import numpy
import librosa
import numpy
from sklearn.preprocessing import normalize


adam = optimizers.adam(lr=0.0001, clipnorm=1.)
filename = 'Adele - Someone Like You-hLQl3WQQoQ0.wav'
filename_2 = 'Adele - Rolling in the Deep-rYEDA3JcQqw.wav'

y, sr = librosa.load(filename, offset=15, duration=5)
y_2, sr_2 = librosa.load(filename_2, offset=15, duration=5)
MFCC_features = numpy.abs(librosa.core.stft(y=y, n_fft=4410, hop_length=812))
MFCC_features_2 = numpy.abs(librosa.core.stft(y=y_2, n_fft=4410, hop_length=812))

MFCC_features_normalized = normalize(MFCC_features, axis=1)
MFCC_features_normalized_2 = normalize(MFCC_features_2, axis=1)

MFCC_input = MFCC_features_normalized.reshape(1,len(MFCC_features_normalized[0]), len(MFCC_features_normalized))
test_input = [x for x in range(500)]

test_output = numpy.reshape(numpy.array(test_input), (500,1))
# test_input = numpy.reshape(numpy.array(test_input), (1,1,500))
test_input = numpy.array(test_input).reshape(1,500)
test_input2 = test_input


encoder_inputs = Input(shape=(136,2206), name='input')
encoded = GRU(551, return_sequences=True)(encoder_inputs)
encoded = GRU(78, return_sequences=True)(encoded)
decoded = Bidirectional(GRU(1103, activation='softmax', return_sequences=True, name='output'))(encoded)

auto_encoder = Model(encoder_inputs, decoded)



# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

auto_encoder.compile(optimizer='adam', loss='mse')
print(test_input.shape, test_output.shape)
auto_encoder.summary()
auto_encoder.fit(MFCC_input, MFCC_input, epochs=1)
encoder = Model(encoder_inputs, encoded)
reduced = encoder.predict(MFCC_input)
print(MFCC_input.shape, reduced.shape)