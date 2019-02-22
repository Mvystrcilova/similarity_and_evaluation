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

filename = 'Adele - Someone Like You-hLQl3WQQoQ0.wav'
filename_3 = 'Black Sabbath - Die Young (lyrics)-CJgHn7MeAwc.wav'
filename_2 = 'Adele - Rolling in the Deep-rYEDA3JcQqw.wav'
y, sr = librosa.load(filename, offset=15, duration=60)
y_2, sr_2 = librosa.load(filename_2, offset=15, duration=60)
y_3, sr_3 = librosa.load(filename_3, offset=15, duration=60)
plt.figure(figsize=(10, 4))

# MFCC = librosa.feature.melspectrogram(y=y, sr=sr)
# MFCC_2 = librosa.feature.melspectrogram(y=y_2, sr=sr_2)
# MFCC_3 = librosa.feature.melspectrogram(y=y_3, sr=sr_3)

MFCC_features = librosa.feature.mfcc(y=y, sr=sr)
MFCC_2_features = librosa.feature.mfcc(y=y_2, sr=sr_2)
MFCC_3_features = librosa.feature.mfcc(y=y_3, sr=sr_3)

spectrogram = numpy.abs(librosa.core.stft(y))
spectrogram_2 = numpy.abs(librosa.core.stft(y_2))
spectrogram_3 = numpy.abs(librosa.core.stft(y_3))

# vector = MFCC.flatten()
# vector_2 = MFCC_2.flatten()
# vector_3 = MFCC_3.flatten()

librosa.display.specshow(librosa.power_to_db(spectrogram),
                         y_axis='log', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()
plt.show()
librosa.display.specshow(librosa.power_to_db(spectrogram_2),
                         y_axis='log', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram 2' )
plt.tight_layout()
plt.show()

librosa.display.specshow(librosa.power_to_db(spectrogram_3),
                         y_axis='log', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram 3')
plt.tight_layout()
plt.show()

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

librosa.display.specshow(librosa.power_to_db(MFCC_features),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
librosa.display.specshow(librosa.power_to_db(MFCC_2_features),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC spectrogram 2')
plt.tight_layout()
plt.show()

librosa.display.specshow(librosa.power_to_db(MFCC_3_features),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC spectrogram 3')
plt.tight_layout()
plt.show()
print(y)