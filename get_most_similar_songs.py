import numpy, pandas
import random
def get_n_most_similar(distance_matrix, index, n):
    songs = pandas.read_csv('/Users/m_vys/Documents/matfyz/rocnikac/djangoApp/rocnikac/rocnikac/useful_songs', names=['title', 'artist'], sep=';', header=None, index_col=False)
    distances = numpy.load(distance_matrix)[index]
    song = songs.ix[index]
    # print(song)
    sorted_distances = numpy.sort(distances)
    top_n = sorted_distances[-n]
    distances[distances < top_n] = 0
    n_most_similar = numpy.nonzero(distances)[0]
    # print(n_most_similar.shape)
    closest_songs = songs.iloc[n_most_similar.tolist()[:n]]
    closest_songs['distances'] = distances[n_most_similar[:n]]
    closest_songs = closest_songs.sort_values(by=['distances'], ascending=False)
    return closest_songs

def assert_list_variability(dataframe):
    length = dataframe.shape[0]
    artists = dataframe.groupby(by='artist').ngroups

    return artists/length

def assert_method_variability(distance_matrix):
    variabilities = []
    for i in random_indexes:
        dataframe = get_n_most_similar(distance_matrix, i, 100)
        variabilities.append(assert_list_variability(dataframe))

    return sum(variabilities)/len(variabilities)


random_indexes = random.sample(range(0, 16593), k=1000)

# value = assert_method_variability('distances/w2v_distances.npy')
# print(value)
# get_n_most_similar('distances/w2v_distances.npy', 9472, 10)
# get_n_most_similar('distances/tf_idf_distances.npy', 9472, 10)
# get_n_most_similar('distances/pca_mel_distances_5717.npy', 9472, 10)
print('pca_mel_5715', assert_method_variability('/mnt/0/pca_mel_distances_5717.npy'))
print('pca_mel_320', assert_method_variability('/mnt/0/pca_melspectrogram_distances.npy'))

print('pca_spec_1106', assert_method_variability('/mnt/0/pca_spec_distances_1106.npy'))
print('pca_spec_320', assert_method_variability('/mnt/0/short_pca_spec_distances.npy'))

print('lstm_mel', assert_method_variability('/mnt/0/lstm_mel_distances_5712.npy'))
print('gru_mel', assert_method_variability('/mnt/0/gru_mel_distances_5712.npy'))

print('lstm_spec_20400', assert_method_variability('/mnt/0/distances/final_lstm_spec_distances.npy'))
print('gru_spec_20400', assert_method_variability('/mnt/0/distances/final_gru_spec_distances.npy'))

print('lstm_spec_5712', assert_method_variability('/mnt/0/short_LSTM_spec_distances.npy'))
print('gru_spec_5712', assert_method_variability('/mnt/0/short_GRU_spec_distances.npy'))

print('lstm_mfcc', assert_method_variability('/mnt/0/lstm_mfcc_distances.npy'))
print('gru_mfcc', assert_method_variability('/mnt/0/gru_mfcc_distances.npy'))

print('tf_idf', assert_method_variability('/mnt/0/distances/tf_idf_distances.npy'))

print('som_w2v', assert_method_variability('/mnt/0/distances/SOM_W2V_batch_5g5i133188_distances.npy'))
print('pca_tf_idf', assert_method_variability('/mnt/0/pca_tf_idf_distances.npy'))

print('w2v', assert_method_variability('/mnt/0/distances/w2v_distances.npy'))