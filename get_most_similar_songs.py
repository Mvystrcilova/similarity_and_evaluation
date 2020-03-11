import numpy, pandas
import random

songs = pandas.read_csv('useful_songs', names=['title', 'artist'], sep=';', header=None, index_col=False)

def get_n_most_similar(distance_matrix, index, n):
    distances = numpy.load(distance_matrix)[index]
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
    if length > 0:
        return artists/length
    return 0

def assert_method_variability(distance_matrix):
    variabilities = []
    j = 0
    for i in range(0,16594):
        dataframe = get_n_most_similar(distance_matrix, i, 50)
        variabilities.append(assert_list_variability(dataframe))
        if (j % 1000) == 0:
            print(j)
        j+=1

    return sum(variabilities)/len(variabilities)


# random_indexes = random.sample(range(0, 16593), k=1000)

# value = assert_method_variability('distances/w2v_distances.npy')
# print(value)
# get_n_most_similar('distances/w2v_distances.npy', 9472, 10)
# get_n_most_similar('distances/tf_idf_distances.npy', 9472, 10)
# get_n_most_similar('distances/pca_mel_distances_5717.npy', 9472, 10)
# print('pca_mel_320', assert_method_variability('/mnt/0/pca_melspectrogram_distances.npy'))

# print('pca_spec_1106', assert_method_variability('/mnt/0/pca_spec_distances_1106.npy'))
# print('pca_spec_320', assert_method_variability('/mnt/0/short_pca_spec_distances.npy'))

# print('lstm_mel', assert_method_variability('/mnt/0/lstm_mel_distances_5712.npy'))
print('gru_mel', assert_method_variability('distances/gru_mel_distances_5712.npy'))

# print('lstm_spec_20400', assert_method_variability('/mnt/0/distances/final_lstm_spec_distances.npy'))
# print('gru_spec_20400', assert_method_variability('/mnt/0/distances/final_gru_spec_distances.npy'))

# print('lstm_spec_5712', assert_method_variability('/mnt/0/short_LSTM_spec_distances.npy'))
# print('gru_spec_5712', assert_method_variability('/mnt/0/short_GRU_spec_distances.npy'))

# print('lstm_mfcc', assert_method_variability('/mnt/0/lstm_mfcc_distances.npy'))
# print('gru_mfcc', assert_method_variability('new_distances/gru_mfcc_distances_30_64.npy'))

# print('tf_idf', assert_method_variability('/mnt/0/distances/tf_idf_distances.npy'))

# print('som_w2v', assert_method_variability('/mnt/0/distances/SOM_W2V_batch_5g5i133188_distances.npy'))
### print('pca_tf_idf', assert_method_variability('distances/pca_tf_idf_distances.npy'))

# print('w2v', assert_method_variability('new_distances/w2v_lyrics_distances.npy'))
# print('pca_mel', assert_method_variability('distances/pca_melspectrogram_distances.npy'))
#1print('tag_tf-idf', assert_method_variability('new_distances/tag_based_TF-IDF.npy'))
#####print('bert', assert_method_variability('new_distances/bert_distances.npy'))