import pandas, numpy, scipy, sklearn, pickle, csv
from scipy import spatial
from sklearn import metrics, preprocessing
# from RepresentationMethod import GRU_Mel_Spectrogram

def save_tf_idf_distances(TF_idf_file):
    vectors = scipy.sparse.load_npz(TF_idf_file)
    tf_idf_distances = metrics.pairwise.cosine_similarity(vectors, dense_output=True)
    numpy.save('tf_idf_distances', tf_idf_distances)


def save_spectrogram_distances(spectrogram_file):
    vectors = numpy.load(spectrogram_file).reshape([16594, 900048])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('spectrogram_distances', distances)

def save_pca_distances(pca_file):
    vectors = numpy.load(pca_file)
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('/mnt/0/som_tf_idf_distances', distances)

def save_mfcc_distances(mffc_file):
    vectors = numpy.load(mffc_file).reshape([16594, 82688])

    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('mfcc_distances', distances)

def save_mel_distances(mel_spec_file):
    vectors = numpy.load(mel_spec_file).reshape([16594, 130560])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('/mnt/0/mel_spectrogram_distances', distances)

def save_bert_distances(bert_encoding_file, save_file):
    vectors = numpy.load(bert_encoding_file).reshape([16594, 768])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save(save_file, distances)

def save_neural_network(representation_file):
    # name_array = representation_file.split('/')[-1].split('_')
    # shape = int(int(name_array[4].split('.')[0])/2)
    vectors = numpy.load(representation_file).reshape([16594, 300])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)

    distance_file = representation_file.replace('representations', 'distances')
    numpy.save(distance_file, distances)

def save_W2V_distances(W2V_file):
    songs = pandas.read_csv(W2V_file, sep=';',
                              names=['somethingWeird',
                                    'songId', 'title',
                                    'artist', 'tf_idf_representation',
                                    'W2V_representation'],
                              usecols=['songId', 'title', 'artist',
                                       'W2V_representation'], header=None)

    w2v_distances = [[0 for x in range(len(songs))] for y in range(len(songs))]

    for i, song_1 in songs.iterrows():
        w2v_repr_1 = numpy.fromstring(song_1['W2V_representation'][1:-1], sep=' ').astype(float)
        print("w2v: ", i)
        for j, song_2 in songs.iterrows():
            w2v_repr_2 = numpy.fromstring(song_2['W2V_representation'][1:-1], sep=' ').astype(float)
            w2v_distances[i][j] = 1 - spatial.distance.cosine(w2v_repr_1, w2v_repr_2)

    numpy.save('w2v_distances', w2v_distances)

def save_SOM_3grid_3it_distances(SOM_file):
    songs = pandas.read_csv(SOM_file, sep=';',
                            names=['somethingWeird',
                                   'songId', 'title',
                                   'artist', 'som_w2v_representation'],
                            header=None)

    som_w2v_distances = [[0 for x in range(len(songs))] for y in range(len(songs))]

    for i, song_1 in songs.iterrows():
        som_w2v_repr_1 = numpy.fromstring(song_1['som_w2v_representation'].replace("(","").replace(')',''), sep=',')
        print(i)
        for j, song_2 in songs.iterrows():
            som_w2v_repr_2 = numpy.fromstring(song_2['som_w2v_representation'].replace("(","").replace(')',''), sep=',')
            som_w2v_distances[i][j] = 1 - spatial.distance.cosine(som_w2v_repr_1, som_w2v_repr_2)

    numpy.save('som_w2v_3g3i', som_w2v_distances)
    print("som_w2v_3g3i saved")


# save_tf_idf_distances('tf_idf_distance_matrix.npz')

def get_som_representation(model_name):
    representation_place = '/Users/m_vys/PycharmProjects/TF_idf_W2V'

    songs_from_file = pandas.read_csv(representation_place, sep=';',
                                    names=['somethingWeird',
                                           'songId', 'title',
                                           'artist', 'tf_idf_representation',
                                           'W2V_representation'],
                                    usecols=['songId', 'title', 'artist',
                                             'W2V_representation'], header=None)
    representations = numpy.empty([16594, 2])
    w2v_representations = []
    scaler = preprocessing.MinMaxScaler()
    with open(model_name, 'rb') as model:
        som = pickle.load(model)

    for i, s in songs_from_file.iterrows():
        w2v_repr_2 = numpy.fromstring(s['W2V_representation'][1:-1], sep=' ').astype(float)
        w2v_representations.append(w2v_repr_2)

    w2v_representations = scaler.fit_transform(w2v_representations)
    for i in range(len(w2v_representations)):
        som_w2v_repr_2 = som.winner(w2v_representations[i])
        print(str(i) + " " + str(som_w2v_repr_2))
        representations[i] = som_w2v_repr_2

    return representations

def save_som_distances_from_array(representations, model_name):
    distances = sklearn.metrics.pairwise.cosine_similarity(representations)
    numpy.save(model_name + '_distances', distances)




# save_mel_distances('/mnt/0/song_mel_spectrograms.npy')
# save_mfcc_distances('mfcc_representations.npy')
# save_pca_distances('short_pca_spec_representations.npy')
# save_neural_network('mnt/0/short_GRU_spec_representations.npy', 5712,'/mnt/0/short_gru_spec_distances')
# save_neural_network('mnt/0/short_LSTM_spec_representations.npy', 5712,'/mnt/0/short_lstm_spec_distances')

# save_neural_network('mnt/0/GRU_mel_representations_5712.npy', 5712,'mnt/0/gru_mel_distances_5712')
# save_neural_network('mnt/0/LSTM_mel_representations_5712.npy', 5712,'mnt/0/final_lstm_mel_distances_5712')
# save_pca_distances('mnt/0/pca_mel_representations_5717.npy')
# save_neural_network('mnt/0/gru_mfcc_representations.npy', 5168, 'mnt/0/gru_mfcc_distances')
# save_pca_distances('/mnt/0/som_tf_idf_representations.npy')
# save_neural_network('new_representations/new_gru_mel_representations_40.npy')
# save_neural_network('mnt/0/new_representations/new_gru_mel_representations_14.npy')
# save_neural_network('mnt/0/new_representations/new_gru_mfcc_representations_16.npy')
# save_neural_network('mnt/0/new_representations/new_gru_mfcc_representations_5.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mel_representations_40.npy')
# save_neural_network('mnt/0/new_representations/new_gru_mel_representations_80.npy')
# save_neural_network('mnt/0/new_representations/new_gru_mfcc_representations_32.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mel_representations_14.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mel_representations_80.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mfcc_representations_32.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mfcc_representations_5.npy')
# save_neural_network('mnt/0/new_representations/new_lstm_mfcc_representations_16.npy')

# save_neural_network('retrained_representations/lstm_mel_representations_bs16_32.npy')
# save_neural_network('retrained_representations/lstm_mel_representations_bs32_32.npy')
# save_neural_network('new_representations/gru_mfcc_representations_30_10.npy')
# save_neural_network('new_representations/gru_mfcc_representations_30_32.npy')
# save_neural_network('new_representations/gru_mfcc_representations_30_64.npy')

# save_neural_network('new_representations/lstm_mfcc_representations_30_32.npy')
# save_neural_network('new_representations/lstm_mfcc_representations_30_64.npy')

# save_neural_network('new_representations/gru_mel_representations_30_28.npy')
# save_neural_network('new_representations/gru_mel_representations_30_.npy')
# save_neural_network('new_representations/gru_mel_representations_30_80.npy')

# save_bert_distances('/Users/m_vys/PycharmProjects/similarity_and_evaluation/bert_model/bert-base-nli-mean-tokens_lyrics_encoding.npy', 'new_distances/bert_distances')
# save_bert_distances('bert_model/roberta-base-nli-stsb-mean-tokens_lyrics_encoding.npy', 'new_distances/roberta_distances')
# save_neural_network('/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_representations/w2v_lyrics_representations.npy')
