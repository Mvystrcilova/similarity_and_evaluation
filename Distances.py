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
    numpy.save('pca_spectrogram_distances', distances)

def save_mfcc_distances(mffc_file):
    vectors = numpy.load(mffc_file).reshape([16594, 82688])

    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('mfcc_distances', distances)

def save_mel_distances(mel_spec_file):
    vectors = numpy.load(mel_spec_file).reshape([16594, 130560])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
    numpy.save('/mnt/0/mel_spectrogram_distances', distances)

def save_neural_networkd(representation_file, shape, distance_file):
    vectors = numpy.load(representation_file).reshape([16594, int(shape)])
    distances = sklearn.metrics.pairwise.cosine_similarity(vectors)
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


# som_repr = get_som_representation('SOM_W2V_batch_5g5i49782')
# with open("som_5g5i_3_representation.txt", "wb") as f:
#     try:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerows(som_repr)
#     except Exception as e:
#         print(e)
# save_som_distances_from_array(som_repr, 'SOM_W2V_batch_5g5i49782')


# save_mel_distances('/mnt/0/song_mel_spectrograms.npy')
# save_mfcc_distances('mfcc_representations.npy')
# save_pca_distances('pca_spec_representations.npy')
save_neural_networkd('representations/final_GRU_Spec_representations.npy', 128520,'distances/final_gru_spec_distances')
save_neural_networkd('representations/final_GRU_mel_representations.npy', 32640,'distances/final_gru_mel_distances')
save_neural_networkd('representations/final_LSTM_mel_representations.npy', 32640,'distances/final_lstm_mel_distances')

