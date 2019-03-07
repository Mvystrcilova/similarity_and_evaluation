import pandas, numpy
from scipy import spatial


def save_tf_idf_distances(TF_idf_file):
    songs = pandas.read_csv(TF_idf_file, sep=';',
                              names=['somethingWeird',
                                    'songId', 'title',
                                    'artist', 'tf_idf_representation',
                                    'W2V_representation'],
                              usecols=['songId', 'title', 'artist',
                                       'tf_idf_representation'], header=None)

    tf_idf_distances = [[0 for x in range(len(songs))] for y in range(len(songs))]

    for i, song_1 in songs.iterrows():
        tf_idf_repr_1 = numpy.fromstring(song_1['tf_idf_representation'].rstrip('/n')[1:-1], sep=' ').astype(float)
        print("tf_idf: ", i)
        for j, song_2 in songs.iterrows():
            tf_idf_repr_2 = numpy.fromstring(song_2['tf_idf_representation'].rstrip('/n')[1:-1], sep=' ').astype(float)
            tf_idf_distances[i][j] = 1 - spatial.distance.cosine(tf_idf_repr_1, tf_idf_repr_2)

    numpy.save('tf_idf_distances', tf_idf_distances)



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

# save_W2V_distances('../TF_idf_W2V')
# print("W2v_saved")
# save_tf_idf_distances('../TF_idf_W2V')
# print("done")