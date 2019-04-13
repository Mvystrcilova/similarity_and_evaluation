import numpy, pandas

def get_n_most_similar(distance_matrix, index, n):
    songs = pandas.read_csv('useful_songs', names=['title', 'artist'], sep=';', header=None, index_col=False)
    distances = numpy.load(distance_matrix)[index]
    song = songs.ix[index]
    print(song)
    sorted_distances = numpy.sort(distances)
    top_n = sorted_distances[-n]
    distances[distances < top_n] = 0
    n_most_similar = numpy.nonzero(distances)[0]
    print(n_most_similar.shape)
    closest_songs = songs.iloc[n_most_similar.tolist()[:n]]
    closest_songs['distances'] = distances[n_most_similar[:n]]
    closest_songs = closest_songs.sort_values(by=['distances'], ascending=False)
    print(closest_songs)


get_n_most_similar('distances/w2v_distances.npy', 9472, 10)
get_n_most_similar('distances/tf_idf_distances.npy', 9472, 10)
get_n_most_similar('pca_mel_distances_5717.npy', 9472, 10)