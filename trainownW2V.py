from time import asctime

import gensim, gzip, logging, pandas, re, numpy as np

bad_chars = ',.?!\'\"-'
rgx = re.compile('[%s]' % bad_chars)

def make_list_of_string(song):
    song = song.replace('\n', ' ')
    song = rgx.sub('', song)
    song = song.split(' ')
    return song

def read_input(input_file):
    list_lyrics = []
    lyrics = pandas.read_csv(input_file, header=None,
                             index_col=False, sep=';', quotechar='\"')
    lyrics = lyrics.iloc[:, 2]
    lyrics = list(lyrics)
    for l in lyrics:
        l = make_list_of_string(l)
        list_lyrics.append(l)

    return list_lyrics

def train_w2v(input_file):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lyrics = read_input(input_file)
    model = gensim.models.Word2Vec(lyrics, size=100, window=5, min_count=1)
    model.save('w2v_lyrics_model_100')


# train_w2v('/Users/m_vys/PycharmProjects/similarity_and_evaluation/not_empty_songs')

def represent_with_lyrics_w2v(model_file, songs_file):
    w2v_representations = []
    w2v_model = gensim.models.Word2Vec.load(model_file)
    lyrics = read_input(songs_file)
    for l in lyrics:
        vector_repr = []
        for word in l:
            repr = w2v_model.wv[word]
            vector_repr.append(repr)
        numpy_list = np.array(vector_repr)
        vector_repr = np.mean(numpy_list, axis=0)
        w2v_representations.append(vector_repr)

    representations = np.array(w2v_representations)
    np.save('new_representations/w2v_lyrics_representations_100', representations)

represent_with_lyrics_w2v('w2v_lyrics_model_100', 'not_empty_songs')