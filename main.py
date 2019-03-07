from gensim.models.keyedvectors import KeyedVectors
from RepresentationMethod import TF_idf, Word2Vec, SOM_W2V, SOM_TF_idf
from Dataset import Dataset
import pandas, sklearn
from sklearn import metrics
import numpy
from Song import Song

def main():
    users_filename = '~/Documents/matfyz/rocnikac/data/songs_with_lyrics'
    # songs_test_filename = '~/Documents/matfyz/rocnikac/'
    # w2v_model = KeyedVectors.load('/Users/m_vys/Documents/matfyz/rocnikac/djangoApp/rocnikac/w2v_subset', mmap='r')
    tf_idf = TF_idf()
    # word2vec = Word2Vec(w2v_model, [])
    # som_tfidf = SOM_TF_idf(1,0.5)
    # som_w2v_2 = SOM_W2V(1, 0.5, grid_size_multiple=2, iterations=10, model_name='som_w2v_4g.p')
    # som_w2v_4 = SOM_W2V(1, 0.5, 4, 10, 'som_w2v_2g.p')
    # som_w2v_5 = SOM_W2V(1, 0.5, 5, 10, 'som_w2v_5g.p')
    dataset = Dataset([tf_idf],['cosine'])
    song_repr_frame = pandas.DataFrame()
    songs = dataset.load_songs(users_filename)
    print(len(songs))

    tf_idf.train(songs)
    # for s in songs:
    #     word2vec.represent_song(s)
    # vectors = sklearn.metrics.pairwise.cosine_similarity(vectors, dense_output=True)
    # for s in songs:
    #     temp_song_frame = pandas.DataFrame(data=[[s.song_id, s.title, s.artist, s.tf_idf_representation,
    #                             s.W2V_representation]])
    #     song_repr_frame = song_repr_frame.append(temp_song_frame)
    #
    # numpy.set_printoptions(threshold=numpy.nan)
    # song_repr_frame.to_csv('TF_idf_W2V', sep=';')

    # som_w2v_2.train(songs)

    # song_repr_frame = pandas.DataFrame()

    # for s in songs:
    #     temp_song_frame = pandas.DataFrame(data=[[s.song_id, s.title, s.artist, s.som_w2v_representation]])
    #     song_repr_frame = song_repr_frame.append(temp_song_frame)
    #
    # song_repr_frame.to_csv('SOM_W2V', sep=';')



main()