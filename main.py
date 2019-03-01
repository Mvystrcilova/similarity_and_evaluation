from gensim.models.keyedvectors import KeyedVectors
from RepresentationMethod import TF_idf, Word2Vec, SOM_W2V, SOM_TF_idf
from Dataset import Dataset
import pandas
from Song import Song

def main():
    users_filename = '~/Documents/matfyz/rocnikac/data/songs_with_lyrics'
    # songs_test_filename = '~/Documents/matfyz/rocnikac/'
    w2v_model = KeyedVectors.load('/Users/m_vys/Documents/matfyz/rocnikac/djangoApp/rocnikac/w2v_subset', mmap='r')
    tf_idf = TF_idf()
    word2vec = Word2Vec(w2v_model, [])
    som_tfidf = SOM_TF_idf(1,0.5)
    som_w2v = SOM_W2V(1,0.5)
    dataset = Dataset([tf_idf, word2vec, som_tfidf, som_w2v],['cosine'])
    song_repr_frame = pandas.DataFrame()
    songs = dataset.load_songs(users_filename)
    print(len(songs))

    tf_idf.train(songs)
    for s in songs:
        word2vec.represent_song(s)

    som_w2v.train(songs)

    for s in songs:
        temp_song_frame = pandas.DataFrame(data=[[s.title, s.artist, s.tf_idf_representation,
                                s.W2V_representation, s.som_w2v_representation]])
        song_repr_frame = song_repr_frame.append(temp_song_frame)

    song_repr_frame.to_csv('TF_idf_W2V_SOM_W2V', sep=';')

main()