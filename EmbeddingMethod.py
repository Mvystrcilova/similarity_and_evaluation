import abc
from Song import Song
import librosa
import numpy, pandas
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from sklearn import decomposition, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RepresentationMethod(abc.ABC):
    @abc.abstractmethod
    def represent_song(self, song):
        pass

    @abc.abstractmethod
    def train(self, songs):
        pass


class TextMethod(RepresentationMethod):
    pass


class TF_idf(TextMethod):

    def represent_song(self, song):
        # the songs need to be represented all together in order to get all the words so there is no
        # reason to put a bag of words representation here
        pass

    def train(self, songs):
        lyrics = []
        for s in songs:
            lyrics.append(s.lyrics)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_train_matrix = (tfidf_vectorizer.fit_transform(songs))

        for i,s in enumerate(songs):
            s.tf_idf_representation = tfidf_train_matrix[i]


class Word2Vec(TextMethod):

    def __init__(self, w2v_model, stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def represent_song(self, song):
        lyrics = song.lyrics.lower()
        words = [w.strip('.,!:?-') for w in lyrics.split(" ") if w.strip(".,!?:-") not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)

            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = numpy.mean(word_vecs, axis=0)
        song.W2V_representation = vector

    def train(self, songs):
        # the model for W2V is already pretrained by Google so no need to implement this
        pass


class AudioMethod(RepresentationMethod):

    def get_spectrogram(self, song):
        filename = song.file_path
        y, sr = librosa.load(filename, offset=15, duration=60)

        return y, sr


class MFCCRepresentation(AudioMethod):

    def represent_song(self, song):
        MFCC = librosa.feature.mfcc(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=128, fmax=8000).flatten()
        return MFCC

    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            mfcc = self.represent_song(s)
            temp_df = pandas.DataFrame(data=mfcc)
            song_frame.append(temp_df)

        for i, s in enumerate(songs):
            s.mfcc_representation = song_frame[i]


class PCA_Mel_spectrogram(AudioMethod):

    def represent_song(self, song):
        mel_spectrogram = librosa.feature.melspectrogram(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=128, fmax=8000).flatten()
        return mel_spectrogram.flatten()

    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            spec = self.represent_song(s)
            temp_df = pandas.DataFrame(data=spec)
            song_frame.append(temp_df)

        pca = decomposition.PCA()
        song_frame_afterPCA = pca.fit_transform(song_frame)
        for i, s in enumerate(songs):
            s.mel_pca_representation = song_frame_afterPCA[i]


class PCA_Spectrogram(AudioMethod):

    def extract_audio(self, song):
        spec = numpy.abs(librosa.stft(self.get_spectrogram(song).y)).flatten()
        return spec


    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            spec = self.represent_song(s)
            temp_df = pandas.DataFrame(data=spec)
            song_frame.append(temp_df)

        pca = decomposition.PCA()
        song_frame_afterPCA = pca.fit_transform(song_frame)
        for i,s in enumerate(songs):
            s.pca_representation = song_frame_afterPCA[i]


class ReccurentRepresentation(AudioMethod):
    pass







