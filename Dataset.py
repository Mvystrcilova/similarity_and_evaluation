from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model, model_from_json
import numpy, pandas
import glob
from Song import Song


class Dataset:

    def __init__(self, embeddings, distances):
        self.embeddings = embeddings
        self.distances = distances

    def load_users(self, filename):
        pass

    def load_songs(self, filename):
        song_frame = pandas.read_csv(filename, sep=';', quotechar='"',
                                     names=['userID', 'songId', 'artist', 'title', 'lyrics'],
                                     engine='python', error_bad_lines=False)
        song_frame = song_frame.drop_duplicates(subset=['title', 'artist'])
        songs = []
        for index, row in song_frame.iterrows():
            title = row['title']
            song = Song(title, row['artist'], row['lyrics'], row['songId'])
            songs.append(song)

        return songs
        pass

    def get_user_songs(self, filename):
        user_frame = pandas.read_csv(filename, sep=';', quotechar='"',
                                     names=['userID', 'songId', 'artist', 'title', 'lyrics'],
                                     engine='python', error_bad_lines=False)
        # select only the users
        users = user_frame.drop_duplicates(subset=['userID'])['userID']

        users_with_songs = {}
        # for each user go through all the songs
        # could be probably done bettter, gotta work on that
        for i,user in users.iterrows():
            for j, song in user_frame.iterrows():
                if user['userID'] == song['userID']:
                    # check if user is already in dictionary
                    if user['userID'] in users_with_songs.keys():
                        # add song to the user
                        users_with_songs[user['userID']].add(user_frame['title'], user_frame['artist'])


def get_song(songs, title, artist):
    for s in songs:
        if s.title == title and s.artist == artist:
            return s

    return None

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_neural_spec_representations(model_file, second_dim,spec_directory, representation_name):
    model = load_model(model_file)
    new_representations = numpy.empty([16594, second_dim])
    i = 0
    for file in sorted(glob.glob(spec_directory + '/*.npy'), key=numericalSort):
        new_repr = numpy.load(file).reshape(1, 408, 2206)
        new_repr = model.predict(new_repr)
        new_representations[i] = new_repr.reshape(1, second_dim)
        print(i)
        i = i +1

    numpy.save(representation_name, new_representations)

def save_neural_mel_representations(model_file, weigths_file, second_dim, mel_specs, representation_name):
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weigths_file)
    old_representations = numpy.load(mel_specs).reshape([16594, 408, 320])
    new_representations = numpy.empty([16594,second_dim])
    for i in range(16594):
        new_repr = model.predict(old_representations[i].reshape([1, 408, 320]))
        new_representations[i] = new_repr.reshape(1,second_dim)
        print(i)

    numpy.save(representation_name, new_representations)


# load_neural_spec_representations('/Users/m_vys/PycharmProjects/similarity_and_evaluation/models/models/final_GRU_Spec_model.h5', 128520, 'spectrograms', 'representations/final_GRU_Spec_representations')
save_neural_mel_representations('/mnt/0/GRU_Mel_model.json', '/mnt/0/GRU_Mel_model.h5', 5712, 'mnt/0/song_mel_spectrograms.npy', 'mnt/0/GRU_mel_representations_5712')
# save_neural_mel_representations('/mnt/0/LSTM_Mel_model.json', '/mnt/0/LSTM_Mel_model.h5', 5712, 'mnt/0/song_mel_spectrograms.npy', 'mnt/0/LSTM_mel_representations_5712')

