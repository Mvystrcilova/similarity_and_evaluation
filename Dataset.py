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

def predict_representations(model_file, repr_string, is_in_gpulab):
    mounted_dir = ''
    if is_in_gpulab:
        mounted_dir = 'mnt/0/'

    model = load_model(model_file)

    name_array = model_file.split('/')[-1].split('_')
    second_dim = int(int(name_array[4].split('.')[0])/2)
    new_repr_name = mounted_dir + "new_representations/new_" + name_array[1] + '_' + name_array[2] + '_representations_' + str(second_dim) + '.npy'
    new_representations = numpy.empty([16594, second_dim])
    if repr_string == "mel":
        mel_repr_file = mounted_dir + 'representations/song_mel_spectrograms.npy'
        old_representations = numpy.load(mel_repr_file).reshape([16594, 408, 320])
        for i in range(16594):
            new_repr = model.predict(old_representations[i].reshape([1,408,320]))
            new_representations[i] = new_repr.reshape(1,second_dim)
    else:
        mfcc_repr_file = mounted_dir + 'representations/mfcc_representations.npy'
        old_representations = numpy.load(mfcc_repr_file).reshape(16594, 646, 128)
        for i in range(16594):
            new_repr = model.predict(old_representations[i].reshape([1, 646, 128]))
            new_representations[i] = new_repr.reshape(1, second_dim)

    numpy.save(new_repr_name, new_representations)



# load_neural_spec_representations('/mnt/0/models/final_GRU_Spec_model.h5', 20400, '/mnt/0/spectrograms/spectrograms', 'mnt/0/final_GRU_Spec_representations')
# save_neural_spec_representations('/mnt/0/models/final_LSTM_Spec_Model.h5', 5712, 'mnt/0/song_mel_spectrograms.npy', 'mnt/0/GRU_mel_representations_5712')
# save_neural_mel_representations('/mnt/0/LSTM_Mel_model.json', '/mnt/0/LSTM_Mel_model.h5', 5712, 'mnt/0/song_mel_spectrograms.npy', 'mnt/0/LSTM_mel_representations_5712')
# save_neural_mel_representations('new_models/gru_mel_autoencoder29.09090909090909.h5', )
# predict_representations('new_models/new_gru_mel_model_80.0.h5', "mel")
predict_representations('new_models/new_gru_mel_model_28.0.h5', "mel", True)
predict_representations('new_models/new_gru_mel_model_160.0.h5', "mel", True)
predict_representations('new_models/new_lstm_mel_model_28.0.h5', "mel", True)
predict_representations('new_models/new_lstm_mel_model_80.0.h5', "mel", True)
predict_representations('new_models/new_lstm_mel_model_160.0.h5', "mel", True)

