from sklearn.metrics.pairwise import cosine_similarity
import pandas
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