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
        users = user_frame.drop_duplicates(subset=['userID'])['userID']
