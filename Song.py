


class Song():

    def __init__(self, title, artist, lyrics, song_id):
        self.title = title
        self.artist = artist,
        # self.YouTube_link = YouTube_link
        # self.file_path = file_path
        self.lyrics = lyrics
        self.song_id = song_id


    def get_users(self, user_data):
        users = []
        for row in user_data.iterrows():
            if row['title'] == self.title and row['artist'] == self.artist:
                users.append(row)
        return users

    




