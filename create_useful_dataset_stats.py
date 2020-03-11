import matplotlib.pyplot as plt
import pandas


useful_playlists = pandas.read_csv('useful_playlists', sep=';', header=None, index_col=False, names=['userID',
                                                       'songId', 'artist', 'title', 'lyrics'])

def get_mean_playlist_lengt():
    useful_playlists_user_groups = useful_playlists.groupby(by=['userID'])
    user_sizes = useful_playlists_user_groups.size()
    user_means = sum(user_sizes)/len(user_sizes)
    print(user_means)

    useful_playlists_song_groups = useful_playlists.groupby(by=['artist', 'title'])
    song_sizes = useful_playlists_song_groups.size()
    song_means = sum(song_sizes)/len(song_sizes)
    print(song_means)

get_mean_playlist_lengt()