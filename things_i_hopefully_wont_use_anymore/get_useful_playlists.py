import pandas, numpy

def get_useful_playlists():
    data = pandas.read_csv('~/Documents/matfyz/rocnikac/data/songs_with_lyrics', sep=';', quotechar='"',
                                         names=['userID', 'songId', 'artist', 'title', 'lyrics'],
                                         engine='python', error_bad_lines=False)

    playlists_lentght_forandmore = data[['userID', 'title', 'artist']].sort_values(
        by=['userID'])[['title', 'userID']].groupby(['userID']).agg('count')
    playlists_lentght_forandmore['userID'] = data.sort_values(by=['userID']).userID.unique()
    playlists_lentght_forandmore = playlists_lentght_forandmore[playlists_lentght_forandmore.iloc[:,0] >=0]
    users_to_use = playlists_lentght_forandmore.userID.tolist()
    print(len(users_to_use))
    artists_per_playlist = numpy.empty([len(users_to_use)])
    print(len(artists_per_playlist))

    i = 0
    # for user in users_to_use:
    #     playlist= data.loc[data['userID'] == user]
    #     shorter_playlist = playlist.groupby('artist').agg('count')
    #     ratio = shorter_playlist.shape[0]/playlist.shape[0]
    #     artists_per_playlist[i] = ratio
    #     i = i+1
    #     print(i, ratio)
    #
    # mean = numpy.mean(artists_per_playlist)
    # print(mean)
    # print(artists_per_playlist)



    # songs = data.drop_duplicates(subset=['title', 'artist'])
    # data = data[data['userID'].isin(users_to_use)]
    #
    #
    data.to_csv('all_playlists', sep=';', index=False, header=False)
    # songs.to_csv('useful_songs', mode='w', sep=';', columns=['title', 'artist'], index=False, header=False)
    # return ()

get_useful_playlists()