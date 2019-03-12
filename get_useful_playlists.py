import pandas

data = pandas.read_csv('~/Documents/matfyz/rocnikac/data/songs_with_lyrics', sep=';', quotechar='"',
                                     names=['userID', 'songId', 'artist', 'title', 'lyrics'],
                                     engine='python', error_bad_lines=False)

playlists_lentght_forandmore = data.sort_values(by=['userID'])[['title', 'userID']].groupby('userID').agg(['count'])
playlists_lentght_forandmore['userID'] = data.sort_values(by=['userID']).userID.unique()
playlists_lentght_forandmore = playlists_lentght_forandmore[playlists_lentght_forandmore.iloc[:,0] >=10]
users_to_use = playlists_lentght_forandmore.userID.tolist()

songs = data.drop_duplicates(subset=['title', 'artist'])
data = data[data['userID'].isin(users_to_use)]


# data.to_csv('useful_playlists_bigger_than_10', sep=';', index=False, header=False)
songs.to_csv('useful_songs', mode='w', sep=';', columns=['title', 'artist'], index=False, header=False)
