from random import seed
import pandas, numpy, pickle
seeds = [23, 57, 11, 8, 76]

def get_masks(useful_playlists_file, song_file):
    useful_playlists = pandas.read_csv(useful_playlists_file, sep=';', names=['userID', 'songId', 'artist', 'title',
                                                                              'lyrics' ],
                                       usecols=[0, 1, 2, 3])

    users = useful_playlists['userID'].drop_duplicates().tolist()
    songs = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])

    x = [x for x in range(songs.shape[0])]
    songs['ind'] = x
    useful_playlists = pandas.merge(useful_playlists, songs, how='left', on=['artist', 'title'])
    all_masks = []
    for j in range(0, 5):
        seed(seeds[j])
        masks = []
        for user in users:
            user_data = useful_playlists.loc[useful_playlists['userID'] == user]
            msk = numpy.random.rand(len(user_data)) < 0.8
            user_train_data = user_data[msk]
            user_test_data = user_data[~msk]

            while (len(user_test_data) == 0):
                msk = numpy.random.rand(len(user_data)) < 0.8
                user_train_data = user_data[msk]
                user_test_data = user_data[~msk]

            masks.append(msk)
        all_masks.append(masks)
    pickle_file = open('masks_file', 'wb')
    pickle.dump(all_masks, pickle_file)

get_masks('useful_playlists', 'useful_songs')


