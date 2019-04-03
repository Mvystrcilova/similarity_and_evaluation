import pandas, numpy
from things_i_hopefully_wont_use_anymore.Evaluation_legacy import User, TF_idfStatistics
from scipy import spatial
from Song import Song
from Dataset import get_song
from RepresentationMethod import TF_idf

# evaluating TF-idf, W2V and Som_W2V
data = pandas.read_csv('~/Documents/matfyz/rocnikac/data/songs_with_lyrics', sep=';', quotechar='"',
                                     names=['userID', 'songId', 'artist', 'title', 'lyrics'],
                                     engine='python', error_bad_lines=False)

playlists_lentght_forandmore = data.sort_values(by=['userID'])[['title', 'userID']].groupby('userID').agg(['count'])
playlists_lentght_forandmore['userID'] = data.sort_values(by=['userID']).userID.unique()
playlists_lentght_forandmore = playlists_lentght_forandmore[playlists_lentght_forandmore.iloc[:,0] >=4]
users_to_use = playlists_lentght_forandmore.userID.tolist()

data = data[data['userID'].isin(users_to_use)]
vectors = pandas.read_csv('../TF_idf_W2V', sep=';', names=['somethingWeird', 'songId', 'title', 'artist', 'tf_idf_representation', 'W2V_representation'])
# split data into 80% train and 20% test data
train_data = pandas.DataFrame()
test_data = pandas.DataFrame()

for user in users_to_use:
    user_data = data.loc[data['userID'] == user]
    msk = numpy.random.rand(len(user_data)) < 0.8
    user_train_data = user_data[msk]
    user_test_data = user_data[~msk]
    train_data = train_data.append(user_train_data)
    test_data = test_data.append(user_test_data)


print(train_data.shape)
print(test_data.shape)


# useful_users = users[users['userID'].isin(users_to_use)]

songs = data.drop_duplicates(subset=['title', 'artist'])
songs = pandas.merge(songs, vectors, on=['title', 'artist'])
tf_idf_distances = [[0 for x in range(len(songs))] for y in range(len(songs))]
w2v_distances = [[0 for x in range(len(songs))] for y in range(len(songs))]
song_instances = []
for i, song_1 in songs.iterrows():
    s = Song(song_1['title'], song_1['artist'], song_1['lyrics'], i)
    s.tf_idf_representation = numpy.fromstring(song_1['tf_idf_representation'].rstrip('/n')[1:-1], sep=' ').astype(float)
    s.W2V_representation = numpy.fromstring(song_1['W2V_representation'][1:-1], sep=' ').astype(float)
    s.index = i
    song_instances.append(s)

for song_1 in song_instances:
    for song_2 in song_instances:
        tf_idf_distances[song_1.index][song_2.index] = 1 - spatial.distance.cosine(song_1.tf_idf_representation, song_2.tf_idf_representation)
        w2v_distances[song_1.index][song_2.index] = 1 - spatial.distance.cosine(song_1.W2V_representation, song_2.W2V_representation)

print(len(users_to_use))

user_instances = []

for user in users_to_use:
    user_instance = User(user)
    user_instance.user_train_frame = train_data.loc[train_data['userID'] == user]
    train_list = []
    test_list = []
    for i,row in user_instance.user_train_frame.iterrows():
        s = get_song(song_instances, row['title'], row['artist'])
        train_list.append(s)
    user_instance.train_songs = train_list
    user_instance.user_test_frame = test_data.loc[test_data['userID'] == user]
    for i, row in user_instance.user_test_frame.iterrows():
        s = get_song(song_instances, row['title'], row['artist'])
        test_list.append(s)
    user_instance.test_songs = test_list

    if user_instance.user_test_frame.shape[0] > 0:
        user_instances.append(user_instance)

tf_idf = TF_idf()
tf_idf_stats = TF_idfStatistics()

# w2v = Word2Vec()
# w2v_stats = TF_idfStatistics()

for usr in user_instance:
    similarities = pandas.DataFrame(columns=['song', 'tf_idf_similarity', 'w2v_similarity'])
    for song in song_instances:
        song_tf_idf_similarity = 0
        song_w2v_similarity = 0
        if song not in usr.train_songs:
            for s in usr.train_songs:
                song_tf_idf_similarity += tf_idf_distances[song.index][s.index]
                song_w2v_similarity += w2v_distances[song.index][s.index]

            song_similarities = pandas.DataFrame([[song, song_tf_idf_similarity, song_w2v_similarity]])
            similarities.append(song_similarities)

    usr.songs_with_distances = similarities
    usr.top_100_tfidf = similarities.sort_values(by=['tf_idf_similarity']).head(100)
    usr.top_100_w2v = similarities.sort_values(by=['w2v_similarity']).head(100)


    print(usr.top_100_tfidf.shape)
    print(usr.top_100_w2v.shape)










