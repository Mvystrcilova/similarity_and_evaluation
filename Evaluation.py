import numpy, pandas, operator, itertools, math

from Distances import save_SOM_3grid_3it_distances

class Evaluation():
    def __init__(self, distance_matrix_file, useful_playlists_file, song_file):
        self.distance_matrix = numpy.load(distance_matrix_file)
        self.useful_playlists = pandas.read_csv(useful_playlists_file, sep=';',
                                                names=['userID',
                                                       'songId', 'artist', 'title', 'lyrics' ],
                                                usecols=[0, 1, 2, 3])

        # a list of users with 4 songs and more in their playlist
        self.users = self.useful_playlists['userID'].drop_duplicates().tolist()
        # a pandas DataFrame with the songs we have
        self.songs = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])
        # assinges and index columns to the pandas dataframe
        self.assing_index()

    def assing_index(self):
        '''this function merges useful playlists with songs so that each song has its index and can
        be found in the distance matrix'''
        x = [x for x in range(self.songs.shape[0])]
        self.songs['ind'] = x
        self.useful_playlists = pandas.merge(self.useful_playlists,self.songs, how='left', on=['artist', 'title'])

    def eval_playlist(self, user):
        # splits user data in two datasets
        user_train_data, user_test_data = self.split_data(user)
        user_distances = pandas.DataFrame()

        # finds the distance from the users train set to every song
        for i, song in self.songs.iterrows():
            song_distance = 0
            for j, user_song in user_train_data.iterrows():
                if (song['ind'] != user_song['ind']):
                    song_distance += self.distance_matrix[song['ind']][user_song['ind']]

            song_distances = pandas.DataFrame([song])
            user_distances = user_distances.append(song_distances)
            # print(song_distance, type(song_distance))
            user_distances.loc[i,'distance'] = song_distance
            # print(user_distances.dtypes)
        # selects 100 songs closest to the user
        # print(user_distances.dtypes)
        user_distances = user_distances.sort_values(ascending=False, by=['distance'])
        x = [x for x in range(len(user_distances))]
        user_distances['ranking'] = x
        # user_distances = user_distances.head(100).reset_index()


        # matches the close songs with actual user test data
        matches = user_distances.merge(user_test_data, on=['artist', 'title'])
        rankings = self.get_rankings(matches)
        top_100_matches = user_distances.head(100).merge(user_test_data, on=['artist', 'title'])
        values = [(len(user_test_data) + len(user_train_data)),len(user_test_data),
                   len(matches), rankings, self.recall_at_n(10, top_100_matches, user_test_data),
                   self.recall_at_n(50, top_100_matches, user_test_data),
                  self.recall_at_n(100, top_100_matches, user_test_data),
                   self.nDCG(top_100_matches, user_test_data)]



        return values


    def get_rankings(self, matches):
        indices = matches['ranking'].tolist()
        return indices

    def recall_at_n(self, n, matches, user_test_data):
        indices = matches['ranking'].tolist()
        # print("the ranking of mathcing songs" , indices)
        return len([x for x in indices if x < n])/len(user_test_data)

    def nDCG(self, matches, user_test_data):
        DGC_score = int(0)
        for i, match in matches.iterrows():
            DGC_score += 1/math.log(match['ranking']+2, 2)
        ideal_DGC = int(0)
        for i in range(len(user_test_data)):
            ideal_DGC += 1/math.log(i+2, 2)

        return DGC_score/ideal_DGC





    def split_data(self, user):
        # returns a users playlists split into two dataframes, 80% is test and 20% is train
        user_data = self.useful_playlists.loc[self.useful_playlists['userID'] == user]
        msk = numpy.random.rand(len(user_data)) < 0.8
        user_train_data = user_data[msk]
        user_test_data = user_data[~msk]

        while (len(user_test_data) == 0):
            msk = numpy.random.rand(len(user_data)) < 0.8
            user_train_data = user_data[msk]
            user_test_data = user_data[~msk]

        return user_train_data, user_test_data

# save_SOM_3grid_3it_distances('SOM_W2V')
evaluation = Evaluation('w2v_distances.npy', 'useful_playlists_bigger_than_10','useful_songs')
results = pandas.DataFrame(columns=['playlist_lenght','test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
i = 0
for user in evaluation.users:
    user_results = evaluation.eval_playlist(user)
    print('user ', i, " out of ", len(evaluation.users))
    print(user_results)
    temp_frame = pandas.DataFrame([user_results], columns=['playlist_lenght','test_list_lenght', 'number_of_matches','match_ranking','recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
    results = results.append(temp_frame)
    i = i+1