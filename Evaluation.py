import numpy, pandas, operator, itertools, math
import sklearn.preprocessing
# from Distances import save_SOM_3grid_3it_distances
# from audio_representations import file_to_spectrogram
# from audio_representations import convert_files_to_mfcc
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
        indices = user_train_data['ind']
        # finds the distance from the users train set to every song
        # for i in range(len(self.songs)):
        #     song_distance = self.distance_matrix[i][indices].sum()
        #     song_distances = pandas.DataFrame([self.songs.loc[i,:]])
        #     user_distances = user_distances.append(song_distances)
        #     # print(song_distance, type(song_distance))
        #     user_distances.loc[i,'distance'] = song_distance
        sums = numpy.sum(self.distance_matrix[:,indices], axis=1)
        # print(sums)
        user_distances = self.songs
        user_distances['distance'] = sums
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

# for j in range(5):
#     # save_SOM_3grid_3it_distances('SOM_W2V')
#     som_evaluation = Evaluation('som_w2v_3g3i.npy', 'useful_playlists','useful_songs')
#     results = pandas.DataFrame(columns=['playlist_lenght','test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
#     i = 0
#     for user in som_evaluation.users:
#         user_results = som_evaluation.eval_playlist(user)
#         print('user ', i, " out of ", len(som_evaluation.users))
#         print(user_results)
#         temp_frame = pandas.DataFrame([user_results], columns=['playlist_lenght','test_list_lenght', 'number_of_matches','match_ranking','recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
#         results = results.append(temp_frame)
#         i = i+1
#
#     print(results.shape)
#     filename = 'results/som_w2v_results/som_w2v_results_' + str(j+1)
#     results.to_csv(filename, sep=';', header=False, index=False)
# for j in range(3,6):
#     evaluation = Evaluation('w2v_distances.npy', 'useful_playlists','useful_songs')
#     results = pandas.DataFrame(columns=['playlist_lenght','test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
#     i = 0
#     for user in evaluation.users:
#             user_results = evaluation.eval_playlist(user)
#             print('user ', i, " out of ", len(evaluation.users))
#             print(user_results)
#             temp_frame = pandas.DataFrame([user_results],
#                                           columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches',
#                                                    'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100',
#                                                    'nDGC'])
#             results = results.append(temp_frame)
#             i = i + 1
#     filename = 'results/w2v_results/w2v_results_' + str(j)
#     print(results.shape)
#     results.to_csv(filename, sep=';', header=False, index=False)

# for j in range(5):
#     evaluation = Evaluation('mnt/0/lstm_mel_distances_5712.npy', 'mnt/0/useful_playlists', 'mnt/0/useful_songs')
#     results = pandas.DataFrame(columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10',
#                  'recall_at_50', 'recall_at_100', 'nDGC'])
#     i = 0
#     for user in evaluation.users:
#         user_results = evaluation.eval_playlist(user)
#         print('user ', i, " out of ", len(evaluation.users))
#         print(user_results)
#         temp_frame = pandas.DataFrame([user_results],
#                                       columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches',
#                                                'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100',
#                                                'nDGC'])
#         results = results.append(temp_frame)
#         i = i + 1
#     filename = 'mnt/0/results/lstm_mel_results_5712/lstm_mel_' + str(j+1)
#     print(results.shape)
#     results.to_csv(filename, sep=';', header=False, index=False)
#
# for j in range(5):
#     evaluation = Evaluation('mnt/0/gru_mel_distances_5712.npy', 'mnt/0/useful_playlists', 'mnt/0/useful_songs')
#     results = pandas.DataFrame(
#         columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10',
#                  'recall_at_50', 'recall_at_100', 'nDGC'])
#     i = 0
#     for user in evaluation.users:
#         user_results = evaluation.eval_playlist(user)
#         print('user ', i, " out of ", len(evaluation.users))
#         print(user_results)
#         temp_frame = pandas.DataFrame([user_results],
#                                       columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches',
#                                                'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100',
#                                                'nDGC'])
#         results = results.append(temp_frame)
#         i = i + 1
#     filename = 'mnt/0/results/gru_mel_results_5712/gru_mel_' + str(j + 1)
#     print(results.shape)
#     results.to_csv(filename, sep=';', header=False, index=False)

for j in range(5):
    evaluation = Evaluation('mnt/0/pca_spec_distances_1106.npy', 'mnt/0/useful_playlists', 'mnt/0/useful_songs')
    results = pandas.DataFrame(columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10',
                 'recall_at_50', 'recall_at_100', 'nDGC'])
    i = 0
    for user in evaluation.users:
        user_results = evaluation.eval_playlist(user)
        print('user ', i, " out of ", len(evaluation.users))
        print(user_results)
        temp_frame = pandas.DataFrame([user_results],
                                      columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches',
                                               'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100',
                                               'nDGC'])
        results = results.append(temp_frame)
        i = i + 1
    filename = 'mnt/0/results/pca_spec_results_1106/pca_spec_' + str(j+1)
    print(results.shape)
    results.to_csv(filename, sep=';', header=False, index=False)

# convert_files_to_mfcc('not_empty_songs', 320)