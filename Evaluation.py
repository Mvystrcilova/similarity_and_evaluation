import numpy, pandas, operator, itertools, math

import os
import sklearn.preprocessing
# from Distances import save_SOM_3grid_3it_distances
# from audio_representations import file_to_spectrogram
# from audio_representations import convert_files_to_mfcc
import pickle
class Evaluation():
    def __init__(self, distance_matrix_file, useful_playlists_file, song_file, are_all_playlists, threshold):
        self.distance_matrix = self.apply_threshold(threshold, distance_matrix_file)
        self.useful_playlists = self.get_useful_playlists(are_all_playlists, useful_playlists_file)
        self.mask = pickle.load(open('masks_file', 'rb'))

        # self.all_playlists = pandas.read_csv('all_playlists', sep=';',
        #                                         names=['userID',
        #                                                'songId', 'artist', 'title', 'lyrics' ],
        #                                         usecols=[0, 1, 2, 3])
        # a list of users with 4 songs and more in their playlist
        self.users = self.useful_playlists['userID'].drop_duplicates().tolist()
        # self.all_users = self.all_playlists['userID'].drop_duplicates().tolist()
        # a pandas DataFrame with the songs we have
        self.songs = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])
        self.threshold = threshold
        # assinges and index columns to the pandas dataframe
        self.assing_index()


    def apply_threshold(self, threshold, distance_matrix_file):
        distance_matrix = numpy.load(distance_matrix_file)
        distance_matrix[distance_matrix < threshold] = 0
        return distance_matrix


    def assing_index(self):
        '''this function merges useful playlists with songs so that each song has its index and can
        be found in the distance matrix'''
        x = [x for x in range(self.songs.shape[0])]
        self.songs['ind'] = x
        self.useful_playlists = pandas.merge(self.useful_playlists,self.songs, how='left', on=['artist', 'title'])


    def get_useful_playlists(self, are_all_playlists, useful_playlists_file):
        if not are_all_playlists:
            return pandas.read_csv(useful_playlists_file, sep=';',
                                                names=['userID',
                                                       'songId', 'artist', 'title', 'lyrics' ],
                                                usecols=[0, 1, 2, 3])
        else:
            return pandas.read_csv('/mnt/0/all_playlists', sep=';',
                                                names=['userID',
                                                       'songId', 'artist', 'title', 'lyrics'],
                                                usecols=[0, 1, 2, 3])

    def eval_playlist(self, user, use_max, j, user_index):
        # splits user data in two datasets
        user_train_data, user_test_data = self.split_data(user, j, user_index)
        user_distances = pandas.DataFrame()
        indices = user_train_data['ind']
        test_indices = user_test_data['ind']

        if not use_max:
            sums = numpy.sum(self.distance_matrix[:, indices], axis=1)
        else:
            try:
                sums = numpy.max(self.distance_matrix[:, indices], axis=1)
            except ValueError:
                sums = 0
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
        matches = user_distances.merge(user_test_data, on=['ind'])
        rankings = self.get_rankings(matches)
        top_100_matches = user_distances.head(100).merge(user_test_data, on=['ind'])
        recall_10, indices_10 = self.recall_at_n(10, user_distances, test_indices)
        recall_50, indices_50 = self.recall_at_n(50, user_distances, test_indices)
        recall_100, indices_100 = self.recall_at_n(100, user_distances, test_indices)
        values = [(len(user_test_data) + len(user_train_data)), len(user_test_data),
                   len(matches), rankings, recall_10, recall_50, recall_100,
                   self.nDCG(top_100_matches, user_test_data)]

        return values, [user_test_data['ind'].tolist(), indices_10, indices_50]


    def get_rankings(self, matches):
        indices = matches['ranking'].tolist()
        return indices

    def recall_at_n(self, n, user_distances, user_test_data):
        head = user_distances.head(n)['ind'].tolist()
        # print("the ranking of mathcing songs" , indices)
        matches = [x for x in user_test_data if x in head]
        return len(matches)/min(10,len(user_test_data)), matches

    def nDCG(self, matches, user_test_data):
        DGC_score = int(0)
        for i, match in matches.iterrows():
            DGC_score += 1/math.log(match['ranking'] + 2, 2)
        ideal_DGC = int(0)
        for i in range(len(user_test_data)):
            ideal_DGC += 1/math.log(i+2, 2)

        return DGC_score/ideal_DGC


    def split_data(self, user, j, user_index):
        # returns a users playlists split into two dataframes, 80% is test and 20% is train
        user_data = self.useful_playlists.loc[self.useful_playlists['userID'] == user]
        msk = self.mask[j][user_index]
        user_train_data = user_data[msk]
        user_test_data = user_data[~msk]

        # while (len(user_test_data) == 0):
        #     msk = numpy.random.rand(len(user_data)) < 0.8
        #     user_train_data = user_data[msk]
        #     user_test_data = user_data[~msk]

        return user_train_data, user_test_data


def evaluate(distance_file, threshold, use_max):
    # mounted_dir = '/mnt/0/'
    max_string = ''
    if use_max:
        max_string = '_max'
    os.makedirs(distance_file.replace('distances', 'results')[:-4] + max_string + '_updated',
                exist_ok=True)
    os.makedirs(distance_file.replace('distances', 'results')[:-4] + max_string + '_indices',
                exist_ok=True)

    for j in range(5):
        filename = distance_file.replace('distances', 'results')[:-4] + max_string + '_updated' + "/" + distance_file.split('/')[-1][:-4] + max_string + '_' + str(j+1)
        ind_file_name = distance_file.replace('distances', 'results')[:-4] + max_string + '_indices' + "/" + distance_file.split('/')[-1][:-4] + max_string + '_' + str(j+1)
        evaluation = Evaluation(distance_file, 'useful_playlists', 'useful_songs', False, threshold)
        results = pandas.DataFrame(columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10',
                     'recall_at_50', 'recall_at_100', 'nDGC'])
        indices = pandas.DataFrame(columns=['test_indices', 'matched_inds_10', 'matched_inds_50'])
        i = 0
        for user in evaluation.users:
            user_results, inds = evaluation.eval_playlist(user, use_max, j, i)
            if (i % 1000) == 0:
                print('user ', i, " out of ", len(evaluation.users))
            # print(user_results)
            # print(indices)
            temp_frame = pandas.DataFrame([user_results],
                                          columns=['playlist_lenght', 'test_list_lenght', 'number_of_matches',
                                                   'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100',
                                                   'nDGC'])
            temp_frame_ind = pandas.DataFrame([inds], columns=['test_indices', 'matched_inds_10', 'matched_inds_50'])
            results = results.append(temp_frame)
            indices = indices.append(temp_frame_ind)
            i = i + 1
        # print(results.shape)
        results.to_csv(filename, sep=';', header=False, index=False)
        indices.to_csv(ind_file_name, sep=';')


def compare_recomendations(method_1, method_2, dm_1_file, dm_2_file, useful_playlists_file, song_file):
    useful_playlists = pandas.read_csv(useful_playlists_file, sep=';', names=['userID', 'songId', 'artist', 'title',
                                                                              'lyrics' ],
                                       usecols=[0, 1, 2, 3])

    users = useful_playlists['userID'].drop_duplicates().tolist()
    dm_1 = numpy.load(dm_1_file)
    dm_2 = numpy.load(dm_2_file)
    songs = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])

    x = [x for x in range(songs.shape[0])]
    songs['ind'] = x
    useful_playlists = pandas.merge(useful_playlists, songs, how='left', on=['artist', 'title'])
    nums_of_same_songs_10 = []
    num_of_same_songs_50 = []
    masks = pickle.load(open('masks_file', 'rb'))
    for j in range(0, 5):
        i = 0
        stats_df = None
        for user in users:

            user_data = useful_playlists.loc[useful_playlists['userID'] == user]
            msk = masks[j][i]
            user_train_data = user_data[msk]
            user_test_data = user_data[~msk]

            # while (len(user_test_data) == 0):
            #     msk = numpy.random.rand(len(user_data)) < 0.8
            #     user_train_data = user_data[msk]
            #     user_test_data = user_data[~msk]

            indices = user_train_data['ind']
            train_indices = user_test_data['ind'].tolist()

            try:
                maxes_1 = numpy.max(dm_1[:, indices], axis=1)
            except ValueError:
                maxes_1 = 0

            try:
                maxes_2 = numpy.max(dm_2[:, indices], axis=1)
            except ValueError:
                maxes_2 = 0

            user_distances1 = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])
            x = [x for x in range(songs.shape[0])]
            user_distances1['ind'] = x
            user_distances1['distance'] = maxes_1

            user_distances2 = pandas.read_csv(song_file, sep=';', names=['title', 'artist'])
            user_distances2['ind'] = x
            user_distances2['distance'] = maxes_2

            user_distances1 = user_distances1.sort_values(ascending=False, by=['distance'])
            user_distances2 = user_distances2.sort_values(ascending=False, by=['distance'])

            user_distances1_50 = user_distances1.head(50)
            user_distances2_50 = user_distances2.head(50)

            user_distances1_10 = user_distances1.head(10)
            user_distances2_10 = user_distances2.head(10)

            songs1_10 = user_distances1_10['ind']
            songs2_10 = user_distances2_10['ind']
            songs1_50 = user_distances1_50['ind']
            songs2_50 = user_distances2_50['ind']

            yes_1_10 = [x for x in train_indices if x in songs1_10]
            yes_2_10 = [x for x in train_indices if x in songs2_10]

            yes_1_50 = [x for x in train_indices if x in songs1_50]
            yes_2_50 = [x for x in train_indices if x in songs2_50]


            lenght_dict = {
                            'yes_1_no_2_10': len([x for x in yes_1_10 if x not in yes_2_10]),
                           'yes_1_no_2_50': len([x for x in yes_1_50 if x not in yes_2_50]),
                           'yes_2_no_1_10': len([x for x in yes_2_10 if x not in yes_1_10]),
                           'yes_2_no_1_50': len([x for x in yes_2_50 if x not in yes_1_50]),
                           'yes_both_10': len([x for x in yes_1_10 if x in yes_2_10]),
                           'yes_both_50': len([x for x in yes_1_50 if x in yes_2_50]),
                           'no_both_10': len([x for x in train_indices if (x not in yes_1_10) and (x not in yes_2_10)]),
                           'no_both_50': len([x for x in train_indices if (x not in yes_1_50) and (x not in yes_2_50)])
                           }

            if stats_df is None:
                stats_df = pandas.DataFrame(lenght_dict, index=[0])
            else:
                stats_df = stats_df.append(lenght_dict, ignore_index=True)

            if (i % 1000) == 0:
                print(i, len(users))
            i += 1
            # same_10 = len(songs1_10.intersection(songs2_10))
            # print(same_10)
            # same_50 = len(songs1_50.intersection(songs2_50))
            # print(same_50)
            # print('##################')
            # num_of_same_songs_50.append(same_50)
            # nums_of_same_songs_10.append(same_10)

        # same_10_np = numpy.array(nums_of_same_songs_10)
        # numpy.save('similarity_of_recommendations/' + method_1 + '_' + method_2 + '_same_in_10', same_10_np)
        # same_50_np = numpy.array(num_of_same_songs_50)
        # numpy.save('similarity_of_recommendations/' + method_1 + '_' + method_2 + '_same_in_50', same_50_np)
        stats_df = stats_df.sum(axis=0)
        stats_df.to_csv('similarity_of_recommendations/' + method_1 + '_' + method_2 + '_same_correct_sums_' + str(j) + '.csv', sep='\t')


def preprocess_df(df, column):
    df = df[column].tolist()
    df = [r.replace('[', '') for r in df]
    df = [r.replace(']', '') for r in df]
    df = [r.replace(' ', '') for r in df]
    # inds = [x for x in df if x != '']
    inds = [x.split(',') for x in df]

    return inds

def consistent_comparison(m1, m2, m1_file_prefix, m2_file_prefix, column):
    df = pandas.DataFrame()
    for i in range(0,5):
        m1_df = pandas.read_csv(m1_file_prefix + str(i+1), sep=';', header=0, index_col=0)
        m2_df = pandas.read_csv(m2_file_prefix + str(i+1), sep=';', header=0, index_col=0)

        results = compare_recommendations_consistently(m1_df, m2_df, column)
        df = df.append([results])
    means = df.mean(axis=0)
    print(m1, m2)
    print(means)
    df = df.append(means, ignore_index=True)
    df.columns = ['both_yes', 'm1_yes_m2_no', 'm2_yes_m1_no']
    df.to_csv('similarity_of_recommendations/comparisons/' + m1 + '_vs_' + m2 + 'corrects.csv', sep=';', index=False)

def compare_recommendations_consistently(m1_df, m2_df, column):
    m1_inds = preprocess_df(m1_df, column)
    m2_inds = preprocess_df(m2_df, column)
    indices = preprocess_df(m1_df, 'test_indices')
    m1_yes_m2_no = []
    m2_yes_m1_no = []
    both_yes = []
    i = 0
    for i in range(11123):
        current_1 = m1_inds[i]
        current_2 = m2_inds[i]
        # print(indices[i], current_1, current_2)
        m1y_m2n = [x for x in current_1 if (x not in current_2) and not (x is '')]
        m2y_m1n = [x for x in current_2 if (x not in current_1) and not (x is '')]
        yes_both = [x for x in current_1 if (x in current_2) and not (x is '')]

        if len(m1y_m2n) > 0:
            for x in m1y_m2n:
                m1_yes_m2_no.append(x)
            # print('m1 yes, m2 no', m2y_m1n)
        if len(m2y_m1n) > 0:
            for x in m2y_m1n:
                m2_yes_m1_no.append(x)
            # print('m2 yes, m1 no', m2y_m1n)
        if len(yes_both) > 0:
            for x in yes_both:
                both_yes.append(x)
            # print('both yes', yes_both)
        # print(i)
        # i += 1
    overall_num = len(both_yes) + len(m1_yes_m2_no) + len(m2_yes_m1_no)
    print(len(both_yes), len(m1_yes_m2_no), len(m2_yes_m1_no))
    return len(both_yes)/overall_num, len(m1_yes_m2_no)/overall_num, len(m2_yes_m1_no)/overall_num



# compare_recommendations_consistently('tags_tf_idf', 'w2v', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_1',
#                                      'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_1', 'matched_inds_10')

# consistent_comparison('tag_tf-idf', 'w2v', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_', 'matched_inds_10')

# consistent_comparison('tag_tf-idf', 'pca_tf-idf', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_','matched_inds_10')
#
# consistent_comparison('tag_tf-idf', 'bert', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'new_results/bert_results_max_indices/bert_distances_max_', 'matched_inds_10')
#
# consistent_comparison('tag_tf-idf', 'pca_mel', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_', 'matched_inds_10')
#
# consistent_comparison('tag_tf-idf', 'gru_mel', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_', 'matched_inds_10')
#
# consistent_comparison('tag_tf-idf', 'gru_mfcc', 'new_results/tag_based_TF-IDF_max_indices/tag_based_TF-IDF_max_',
#                       'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_', 'matched_inds_10')
#
# consistent_comparison('w2v', 'pca_tf-idf', 'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_',
#                       'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_', 'matched_inds_10')
#
# consistent_comparison('w2v', 'bert', 'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_',
#                       'new_results/bert_results_max_indices/bert_distances_max_', 'matched_inds_10')
#
# consistent_comparison('w2v', 'pca_mel', 'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_',
#                       'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_', 'matched_inds_10')
#
# consistent_comparison('w2v', 'gru_mel', 'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_',
#                       'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_', 'matched_inds_10')
#
# consistent_comparison('w2v', 'gru_mfcc', 'new_results/w2v_lyrics_results_max_indices/w2v_lyrics_distances_max_',
#                       'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_',
#                       'matched_inds_10')
#
# consistent_comparison('pca_tf-idf', 'bert', 'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_',
#                         'new_results/bert_results_max_indices/bert_distances_max_',
#                       'matched_inds_10')
# consistent_comparison('pca_tf-idf', 'pca_mel', 'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_',
#                         'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_',
#                       'matched_inds_10')
# consistent_comparison('pca_tf-idf', 'gru_mel', 'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_',
#                         'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_',
#                       'matched_inds_10')
# consistent_comparison('pca_tf-idf', 'gru_mfcc', 'results/pca_tf_idf_results_max_indices/pca_tf_idf_distances_max_',
#                         'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_',
#                       'matched_inds_10')
#
# consistent_comparison('bert', 'pca_mel', 'new_results/bert_results_max_indices/bert_distances_max_',
#                       'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_', 'matched_inds_10')
#
# consistent_comparison('bert', 'gru_mel', 'new_results/bert_results_max_indices/bert_distances_max_',
#                       'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_', 'matched_inds_10')
#
# consistent_comparison('bert', 'gru_mfcc', 'new_results/bert_results_max_indices/bert_distances_max_',
#                       'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_', 'matched_inds_10')
#
# consistent_comparison('pca_mel', 'gru_mel', 'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_',
#                       'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_',
#                       'matched_inds_10')
#
# consistent_comparison('pca_mel', 'gru_mfcc', 'results/pca_melspectrogram_results_max_indices/pca_melspectrogram_distances_max_',
#                       'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_',
#                       'matched_inds_10')
#
# consistent_comparison('gru_mel', 'gru_mfcc', 'results/gru_mel_results_5712_max_indices/gru_mel_distances_5712_max_',
#                       'new_results/gru_mfcc_results_30_64_max_indices/gru_mfcc_distances_30_64_max_',
#                       'matched_inds_10')

# compare_recomendations('tags_tf-idf', 'w2v_lyrics',
#                        'new_distances/tag_based_TF-IDF.npy', 'new_distances/w2v_lyrics_distances.npy',
#                        'useful_playlists', 'useful_songs')

# compare_recomendations('tags_tf-idf', 'pca_tf-idf',
#                        'new_distances/tag_based_TF-IDF.npy',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/distances/pca_tf_idf_distances.npy',
#                        'useful_playlists', 'useful_songs')
#
# compare_recomendations('tags_tf-idf', 'gru_mel',
#                        'new_distances/tag_based_TF-IDF.npy',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/distances/gru_mel_distances_5712.npy',
#                        'useful_playlists', 'useful_songs')
#
# compare_recomendations('tags_tf-idf', 'gru_mfcc',
#                        'new_distances/tag_based_TF-IDF.npy',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/gru_mfcc_distances_30_64.npy',
#                        'useful_playlists', 'useful_songs')
#
#
# compare_recomendations('W2V', 'gru_mfcc',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/w2v_lyrics_distances.npy',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/gru_mfcc_distances_30_64.npy',
#                        'useful_playlists', 'useful_songs')
#
# compare_recomendations('W2V', 'pca_tf-idf',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/w2v_lyrics_distances.npy',
#                        'distances/pca_tf_idf_distances.npy', 'useful_playlists', 'useful_songs')
#
# compare_recomendations('W2V', 'gru_mel',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/w2v_lyrics_distances.npy',
#                        '/Users/m_vys/PycharmProjects/similarity_and_evaluation/distances/gru_mel_distances_5712.npy')
#
# compare_recomendations('pca_tf-idf', 'gru_mel',
#                        'distances/pca_tf_idf_distances.npy',
#                        'distances/gru_mel_distances_5712.npy', 'useful_playlists', 'useful_songs')
#
# compare_recomendations('pca_tf-idf', 'gru_mfcc',
#                        'distances/pca_tf_idf_distances.npy',
#                        'new_distances/gru_mfcc_distances_30_64.npy', 'useful_playlists', 'useful_songs' )
#
# compare_recomendations('gru_mfcc', 'gru_mel',
#                        'new_distances/gru_mfcc_distances_30_64.npy',
#                        'distances/gru_mel_distances_5712.npy', 'useful_playlists','useful_songs')


# evaluate('mnt/0/new_distances/new_gru_mel_distances_14.npy', 0.9999999999984833)
# evaluate('mnt/0/new_distances/new_gru_mel_distances_80.npy', 0.9999999999965073)
# evaluate('mnt/0/new_distances/new_gru_mfcc_distances_16.npy', 0.9948029534716075)
# evaluate('mnt/0/new_distances/new_gru_mfcc_distances_32.npy', 0.9926800385717636)
# evaluate('mnt/0/new_distances/new_gru_mfcc_distances_5.npy', 0.9985064524053973)
# evaluate('mnt/0/new_distances/new_lstm_mel_distances_14.npy', 0.9999999999942446)
# evaluate('mnt/0/new_distances/new_lstm_mel_distances_40.npy', 0.9999999999998231)
# evaluate('mnt/0/new_distances/new_lstm_mel_distances_80.npy', 0.9999999999999124)
# evaluate('mnt/0/new_distances/new_lstm_mfcc_distances_16.npy', 0.9984163067049169)
# evaluate('mnt/0/new_distances/new_lstm_mfcc_distances_32.npy',0.9966145573465457)
# evaluate('mnt/0/new_distances/new_lstm_mfcc_distances_5.npy', 0.9998160194542247)

# evaluate('mnt/0/new_distances/gru_mfcc_distances_30_10.npy', 0.9979790719725744)
# evaluate('mnt/0/new_distances/gru_mfcc_distances_30_32.npy', 0.9951220311670839)
# evaluate('mnt/0/new_distances/gru_mfcc_distances_30_64.npy', 0.992621772026671)
# evaluate('mnt/0/retrained_distances/lstm_mel_distances_bs16_32.npy', 0.999999999997554)
# evaluate('mnt/0/retrained_distances/lstm_mel_distances_bs32_32.npy', 0.999999999997234)

# evaluate('new_distances/lstm_mfcc_distances_30_32.npy', 0.9985312636518952)
# evaluate('mnt/0/new_distances/lstm_mfcc_distances_30_64.npy', 0.9973641183864284)

# evaluate('new_distances/gru_mel_distances_30_28.npy', 0.9999999999995358)
# evaluate('new_distances/gru_mel_distances_30_160.npy', 0.9999999999963747)
# evaluate('new_distances/gru_mel_distances_30_80.npy',0.9999999999973926)


# evaluate('new_distances/bert_distances.npy', 0.8402684, True)
# evaluate('new_distances/roberta_distances.npy', 0.67700815)

# evaluate('new_distances/tag_based_BOW.npy', 0.19245008972987532)
# evaluate('new_distances/tag_based_TF-IDF.npy', 0.0005135140607123168)

# evaluate('new_distances/bert_distances.npy', 0.8402684, True)
# evaluate('distances/pca_mel_distances.npy, '', True)
# evaluate('/Volumes/LaCie/rocnikac/rocnikac/distances/pca_tf_idf_distances.npy', True)
# evaluate('/Volumes/LaCie/rocnikac/rocnikac/distances/gru_mel_distances_5712.npy', True)
# evaluate(gru_mfcc, True)
# # evaluate('new_distances/tag_based_BOW.npy', 0.19245008972987532, True)
# evaluate('new_distances/tag_based_TF-IDF.npy', 0.0005135140607123168, True)
# # # # evaluate('new_distances/w2v_lyrics_distances.npy', 0.9763697, False)
# evaluate('new_distances/w2v_lyrics_distances.npy', 0.9763697, True)
evaluate('distances/w2v_distances.npy', 0, True)
# evaluate('distances/pca_tf_idf_distances.npy', 0, True)
# evaluate('distances/pca_melspectrogram_distances.npy', 0, True)
# evaluate('new_distances/gru_mfcc_distances_30_64.npy', 0, True)
# evaluate('new_distances/bert_distances.npy', 0, True)
# evaluate('distances/gru_mel_distances_5712.npy', 0, True)
# evaluate('/Users/m_vys/PycharmProjects/similarity_and_evaluation/distances/w2v_distances.npy',0, True)
