import numpy, pandas, os

def get_results(filename):
    filename_1 = filename + '1'
    filename_2 = filename + '2'
    filename_3 = filename + '3'
    filename_4 = filename + '4'
    filename_5 = filename + '5'

    cross_1, c1 = read_cross_file(filename_1)
    cross_2, c2 = read_cross_file(filename_2)
    cross_3, c3 = read_cross_file(filename_3)
    cross_4, c4 = read_cross_file(filename_4)
    cross_5, c5 = read_cross_file(filename_5)

    means_1 = get_means_for_playlists(2, float("inf"), cross_1)
    means_2 = get_means_for_playlists(2, float("inf"), cross_2)
    means_3 = get_means_for_playlists(2, float("inf"), cross_3)
    means_4 = get_means_for_playlists(2, float("inf"), cross_4)
    means_5 = get_means_for_playlists(2, float("inf"), cross_5)
    general_means = pandas.DataFrame(data=[means_1, means_2, means_3, means_4, means_5])
    general_means.to_csv(filename + 'general_means', sep=';', header=None, index=False)

    means_1_4_6 = get_means_for_playlists(4, 7, cross_1)
    means_2_4_6 = get_means_for_playlists(4, 7, cross_2)
    means_3_4_6 = get_means_for_playlists(4, 7, cross_3)
    means_4_4_6 = get_means_for_playlists(4, 7, cross_4)
    means_5_4_6 = get_means_for_playlists(4, 7, cross_5)
    means_4_6 = pandas.DataFrame(data=[means_1_4_6, means_2_4_6, means_3_4_6, means_4_4_6, means_5_4_6])
    means_4_6.to_csv(filename + 'means_4_6', sep=';', header=None, index=False)

    means_1_7_10 = get_means_for_playlists(7, 11, cross_1)
    means_2_7_10 = get_means_for_playlists(7, 11, cross_2)
    means_3_7_10 = get_means_for_playlists(7, 11, cross_3)
    means_4_7_10 = get_means_for_playlists(7, 11, cross_4)
    means_5_7_10 = get_means_for_playlists(7, 11, cross_5)
    means_7_10 = pandas.DataFrame(data=[means_1_7_10, means_2_7_10, means_3_7_10, means_4_7_10, means_5_7_10])
    means_7_10.to_csv(filename + 'means_7_10', sep=';', header=None, index=False)

    means_1_11_15 = get_means_for_playlists(11, 16, cross_1)
    means_2_11_15 = get_means_for_playlists(11, 16, cross_2)
    means_3_11_15 = get_means_for_playlists(11, 16, cross_3)
    means_4_11_15 = get_means_for_playlists(11, 16, cross_4)
    means_5_11_15 = get_means_for_playlists(11, 16, cross_5)
    means_7_10 = pandas.DataFrame(data=[means_1_11_15, means_2_11_15, means_3_11_15, means_4_11_15, means_5_11_15])
    means_7_10.to_csv(filename + 'means_11_15', sep=';', header=None, index=False)

    means_1_16_20 = get_means_for_playlists(16, 21, cross_1)
    means_2_16_20 = get_means_for_playlists(16, 21, cross_2)
    means_3_16_20 = get_means_for_playlists(16, 21, cross_3)
    means_4_16_20 = get_means_for_playlists(16, 21, cross_4)
    means_5_16_20 = get_means_for_playlists(16, 21, cross_5)
    means_16_20 = pandas.DataFrame(data=[means_1_16_20, means_2_16_20, means_3_16_20, means_4_16_20, means_5_16_20])
    means_16_20.to_csv(filename + 'means_16_20', sep=';', header=None, index=False)

    means_1_21_more = get_means_for_playlists(21, float("inf"), cross_1)
    means_2_21_more = get_means_for_playlists(21, float("inf"), cross_2)
    means_3_21_more = get_means_for_playlists(21, float("inf"), cross_3)
    means_4_21_more = get_means_for_playlists(21, float("inf"), cross_4)
    means_5_21_more = get_means_for_playlists(21, float("inf"), cross_5)
    means_21_more = pandas.DataFrame(data=[means_1_21_more, means_2_21_more, means_3_21_more, means_4_21_more, means_5_21_more])
    means_21_more.to_csv(filename + 'means_21_more', sep=';', header=None, index=False)
    print(filename)
    # print(means_1, means_2, means_3, means_4, means_5)
    # print(means_1_4_6, means_2_4_6, means_3_4_6, means_4_4_6, means_5_4_6)
    # print(means_1_7_10, means_2_7_10, means_3_7_10, means_4_7_10, means_5_7_10)
    # print(means_1_11_15, means_2_11_15, means_3_11_15, means_4_11_15, means_5_11_15)
    # print(means_1_16_20, means_2_16_20, means_3_16_20, means_4_16_20, means_5_16_20)
    # print(means_1_21_more, means_2_21_more, means_3_21_more, means_4_21_more, means_5_21_more)

    overall_means = pandas.DataFrame(data=[means_1, means_2, means_3, means_4, means_5])
    overall_means.to_csv(filename + 'overall_means', sep=';', header=None, index=False)
    print(overall_means.mean(axis=0))


def get_num_of_correct(df_file):
    df = pandas.read_csv(df_file, sep=';', header=None, index_col=False)
    ranks = df.iloc[:, 3]
    ranks = ranks.tolist()
    ranks = [r.replace('[', '') for r in ranks]
    ranks = [r.replace(']', '') for r in ranks]
    ranks = [r.replace(' ', '') for r in ranks]
    num = 0
    for r in ranks:
        x = [number for number in r.split(',') if int(number) < 10]
        num += len(x)
    print(df_file, num)
    return num

def read_cross_file(filename):
    df = pandas.read_csv(filename, sep=';', header=None)
    ds = pandas.read_csv(filename, sep=';', header=None, usecols=[0,3])
    df[3] = df[3].replace({']': ''}, regex=True)
    df[3] = df[3].replace({'\[': ''}, regex=True)
    # df[3] = df[3].replace({',': ''}, regex=True)
    new_column = []
    for i, row in df.iterrows():
        a = numpy.fromstring(row[3], sep=',').astype(int)
        new_column.append(numpy.mean(a))

    df[3] = new_column
    # print(df.dtypes)
    return df, ds

def get_means_for_playlists(min_length, max_lenght, df):
    df = df.loc[(df[0] >= min_length) & (df[0] < max_lenght)]
    means = []
    means.append(numpy.mean(df[4]))
    means.append(numpy.mean(df[5]))
    means.append(numpy.mean(df[6]))
    means.append(numpy.mean(df[3]))
    means.append(numpy.mean(df[7]))

    # print(means)
    return(means)


def get_distribution(df, min_length, max_length):
    df = df.loc[(df[0]>=min_length) & (df[0] < max_length)]
    df[3] = df[3].replace({']': ''}, regex=True)
    df[3] = df[3].replace({'\[': ''}, regex=True)
    df[3] = df[3].replace({',': ''}, regex=True)
    rankings = []
    for i, row in df.iterrows():
        a = row[3].split(' ')
        for j in range(len(a)):
            rankings.append(int(a[j]))

    return rankings

# filename_1 = '/Users/m_vys/PycharmProjects/similarity_and_evaluation/results/som_w2v_results/som_w2v_results_1'
# cross_1, c1 = read_cross_file(filename_1)
# ranks = get_distribution(c1)
# print(ranks, numpy.mean(ranks))


#
# print('tf_idf_results')
# get_results('results/tf_idf_results/chopped_tf_idf_')
# #
#
# #
# print('pca_spec_results')
# get_results('results/pca_spec_results_1106/chopped_pca_spec_')
# #
# print('pca_mel_results')
# get_results('results/pca_mel_results_5717/chopped_pca_mel_')
#
# print('short pca_spec_results with threshold')
# get_results('results/short_pca_spec_results/chopped_pca_spec_')
#
# print('short pca_spec_results without threshold')
# get_results('results/short_pca_spec_results/pca_spec_')
#
# print('lstm mfcc results')
# get_results('results/lstm_mfcc_results/chopped_lstm_mfcc_')
#
#
# print('lstm mel results')
# get_results('results/lstm_mel_results_5712/chopped_lstm_mel_')
#
# print('gru_spec_results')
# get_results('results/gru_spec_results/chopped_gru_spec_')
#
# print('gru_mfcc_results')
# get_results('results/gru_mfcc_results/chopped_gru_mfcc_')
#
# print('gru_mel_results')
# get_results('results/gru_mel_results_5712/chopped_gru_mel_')
#
#
# print('short gru spec results')
# get_results('results/short_GRU_spec_results/chopped_GRU_spec_')
# #
# print('pca tf_idf_results')
# get_results('results/pca_tf_idf_results/chopped_pca_tf_idf_')
# # #
# print('short lstm spec results')
# get_results('results/short_LSTM_spec_results/chopped_LSTM_spec_')
#
#
# print('pca 320 mel results with threshold 2 ')
# get_results('results/pca_mel_results/chopped_pca_mel_threshold_0.383_')
#
# print('som_tf_idf_results without threshold')
# get_results('results/som_tf_idf_results/chopped_som_tf_idf_')
#
# print('lstm spec results')
# get_results('results/lstm_spec_results/chopped_lstm_spec_')
#
# print('som_w2v_results')
# get_results('results/som_w2v_results/chopped_som_w2v_')
#
# print('w2v results')
# get_results('results/w2v_results/chopped_w2v_')

# for name in os.listdir("new_results"):
#         if (not 'graphs' in name) and (not ('DS') in name):
#             sub_dirs = 'new_results/' + name +'/' + name.replace('results', 'distances') + '_'
#             print(sub_dirs.split('/')[-1][:-1])
#             get_results(sub_dirs)

# get_results('new_results/bert_results/bert_distances_')

# print('tag_bow')
# get_results('new_results/tag_based_BOW/tag_based_BOW_')
# print('tag_tfidf')
# get_results('new_results/tag_based_TF-IDF_max_updated/tag_based_TF-IDF_max_')

# get_results('/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_results/new_gru_mfcc_results_5/new_gru_mfcc_distances_5_')
# get_results('new_results/bert_results_max/bert_distances_max_')
# get_results('new_results/w2v_lyrics_results_max_updated/w2v_lyrics_distances_max_')
# get_num_of_correct('/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_results/tag_based_TF-IDF_max_updated/tag_based_TF-IDF_max_2')
# get_num_of_correct('new_results/w2v_lyrics_results_max_updated/w2v_lyrics_distances_max_1')
# get_num_of_correct('results/pca_tf_idf_results_max_updated/pca_tf_idf_distances_max_2')
# get_results('new_results/tag_based_TF-IDF_max_updated/tag_based_TF-IDF_max_')
# get_results('new_results/w2v_lyrics_results_max_updated/w2v_lyrics_distances_max_')
# get_results('results/pca_tf_idf_results_max_updated/pca_tf_idf_distances_max_')
# get_results('new_results/bert_results_max_updated/bert_distances_max_')
# get_results('results/pca_melspectrogram_results_max_updated/pca_melspectrogram_distances_max_')
# get_results('results/gru_mel_results_5712_max_updated/gru_mel_distances_5712_max_')
# get_results('new_results/gru_mfcc_results_30_64_max_updated/gru_mfcc_distances_30_64_max_')
get_results('/Users/m_vys/PycharmProjects/similarity_and_evaluation/results/w2v_results_max_updated/w2v_distances_max_')
