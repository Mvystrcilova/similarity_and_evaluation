import numpy, pandas

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

    means_1 = get_means_for_playlists(0, float("inf"), cross_1)
    means_2 = get_means_for_playlists(0, float("inf"), cross_2)
    means_3 = get_means_for_playlists(0, float("inf"), cross_3)
    means_4 = get_means_for_playlists(0, float("inf"), cross_4)
    means_5 = get_means_for_playlists(0, float("inf"), cross_5)
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

    print(means_1, means_2, means_3, means_4, means_5)
    print(means_1_4_6, means_2_4_6, means_3_4_6, means_4_4_6, means_5_4_6)
    print(means_1_7_10, means_2_7_10, means_3_7_10, means_4_7_10, means_5_7_10)
    print(means_1_11_15, means_2_11_15, means_3_11_15, means_4_11_15, means_5_11_15)
    print(means_1_16_20, means_2_16_20, means_3_16_20, means_4_16_20, means_5_16_20)
    print(means_1_21_more, means_2_21_more, means_3_21_more, means_4_21_more, means_5_21_more)


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
    print(df.dtypes)
    return df, ds

def get_means_for_playlists(min_length, max_lenght, df):
    df = df.loc[(df[0]>=min_length) & (df[0] < max_lenght)]
    means = []
    means.append(numpy.mean(df[4]))
    means.append(numpy.mean(df[5]))
    means.append(numpy.mean(df[6]))
    # a = []
    #     # df[3] = df[3].replace({']': ''}, regex=True)
    #     # df[3] = df[3].replace({'\[': ''}, regex=True)
    #     # df[3] = df[3].replace({',': ''}, regex=True)
    #     # for i, row in df.iterrows():
    #     #     ranks = row[3].split(' ')
    #     #     for j in range(len(ranks)):
    #     #         a.append(int(ranks[j]))
    #     # a = numpy.array(a)
    #     # print(a)
    means.append(numpy.mean(df[3]))
    means.append(numpy.mean(df[6]))

    print(means)
    return(means)

# get_results('/Users/m_vys/PycharmProjects/similarity_and_evaluation/results/som_w2v_results/som_w2v_results_')


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


# from audio_representations import convert_files_to_specs

# convert_files_to_specs('not_empty_songs', 4410, 812)