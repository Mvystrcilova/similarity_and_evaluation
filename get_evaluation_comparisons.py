import numpy, pandas
import matplotlib.pyplot as plt
import seaborn as sns
# from visualize_model_training import get_nn_training_loss
import pickle, os

def read_means(file):
    df = pandas.read_csv(file, sep=';', header=None, index_col=None, names=['R@10', 'R@50', 'R@100', 'average rank', 'ndGC'])

    means = df.mean()
    return means[0], means[1], means[2], means[3], means[4]

def read_data(directory, file_names):
    chopped_file = 'chopped_'+ file_names
    chopped_general_means = read_means(directory + '/' + chopped_file + '_general_means')
    chopped_short = read_means(directory + '/' + chopped_file + '_means_4_6')
    chopped_long = read_means(directory + '/' + chopped_file + '_means_21_more')

    general_means = read_means(directory + '/' + file_names + '_general_means')
    short = read_means(directory + '/' + file_names + '_means_4_6')
    long = read_means(directory + '/' + file_names + '_means_21_more')

    return general_means, short, long, chopped_general_means, chopped_short, chopped_long

def read_category_data(directory, file_names):
    chopped_file = file_names
    chopped_general_means = read_means(directory + '/' + chopped_file + '_general_means')
    chopped_general_means = [float(x) for x in chopped_general_means]
    chopped_A = read_means(directory + '/' + chopped_file + '_means_4_6')
    chopped_A = [float(x) for x in chopped_A]
    chopped_B_1 = read_means(directory + '/' + chopped_file + '_means_7_10')
    chopped_B_2 = read_means(directory + '/' + chopped_file + '_means_11_15')
    chopped_B = [(x+y)/2 for x,y in zip(chopped_B_1, chopped_B_2)]
    chopped_C_1 = read_means(directory + '/' + chopped_file + '_means_16_20')
    chopped_C_2 = read_means(directory + '/' + chopped_file + '_means_21_more')
    chopped_C = [(x+y)/2 for x,y in zip(chopped_C_1, chopped_C_2)]
    print('general means:')
    print(chopped_general_means)
    print('short means')
    print(chopped_A)
    print('medium means')
    print(chopped_B)
    print('long_means')
    print(chopped_C)

    all_values = chopped_general_means + chopped_A + chopped_B + chopped_C
    return all_values

def get_means_dataframe():
    means_df = pandas.DataFrame(columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    temp_dataframe = pandas.DataFrame([read_category_data('results/tf_idf_results', 'tf_idf')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)

    temp_dataframe = pandas.DataFrame([read_category_data('results/w2v_results', 'w2v')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/w2v', sep=';')


    temp_dataframe = pandas.DataFrame([read_category_data('results/pca_tf_idf_results', 'pca_tf_idf')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/pca_tf_idf', sep=';')

    # temp_dataframe = pandas.DataFrame([read_category_data('results/som_w2v_results', 'som_w2v')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)

    temp_dataframe = pandas.DataFrame([read_category_data('results/pca_mel_results', 'pca_mel')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/pca_mel_320', sep=';')

    temp_dataframe = pandas.DataFrame([read_category_data('results/pca_mel_results_5717', 'pca_mel')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/pca_mel_5717', sep=';')

    temp_dataframe = pandas.DataFrame([read_category_data('results/pca_spec_results_1106', 'pca_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/pca_spec_results_1106', sep=';')

    # temp_dataframe = pandas.DataFrame([read_category_data('results/short_pca_spec_results', 'pca_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)
    #
    # temp_dataframe = pandas.DataFrame([read_category_data('results/gru_spec_results', 'gru_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)
    #
    # temp_dataframe = pandas.DataFrame([read_category_data('results/short_GRU_spec_results', 'GRU_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)
    #
    # temp_dataframe = pandas.DataFrame([read_category_data('results/lstm_spec_results', 'lstm_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)
    #
    # temp_dataframe = pandas.DataFrame([read_category_data('results/short_LSTM_spec_results', 'LSTM_spec')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)

    temp_dataframe = pandas.DataFrame([read_category_data('results/gru_mel_results_5712', 'gru_mel')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/gru_mel_results_5712', sep=';')


    # temp_dataframe = pandas.DataFrame([read_category_data('results/lstm_mel_results_5712', 'lstm_mel')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)

    # temp_dataframe = pandas.DataFrame([read_category_data('results/gru_mfcc_results', 'gru_mfcc')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    # means_df = means_df.append(temp_dataframe)

    temp_dataframe = pandas.DataFrame([read_category_data('results/lstm_mfcc_results', 'lstm_mfcc')], columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    means_df = means_df.append(temp_dataframe)
    temp_dataframe.to_csv('means_files/lstm_mfcc_results', sep=';', )

    for name in os.listdir("new_results"):
        if (not 'graphs' in name) and (not 'DS' in name):
            method_name = name.split('/')[-1].replace('results','distances')
            dir_name = 'new_results/' + name
            temp_dataframe = pandas.DataFrame([read_category_data(dir_name, method_name)],columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
            dataframe_dir = 'means_files/' + name
            temp_dataframe.to_csv(dataframe_dir, sep=';', )


    return means_df

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means, data_to_assign):
    general_means.append(data_to_assign[0])
    short_means.append(data_to_assign[1])
    long_means.append(data_to_assign[2])
    chopped_general_means.append(data_to_assign[3])
    chopped_short_means.append(data_to_assign[4])
    chopped_long_means.append(data_to_assign[5])

def plot_comparisons():
    names = ['Tf-idf', 'W2V', 'PCA_Tf-idf', 'SOM_W2V', 'PCA_mel_320', 'PCA_mel_5715', 'PCA_spec_1106', 'PCA_spec_320',
             'GRU_spec_20400', 'GRU_spec_5712', 'LSTM_spec_20400', 'LSTM_spec_5712', 'GRU_mel', 'LSTM_mel', 'GRU_MFCC',
             'LSTM_MFCC']

    colors = ['firebrick', 'darkgreen', 'navy', 'darkcyan','darkorchid', 'darkorange', 'deeppink', 'gold', 'indigo', 'darkolivegreen', 'blue', 'black', 'dodgerblue', 'darkgreen', 'darkred', 'sienna']
    general_means = []
    short_means = []
    long_means = []

    chopped_general_means = []
    chopped_short_means = []
    chopped_long_means = []


    tf_idf_data = read_data('results/tf_idf_results', 'tf_idf')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means, tf_idf_data)

    w2v_data = read_data('results/w2v_results', 'w2v')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means, w2v_data)

    pca_tf_idf_data = read_data('results/pca_tf_idf_results', 'pca_tf_idf')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means, pca_tf_idf_data)

    data = read_data('results/som_w2v_results', 'som_w2v')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/pca_mel_results', 'pca_mel')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/pca_mel_results_5717', 'pca_mel')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/pca_spec_results_1106', 'pca_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/short_pca_spec_results', 'pca_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/gru_spec_results', 'gru_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/short_GRU_spec_results', 'GRU_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/lstm_spec_results', 'lstm_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/short_LSTM_spec_results', 'LSTM_spec')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/gru_mel_results_5712', 'gru_mel')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/lstm_mel_results_5712', 'lstm_mel')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/gru_mfcc_results', 'gru_mfcc')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)

    data = read_data('results/lstm_mfcc_results', 'lstm_mfcc')
    assign_means(general_means, short_means, long_means, chopped_general_means, chopped_short_means, chopped_long_means,
                 data)
    fig = plt.figure(figsize=(10, 20))
    fig.suptitle('Recalls compared for methods on playlists of different lengths', fontsize=16)

    grid = plt.GridSpec(1, 3, hspace=0.2, wspace=0.3)

    general_means_plot = fig.add_subplot(grid[0,0])
    short_means_plot = fig.add_subplot(grid[0,1])
    long_means_plot = fig.add_subplot(grid[0,2])



    x = ['R@10', 'R@50', 'R@100']

    for i in range(16):
        # general_means_plot.plot(x, general_means[i], color=colors[i], label=names[i], linestyle='--')
        general_means_plot.plot(x, chopped_general_means[i], color=colors[i], label=(names[i] + ' with threshold'))

    general_means_plot.set_title('All playlist lengths')
    general_means_plot.set_ylim(0, 0.14)

    for i in range(16):
        # short_means_plot.plot(x, short_means[i], color=colors[i], label=names[i], linestyle='--')
        short_means_plot.plot(x, chopped_short_means[i], color=colors[i], label=(names[i] + ' with threshold'))

    short_means_plot.set_title('Playlists lengths 4 to 6')
    short_means_plot.set_ylim(0, 0.14)

    for i in range(16):
        # long_means_plot.plot(x, long_means[i], color=colors[i], label=names[i], linestyle='--')
        long_means_plot.plot(x, chopped_long_means[i], color=colors[i], label=(names[i] + ' with threshold'))

    long_means_plot.set_title('Playlist lengths 21 and more')
    long_means_plot.set_ylim(0, 0.14)

    fig.subplots_adjust(top=0.95, bottom=0.15)

    handles, labels = long_means_plot.get_legend_handles_labels()
    fig.legend(labels, loc='lower center', ncol=3, prop={'size':10})
    # fig.legend()
    # plt.legend(loc='lower center', bbox_to_anchor=(-1.0, -0.5), ncol=3, prop={'size':10})
    #
    plt.savefig('threshold_comparison_simplified', dpi=500)


    plt.show()

def rank_dataframe(file):
    df = pandas.read_csv(file, index_col=None, sep=';')
    df = df.apply(lambda x: [-1* y if y > 100 else y for y in x])
    new_df = df.rank(axis=0, ascending=False)
    print(new_df)
    fig = plt.figure(figsize=(12, 12))
    r = sns.heatmap(new_df, cmap='RdYlGn_r', linecolor="white", linewidths=0.1,)
    r.set_yticklabels(df['names'], rotation=30)
    r.set_xticklabels(list(new_df), rotation=30)
    r.xaxis.tick_top() # x axis on top
    r.xaxis.set_label_position('top')
    r.set_title("Ranking of the tested methods in various categories without threshold")
    fig.savefig('no_threshold_method_ranking.png', dpi=700)
    plt.show()
# plot_comparisons()
def rank_dataframe_not_from_file(dataframe):
    df = dataframe
    df = df.apply(lambda x: [-1 * y if y > 100 else y for y in x])
    new_df = df.rank(axis=0, ascending=False)
    print(new_df)
    fig = plt.figure(figsize=(12, 12))
    r = sns.heatmap(new_df, cmap='RdYlGn_r', linecolor="white", linewidths=0.1, )
    r.set_yticklabels(
        list(df.index), rotation=30)
    r.set_xticklabels(list(new_df), rotation=30)
    r.xaxis.tick_top()  # x axis on top
    r.xaxis.set_label_position('top')
    r.set_title("Ranking of the tested methods in various categories without threshold")
    fig.savefig('no_threshold_method_ranking.png', dpi=700)
    plt.show()
# df = get_means_dataframe()
# df.to_csv('no_threshold_means_data',sep=';')

def get_min_and_max():
    df_threshold = pandas.read_csv('means_data', index_col=None, sep=';', usecols=['R@10', 'R@50', 'R@100', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'nCDG_L'])
    df_no_threshold = pandas.read_csv('no_threshold_means_data', index_col=None, sep=';', usecols=['R@10', 'R@50', 'R@100', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'nCDG_L'])

    df_threshold_max = df_threshold.max(axis=0)
    df_threshold_min = df_threshold.min(axis=0)
    df_threshold_avg = df_threshold.mean(axis=0)

    df_no_threshold_max = df_no_threshold.max(axis=0)
    df_no_threshold_min = df_no_threshold.min(axis=0)
    df_no_threshold_avg = df_no_threshold.mean(axis=0)



    barWidth = 0.15

    # set height of bar
    bars1 = df_threshold_max
    bars2 = df_no_threshold_max

    bars3 = df_threshold_avg
    bars4 = df_no_threshold_avg

    bars5 = df_threshold_min
    bars6 = df_no_threshold_min

    # Set position of bar on X axis
    r1 = numpy.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]

    fig = plt.figure(figsize=(20, 10))
    # Make the plot
    plt.bar(r1, bars1, color='limegreen', width=barWidth, edgecolor='white', label='max value with threshold')
    plt.bar(r2, bars2, color='lime', width=barWidth, edgecolor='white', label='max value without threshold')
    plt.bar(r3, bars3, color='dodgerblue', width=barWidth, edgecolor='white', label='average value with threhsold')
    plt.bar(r4, bars4, color='deepskyblue', width=barWidth, edgecolor='white', label='average value without threshold')
    plt.bar(r5, bars5, color='tomato', width=barWidth, edgecolor='white', label='min value with threshold')
    plt.bar(r6, bars6, color='r', width=barWidth, edgecolor='white', label='min value without threshold')
    # Add xticks on the middle of the group bars
    plt.ylabel('value', fontweight='bold', fontsize=12)
    plt.xlabel('evaluation measure', fontweight='bold', fontsize=12)
    plt.xticks([r + barWidth for r in range(len(bars1))], ['R@10', 'R@50', 'R@100', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'nCDG_L'], rotation=30, fontsize=12)

    # Create legend & Show graphic
    plt.legend(fontsize=12)
    plt.title('Comparison of the max, average and min values for each evaluation measure with and without threshold', fontsize=15)
    # fig.savefig('results/graphs/mean_min_max_measure_comparison.png', dpi=700)
    plt.show()

    # ax.sns.boxplot()


# rank_dataframe('no_threshold_means_data')
# get_min_and_max()
def get_absolute_comparison():
    df_threshold = pandas.read_csv('means_data', index_col=None, sep=';', usecols=['R@10', 'R@50', 'R@100', 'R@10_S', 'R@50_S', 'R@100_S', 'R@10_M', 'R@50_M', 'R@100_M', 'R@10_L', 'R@50_L', 'R@100_L'])
    df_threshold[df_threshold.columns[::3]] /= 0.0006
    df_threshold[df_threshold.columns[1::3]] /= 0.003
    df_threshold[df_threshold.columns[2::3]] /= 0.006
    df_threshold = df_threshold.round(1)
    # df_threshold['names'] = ['Tf-idf', 'W2V', 'PCA_Tf-idf', 'SOM_W2V', 'PCA_mel_320', 'PCA_mel_5715', 'PCA_spec_1106', 'PCA_spec_320', 'GRU_spec_20400', 'GRU_spec_5712', 'LSTM_spec_20400', 'LSTM_spec_5712', 'GRU_mel', 'LSTM_mel', 'GRU_MFCC', 'LSTM_MFCC']
    fig = plt.figure(figsize=(15, 12))
    r = sns.heatmap(df_threshold, cmap='coolwarm', annot=True, fmt="g", linecolor="white", linewidths=0.1, )
    r.set_yticklabels(
        ['Tf-idf', 'W2V', 'PCA_Tf-idf', 'SOM_W2V', 'PCA_mel_320', 'PCA_mel_5715', 'PCA_spec_1106', 'PCA_spec_320',
         'GRU_spec_20400', 'GRU_spec_5712', 'LSTM_spec_20400', 'LSTM_spec_5712', 'GRU_mel', 'LSTM_mel', 'GRU_MFCC',
         'LSTM_MFCC'], rotation=30)
    r.xaxis.tick_top()  # x axis on top
    r.xaxis.set_label_position('top')
    fig.savefig('results/graphs/absolute_evaluation_comparison.png', dpi=700)
    plt.show()
# get_absolute_comparison()

def get_nn_training_loss(gru_spec_file, short_gru_spec_file, lstm_spec_file, short_lstm_spec_file, gru_mel_file, lstm_mel_file, gru_mfcc_file, lstm_mfcc_file):
    gru_spec_file = open(gru_spec_file, 'rb')
    short_gru_spec_file = open(short_gru_spec_file, 'rb')
    lstm_spec_file = open(lstm_spec_file, 'rb')
    short_lstm_spec_file = open(short_lstm_spec_file, 'rb')
    gru_mel_file = open(gru_mel_file, 'rb')
    lstm_mel_file = open(lstm_mel_file, 'rb')
    gru_mfcc_file = open(gru_mfcc_file, 'rb')
    lstm_mfcc_file = open(lstm_mfcc_file, 'rb')

    history_1 = pickle.load(gru_spec_file)
    history_2 = pickle.load(short_gru_spec_file)
    lstm_spec_history = pickle.load(lstm_spec_file)
    short_lstm_spec_history = pickle.load(short_lstm_spec_file)
    gru_mel_history = pickle.load(gru_mel_file)
    lstm_mel_history = pickle.load(lstm_mel_file)
    gru_mfcc_history = pickle.load(gru_mfcc_file)
    lstm_mfcc_history = pickle.load(lstm_mfcc_file)

    a = [history_1['loss'][-1]]
    b = [history_2['loss'][-1]]
    c = [lstm_spec_history['loss'][-1]]
    d = [short_lstm_spec_history['loss'][-1]]
    e = [gru_mel_history['loss'][-1]]
    f = [lstm_mel_history['loss'][-1]]
    g = [gru_mfcc_history['loss'][-1]]
    h = [lstm_mfcc_history['loss'][-1]]
    return a + b + c + d + e + f + g + h


def get_just_nns():
    exclude = [1,2,3,4,5,6,7,8]
    df_threshold = pandas.read_csv('means_data', usecols=['R@10', 'R@50', 'R@100', 'nCDG'], index_col=None, sep=';',skiprows=exclude)
    R10 = df_threshold['R@10'].tolist()
    R50 = df_threshold['R@50'].tolist()
    R100 = df_threshold['R@100'].tolist()
    nCDG = df_threshold['nCDG'].tolist()
    second_col = ['R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG']
    new_df = pandas.DataFrame(R10 + R50 + R100 + nCDG, columns=['values'])
    new_df['eval measure'] = second_col
    return new_df

def loss_vs_performance():
    df = get_just_nns()
    smallest_loss = get_nn_training_loss('model_histories/GRU_SPEC_history','model_histories/short_GRU_SPEC_history','model_histories/LSTM_SPEC_history', 'model_histories/short_LSTM_SPEC_history', 'model_histories/GRU_MEL_history', 'model_histories/LSTM_MEL_history', 'model_histories/GRU_MFCC_history','model_histories/LSTM_MFCC_history')
    smallest_loss = smallest_loss + smallest_loss + smallest_loss + smallest_loss
    df['loss'] = smallest_loss
    print(df)
    # sns.set(rc={'figure.figsize': (5, 30)})

    # sns.set(style="darkgrid")
    g = sns.relplot(x='loss', y='values', hue='eval measure', data=df)

    leg = g._legend
    leg.set_bbox_to_anchor([0.65, 0.85])  # coordinates of lower left of bounding box
    # leg._loc = 3  # if required you can set the loc

    # labels = ('R@10', 'R@50', 'R@100')
    # plt.legend(labels, loc='upper center')
    # sns.lmplot(x='R@50', y='loss', data=df, fit_reg=False, scatter_kws={"s": 100})
    # sns.lmplot(x='R@100', y='loss', data=df, fit_reg=False, scatter_kws={"s": 100})
    # sns.end(['R@10' 'R@50', 'R@100'], loc='lower center')
    g.savefig('results/graphs/nn_training_performance_correlation.png', dpi=500)
    plt.show()

def threshold_vs_performance():
    df_threshold = pandas.read_csv('means_data', usecols=['R@10', 'R@50', 'R@100', 'nCDG'], index_col=None, sep=';')
    R10 = df_threshold['R@10'].tolist()
    R50 = df_threshold['R@50'].tolist()
    R100 = df_threshold['R@100'].tolist()
    nCDG = df_threshold['nCDG'].tolist()
    threshold = pandas.read_csv('/Users/m_vys/Documents/matfyz/bakalarka/intermethod_distances.txt', usecols=['50'], sep=';')['50'].tolist()
    second_col = ['R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10','R@10', 'R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50','R@50', 'R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100','R@100', 'nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG','nCDG']

    new_df = pandas.DataFrame(R10 + R50 + R100 + nCDG, columns=['values'])
    new_df['eval measure'] = second_col
    new_df['threshold'] = threshold + threshold + threshold + threshold

    # sns.set(style="darkgrid")
    g = sns.relplot(x='threshold', y='values', hue='eval measure', data=new_df)

    leg = g._legend
    leg.set_bbox_to_anchor([0.7, 0.85])
    g.savefig('results/graphs/nn_threshold_performance_correlation.png', dpi=500)


    plt.show()
def create_dataframe(directory):
    means_df = pandas.DataFrame(columns=['R@10', 'R@50', 'R@100', 'avg_rank', 'nCDG', 'R@10_S', 'R@50_S', 'R@100_S', 'avg_rank_S', 'nCDG_S', 'R@10_M', 'R@50_M', 'R@100_M', 'avg_rank_M', 'nCDG_M', 'R@10_L', 'R@50_L', 'R@100_L', 'avg_rank_L', 'nCDG_L'])
    names = []
    for name in os.listdir(directory):
        names.append(name)
        relative_path = directory + '/' + name
        cols = [x for x in range(1,21)]
        temp_frame = pandas.read_csv(relative_path, sep=';', usecols=cols)
        means_df = means_df.append(temp_frame)
    means_df['names'] = names
    means_df = means_df.set_index('names')
    return means_df

# get_means_dataframe()
df = create_dataframe('means_files')
rank_dataframe_not_from_file(df)