import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from scipy import stats
from evaluation_results import get_distribution, read_cross_file

def ranking_distribution_plot(filename, axlabel, plot_tile, graph_name):
    filename_1 = filename + '1'
    cross_1, c1 = read_cross_file(filename_1)
    filename_2 = filename + '2'
    cross_2, c2 = read_cross_file(filename_2)
    filename_3 = filename + '3'
    cross_3, c3 = read_cross_file(filename_3)
    filename_4 = filename + '4'
    cross_4, c4 = read_cross_file(filename_4)
    filename_5 = filename + '5'
    cross_5, c5 = read_cross_file(filename_5)

    ranks_1 = get_distribution(c1, 1, float("inf"))
    ranks_2 = get_distribution(c2, 1, float("inf"))
    ranks_3 = get_distribution(c3, 1, float("inf"))
    ranks_4 = get_distribution(c4, 1, float("inf"))
    ranks_5 = get_distribution(c5, 1, float("inf"))

    # ranks_1_0_3 = get_distribution(c1, 0, 3)
    # ranks_2_0_3 = get_distribution(c2, 0, 3)
    # ranks_3_0_3 = get_distribution(c3, 0, 3)
    # ranks_4_0_3 = get_distribution(c4, 0, 3)
    # ranks_5_0_3 = get_distribution(c5, 0, 3)

    ranks_1_4_6 = get_distribution(c1, 4, 6)
    ranks_2_4_6 = get_distribution(c2, 4, 6)
    ranks_3_4_6 = get_distribution(c3, 4, 6)
    ranks_4_4_6 = get_distribution(c4, 4, 6)
    ranks_5_4_6 = get_distribution(c5, 4, 6)

    ranks_1_7_10 = get_distribution(c1, 7, 11)
    ranks_2_7_10 = get_distribution(c2, 7, 11)
    ranks_3_7_10 = get_distribution(c3, 7, 11)
    ranks_4_7_10 = get_distribution(c4, 7, 11)
    ranks_5_7_10 = get_distribution(c5, 7, 11)

    ranks_1_11_15 = get_distribution(c1, 11, 16)
    ranks_2_11_15 = get_distribution(c2, 11, 16)
    ranks_3_11_15 = get_distribution(c3, 11, 16)
    ranks_4_11_15 = get_distribution(c4, 11, 16)
    ranks_5_11_15 = get_distribution(c5, 11, 16)

    ranks_1_16_20 = get_distribution(c1, 16, 21)
    ranks_2_16_20 = get_distribution(c2, 16, 21)
    ranks_3_16_20 = get_distribution(c3, 16, 21)
    ranks_4_16_20 = get_distribution(c4, 16, 21)
    ranks_5_16_20 = get_distribution(c5, 16, 21)

    ranks_1_21_more = get_distribution(c1, 21, float("inf"))
    ranks_2_21_more = get_distribution(c2, 21, float("inf"))
    ranks_3_21_more = get_distribution(c3, 21, float("inf"))
    ranks_4_21_more = get_distribution(c4, 21, float("inf"))
    ranks_5_21_more = get_distribution(c5, 21, float("inf"))

    ranks = ranks_1 + ranks_2 + ranks_3 + ranks_4 + ranks_5
    # ranks_0_3 = ranks_1_0_3 + ranks_2_0_3 + ranks_3_0_3 + ranks_4_0_3 + ranks_5_0_3
    ranks_4_6 = ranks_1_4_6 + ranks_2_4_6 + ranks_3_4_6 + ranks_4_4_6 + ranks_5_4_6
    ranks_7_10 = ranks_1_7_10 + ranks_2_7_10 + ranks_3_7_10 + ranks_4_7_10 + ranks_5_7_10
    ranks_11_15 = ranks_1_11_15 + ranks_2_11_15 + ranks_3_11_15 + ranks_4_11_15 + ranks_5_11_15
    ranks_16_20 = ranks_1_16_20 + ranks_2_16_20 + ranks_3_16_20 + ranks_4_16_20 + ranks_5_16_20
    ranks_21_more = ranks_1_21_more + ranks_2_21_more + ranks_3_21_more + ranks_4_21_more + ranks_5_21_more

    palette = sns.color_palette('husl', 6)

    sns.set_palette(palette)

    distribution = get_ranks_distribution_for_lineplot(ranks)
    # distribution_0_3 = get_ranks_distribution_for_lineplot(ranks_0_3)
    distribution_4_6 = get_ranks_distribution_for_lineplot(ranks_4_6)
    distribution_7_10 = get_ranks_distribution_for_lineplot(ranks_7_10)
    distribution_11_15 = get_ranks_distribution_for_lineplot(ranks_11_15)
    distribution_16_20 = get_ranks_distribution_for_lineplot(ranks_16_20)
    distribution_21_more = get_ranks_distribution_for_lineplot(ranks_21_more)

    # g = sns.distplot(ranks_21_more, bins=5000, axlabel=axlabel, hist_kws=dict(edgecolor="black", linewidth=0.5), kde=True, kde_kws=dict(linestyle=''))
    # g.set_xscale('log')

    # sns.kdeplot(ranks_4_6, label='4 to 6 songs', linestyle='--')
    # sns.kdeplot(ranks_7_10, label='7 to 10 songs', linestyle="--")
    # sns.kdeplot(ranks_11_15, label='11 to 15 songs', linestyle="--")
    # sns.kdeplot(ranks_16_20, label='16 to 20 songs', linestyle="--")
    # g = sns.distplot(ranks_21_more, label='21 and longer playlists', linestyle="--")
    g = sns.lineplot([x for x in range(16594)], distribution.reshape([16594]), label='overall distribution')
    sns.lineplot([x for x in range(16594)], distribution_4_6.reshape([16594]), label='4 to 6 songs')
    sns.lineplot([x for x in range(16594)], distribution_7_10.reshape([16594]), label='7 to 10 songs')
    sns.lineplot([x for x in range(16594)], distribution_11_15.reshape([16594]), label='11 to 15 songs')
    sns.lineplot([x for x in range(16594)], distribution_16_20.reshape([16594]), label='16 to 20 songs')
    sns.lineplot([x for x in range(16594)], distribution_21_more.reshape([16594]), label='21 and more')
    # plt.ylim(0, 0.002)
    plt.gca().set_ylim(bottom=0)
    g.set_xscale('log')

    plt.title(plot_tile)
    g_name = filename.split('/')[0] + '/' + 'graphs' + '/' + graph_name

    plt.savefig(g_name, dpi=500)


    plt.show()


def get_ranks_distribution_for_lineplot(ranks):
    distribution = numpy.zeros([16594, 1])
    print(len(ranks))
    for i in range(len(ranks)):
        distribution[ranks[i]] += 1
    print(distribution.min(), distribution.max())
    distribution = numpy.divide(distribution, len(ranks))

    return distribution

# ranking_distribution_plot('results/pca_mel_results_5717/chopped_pca_mel_', 'rankings', 'RDG of the PCA mel 5715 method', 'pca_mel_5715_graph.png')
# ranking_distribution_plot('results/pca_spec_results_1106/chopped_pca_spec_', 'rankings', 'RDG of the PCA_spec_1106 method', 'pca_spec_1106_graph.png')
# ranking_distribution_plot('results/lstm_mel_results_5712/chopped_lstm_mel_', 'rankings', 'RDG of the LSTM_mel method', 'lstm_mel_5712_graph.png')
# ranking_distribution_plot('results/gru_spec_results/chopped_gru_spec_', 'rankings', 'RDG of the GRU_spec_20400  method', 'gru_spec_20400_graph.png')
# ranking_distribution_plot('results/short_GRU_spec_results/chopped_GRU_spec_', 'rankings', 'RDG of the GRU_spec_5712', 'gru_spec_5712_graph.png')
# ranking_distribution_plot('results/gru_mel_results_5712/chopped_gru_mel_', 'rankings', 'RDG of the GRU_mel method', 'gru_mel_graph.png')
#
# ranking_distribution_plot('results/som_tf_idf_results/som_tf_idf_', 'rankings', 'RDG of the SOM_TF-idf method', 'som_tf_idf_graph.png')
# ranking_distribution_plot('results/lstm_spec_results/chopped_lstm_spec_', 'rankings', 'The RDG of the LSTM_spec_20400 network', 'lstm_spec_20400_graph.png')
# ranking_distribution_plot('results/short_LSTM_spec_results/chopped_lstm_spec_', 'rankings', 'The RDG of the LSTM_spec_5712 network', 'lstm_spec_5712.png')
# ranking_distribution_plot('results/gru_mfcc_results/chopped_gru_mfcc_', 'rankings', 'RDG of the GRU_MFCC method', 'gru_mfcc_graph.png')
# ranking_distribution_plot('results/lstm_mfcc_results/chopped_lstm_mfcc_', 'rankings', 'RDG of the LSTM_MFCC network', 'lstm_mfcc_graph.png')
#
# ranking_distribution_plot('results/pca_tf_idf_results/chopped_pca_tf_idf_', 'rankings', 'RDG of the PCA_Tf-idf method', 'pca_tf_idf_graph.png')
# ranking_distribution_plot('results/tf_idf_results/chopped_tf_idf_', 'rankings', 'RDG of the Tf-idf method', 'tf_idf_graph.png')
# ranking_distribution_plot('results/w2v_results/chopped_w2v_', 'rankings', 'RDG of the W2V method', 'w2v_graph.png')
# ranking_distribution_plot('results/som_w2v_results/chopped_som_w2v_', 'rankings', 'RDG of the SOM W2V method', 'som_w2v_graph.png')
# ranking_distribution_plot('results/pca_mel_results/chopped_pca_mel_threshold_0.383_', 'rankings', 'RDG of the PCA Mel 320 method', 'pca_mel_320_graph.png')

ranking_distribution_plot('new_results/new_gru_mel_results_40/new_gru_mel_distances_40_', 'rankings', 'RDG of the gru_mel_40', 'gru_mel_40_graph.png')
# ranking_distribution_plot('results/short_pca_spec_results/chopped_pca_spec_', 'rankings', 'RDG of the PCA spec 320 method', 'pca_spec_320_graph.png')
ranking_distribution_plot('new_results/new_gru_mel_results_14/new_gru_mel_distances_14_', 'rankings', 'RDG of the gru_mel_14', 'gru_mel_14_graph.png')
ranking_distribution_plot('new_results/new_gru_mel_results_80/new_gru_mel_distances_80_', 'rankings', 'RDG of the gru_mel_80', 'gru_mel_80_graph.png')
ranking_distribution_plot('new_results/new_gru_mfcc_results_16/new_gru_mfcc_distances_16_', 'rankings', 'RDG of the gru_mfcc_16', 'gru_mfcc_16_graph.png')
ranking_distribution_plot('new_results/new_lstm_mel_results_14/new_lstm_mel_distances_14_', 'rankings', 'RDG of the lstm_mel_14', 'lstm_mel_14_graph.png')
ranking_distribution_plot('new_results/new_lstm_mel_results_40/new_lstm_mel_distances_40_', 'rankings', 'RDG of the lstm_mel_40', 'lstm_mel_40_graph.png')
ranking_distribution_plot('new_results/new_lstm_mel_results_80/new_lstm_mel_distances_80_', 'rankings', 'RDG of the lstm_mel_80', 'lstm_mel_80_graph.png')

# ranking_distribution_plot('new_results/new_lstm_mfcc_results_16/new_lstm_mfcc_distances_16_', 'rankings', 'RDG of the lstm_mfcc_16', 'lstm_mfcc_16_graph.png')
# ranking_distribution_plot('new_results/new_lstm_mfcc_results_32/new_lstm_mfcc_distances_32_', 'rankings', 'RDG of the lstm_mfcc_32', 'lstm_mfcc_32_graph.png')
# ranking_distribution_plot('new_results/new_lstm_mfcc_results_5/new_lstm_mfcc_distances_5_', 'rankings', 'RDG of the lstm_mfcc_5', 'lstm_mfcc_5_graph.png')


#SOM A W2V jeste


