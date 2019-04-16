import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from scipy import stats
from evaluation_results import get_distribution, read_cross_file

def ranking_distribution_plot(filename, axlabel, plot_tile):
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

    ranks_1 = get_distribution(c1, 0, float("inf"))
    ranks_2 = get_distribution(c2, 0, float("inf"))
    ranks_3 = get_distribution(c3, 0, float("inf"))
    ranks_4 = get_distribution(c4, 0, float("inf"))
    ranks_5 = get_distribution(c5, 0, float("inf"))

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

    g = sns.distplot(ranks, bins=5000, axlabel=axlabel, hist_kws=dict(edgecolor="black", linewidth=0.5), kde=True, kde_kws=dict(linestyle=''))
    g.set_xscale('log')

    # sns.kdeplot(ranks_4_6, label='4 to 6 songs', linestyle='--')
    # sns.kdeplot(ranks_7_10, label='7 to 10 songs', linestyle="--")
    # sns.kdeplot(ranks_11_15, label='11 to 15 songs', linestyle="--")
    # sns.kdeplot(ranks_16_20, label='16 to 20 songs', linestyle="--")
    # sns.kdeplot(ranks_21_more, label='21 and longer playlists', linestyle="--")
    sns.lineplot([x for x in range(16594)], distribution.reshape([16594]), label='over all distribution')
    sns.lineplot([x for x in range(16594)], distribution_4_6.reshape([16594]), label='4 to 6 songs')
    sns.lineplot([x for x in range(16594)], distribution_7_10.reshape([16594]), label='7 to 10 songs')
    sns.lineplot([x for x in range(16594)], distribution_11_15.reshape([16594]), label='11 to 15 songs')
    sns.lineplot([x for x in range(16594)], distribution_16_20.reshape([16594]), label='16 to 20 songs')
    sns.lineplot([x for x in range(16594)], distribution_21_more.reshape([16594]), label='21 and more')
    # plt.ylim(0, 0.002)
    g.set_xscale('log')
    plt.title(plot_tile)
    if filename == 'results/gru_spec_results/gru_spec_':
        plt.savefig('results/gru_spec_results/gru_spec_20400_graph.png', dpi=500)
    else:
        plt.savefig('results/pca_mel_results_5717/pca_mel_5715_graph.png', dpi=500)

    plt.show()


def get_ranks_distribution_for_lineplot(ranks):
    distribution = numpy.zeros([16594, 1])
    print(len(ranks))
    for i in range(len(ranks)):
        distribution[ranks[i]] += 1
    print(distribution.min(), distribution.max())
    distribution = numpy.divide(distribution, len(ranks))

    return distribution
# ranking_distribution_plot('results/pca_mel_results_5717/pca_mel_', 'rankings', 'PCA Mel 5717 ranking distribution')
# ranking_distribution_plot('results/lstm_mel_results_5712/lstm_mel_', 'rankings', 'LSTM MEL ranking distribution')
# ranking_distribution_plot('results/gru_spec_results/gru_spec_', 'rankings', 'GRU SPEC  ranking distribution')

# ranking_distribution_plot('results/som_w2v_results/som_w2v_b_5g5i_1_results_', 'rankings', 'som W2V ranking distribution')

ranking_distribution_plot('results/gru_spec_results/gru_spec_', 'rankings', 'Distribution of predicted ranks using GRU_20400 network with spectrograms')
# ranking_distribution_plot('results/pca_mel_results_5717/pca_mel_', 'rankings', 'Distribution of predicted ranks using PCA_5715 with Mel spectrograms')