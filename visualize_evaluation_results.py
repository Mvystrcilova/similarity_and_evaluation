import seaborn as sns
import matplotlib.pyplot as plt
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
    ranks_4_6 = ranks_1_4_6 + ranks_2_4_6 + ranks_3_4_6 + ranks_4_4_6 + ranks_5_4_6
    ranks_7_10 = ranks_1_7_10 + ranks_2_7_10 + ranks_3_7_10 + ranks_4_7_10 + ranks_5_7_10
    ranks_11_15 = ranks_1_11_15 + ranks_2_11_15 + ranks_3_11_15 + ranks_4_11_15 + ranks_5_11_15
    ranks_16_20 = ranks_1_16_20 + ranks_2_16_20 + ranks_3_16_20 + ranks_4_16_20 + ranks_5_16_20
    ranks_21_more = ranks_1_21_more + ranks_2_21_more + ranks_3_21_more + ranks_4_21_more + ranks_5_21_more

    palette = sns.color_palette('husl', 6)
    sns.set_palette(palette)

    sns.distplot(ranks, bins=50, axlabel=axlabel, hist_kws=dict(edgecolor="black", linewidth=0.5))
    sns.kdeplot(ranks_4_6, bw=0.2,  label='4 to 6 songs' )
    sns.kdeplot(ranks_7_10, bw=0.2, label='7 to 10 songs')
    sns.kdeplot(ranks_11_15, bw=0.2, label='11 to 15 songs')
    sns.kdeplot(ranks_16_20, bw=0.2, label='16 to 20 songs')
    sns.kdeplot(ranks_21_more, bw=0.2, label='21 and longer playlists')

    plt.xlim(0,16594)
    plt.title(plot_tile)

    plt.show()

ranking_distribution_plot('results/w2v_results/w2v_results_', 'rankings', 'W2V ranking distribution')
ranking_distribution_plot('results/tf_idf_results/tf_idf_results_', 'rankings', 'TF-idf ranking distribution')
ranking_distribution_plot('results/som_w2v_results/som_w2v_results_', 'rankings', 'SOM with W2V ranking distribution')

