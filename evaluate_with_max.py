# from Evaluation import evaluate
import numpy, pandas
# evaluate('new_distances/bert_distances.npy', 0.8402684, True)
# evaluate(
#     '/Users/m_vys/Documents/matfyz/bakalar/bakalarka/songRecommender_project/songRecommender_project/distances/pca_melspectrogram_distances.npy',
#     0.383, True)
# evaluate('distances/gru_mel_distances_5712.npy', 0.363459, True)
#
# evaluate('/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_distances/gru_mfcc_distances_30_64.npy', 0.9926,
#          True)
# evaluate('/Users/m_vys/PycharmProjects/similarity_and_evaluation/distances/w2v_distances.npy', 0.9567, True)
# evaluate('new_distances/tag_based_BOW.npy', 0.19245008972987532, True)
# evaluate('new_distances/tag_based_TF-IDF.npy', 0.0005135140607123168, True)
# evaluate('distances/pca_tf_idf_distances.npy', 0.1899072240495832, True)

from evaluation_results import get_results

# get_results('new_results/tag_based_TF-IDF_max/tag_based_TF-idf_max_')
# get_results('new_results/tag_based_BOW_max/tag_based_BOW_max_')
# get_results(
#     '/Users/m_vys/PycharmProjects/similarity_and_evaluation/new_results/gru_mfcc_results_30_64_max/gru_mfcc_distances_30_64_max_')
# get_results(
#     '/Users/m_vys/Documents/matfyz/bakalar/bakalarka/songRecommender_project/songRecommender_project/results/pca_melspectrogram_rpesults_max/pca_melspectrogram_distances_max_')
# get_results('/Users/m_vys/Documents/matfyz/bakalar/bakalarka/songRecommender_project/songRecommender_project/results/pca_melspectrogram_results_max/pca_melspectrogram_distances_max_')
# get_results('results/gru_mel_results_5712_max/gru_mel_distances_5712_max_')
# get_results('results/w2v_results_max/w2v_distances_max_')
# get_results('/Users/m_vys/PycharmProjects/similarity_and_evaluation/results/pca_tf_idf_results_max/pca_tf_idf_distances_max_')

def compare_recs_similarities(same_10, same_50):
    print(same_10[:-15])
    array_10 = numpy.load(same_10)
    mean = numpy.mean(array_10)
    median = numpy.median(array_10)
    print('mean', mean)
    print('median', median)

    print(same_50[:-15])
    array_50 = numpy.load(same_50)
    mean = numpy.mean(array_50)
    median = numpy.median(array_50)
    print('mean', mean)
    print('median', median)

def compare_correct_recs_similarities(same_df_file):
    print(same_df_file)
    df = pandas.read_csv(same_df_file,  sep='\t', header=0, index_col=0)
    sums = df.sum(axis=0)
    print(sums)

# compare_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_gru_mel_same_in_10.npy',
#                           '/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_gru_mel_same_in_50.npy')
#
# compare_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_mel_same_in_10.npy',
#                           '/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_mel_same_in_50.npy')
#
# compare_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_tf-idf_same_in_10.npy',
#                           '/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_tf-idf_same_in_50.npy')
#
# compare_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_w2v_lyrics_same_in_10.npy',
#                           '/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_w2v_lyrics_same_in_50.npy')
#
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_w2v_lyrics_same_correct_recs.csv')
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_tf-idf_same_correct_recs.csv')
#
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_gru_mel_same_correct_recs.csv')
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_pca_mel_same_correct_recs.csv')
#
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_gru_mfcc_same_correct_recs.csv')
# compare_correct_recs_similarities('/Users/m_vys/PycharmProjects/similarity_and_evaluation/similarity_of_recommendations/tags_tf-idf_bert_same_correct_recs.csv')

# compare_correct_recs_similarities('similarity_of_recommendations/tags_tf-idf_w2v_lyrics_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/tags_tf-idf_pca_tf-idf_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/tags_tf-idf_gru_mel_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/tags_tf-idf_gru_mfcc_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/W2V_gru_mfcc_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/W2V_pca_tf-idf_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/W2V_gru_mel_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/pca_tf-idf_gru_mel_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/pca_tf-idf_gru_mfcc_same_correct_recs.csv')
# compare_correct_recs_similarities('similarity_of_recommendations/gru_mfcc_gru_mel_same_correct_recs.csv')