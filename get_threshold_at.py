import numpy


def get_threshold_at(position, distance_matrix):
    distances = numpy.load(distance_matrix)
    distances = distances.flatten()
    # distances[distances > 0.99] = 0
    distances.sort()
    return distances[-(position + 16594)]

# print('Value at position 829700')
# print('pca_mel_5715_50', get_threshold_at(829700, '/mnt/0/pca_mel_distances_5717.npy'))
# print('pca_mel_320_50', get_threshold_at(829700, '/mnt/0/pca_mel_distances_5717.npy'))
#
# print('pca_spec_1106_50', get_threshold_at(829700, '/mnt/0/pca_spec_distances_1106.npy'))
# print('pca_spec_320_50', get_threshold_at(829700, '/mnt/0/short_pca_spec_distances.npy'))
#
# print('lstm_mel_50', get_threshold_at(829700, '/mnt/0/lstm_mel_distances_5712.npy'))
# print('gru_mel_50', get_threshold_at(829700, '/mnt/0/gru_mel_distances_5712.npy'))
#
# print('lstm_spec_20400_50', get_threshold_at(829700, '/mnt/0/distances/final_lstm_spec_distances.npy'))
# print('gru_spec_20400_50', get_threshold_at(829700, '/mnt/0/distances/final_lstm_spec_distances.npy'))
#
# print('lstm_spec_5712_50', get_threshold_at(829700, '/mnt/0/short_LSTM_spec_distances.npy'))
# print('gru_spec_5712_50', get_threshold_at(829700, '/mnt/0/short_GRU_spec_distances.npy'))
#
# print('lstm_mfcc_50', get_threshold_at(829700, '/mnt/0/lstm_mfcc_distances.npy'))
# print('gru_mfcc_50', get_threshold_at(829700, '/mnt/0/gru_mfcc_distances.npy'))
#
# print('tf_idf_50', get_threshold_at(829700, '/mnt/0/distances/tf_idf_distances.npy'))
#
# print('som_w2v_50', get_threshold_at(829700, '/mnt/0/distances/SOM_W2V_batch_5g5i133188_distances.npy'))
# print('pca_tf_idf_50', get_threshold_at(829700, '/mnt/0/pca_tf_idf_distances.npy'))
#
# print('w2v_50', get_threshold_at(829700, '/mnt/0/distances/w2v_distances.npy'))
#
# print('Value at position 16594000')
# print('pca_mel_5715_1000', get_threshold_at(16594000, '/mnt/0/pca_mel_distances_5717.npy'))
# print('pca_mel_320_1000', get_threshold_at(16594000, '/mnt/0/pca_mel_distances_5717.npy'))
#
# print('pca_spec_1106_1000', get_threshold_at(16594000, '/mnt/0/pca_spec_distances_1106.npy'))
# print('pca_spec_320_1000', get_threshold_at(16594000, '/mnt/0/short_pca_spec_distances.npy'))
#
# print('lstm_mel_1000', get_threshold_at(16594000, '/mnt/0/lstm_mel_distances_5712.npy'))
# print('gru_mel_1000', get_threshold_at(16594000, '/mnt/0/gru_mel_distances_5712.npy'))
#
# print('lstm_spec_20400_1000', get_threshold_at(16594000, '/mnt/0/distances/final_lstm_spec_distances.npy'))
# print('gru_spec_20400_1000', get_threshold_at(16594000, '/mnt/0/distances/final_lstm_spec_distances.npy'))
#
# print('lstm_spec_5712_1000', get_threshold_at(16594000, '/mnt/0/short_LSTM_spec_distances.npy'))
# print('gru_spec_5712_1000', get_threshold_at(16594000, '/mnt/0/short_GRU_spec_distances.npy'))
#
# print('lstm_mfcc_1000', get_threshold_at(16594000, '/mnt/0/lstm_mfcc_distances.npy'))
# print('gru_mfcc_1000', get_threshold_at(16594000, '/mnt/0/gru_mfcc_distances.npy'))
#
# print('tf_idf_1000', get_threshold_at(16594000, '/mnt/0/distances/tf_idf_distances.npy'))
#
# print('som_w2v_1000', get_threshold_at(16594000, '/mnt/0/distances/SOM_W2V_batch_5g5i133188_distances.npy'))
# print('pca_tf_idf_1000', get_threshold_at(16594000, '/mnt/0/pca_tf_idf_distances.npy'))
#
# print('w2v_1000', get_threshold_at(16594000, '/mnt/0/distances/w2v_distances.npy'))
#
# print('Value at last position')
# print('pca_mel_5715_last', get_threshold_at((-1*16593), '/mnt/0/pca_mel_distances_5717.npy'))
# print('pca_mel_320_last', get_threshold_at((-1*16593), '/mnt/0/pca_mel_distances_5717.npy'))
#
# print('pca_spec_1106_last', get_threshold_at((-1*16593), '/mnt/0/pca_spec_distances_1106.npy'))
# print('pca_spec_320_last', get_threshold_at((-1*16593), '/mnt/0/short_pca_spec_distances.npy'))
#
# print('lstm_mel_last', get_threshold_at((-1*16593), '/mnt/0/lstm_mel_distances_5712.npy'))
# print('gru_mel_last', get_threshold_at((-1*16593), '/mnt/0/gru_mel_distances_5712.npy'))
#
# print('lstm_spec_20400_last', get_threshold_at((-1*16593), '/mnt/0/distances/final_lstm_spec_distances.npy'))
# print('gru_spec_20400_last', get_threshold_at((-1*16593), '/mnt/0/distances/final_lstm_spec_distances.npy'))
#
# print('lstm_spec_5712_last', get_threshold_at((-1*16593), '/mnt/0/short_LSTM_spec_distances.npy'))
# print('gru_spec_5712_last', get_threshold_at((-1*16593), '/mnt/0/short_GRU_spec_distances.npy'))
#
# print('lstm_mfcc_last', get_threshold_at((-1*16593), '/mnt/0/lstm_mfcc_distances.npy'))
# print('gru_mfcc_last', get_threshold_at((-1*16593), '/mnt/0/gru_mfcc_distances.npy'))
#
# print('tf_idf_last', get_threshold_at((-1*16593), '/mnt/0/distances/tf_idf_distances.npy'))
#
# print('som_w2v_last', get_threshold_at((-1*16593), '/mnt/0/distances/SOM_W2V_batch_5g5i133188_distances.npy'))
# print('pca_tf_idf_last', get_threshold_at((-1*16593), '/mnt/0/pca_tf_idf_distances.npy'))
#
# print('w2v_last', get_threshold_at((-1*16593), '/mnt/0/distances/w2v_distances.npy'))

# print(get_threshold_at((16594000), 'new_distances/new_gru_mel_distances_40.npy'))
print(get_threshold_at((829700), 'new_distances/new_gru_mel_distances_40.npy'))