from Evaluation import Evaluation
import pandas

for j in range(4,5):
    evaluation = Evaluation('mnt/0/gru_mfcc_distances.npy', 'mnt/0/useful_playlists', 'mnt/0/useful_songs', True)
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
    filename = 'mnt/0/results/all_gru_mfcc_results/gru_mfcc' + str(j+1)
    print(results.shape)
    results.to_csv(filename, sep=';', header=False, index=False)

for j in range(5):
    evaluation = Evaluation('mnt/0/gru_mfcc_distances.npy', 'mnt/0/useful_playlists', 'mnt/0/useful_songs', False)
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
    filename = 'mnt/0/results/gru_mfcc_results/gru_mfcc_' + str(j+1)
    print(results.shape)
    results.to_csv(filename, sep=';', header=False, index=False)