import math, pandas
from Evaluation import Evaluation

class User():

    def __init(self, userID):
        self.userID = userID

    def calculate_NDCG(self, usr):
        DCG_score  = 0
        for i in range(100):
            if self.first_100_list[i] in self.songs_in_first_100:
                DCG_score += 1 / math.log(i+1, base=2)

        ideal_DCG = 0
        for i in range(len(usr.test_list)):
            ideal_DCG += 1/math.log(i+1, base=2)

        return DCG_score/ideal_DCG





evaluation = Evaluation('w2v_distances.npy', 'useful_playlists_bigger_than_10','useful_songs')
results = pandas.DataFrame(columns=['playlist_lenght','test_list_lenght', 'number_of_matches', 'match_ranking', 'recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
i = 0
for user in evaluation.users:
    user_results = evaluation.eval_playlist(user)
    print('user ', i, " out of ", len(evaluation.users))
    print(user_results)
    temp_frame = pandas.DataFrame([user_results], columns=['playlist_lenght','test_list_lenght', 'number_of_matches','match_ranking','recall_at_10', 'recall_at_50', 'recall_at_100', 'nDGC'])
    results = results.append(temp_frame)
    i = i+1

print(results.shape)
results.to_csv('w2v_resutls', sep=';', header=False, index=False)