from minisom import MiniSom
import pandas, numpy, math
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

from pylab import bone, pcolor, colorbar, plot, show
# test_input = pandas.DataFrame([[2,3,9,3,0,7,1],
#               [6,9,2,0,2,1,2],
#               [3,9,2,8,0,5,2],
#               [1,4,9,23,95048, 5, 2],
#               [2903, 848, 9395,9330,3839,3,909],
#               [4,3,2,8,0,2,8],
#               [9,389239, 8493,48458,3929,283,9393],
#               [4,2,9,0,1,1,2],
#               [3,8,0,0,2,1,5]])
# print(test_input.shape)
#
# som = MiniSom(15,15,len(test_input.iloc[0]))
# scaler = preprocessing.MinMaxScaler((0,1))
# test_input = scaler.fit_transform(test_input)
#
# som.random_weights_init(test_input)
# som.train_random(test_input,100)
#
# for i,vec in enumerate(test_input):
#     winning_position = som.winner(vec)
#     plt.text(winning_position[0], winning_position[1], i)
#
# plt.xticks(range(15))
# plt.yticks(range(15))
# plt.grid()
# plt.xlim([0,15])
# plt.ylim([0,15])
# plt.plot()
# plt.show()

som_vectors = pandas.read_csv('SOM_W2V', sep=';',
                              names=['somethingWeird', 'songId',
                                     'title', 'artist', 'som_representation'],
                              usecols=[2,3,4])
print(som_vectors.shape)

# som_distances = [[0 for x in range(len(som_vectors))] for y in range(len(som_vectors))]
plt.figure(figsize=(30,30))

for i, song_1 in som_vectors.iterrows():
    # if i < 1000:
        som_repr_1 = numpy.fromstring(song_1['som_representation'].replace("(","").replace(')',''), sep=',')
        plt.text(som_repr_1[0]+numpy.random.rand()*20, som_repr_1[1]+numpy.random.rand()*20, song_1['title'])

plt.grid()
plt.xlim([0, int(3 * math.sqrt(len(som_vectors)))])
plt.ylim([0, int(3 * math.sqrt(len(som_vectors)))])
plt.plot()
plt.show()



