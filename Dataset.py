from sklearn.metrics.pairwise import cosine_similarity



class Dataset:

    def __init__(self, songs, embeddings, distances):
        self.songs = songs
        self.embeddings = embeddings
        self.distances = distances
