import pandas
from pathlib import Path
all_songs = pandas.read_csv('all_songs_with_file_paths_without_apostrophes_and_',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])

for i, row in all_songs.iterrows():
    mp3_path = row['path']
    try:
        mp3_file = Path(mp3_path)
        path = mp3_path.split('/')
        song_name = path[5]
        # if len(path) > 6:
        #     for j in range(6,len(path)):
        #         song_name = song_name + "/" + path[j]
        #     song_name = song_name.replace('/', '_')
        mp3_path = '/Users/m_vys/PycharmProjects/mp3_files/' + song_name
        wav_file = Path('/Users/m_vys/PycharmProjects/wav_files/' + song_name[:-3] + 'wav')
        used_wav_file = Path('/Users/m_vys/PycharmProjects/used_wav_files/' + song_name[:-3] + 'wav')
        # mp3_file = Path('/Users/m_vys/PycharmProjects/mp3_files/' + path[5])
        all_songs.at[i,'path'] = mp3_path.replace("?", "")
        print(mp3_path)
    except Exception as e:
        print(e)

all_songs.to_csv('all_songs_with_file_paths_without_apostrophes_and_and_questionmarks', sep=';', header=None, index=False)