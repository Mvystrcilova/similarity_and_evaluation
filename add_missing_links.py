import pandas

all_songs = pandas.read_csv('all_songs_with_file_paths',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'], error_bad_lines=False, warn_bad_lines=True, engine='python')

missing_songs = pandas.read_csv('missing_wav_files',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'new_path'], engine='python')

# missing_songs = missing_songs.append(missing_songs_naopak)
# missing_songs = missing_songs.drop_duplicates(subset=['artist', 'title'], keep="last")

for i, row in missing_songs.iterrows():
    selected_song_index = all_songs[(all_songs['artist'] == row['artist'] )& (all_songs['title'] == row['title'])].index
    if len(selected_song_index.values) > 0:
        print(selected_song_index.values[0])
        index = selected_song_index.values[0]
        print(row['artist']," ", row['title'])
        all_songs.at[index, 'path'] = row['new_path']
        print(all_songs.loc[index])
    else:
        print(row)

all_songs.to_csv('all_songs_with_file_paths', sep=';', header=None, index=False)