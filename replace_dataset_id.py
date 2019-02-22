import pandas, difflib

ids = pandas.read_csv("/Users/m_vys/Documents/matfyz/rocnikac/data/msd_to_echoNest_id.txt", delimiter="<SEP>", names=['MSD_id','EchoNest_id', 'artist', 'title'])
songs = pandas.read_csv("/Users/m_vys/Documents/matfyz/rocnikac/data/train_triplets.txt", delimiter='\t', names=['user_id', 'EchoNest_id', 'play_count'])
lyrics = pandas.read_csv("/Users/m_vys/Documents/matfyz/rocnikac/data/songdata.csv", delimiter=',', quotechar="\"", usecols=[0,1,3], names=['artist', 'title', 'lyrics'])
matches = open('matches_2.txt', 'a', encoding='utf-8')
frame1 = (pandas.merge(ids, songs, on='EchoNest_id')).drop_duplicates(subset='EchoNest_id')
frame1 = frame1.reset_index(drop=True)
lyrics_index = 0
frame1_index = 0
for index_a, (artist_a, title_a) in enumerate(lyrics[['artist', 'title']].values):
    for index, (artist, title) in enumerate(frame1[['artist', 'title']].values):
        artist_difference = difflib.SequenceMatcher(None, str(artist).lower(), str(artist_a).lower()).ratio()
        song_difference = difflib.SequenceMatcher(None, str(title).lower(), str(title_a).lower()).ratio()
        if artist_difference > 0.75 and song_difference > 0.75:
            matches.write((str(artist_difference) + " -- " + str(song_difference) + "\n"))
            s = " first string: " + artist_a + " - " + title_a + " second string: " + artist + " - " + title + "\n"
            matches.write(s)
            print(s)
            frame1.at[frame1_index, 'lyrics'] = lyrics.at[lyrics_index, 'lyrics']
        frame1_index += 1
    lyrics_index += 1
    frame1_index = 0


frame1 = frame1[frame1.lyrics != 'nan']
frame1.to_csv('unique_songs.txt')
matches.close()