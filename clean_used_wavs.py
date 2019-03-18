from __future__ import unicode_literals

import pandas, os
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play



import pandas
import string
import urllib.request
from bs4 import BeautifulSoup
import time
import re, os
import youtube_dl
from pydub import AudioSegment
from pathlib import Path
from replace_empty_wavs import download_song_at_i

def add_empty_mp3():
    all_songs = pandas.read_csv('not_empty_songs',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])
    h = open('missing_wavs', 'a')

    directory = os.fsencode('/Users/m_vys/PycharmProjects/used_wav_files/')
    double_paths = pandas.DataFrame()
    for file in os.listdir(directory):
        try:
            new_file = ''
            filename = os.fsdecode(file)
            mp3_file_name = '/Users/m_vys/PycharmProjects/mp3_files/' + filename[:-3] + 'mp3'
            ds = all_songs[all_songs['path'] == mp3_file_name]
            index = all_songs[all_songs['path'] == mp3_file_name].index.values[0]
            textToSearch = ds.at[index,'artist'] + ' ' + ds.at[index,'title']
            query = urllib.parse.quote(textToSearch)
            url = "https://www.youtube.com/results?search_query=" + query
            response = urllib.request.urlopen(url)
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            # for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
            #     print('https://www.youtube.com' + vid['href'])
            results = soup.findAll(attrs={'class': 'yt-uix-tile-link'})
            position = 0
            length = 0
            while length == 0:
                length, current_file = download_song_at_i(position, results, all_songs, index)
                position = position + 1
                new_file = current_file

            all_songs.at[index, 'path'] = new_file
        except Exception as e:
            print(e)

    all_songs.to_csv('not_empty_songs', sep=';', header=None, index=False)

def clean_wav_files(directory, all_songs):
    for file in os.listdir(directory):
        try:
            new_file = ''
            filename = os.fsdecode(file)
            mp3_file_name = '/Users/m_vys/PycharmProjects/mp3_files/' + filename[:-3] + 'mp3'
            wav_file_name = '/Users/m_vys/PycharmProjects/used_wav_files/' + filename[:-3] + 'wav'
            ds = all_songs[all_songs['path'] == mp3_file_name]
            # index = all_songs[all_songs['path'] == mp3_file_name].index.values[0]

            if ds.size > 0:
                os.rename(wav_file_name, '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[:-3] + 'wav')
            if ds.size > 5:
                print(filename, 'File has two entries in dataframe', str(ds.size))

            if ds.size == 0:
                print(filename, "file not in not_empty_songs")
        except Exception as e:
            print(e)



# exception = 0
# empty_indexes = []
# broken_files = []
# for i, row in all_songs.iterrows():
#     mp3_path = row['path']
#
#     mp3_file = Path(mp3_path)
#     path = mp3_path.split('/')
#     used_wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + path[5][:-3] + 'wav'
#     if not os.path.exists(used_wav_file):
#         # print(row)
#         h.write(str(i) + ' '+ row['title'] + ' ' + row['artist'])
#         empty_indexes.append(i)
#     else:
#         try:
#             song = AudioSegment.from_wav(used_wav_file)
#         except Exception as e:
#             print("Following song could not be played")
#             print(str(e), str(i))
#             os.rename('/Users/m_vys/PycharmProjects/cleaned_wav_files/' + path[5][:-3] + 'wav', '/Users/m_vys/PycharmProjects/broken_wav_files/' + path[5][:-3] + 'wav')
#             exception = exception + 1
#             broken_files.append(i)
#
# print(exception)
# print(empty_indexes)
# h.close()

all_songs = pandas.read_csv('not_empty_songs',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])
clean_wav_files('/Users/m_vys/PycharmProjects/used_wav_files/', all_songs)