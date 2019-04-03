from __future__ import unicode_literals

import pandas
import string
import urllib.request
from bs4 import BeautifulSoup
import time
import re, os
import youtube_dl
from pydub import AudioSegment
from pathlib import Path
import re
all_songs = pandas.read_csv('downloaded_extra_songs',sep=';', header=None, index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])

def download_song_at_i(position, results, song_frame, index):
    video = results[position]
    l = 'https://www.youtube.com' + video['href']
    name = str(song_frame.at[index, 'artist']) + ' - ' + str(all_songs.at[index, 'title'])
    regex = re.compile('[^A-Za-z0-9. -]')
    song_frame.at[index, 'link'] = l
    name = regex.sub("", name)
    print(name)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],

        'outtmpl': '/Users/m_vys/PycharmProjects/empty_mp3_files/' + name + '.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(l, download=False)
        if 'entries' in info_dict:
            l = info_dict['entries'][0]['url']
        else:
            l = info_dict['url']

        ydl.download([l])
        new_mp3_file = '/Users/m_vys/PycharmProjects/empty_mp3_files/' + name + '.mp3'
        song = AudioSegment.from_mp3(new_mp3_file)

        print(info_dict.get('filename', None))
        song_frame.at[index, 'path'] = '/Users/m_vys/PycharmProjects/mp3_files/' + name + '.mp3'


        return len(song), new_mp3_file


# for i, row in all_songs.iterrows():
#     mp3_path = row['path']
#     mp3_file = Path(mp3_path)
#     path = mp3_path.split('/')
#     empty_wav_file = Path('/Users/m_vys/PycharmProjects/empty_wavs/' + path[5][:-3] + 'wav')
#     try:
#         song = AudioSegment.from_wav(empty_wav_file)
#         if len(song) == 0:
#             textToSearch = row['artist'] + ' ' + row['title']
#             query = urllib.parse.quote(textToSearch)
#             url = "https://www.youtube.com/results?search_query=" + query
#             response = urllib.request.urlopen(url)
#             html = response.read()
#             soup = BeautifulSoup(html, 'html.parser')
#             # for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
#             #     print('https://www.youtube.com' + vid['href'])
#             if ((i % 500) == 0) and (i != 0):
#                 time.sleep(600)
#             results = soup.findAll(attrs={'class': 'yt-uix-tile-link'})
#             vid = results[0]
#             # all_songs.at[i, 'link'] = l
#             name = str(all_songs.at[i, 'title']) + ' - ' + str(all_songs.at[i, 'artist'])
#             regex = re.compile('[^A-Za-z0-9. -]')
#             name = regex.sub("", name)
#             print(name)
#
#             ydl_opts = {
#                 'format': 'bestaudio/best',
#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': 'mp3',
#                     'preferredquality': '192',
#                 }],
#
#                 'outtmpl': '/Users/m_vys/PycharmProjects/empty_mp3_files/' + name + '.%(ext)s'
#             }
#             with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#                 info_dict = ydl.extract_info(l, download=False)
#                 if 'entries' in info_dict:
#                     l = info_dict['entries'][0]['url']
#                 else:
#                     l = info_dict['url']
#
#                 ydl.download([l])
#                 new_mp3_file = '/Users/m_vys/PycharmProjects/missing_mp3_files/' + name + '.mp3'
#
#                 position = 1
#                 song = AudioSegment.from_mp3(new_mp3_file)
#                 if len(song) == 0:
#                     length = download_song_at_i(position, results, all_songs)[0]
#
#                 position = position + 1
#                 current_path = ''
#                 while (length == 0):
#                     length, current_path = download_song_at_i(position, results, all_songs)
#                     position = position + 1
#
#                 print(info_dict.get('filename', None))
#                 all_songs.at[i, 'path'] = current_path
#                 print(all_songs.at[i, 'path'])
#
#             # df.at[i,'link_on_disc']
#
#             print(l)
#     except FileNotFoundError:
#         print(row)



