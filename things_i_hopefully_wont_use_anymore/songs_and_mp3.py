from __future__ import unicode_literals
import pandas
import string
import urllib.request
from bs4 import BeautifulSoup
import time
import re
import youtube_dl
# from rocnikac.settings import MP3FILES_DIR
import os

songs_1 = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_2',
                          sep=";", quotechar='"',  names=['title', 'artist', 'lyrics', 'link', 'path'],
                          index_col=False )
songs_2 = pandas.read_csv('~/PycharmProjects/song_files/fixed_songs_for_database',
                          sep=';',quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
# songs_2_copy = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_2 copy',
#                                sep=';',quotechar='"',
#                                names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
songs_3 = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_3',
                          sep=";",quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
songs_4 = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_4',
                          sep=";",quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
songs_5 = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_5',
                          sep=";",quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
songs_6 = pandas.read_csv('~/PycharmProjects/song_files/songs_for_database_6',
                          sep=";",quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)
songs_7 = pandas.read_csv('songs_for_database_7',
                          sep=";",quotechar='"',
                          names=['title', 'artist', 'lyrics', 'link', 'path'], index_col=False)

songs_1 = songs_1.append(songs_2)
# songs_1 = songs_1.append(songs_2_copy)
songs_1 = songs_1.append(songs_3)
songs_1 = songs_1.append(songs_4)
songs_1 = songs_1.append(songs_5)
songs_1 = songs_1.append(songs_6)
songs_1 = songs_1.append(songs_7)

print(songs_1.shape)
songs_1 = songs_1.drop_duplicates()
print(songs_1.shape)

df = pandas.read_csv('~/Documents/matfyz/rocnikac/data/songs_with_lyrics', sep=';', quotechar='"',
                     names=['artist', 'title', 'lyrics'], engine='python',
                     error_bad_lines=False, usecols=[2, 3, 4])
df = df.drop_duplicates(subset=['artist', 'title'])

common = df.merge(songs_1, how='inner', on=['artist', 'title'])
common = common.drop_duplicates(subset=['artist', 'title'])

print(common.shape)

common.to_csv('all_songs_with_file_paths', sep=';', index=False, header=None)
# h = open('songs_for_database_7', 'a', encoding='utf8')
#
# for i, row in missing.iterrows():
#
#     if i > 0:
#         try:
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
#             vid = soup.findAll(attrs={'class':'yt-uix-tile-link'})[0]
#             l = 'https://www.youtube.com' + vid['href']
#             missing.at[i, 'link'] = l
#             name = str(missing.at[i,'title']) + '-' + str(missing.at[i, 'artist']) + ".mp3"
#             ydl_opts = {
#                 'format': 'bestaudio/best',
#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': 'mp3',
#                     'preferredquality': '192',
#                 }],
#                 'outtmpl': 'Downloads/' + "%(title)s.%(ext)s"
#             }
#             with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([l])
#                 info_dict = ydl.extract_info(l, download=False)
#                 missing.at[i, 'link_on_disc'] = '~/Downloads/' + info_dict.get('title', None) + ".mp3"
#             # df.at[i,'link_on_disc']
#             h.write(str(missing.at[i, 'title']) + ';' + str(missing.at[i, 'artist']) + ';' + "\"" + missing.at[i, 'lyrics_x'] + "\""
#                     ';' + missing.at[i, 'link'] + ';' + missing.at[i, 'link_on_disc'] + ";\n")
#             print(l)
#
#         except youtube_dl.utils.DownloadError:
#             h.write(str(missing.at[i, 'title']) + ';' + str(missing.at[i, 'artist']) + ';' + "\"" + missing.at[
#                 i, 'lyrics_x'] + "\""
#                                  ';' + missing.at[i, 'link'] + ';' + "INSERT VIDEO PATH HERE" + ";\n")
#             print(i)
#             print(row['artist'] + ' ' + row['title'])