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
h = open('missing_files', 'a')

exception = 0
for i, row in all_songs.iterrows():
    mp3_path = row['path']
    try:
        mp3_file = Path(mp3_path)
        path = mp3_path.split('/')
        wav_file = Path('/Users/m_vys/PycharmProjects/wav_files/' + path[5][:-3] + 'wav')
        used_wav_file = Path('/Users/m_vys/PycharmProjects/used_wav_files/' + path[5][:-3] + 'wav')
        # mp3_file = Path('/Users/m_vys/PycharmProjects/mp3_files/' + path[5])
        if (mp3_file.is_file and (not wav_file.is_file()) and (not used_wav_file.is_file())):
             try:
                sound = AudioSegment.from_mp3(mp3_path)
                sound = sound.set_channels(1)
                beginning = sound[20000:25000]
                middle = sound[60000:65000]
                end = sound[-15000:-10000]
                song = beginning + middle + end
                filename = path[5][:-3]
                new_file_name = "/Users/m_vys/PycharmProjects/wav_files/" + filename + "wav"
                wav_file = Path(new_file_name)
                if not wav_file.exists():
                    print(new_file_name)
                    # play(song)
                    # play(sound)
                    song.export(new_file_name, format="wav")
                else:
                    print(new_file_name + ' is already a wav file ')
                print(i)
             except Exception as e:
                print(e)
                exception = exception+1
             i = i + 1
        elif (not wav_file.is_file()) and (not used_wav_file.is_file()):
            h.write(str(all_songs.at[i, 'artist']) + ';' + str(all_songs.at[i, 'title']) + ';' + "\"" +
                    all_songs.at[i, 'lyrics'] + "\"" + ';' +
                    all_songs.at[i, 'link'] + ';' + "INSERT VIDEO PATH HERE" + ";\n")
                #             ';' + all_songs.at[i, 'link'] + ';' + all_songs.at[i, 'link_on_disc'] + ";\n")
             #    textToSearch = row['artist'] + ' ' + row['title']
             #    query = urllib.parse.quote(textToSearch)
             #    url = "https://www.youtube.com/results?search_query=" + query
             #    response = urllib.request.urlopen(url)
             #    html = response.read()
             #    soup = BeautifulSoup(html, 'html.parser')
             #    # for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
             #    #     print('https://www.youtube.com' + vid['href'])
             #    if ((i % 500) == 0) and (i != 0):
             #        time.sleep(600)
             #    vid = soup.findAll(attrs={'class':'yt-uix-tile-link'})[0]
             #    l = 'https://www.youtube.com' + vid['href']
             #    all_songs.at[i, 'link'] = l
             #    name = str(all_songs.at[i,'title']) + ' - ' + str(all_songs.at[i, 'artist'])
             #    regex = re.compile('[^A-Za-z0-9. -]')
             #    name = regex.sub("", name)
             #    print(name)
             #
             #    ydl_opts = {
             #        'format': 'bestaudio/best',
             #        'postprocessors': [{
             #            'key': 'FFmpegExtractAudio',
             #            'preferredcodec': 'mp3',
             #            'preferredquality': '192',
             #        }],
             #
             #        'outtmpl': '/Users/m_vys/PycharmProjects/missing_mp3_files/' + name + '.%(ext)s'
             #    }
             #    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
             #        ydl.download([l])
             #        info_dict = ydl.extract_info(l, download=False)
             #        print(info_dict.get('filename', None))
             #        all_songs.at[i, 'path'] = '/Users/m_vys/PycharmProjects/missing_mp3_files/' + name + '.mp3'
             #        print(all_songs.at[i, 'path'])
             #    # df.at[i,'link_on_disc']
             #    h.write(str(all_songs.at[i, 'artist']) + ';' + str(all_songs.at[i, 'title']) + ';' + "\"" + all_songs.at[i, 'lyrics'] + "\""
             #            ';' + all_songs.at[i, 'link'] + ';' + all_songs.at[i, 'path'] + ";\n")
             #    print(l)
             # except youtube_dl.utils.DownloadError:
             #    h.write(str(all_songs.at[i, 'artist']) + ';' + str(all_songs.at[i, 'title']) + ';' + "\"" + all_songs.at[
             #        i, 'lyrics'] + "\""
             #                         ';' + all_songs.at[i, 'link'] + ';' + "INSERT VIDEO PATH HERE" + ";\n")
             #    print(i)
             #    print(row['artist'] + ' ' + row['title'])t
        else:
            if wav_file.is_file():
                os.rename('/Users/m_vys/PycharmProjects/wav_files/' + path[5][:-3] + 'wav', '/Users/m_vys/PycharmProjects/used_wav_files/' + path[5][:-3] + 'wav')
    except Exception as e:
        print(e)
        print("!!!EXCEPTION!!! ", row, mp3_path)

h.close()
print(str(exception))
# all_songs.to_csv('downloaded_extra_songs', sep=';', header=None, index=False)