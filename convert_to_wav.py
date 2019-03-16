import os
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path

directory = os.fsencode('/Users/m_vys/PycharmProjects/missing_mp3_files')
i = 0
for file in os.listdir(directory):
     pathname = '/Users/m_vys/PycharmProjects/missing_mp3_files/' + os.fsdecode(file)
     filename = os.fsdecode(file)
     try:
          sound = AudioSegment.from_mp3(pathname)
          sound = sound.set_channels(1)
          beginning = sound[20000:25000]
          middle = sound[60000:65000]
          end = sound[-15000:-10000]
          song = beginning + middle + end
          filename = filename[:-3]
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
          os.rename(pathname, '/Users/m_vys/PycharmProjects/mp3_files/' + os.fsdecode(file) )
     except:
          print(pathname)
     i = i + 1

