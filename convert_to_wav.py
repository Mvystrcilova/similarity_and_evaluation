import os
from pydub import AudioSegment
from pydub.playback import play

directory = os.fsencode('/Volumes/NO NAME/Miska/mp3_files')
i = 0
for file in os.listdir(directory):
     if i > 29:
          pathname = '/Volumes/NO NAME/Miska/mp3_files/' + os.fsdecode(file)
          filename = os.fsdecode(file)
          try:
               sound = AudioSegment.from_mp3(pathname)
               sound = sound.set_channels(1)
               beginning = sound[20000:25000]
               middle = sound[60000:65000]
               end = sound[-15000:-10000]
               song = beginning + middle + end
               filename = filename[:-3]
               new_file_name = "/Volumes/NO NAME/Miska/wav_files/" + filename + "wav"
               print(new_file_name)
               # play(song)
               # play(sound)
               song.export(new_file_name, format="wav")
               print(i)
          except:
               print(pathname)
     i = i + 1

