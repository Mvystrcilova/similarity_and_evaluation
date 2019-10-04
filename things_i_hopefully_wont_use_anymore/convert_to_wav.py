import os
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path

def get_15():
    directory = os.fsencode('/Users/m_vys/PycharmProjects/empty_mp3_files')
    i = 0
    for file in os.listdir(directory):
        pathname = '/Users/m_vys/PycharmProjects/empty_mp3_files/' + os.fsdecode(file)
        filename = os.fsdecode(file)
        try:
            sound = AudioSegment.from_mp3(pathname)
            if len(sound) < 65000:
                sound = sound.set_channels(1)
                beginning = sound[10000:15000]
                middle = sound[30000:35000]
                end = sound[-10000:-5000]

            else:
                sound = sound.set_channels(1)
                beginning = sound[20000:25000]
                middle = sound[60000:65000]
                end = sound[-15000:-10000]
            song = beginning + middle + end
            play(song)
            filename = filename[:-3]
            new_file_name = "/Users/m_vys/PycharmProjects/cleaned_wav_files/" + filename + "wav"
            wav_file = Path(new_file_name)

            song.export(new_file_name, format="wav")

        except:
            print(pathname)
        i = i + 1

def create_song_segment(path):
    try:
        sound = AudioSegment.from_mp3(path)
        if len(sound) < 70000:
            sound = sound.set_channels(1)
            beginning = sound[10000:20000]
            middle = sound[30000:40000]
            end = sound[-20000:-10000]

        else:
            sound = sound.set_channels(1)
            beginning = sound[20000:30000]
            middle = sound[60000:70000]
            end = sound[-20000:-10000]
        song = beginning + middle + end
        # play(song)
        filename = path.split('/')[1][:-3]
        new_file_name = "wav_files/" + filename + "wav"
        # wav_file = Path(new_file_name)

        song.export(new_file_name, format="wav")

        return new_file_name

    except:
        print(path)

def get_mono_and_crop_to_15():
    directory = os.fsencode('/Users/m_vys/PycharmProjects/cleaned_wav_files')
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename
        try:
            sound = AudioSegment.from_wav(filename)
            if len(sound) > 15000:

                beginning = sound[20000:25000]
                middle = sound[60000:65000]
                end = sound[-15000:-10000]
                sound = beginning + middle + end

            sound = sound.set_channels(1)
            # filename = "/Users/m_vys/PycharmProjects/cleaned_wav_files/" + filename
            # wav_file = Path(new_file_name)

            sound.export(filename, format="wav")

        except Exception as e:
            print(filename, e)

# get_15()