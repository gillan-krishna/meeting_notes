# enter audio file path
path = 'audio_files/fbd_meeting.m4a'

import time
start = time.time()

import os
from setEnv import num_speakers, model_size, language
import subprocess
import whisper
import contextlib
import wave
import datetime
from pyannote.audio import Audio
from pyannote.core import Segment
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

def segment_embedding(segment):
  start = segment['start']
  end = min(duration, segment['end'])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def time(secs):
  return datetime.timedelta(seconds=round(secs))

#audio file path

#convert to .wav format
if path[-3:] != '.wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'

#transcription
model_name = model_size
if language == 'English ' and model_size != 'large':
  model_name += '.en'
model = whisper.load_model(model_size)
result = model.transcribe(path)
segments = result['segments']


with contextlib.closing(wave.open(path, 'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames/float(rate)

audio = Audio()

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

#identify speakers
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_

with contextlib.closing(wave.open('audio.wav', 'r')) as f:
    # Read the entire file into a numpy array
    audio = np.frombuffer(f.readframes(-1), np.int16)
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

input('Identify Speakers, enter 0 to play another clip')
speaker = ['0']*num_speakers
input_val = '0'

for i in range(num_speakers):
  sp = [x for x in segments if x['speaker'] == f'SPEAKER {i+1}']
  j =-1
  while input_val=='0':
    j += 1
    seg = sp[j]
    start_time = seg['start']
    end_time = seg['end']
    start_sample = int(start_time * f.getframerate())
    end_sample = int(end_time * f.getframerate())
    cropped_audio = audio[start_sample:end_sample]

    # Play the cropped audio
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    sound = AudioSegment.from_file(path, format="wav")
    splice = sound[start_ms:end_ms]
    play(splice)

    # Get speaker name
    input_val = input(f'Identify Speaker {i+1}:')
    if input_val!=0:
      speaker[i] = input_val
  input_val = '0'

os.remove('audio.wav')


print('Identified speakers are', speaker)
for i in range(len(segments)):
  segments[i]["speaker"] = speaker[labels[i]]

#write transcript
with open("transcript.txt", "w") as f:
  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      f.write("\n" + segment["speaker"] + ': ')
    f.write(segment["text"][1:] + ' ')
# print(open('transcript.txt', 'r').read())

print('Runtime: %f', time.time()-start)