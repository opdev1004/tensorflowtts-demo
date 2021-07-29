import os
import sys
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from scipy.io.wavfile import write

# __location__ = os.path.realpath(os.path.dirname(__file__))

# Tacotron2
taco_model_path = ""
taco_model_config_path = ""
taco_config = AutoConfig.from_pretrained(taco_model_config_path)
tacotron2 = TFAutoModel.from_pretrained(taco_model_path, config=taco_config, name="tacotron2")

# Fastspeech2
fs_model_path = ""
fs_model_config_path = ""
fs_config = AutoConfig.from_pretrained(fs_model_config_path)
fastspeech2 = TFAutoModel.from_pretrained(fs_model_path, config=fs_config, name="fastspeech2")

# Mel generator
mg_model_path = ""
mg_model_config_path = ""
mg_config = AutoConfig.from_pretrained(mg_model_config_path)
mb_melgan = TFAutoModel.from_pretrained(mg_model_path, config=mg_config, name="mb_melgan")

# Used Tacotron2 processor.json
processor = AutoProcessor.from_pretrained("")


def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text)

  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  elif text2mel_name == "FASTSPEECH2":
    mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

  # vocoder part
  if vocoder_name == "MB-MELGAN":
    audio = vocoder_model.inference(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
  else:
    return mel_outputs.numpy(), audio.numpy()

def visualize_attention(alignment_history):
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.set_title(f'Alignment steps')
  im = ax.imshow(
      alignment_history,
      aspect='auto',
      origin='lower',
      interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.show()
  plt.close()

def visualize_mel_spectrogram(mels):
  mels = tf.reshape(mels, [-1, 80]).numpy()
  fig = plt.figure(figsize=(10, 8))
  ax1 = fig.add_subplot(311)
  ax1.set_title(f'Predicted Mel-after-Spectrogram')
  im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
  fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
  plt.show()
  plt.close()

input_text = ""

# setup window for tacotron2 if you want to try
tacotron2.setup_window(win_front=10, win_back=10)

# Tacotron2 + MB-MELGAN
mels, alignment_history, audio = do_synthesis(input_text, tacotron2, mb_melgan, "TACOTRON", "MB-MELGAN")
write("OUTPUT PATH", 22050, audio)
print("Tacotron2 + MB-MELGAN is done!")
# FastSpeech2 + MB-MELGAN
mels, audio = do_synthesis(input_text, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")
write("OUTPUT PATH", 22050, audio)
print("FastSpeech2 + MB-MELGAN is done!")

input("Press Enter to continue...")