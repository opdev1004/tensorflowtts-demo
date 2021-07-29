# Tensorflow TTS Demo
This is the demonstration of using Tensorflow TTS without Collab or Notebook.

Python 3.8.x is recomended.

## Demonstration PC Spec:
| Parts | PC Spec |
| - | - |
OS | Windows 10
Python | 3.8.10
CPU | i7-4710MQ @ 2.50GHz
Memory | DDR3 16GB
GPU | Released in 2013ish.

Read details from Tensorflow TTS Repo: https://github.com/TensorSpeech/TensorFlowTTS

My code is nothing much different from the Collab version of Tensorflow TTS.

# Installation &Execution

```
pip install TensorFlowTTS ipython scipy git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate h5py

```

Download pretrained models from: https://huggingface.co/tensorspeech
They are maintained by Tensorflow TTS. Make sure download model.h5, config.yml and processor.json. You would want to try all the models and you have to use combination like Tacotron2 + MB-MELGAN or FastSpeech2 + MB-MELGAN.

Edit the multiple paths inside of app.py so you can load pretrained models and export the wav files. All the paths will be empty like "".

Run the script once everything is done.

# License
Apache License 2.0. Same as Tensorflow TTS.
