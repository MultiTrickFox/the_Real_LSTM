from glob import glob
from pydub import AudioSegment
import pydub

import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wave



for file in glob('*.wav'):


    # sound = AudioSegment.from_file(file, format="wav")
    #
    #
    # samples = sound.get_array_of_samples()
    # mono_channels = []
    #
    # for i in range(sound.channels):
    #   samples_for_current_channel = samples[i::sound.channels]
    #
    #   try:
    #     mono_data = samples_for_current_channel.tobytes()
    #   except AttributeError:
    #     mono_data = samples_for_current_channel.tostring()
    #
    #   mono_channels.append(
    #     sound._spawn(mono_data, overrides={"channels": 1})
    #   )
    #
    #
    # sound_l = AudioSegment.from_mono_audiosegments(mono_channels[0])
    # sound_r = AudioSegment.from_mono_audiosegments(mono_channels[1])
    #
    #
    # sound_l.export("l.wav", format="wav")
    # sound_r.export("r.wav", format="wav")

    rate, data = wave.read(file)

    ch0 = data[:,0]
    ch1 = data[:,1]


    freqs1, time1, amps1 = signal.spectrogram(ch0)
    freqs2, time2, amps2 = signal.spectrogram(ch1)





    print(freqs1[98])

    print(len(freqs1), len(amps1))
    print(len(time1), len(time2))
    print(len(freqs2), len(amps2))
