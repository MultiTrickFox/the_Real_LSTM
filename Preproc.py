from glob import glob
from numpy import argmax
from random import choices

from scipy import signal
import scipy.io.wavfile as wave





hm_channels  = 3    # 3 dominant frequency channels
channel_size = 2    # frequency, amplitude per channel

max_vals = [1, 1]   # frequency, amplitude normalization



hm_altered        = 5       # altered data, timesteps lost.
drop_rate_time    = 0.3     # likely to be lost.



dataset = []


for file in glob('*.wav'):

    rate, data = wave.read(file)

    hm_tracks = data.shape[1]
    tracks = [data[:,tr] for tr in range(hm_tracks)]

    tracks_converted = []

    for track in tracks:

        freqs, times, amps = signal.spectrogram(track)

        track_converted = []

        hm_frequencies = amps.shape[0]
        hm_timesteps = amps.shape[1]

        for t in range(hm_timesteps):
            amps_at_t = amps[:,t]

            channels_converted = []

            for ch in range(hm_channels):
                amp_max = argmax(amps_at_t)

                max_freq = freqs[amp_max]
                that_amp = amps_at_t[amp_max]

                channels_converted.append((max_freq, that_amp))

                amps_at_t = [e for _,e in enumerate(amps_at_t) if _ != amp_max]  # del amps_at_t[fr_max]

            track_converted.append(channels_converted)

        tracks_converted.append(track_converted)

    for _,track in enumerate(tracks_converted):

        for track2 in tracks_converted[_+1:]:

            for __ in range(hm_altered):

                to_drop1 = choices(range(len(track)), k=int(len(track)*drop_rate_time))
                to_drop2 = choices(range(len(track2)), k=int(len(track2)*drop_rate_time))

                new_track = [e for _,e in enumerate(track) if _ not in to_drop1]
                new_track2 = [e for _,e in enumerate(track2) if _ not in to_drop2]


                dataset.append((new_track, track2))
                dataset.append((new_track2, track))

import pickle
print(f'Total of {len(dataset)} samples.')
with open('dataset.pkl', 'wb') as file:
    pickle.dump(dataset, file)
