import pickle

from glob import glob
from numpy import argmax
from scipy import signal
from random import choices

from numpy.linalg import norm as n
from scipy.io.wavfile import read
from multiprocessing import Pool, cpu_count



hm_channels  = 3    # dominant frequency channels

channel_size = 2    # frequency, amplitude per channel
max_vals = [1, 1]   # per data normalization

hm_altered          = 5       # altered data, timesteps lost.
drop_rate           = 0.2     # likely to be lost.

max_timesteps       = 20_000  # memory limitations :(



def run(inp):
    file, id = inp

    rate, data = read(file)

    dataset = []

    hm_tracks = data.shape[1]

    if hm_tracks >= 2:

        tracks = [data[:, tr_nr] for tr_nr in range(hm_tracks)]
        tracks_converted = []

        for track in tracks:
            track_converted = []

            freqs, times, amps = signal.spectrogram(track)
            hm_timesteps = amps.shape[1]
            if hm_timesteps > max_timesteps: hm_timesteps = max_timesteps
            # hm_frequencies = amps.shape[0]

            amps = normalize(amps)

            for t in range(hm_timesteps):
                channels_converted = []

                amps_at_t = amps[:, t]

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

                    to_drop1 = choices(range(len(track)), k=int(len(track) * drop_rate))
                    to_drop2 = choices(range(len(track2)), k=int(len(track2) * drop_rate))

                    new_track = [e for _,e in enumerate(track) if _ not in to_drop1]
                    new_track2 = [e for _,e in enumerate(track2) if _ not in to_drop2]

                    dataset.append((new_track, track2))
                    dataset.append((new_track2, track))


        print(f'obtained {len(dataset)} samples.')
        with open('dataset'+str(id)+'.pkl', 'wb') as file:
            pickle.dump(dataset, file)

    else: print(f'insufficient tracks in file {file}')

def normalize(v):
    norm = n(v)
    if norm == 0:
       return v
    return v / norm



if __name__ == '__main__':

    with Pool(cpu_count()) as P:
        P.map_async(run, [(f,i+1) for i,f in enumerate(glob('samples/*.wav'))])
        P.close() ;P.join()

