from glob import glob
from random import shuffle, choices

from music21 import *
from multiprocessing import Pool, cpu_count





hm_channels  = 2
channel_size = 2

max_vals = [127, 8]


hm_altered        = 4
drop_rate_time    = 0.2
drop_rate_element = 0.1


show_exceptions = True





def process_file(file):
    datapoints = []


    try:
        file = converter.parse(file)
        parts = instrument.partitionByInstrument(file)
        parts_processed = []

        if len(parts) >= hm_channels:

            for part in parts:
                part_processed = []

                for element in part.flat.elements:
                    element_processed = []

                    if element.isNote():
                        if 0.0 < element.duration.quarterLength <= max_vals[1]:
                            element_processed.append(float(element.pitch.midi))
                            element_processed.append(float(element.duration.quarterLength))

                    elif element.isChord():
                        for e in element:
                            if e.isNote():
                                if 0.0 < element.duration.quarterLength <= max_vals[1]:
                                    element_processed.append(float(element.pitch.midi))
                                    element_processed.append(float(element.duration.quarterLength))

                    part_processed.append(element)
                parts_processed.append(part_processed)


            parts_modified = []
            for _ in range(hm_altered):

                for part in parts_processed:

                    part_modified = []
                    for element in part:

                        # for to_drop in choices(range(len(element)), k=len(element)*drop_rate_element):
                        #     element[to_drop] = 0

                        part_modified.append([e/max_val for e,max_val in zip(element,max_vals)])

                    for to_drop in choices(range(len(part_modified)), k=len(parts_modified)*drop_rate_time):
                        parts_modified.pop(to_drop)

                    parts_modified.append(part_modified)


            for i in parts_modified:
                for j in parts_processed:
                    datapoints.append((i, j))
            shuffle(datapoints)

        return datapoints



    except Exception as e:
        if show_exceptions:
            print(f'Encountered exception : {e}')



note_dict = {
    'A' : 0,
    'A#': 1, 'B-': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4, 'D-': 4,
    'D' : 5,
    'D#': 6, 'E-': 6,
    'E' : 7,
    'F' : 8,
    'F#': 9, 'G-': 9,
    'G' :10,
    'G#':11, 'A-': 11,
    'R' :12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10:'G',
    11:'G#',
    12:'R'
}



    # Global Helpers #

import pickle

def pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(MacOSFile(f))
    except: return None


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size





if __name__ == '__main__':
    data = []

    with Pool(cpu_count()) as P:
        results = P.map_async(process_file, glob('samples/*.mid'))

        P.close() ; P.join()

    [data.extend(result) for result in results.get()]

    print(f'Obtained: {len(data)} samples.')
    pickle_save(data, 'samples_1.pkl')
