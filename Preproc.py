from glob import glob
from random import shuffle, choices

from music21 import *
from multiprocessing import Pool, cpu_count





hm_channels  = 2
channel_size = 2

max_vals = [127, 32]


hm_altered        = 5
drop_rate_time    = 0.3


show_exceptions = True





def process_file(filename):
    datapoints = []


    try:
        file = converter.parse(filename)
        parts = instrument.partitionByInstrument(file)

        parts_processed = []

        if len(parts) >= hm_channels*2:

            for part in parts:
                part_processed = []

                for element in part.flat.elements:
                    element_processed = []

                    try:
                        if element.isNote:
                            try:
                                if 0.0 < element.duration.quarterLength <= max_vals[1]:
                                    element_processed.append(int(element.pitch.midi))
                                    element_processed.append(float(element.duration.quarterLength))

                            except Exception as e:
                                pass
                                # if show_exceptions:
                                #     print(f'Inner Inner Exception ; File {filename} : {e}')

                        elif element.isChord:
                            pass
                            # for e in element:
                            #     try:
                            #         if e.isNote:
                            #             if 0.0 < e.duration.quarterLength <= max_vals[1]:
                            #                 element_processed.append(int(e.pitch.midi))
                            #                 element_processed.append(float(e.duration.quarterLength))

                                # except Exception as e:
                                #     pass
                                    # if show_exceptions:
                                    #     print(f'Inner Inner Exception ; File {filename} : {e}')

                    except Exception as e:
                        pass
                        # if show_exceptions:
                        #     print(f'Inner Exception ; File {filename} : {e}')


                    if len(element_processed) != 0: part_processed.append(element_processed)
                if len(part_processed)        != 0: parts_processed.append(part_processed)

            parts_modified = []
            for _ in range(hm_altered):

                for part in parts_processed:

                    part_modified = []
                    for element in part:

                        # for to_drop in choices(range(len(element)), k=len(element)*drop_rate_element):
                        #     element[to_drop] = 0

                        part_modified.append([e/max_val for e,max_val in zip(element,max_vals)])

                    to_drop = choices(range(len(part_modified)), k=int(len(parts_modified)*drop_rate_time))
                    parts_modified = [e for _,e in enumerate(parts_modified) if _ not in to_drop]

                    parts_modified.append(part_modified)


            for _,i in enumerate(parts_modified[:-1]):
                for __,j in enumerate(parts_processed[:-1]):
                    for i2 in parts_modified[_+1:]:
                        for j2 in parts_processed[__+1:]:

                            size_smallest_i = min(len(i), len(i2))
                            size_smallest_j = min(len(j), len(j2))

                            serie_over_time = ([], [])

                            for t_i in range(size_smallest_i):
                                serie_over_time[0].append((i[t_i], i2[t_i]))

                            for t_j in range(size_smallest_j):
                                serie_over_time[1].append((j[t_j], j2[t_j]))

                            datapoints.append(serie_over_time)
            shuffle(datapoints)

            return datapoints


        else:
            if show_exceptions:
                print(f'File {filename} parts < {hm_channels*2}.')

    except Exception as e:
        if show_exceptions:
            print(f'Outer Exception ; File {filename} : {e}')



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

    [data.extend(result) for result in results.get() if result is not None]

    print(f'Obtained: {len(data)} samples.')
    pickle_save(data, 'samples_1.pkl')
