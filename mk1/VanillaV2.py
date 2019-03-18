import The_Real_LSTM as gstm

from torch import optim
from torch import save, load
from torch import cuda, set_default_tensor_type

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module ; from random import shuffle

set_default_tensor_type('torch.cuda.FloatTensor' if cuda.is_available() else 'torch.FloatTensor')




def make_model(hm_channels, vector_size, memory_size, blueprints=None):

    if blueprints is None: blueprints = tuple([(
        (int(memory_size * 3/5), memory_size),   # module : intermediate state
        (int(memory_size * 3/5), memory_size),   # module : global state
        (int(vector_size * 3/5), vector_size),   # module : global output
    ) for _ in range(2)])
    else: blueprints = tuple([[module + tuple([size]) if len(module) == 0 or size != module[-1] else module
                         for _, (module, size) in enumerate(zip(structure, [memory_size, memory_size, vector_size]))]
                        for structure in blueprints])

    model = gstm.create_networks(blueprints, vector_size, memory_size, hm_channels)
    params = gstm.get_params(model)


    return GSTM(model, params)



class GSTM(Module):

    def __init__(self, model, params):
        super(GSTM, self).__init__()

        self.model = model
        self.params, self.names = params

    def forward(self, sequence, hm_timestep=None, drop=0.0):
        return gstm.propogate_model(self.model, sequence, gen_iterations=hm_timestep, dropout=drop)



class Dataset(Dataset):

    def __init__(self, hm_channels, channel_size, min_seq_len, max_seq_len, hm_data, file, obj):


        if obj is not None:
            self.data = obj
            self.hm_data = len(obj)

        elif file is not None:
            from glob import glob

            raw_files = glob(file)
            self.data = []
            for file in raw_files:
                self.data.extend(pickle_load(file))

            shuffle(self.data) ; self.hm_data = hm_data
            self.data = self.data[:hm_data]


        else:
            import random

            self.hm_data      = hm_data
            self.min_seq_len  = min_seq_len
            self.max_seq_len  = max_seq_len
            self.hm_channels  = hm_channels
            self.channel_size = channel_size

            data_fn = lambda : [random.random() for _ in range(channel_size)]
            len_fn  = lambda :  random.randint(min_seq_len,max_seq_len)
            generate= lambda : [[data_fn() for e in range(self.hm_channels)] for _ in range(len_fn())]

            self.data = [(generate(), generate())
                        for _ in range(hm_data)]

        self.shuffle = lambda : shuffle(self.data)

    def split(self, dev_ratio=0.0, test_ratio=0.0):
        hm_train = int((1-dev_ratio-test_ratio) * self.hm_data)
        hm_dev = int(dev_ratio * self.hm_data)
        hm_test = int(test_ratio * self.hm_data)

        dev = Dataset(0, 0, 0, 0, 0, 0,
                obj=self.data[hm_dev:-hm_test])
        test = Dataset(0, 0, 0, 0, 0, 0,
                obj=self.data[-hm_test:])
        self.data = self.data[:hm_dev]
        self.hm_data = hm_train

        returns = [self, dev, test]
        return tuple([r for r in
             returns if len(r) > 0])

    def batchify(self, batch_size):
        hm_batches = int(self.hm_data / batch_size)
        hm_leftover = int(self.hm_data % batch_size)
        batched_resource = [self.data[_ * batch_size : (_+1) * batch_size]
                            for _ in range(hm_batches)]
        if hm_leftover != 0:
            batched_resource.append(self.data[-hm_leftover:])

        return batched_resource

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self): return self.hm_data



def make_data(hm_channels=1, channel_size=5, min_seq_len=50, max_seq_len=75, data_size=200, from_file=None, from_obj=None):
    return Dataset(hm_channels, channel_size, min_seq_len, max_seq_len, data_size, from_file, from_obj)

def propogate(model, input, target_length=None, dropout=0.0):
    return model.forward(input, target_length, drop=dropout)

def make_grads(output, target):
    loss = gstm.loss(output, target)
    loss.backward()
    return float(loss)

def make_optimizer(model, lr, type=None):
    if type == 'adam':
        return optim.Adam(model.params, lr)
    elif type == 'rms':
        return optim.RMSprop(model.params, lr)
    else: return optim.SGD(model.params, lr)

def take_step(optimizer):
    optimizer.step() ; optimizer.zero_grad()


    # Helpers #


def save_session(model, optimizer=None):
    pickle_save(model.model, 'model.pkl')
    if optimizer is not None:
        save(optimizer.state_dict(), 'meta.pkl')


def load_session():
    model = pickle_load('model.pkl')
    if model is None: params = None
    else: params = gstm.get_params(model)

    if model is not None:
        mtorch = GSTM(model, params)
        try: meta = load('meta.pkl')
        except Exception:
            return mtorch, make_optimizer(mtorch, 0.001)
        type = get_opt_type(meta)
        opt = make_optimizer(mtorch, 0, type)
        opt.load_state_dict(meta)

    else: return None, None
    return mtorch, opt


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


def get_opt_type(meta):
    # print(f'Opt params: {meta['param_groups'][0].keys()}')
    for key in meta['param_groups'][0].keys():
        if key == 'dampening': return None
        elif key == 'alpha': return 'rms'
        elif key == 'amsgrad':return 'adam'


def plot(losses):
    try:
        import matplotlib.pyplot as plot

        if len(losses) <= 10:
            hm_epochs = len(losses[0])
            for _, color in enumerate(('r', 'g', 'b')):
                try:
                    plot.plot(range(hm_epochs), losses[_], color)
                except : pass
        else: plot.plot(range(len(losses)), losses, 'k')

        plot.show()
    except: print('graph unavailable.')
