import The_Real_LSTM as gstm
from torch.nn import Module

from torch import optim, device, cuda
from torch import save, load


device = device("cuda" if cuda.is_available() else "cpu")



def make_model(hm_channels, vector_size, memory_size, network_structs=None):

    if network_structs is None: network_structs = [(
        (int(memory_size * 3/5), memory_size),   # module : intermediate state
        (int(memory_size * 3/5), memory_size),   # module : global state
        (int(vector_size * 3/5), vector_size),   # module : global output
    ) for _ in range(2)]

    else: network_structs = [[module + tuple([size]) if len(module) == 0 or size != module[-1] else module
                              for _, (module, size) in enumerate(zip(struct, [memory_size, memory_size, vector_size]))]
                              for struct in network_structs]

    internal_model = gstm.create_networks(network_structs, vector_size, memory_size, hm_channels)
    internal_params = gstm.get_params(internal_model)


    return GSTM(internal_model, internal_params).to(device)



class GSTM(Module):

    def __init__(self, internal_model, internal_params):
        super(GSTM, self).__init__()

        self.model = internal_model
        self.params, self.names = internal_params

    def forward(self, sequence, hm_timestep=None):
        return gstm.propogate_model(self.model, sequence, gen_iterations=hm_timestep)



def make_optimizer(model, lr, which=None):
    if which == 'adam':
        return optim.Adam(model.params, lr)
    elif which == 'rms':
        return optim.RMSprop(model.params, lr)
    else: return optim.SGD(model.params, lr)

def propogate(model, input, hm_timesteps=None):
    return model.forward(input, hm_timesteps)

def make_grads(output, target):
    loss = gstm.loss(output, target)
    loss.backward()
    return float(loss)

def take_a_step(optimizer):
    optimizer.step()
    optimizer.zero_grad()


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
