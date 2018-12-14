import The_Real_LSTM as gstm

from torch.nn import Module
from torch import optim, device, cuda
from torch import save, load


device = device("cuda" if cuda.is_available() else "cpu")



def make_model(hm_channels, vector_size, memory_size,network_struct=None):

    if network_struct is None: network_struct = (
        (int(memory_size * 3/5), int(memory_size * 4/5)),   # module : intermediate state
        (int(memory_size * 3/5), int(memory_size * 4/5)),   # module : global state
        (int(vector_size * 3/5), int(vector_size * 4/5)),   # module : global output
    )

    internal_model = gstm.create_model(network_struct, vector_size, memory_size, hm_channels)
    internal_params = gstm.get_params(internal_model)


    return GSTM(internal_model, internal_params).to(device)



class GSTM(Module):

    def __init__(self, internal_model, internal_params):
        super(GSTM, self).__init__()

        self.model = internal_model
        self.params = internal_params

    def forward(self, sequence, hm_timestep=None):
        return gstm.propogate_model(self.model, sequence, gen_iterations=hm_timestep)



def make_optimizer(model, lr, which=None):
    if which == 'adam':
        return optim.Adam(model.params, lr)
    elif which == 'rms':
        return optim.RMSprop(model.params, lr)
    else: return optim.SGD(model.params, lr)

def propogate(input, model, hm_timesteps=None):
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
        meta = load('meta.pkl')
        opt = make_optimizer(mtorch, 0.001)
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
