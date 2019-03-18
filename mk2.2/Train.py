from torch import no_grad
from gstm import *


hm_vectors   = 4
vector_size  = 13
storage_size = 12

is_layers = [25,15]
gs_layers = [15]
go_layers = [15]

hm_data = 50
hm_epochs = 50
learning_rate = 0.0002


(enc,dec), (zerostate_enc, zerostate_dec) = mk(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)

# sample_input = [[randn(1,vector_size) for _ in range(hm_vectors)] for t in range(50)]
#
# output = prop(enc, dec, state_enc, sample_input, 50)
#
# sample_label = [[randn(1,vector_size) for _ in range(hm_vectors)] for t in range(50)]

data = []

for i in range(hm_data):
    sample_x = [[randn(1,vector_size) for _ in range(hm_vectors)] for t in range(16)]
    sample_y = [[randn(1,vector_size) for _ in range(hm_vectors)] for t in range(16)]
    data.append((sample_x, sample_y))


def runner_fn():
    for i in range(hm_epochs):
        ep_loss = 0.0
        for x,y in data:
            loss = seq_loss(prop(enc, dec, zerostate_enc, x, len(y)), y)
            # loss.backward()
            ep_loss +=float(loss)

            with no_grad():
                for model in (enc,dec):
                    for network in model:
                        for layer in network:
                            for param in layer.values():
                                if param.grad is not None:
                                    param -= param.grad * learning_rate
                                    param.grad = None
                                else: # pass
                                    print(f"{param} none grad.")

        print(f"Epoch: {i} Loss: {ep_loss}")

import timeit
print("time:",timeit.timeit(runner_fn, number=1))
