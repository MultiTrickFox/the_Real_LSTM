from torch import zeros, ones, zeros_like, ones_like, randn
from torch import matmul, sigmoid, tanh, relu, exp
from torch import Tensor, stack                            ; import random


max_prop_time = 55



def create_model(network_structs, vector_size, storage_size, hm_vectors):

    return (create_enc_network(network_structs[0], hm_vectors, vector_size, storage_size),
            create_dec_network(network_structs[1], hm_vectors, vector_size, storage_size))



def create_enc_network(network_struct, hm_vectors, vector_size, storage_size):

    encoder = []

    # module : intermediate state

    encoder.append(create_module(network_struct[0], (
        (hm_vectors, storage_size),
        (hm_vectors, vector_size)
    )))

    # module : global state

    encoder.append(create_module(network_struct[1], (
        (hm_vectors, storage_size),
        (1, storage_size)
    )))

    # module : global output

    encoder.append(create_module(network_struct[-1], (
        (1, storage_size),
        (hm_vectors, vector_size)
    )))


    return encoder


def create_dec_network(network_struct, hm_vectors, vector_size, storage_size):

    decoder = []

    # module : intermediate state

    decoder.append(create_module(network_struct[0], (
        (hm_vectors, storage_size),
        (hm_vectors, vector_size)
    )))

    # module : global state

    decoder.append(create_module(network_struct[1], (
        (hm_vectors, storage_size),
        (1, storage_size)
    )))

    # module : global output

    decoder.append(create_module(network_struct[-1], (
        (1, storage_size),
        (hm_vectors, vector_size)
    )))


    return decoder


def create_module(module_layers, input_sizes):

    module = []
    for _,layer_size in enumerate(module_layers):

        layer = {}
        for __,input_size in enumerate(input_sizes):
            str_ = '_' + str(__)

            if _ == 0:
                size_left = input_size[0]
                size_right = input_size[1]
            else:
                if __ == 0:
                    size_left = 1
                    size_right = module_layers[_ - 1]
                else:
                    size_left = input_size[0]
                    size_right = input_size[1]


            layer['vol' + str_] = randn([1, size_left], requires_grad=True)
            layer['vor' + str_] = randn([size_right, layer_size], requires_grad=True)
            layer['vkl' + str_] = randn([1, size_left], requires_grad=True)
            layer['vkr' + str_] = randn([size_right, layer_size], requires_grad=True)

        layer['uo'] = randn([layer_size, layer_size], requires_grad=True)
        layer['bo'] = randn([1, layer_size], requires_grad=True)
        layer['uk'] = randn([layer_size, layer_size], requires_grad=True)
        layer['bk'] = randn([1, layer_size], requires_grad=True)


        module.append(layer)


    return module


def propogate_module(module, module_state, input1, input2, dropout):

    produced_outputs = []
    produced_states = []

    for _,(layer,layer_state) in enumerate(zip(module, module_state)):

        input1 = produced_outputs[-1] if _ != 0 else input1

        if dropout != 0.0:
            for inp in [input1, input2]:
                drop = random.choices(range(len(inp)), k=int(len(inp) * dropout))
                for _ in drop: inp[_] = 0

        out = tanh(
            matmul(layer['vol_0'], tanh(matmul(input1, layer['vor_0']))) +
            matmul(layer['vol_1'], tanh(matmul(input2, layer['vor_1']))) +
            matmul(layer_state, layer['uo']) +
            layer['bo']
        )

        keep = sigmoid(
            matmul(layer['vkl_0'], tanh(matmul(input1, layer['vkr_0']))) +
            matmul(layer['vkl_1'], tanh(matmul(input2, layer['vkr_1']))) +
            matmul(layer_state, layer['uk']) +
            layer['bk']
        )

        produced_outputs.append(out)
        produced_states.append(keep * out + (1-keep) * layer_state)


    return produced_outputs[-1], produced_states


def propogate_enc_network(network, network_state, input1, input2, dropout):

    keys = []
    values = []

    produced_states = []

    hm_vectors = len(input2)

    input1 = stack([Tensor(e) for e in input1], 0)
    input2 = stack([Tensor(e) for e in input2], 0)

    # module : intermediate state

    module = network[0]
    module_state = network_state[0]

    output, states = propogate_module(module, module_state, input1, input2, dropout)

    intermediate_state = output
    produced_states.append(states)

    # module : global state alter

    module = network[1]
    module_state = network_state[1]

    produced_states.append([])

    for _ in range(hm_vectors):

        output, states = propogate_module(module, module_state[_], input1, intermediate_state, dropout)

        filter = sigmoid(output)
        keys.append((filter * intermediate_state + (1-filter) * input1[_]).squeeze(0))
        produced_states[-1].append(states)

    # module : global output

    module = network[-1]
    module_state = network_state[-1]

    produced_states.append([])

    for _ in range(hm_vectors):

        output, states = propogate_module(module, module_state[_], intermediate_state, input2, dropout)

        values.append(softmax(output.squeeze(0)))
        produced_states[-1].append(states)


    return (keys, values), produced_states


def propogate_dec_network(network, network_state, input1, input2, attended, dropout):

    keys = []
    values = []
    produced_states = []

    hm_vectors = len(input2)
    input1 = stack([Tensor(e) for e in input1], 0)
    input2 = stack([Tensor(e) for e in input2], 0)

    # module : intermediate state

    module = network[0]
    module_state = network_state[0]

    output, states = propogate_module(module, module_state, input1, attended, dropout)

    intermediate_state = output
    produced_states.append(states)

    # module : global state

    module = network[1]
    module_state = network_state[1]

    produced_states.append([])

    for _ in range(hm_vectors):

        output, states = propogate_module(module, module_state[_], input1, intermediate_state, dropout)

        filter = sigmoid(output)
        keys.append((filter * intermediate_state + (1-filter) * input1[_]).squeeze(0))
        produced_states[-1].append(states)

    # module : global output

    module = network[-1]
    module_state = network_state[-1]

    produced_states.append([])

    for _ in range(hm_vectors):

        output, states = propogate_module(module, module_state[_], intermediate_state, input2, dropout)

        values.append(softmax(output.squeeze(0)))
        produced_states[-1].append(states)


    return [keys, values], produced_states





def propogate_model(model, sequence, context=None, gen_seed=None, gen_iterations=None, dropout=0.0):

    encoder, decoder = model


    produced_outputs = init_network_outs(encoder)
    produced_states  = init_network_states(encoder) \
        if context is None else [context]


    for sequence_t in sequence:

        outs, states = propogate_enc_network(encoder, produced_states[-1], produced_outputs[-1][0], sequence_t, dropout)

        produced_outputs.append(outs)
        produced_states.append(states)

    out_keys, out_values = pre_attention(produced_outputs)

    # produced_states = init_network_states(decoder)

    if gen_seed is not None: produced_outputs.append(gen_seed)

    if gen_iterations is None:

        while True:

            attended = pay_attention(produced_outputs[-1][0], out_keys, out_values)

            output, states = propogate_dec_network(decoder, produced_states[-1], produced_outputs[-1][0], produced_outputs[-1][-1], attended, dropout)

            produced_outputs.append(output)
            produced_states.append(states)

            if stop_cond(produced_outputs[-1]): break

    else:

        for t in range(gen_iterations):

            attended = pay_attention(produced_outputs[-1][0], out_keys, out_values)

            output, states = propogate_dec_network(decoder, produced_states[-1], produced_outputs[-1][0], produced_outputs[-1][-1], attended, dropout)

            produced_outputs.append(output)
            produced_states.append(states)

    del produced_outputs[0]
    produced_outputs = [e[-1] for e in produced_outputs]

    return produced_outputs


# math ops


def pre_attention(produced_outputs):
    out_keys = []
    out_values = []

    for t in range(max_prop_time):
        try:
            out_keys.append(stack(produced_outputs[t][0], 0))
            out_values.append(stack(produced_outputs[t][-1], 0))
        except:
            out_keys.append(zeros_like(out_keys[-1], requires_grad=False))
            out_values.append(zeros_like(out_values[-1], requires_grad=False))

    return out_keys, out_values


def softmax(vector): return (lambda e_x: e_x / e_x.sum())(exp(vector))


def pay_attention(out_key, enc_out_keys, enc_out_values):

    out_key = stack(out_key, 0)
    enc_out_values = stack(enc_out_values,2)

    cos_similarity = stack([out_key * enc_out_key for enc_out_key in enc_out_keys], 0)
    cos_similarity = softmax(cos_similarity.sum(1).sum(1).unsqueeze(1))

    attended = matmul(enc_out_values, cos_similarity).squeeze(2)

    return attended


# inside helpers


def init_network_outs(network):
    hm_vectors = network[0][0]['vol_1'].size()[1]
    vector_size = network[0][0]['vor_1'].size()[0]
    storage_size = network[0][0]['vor_0'].size()[0]

    out1_initial = [zeros(storage_size, requires_grad=False)] * hm_vectors
    out2_initial = [zeros(vector_size, requires_grad=False)] * hm_vectors

    return [[out1_initial, out2_initial]]


def init_network_states(network):
    hm_vectors = network[0][0]['vol_1'].size()[1]

    network_states = []

    for _,module in enumerate(network):
        module_state = []

        if _ == 0:
            for __,layer in enumerate(module):
                module_state.append(zeros_like(layer['bo'], requires_grad=False))

        else:
            for ___ in range(hm_vectors):
                module_state.append([])

                for __, layer in enumerate(module):
                    module_state[-1].append(zeros_like(layer['bo'], requires_grad=False))

        network_states.append(module_state)


    return [network_states]



def stop_cond(output_t):
    return True
# todo :  do stop sometime.


# outside helpers


def loss(output_seq, target_seq):
    output_seq = [stack(e, 0) for e in output_seq]
    target_seq = [Tensor(e) for e in target_seq]

    arr = [(out_t - trg_t).pow(2).sum() for out_t, trg_t in zip(output_seq, target_seq)]

    return sum(arr)


def get_params(model):
    all_values = []
    all_keys = []
    for _,network in enumerate(model):
        for __,module in enumerate(network):
            for ___,layer in enumerate(module):
                all_values.extend(layer.values())
                all_keys.extend(['N'+str(_)+'-'+'M'+str(__)+'-'+'L'+str(___)+'-'+'W'+key for key in layer.keys()])

    return all_values, all_keys
