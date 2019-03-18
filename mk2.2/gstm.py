from torch import zeros, ones, zeros_like, ones_like, randn
from torch import mul, div, add, sub, matmul, exp, pow
from torch import sigmoid, tanh

from torch import Tensor, tensor, no_grad      ; import random
from torch import squeeze, unsqueeze, cat, stack
from torch import argmax ; from math import sqrt



def soft(tensor):
    tensor = exp(tensor)
    return div(tensor,tensor.sum())


def attend(dec_key, enc_keys, enc_values):
    similarities = mul(enc_keys, dec_key)
    similarities = soft(similarities.sum(1)).unsqueeze(1).unsqueeze(2)
    attended_vec = mul(enc_values, similarities).sum(0)

    return attended_vec


def seq_loss(output, label):
    loss = 0
    for t_o,t_l in zip(output, label):
        # for e in t_o:
        #     print("*******",e.requires_grad)
        t_o = cat(t_o,0)
        t_l = cat(t_l,0)
        # print(t_o.size(), t_l.size())
        result = pow(sub(t_o,t_l), 2).sum()
        loss += result
    loss.backward()

    return float(loss)


def mk_layer(in_size1, in_size2, layer_size):
    layer = {}

    layer["wr1"] = randn(in_size1,layer_size,requires_grad=True)
    layer["wr2"] = randn(in_size2,layer_size,requires_grad=True)
    layer["wrs"] = randn(layer_size,layer_size,requires_grad=True)
    layer["wa1"] = randn(in_size1,layer_size,requires_grad=True)
    layer["wa2"] = randn(in_size2,layer_size,requires_grad=True)
    layer["was"] = randn(layer_size,layer_size,requires_grad=True)
    layer["wi1"] = randn(in_size1,layer_size,requires_grad=True)
    layer["wi2"] = randn(in_size2,layer_size,requires_grad=True)

    sq = sqrt(2/((in_size1+in_size2)/2+layer_size))

    with no_grad():
        for param in layer.values():
            param *= 2*sq
            param -= sq

    state = zeros(1,layer_size)

    return layer, state

def prop_layer(layer, state, in_1, in_2):
    # print(in_1.size(), layer["wo1"].size(), in_2.size(), layer["wo2"].size(), state.size(), layer["wos"].size())
    remember = sigmoid(add(add(matmul(in_1,layer["wr1"]),matmul(in_2,layer["wr2"])),matmul(state,layer["wrs"])))
    attention = sigmoid(add(add(matmul(in_1,layer["wa1"]),matmul(in_2,layer["wa2"])),matmul(state,layer["was"])))
    interm = tanh(add(add(matmul(in_1,layer["wi1"]),matmul(in_2,layer["wi2"])),mul(state,attention)))
    state = add(mul(remember,state),mul(sub(1,remember),interm))

    return state, state


def mk_is(in_size1, hm_vectors, vector_size, layer_sizes, out_size):
    hm_layers = len(layer_sizes)+1
    layers, states = [], []
    for _ in range(hm_layers):
        if _ == 0:
            layer, state = mk_layer(in_size1, hm_vectors*vector_size, layer_sizes[_])
        elif _ == hm_layers-1:
            layer, state = mk_layer(in_size1, layer_sizes[-1], out_size)
        else:
            layer, state = mk_layer(in_size1, layer_sizes[_-1], layer_sizes[_])
        layers.append(layer)
        states.append(state)

    return layers, states

def prop_is(module, states, in_1, in_2):
    new_states = []
    for _,(layer,state) in enumerate(zip(module,states)):
        if _ == 0:
            out, state = prop_layer(layer, state, in_1, in_2.reshape(1,-1))
        else:
            out, state = prop_layer(layer, state, in_1, out)
        new_states.append(state)

    return out, new_states


def mk_gs(in_size1, in_size2, layer_sizes, out_size):
    hm_layers = len(layer_sizes)+1
    layers, states = [], []
    for _ in range(hm_layers):
        if _ == 0:
            layer, state = mk_layer(in_size1, in_size2, layer_sizes[_])
        elif _ == hm_layers-1:
            layer, state = mk_layer(in_size1, layer_sizes[-1], out_size)
        else:
            layer, state = mk_layer(in_size1, layer_sizes[_-1], layer_sizes[_])
        layers.append(layer)
        states.append(state)

    return layers, states

def prop_gs(module, states, in_1, in_2):
    new_states = []
    for _,(layer,state) in enumerate(zip(module,states)):
        if _ == 0:
            out, state = prop_layer(layer, state, in_1, in_2)
        else:
            out, state = prop_layer(layer, state, in_1, out)
        new_states.append(state)

    return out, new_states


def mk_go(in_size1, in_size2, layer_sizes, out_size):
    hm_layers = len(layer_sizes)+1
    layers, states = [], []
    for _ in range(hm_layers):
        if _ == 0:
            layer, state = mk_layer(in_size1, in_size2, layer_sizes[_])
        elif _ == hm_layers-1:
            layer, state = mk_layer(in_size1, layer_sizes[-1], out_size)
        else:
            layer, state = mk_layer(in_size1, layer_sizes[_-1], layer_sizes[_])
        layers.append(layer)
        states.append(state)

    return layers, states

def prop_go(module, states, in_1, in_2):
    new_states = []
    for _,(layer,state) in enumerate(zip(module,states)):
        if _ == 0:
            out, state = prop_layer(layer, state, in_1, in_2)
        else:
            out, state = prop_layer(layer, state, in_1, out)
        new_states.append(state)

    return out, new_states


def mk_encdec(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers):
    is_module, is_state = mk_is(storage_size, hm_vectors, vector_size, is_layers, storage_size)
    gs_module, gs_state = mk_gs(storage_size, storage_size, gs_layers, storage_size)
    go_module, go_state = mk_go(storage_size, vector_size, go_layers, vector_size)
    encdec = [is_module, gs_module, go_module]
    state  = [is_state, gs_state, [go_state for _ in range(hm_vectors)]]

    return encdec, state

def prop_enc(enc, states, in_1, in_2):
    is_out, is_state = prop_is(enc[0], states[0], in_1, cat(in_2, 0))
    gs_out, gs_state = prop_gs(enc[1], states[1], in_1, is_out)
    go_out, go_state = [], []
    for vector, state_v in zip(in_2, states[2]):
        go_out_v, go_state_v = prop_go(enc[2], state_v, is_out, vector)
        go_out.append(go_out_v)
        go_state.append(go_state_v)

    output = [gs_out, go_out]
    new_states = [is_state, gs_state, go_state]

    return output, new_states

def prop_dec(dec, states, in_1, in_2, enc_keys, enc_values):
    attended = attend(in_1, enc_keys, enc_values)
    is_out, is_state = prop_is(dec[0], states[0], in_1, attended)
    gs_out, gs_state = prop_gs(dec[1], states[1], in_1, is_out)
    go_out, go_state = [], []
    for (vector, state_v) in zip(in_2, states[2]):
        go_out_v, go_state_v = prop_go(dec[2], state_v, is_out, vector)
        go_out.append(go_out_v)
        go_state.append(go_state_v)
    output = [gs_out, go_out]
    new_states = [is_state, gs_state, go_state]
    return output, new_states


def mk(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers):
    enc, enc_state = mk_encdec(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)
    dec, dec_state = mk_encdec(hm_vectors, vector_size, storage_size, is_layers, gs_layers, go_layers)

    return (enc,dec), (enc_state,dec_state)

def prop(enc, dec, enc_state, x, length_y, dec_state=None):
    enc_outs = [zeros(1, enc_state[1][-1].size(1)), None]
    enc_gs_time, enc_go_time = [], []
    for timestep in x:
        enc_outs, enc_state = prop_enc(enc, enc_state, enc_outs[0], timestep)
        enc_gs_time.append(enc_outs[0])
        enc_go_time.append(enc_outs[1])

    enc_go_time = stack([cat(e,0) for e in enc_go_time],0)
    enc_gs_time = cat(enc_gs_time,0)

    dec_outs = enc_outs
    if dec_state is None: dec_state = enc_state
    dec_go_time = []
    for timestep in range(length_y):
        dec_outs, dec_state = prop_dec(dec, dec_state, dec_outs[0], dec_outs[1], enc_gs_time, enc_go_time)
        dec_go_time.append(dec_outs[1])

    return dec_go_time

