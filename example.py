def generate_test_data(in_len, out_len, hm_channels, channel_size):
    import random
    sample_vector = [random.random()] * channel_size ; sample_vector_2 = [random.random()] * 12
    return [[sample_vector for e in range(hm_channels)] for _ in range(in_len)], \
           [[sample_vector_2 for e in range(hm_channels)] for _ in range(out_len)]

list = [[], []]
for _ in range(40):
    for e,ee in zip(list, generate_test_data(10, 20, 4, 12)): e.append(ee)
inputs, targets = list


network1 = (
    (8,  6, 10),   # module : intermediate state
    (8, 10),       # module : global state
    (10, 8, 12),   # module : global output
)

network2 = (
    (8,  6, 10),   # module : intermediate state
    (8, 10),       # module : global state
    (10, 8, 12),   # module : global output
)




hm_epochs = 3
learning_rate = 0.001


hm_channels = 4
channel_size = 12
channel_storage_size = 10



import VanillaV2 as v

model = v.make_model(hm_channels,
                     channel_size,
                     channel_storage_size,
                     network_structs=(network1, network2))  # optional

optimizer = v.make_optimizer(model, learning_rate)


for i in range(hm_epochs):
    loss = 0

    for input, target in zip(inputs, targets):

        output = v.propogate(input, model, len(target))
        loss += v.make_grads(output, target)

        v.take_a_step(optimizer)

    print(f'epoch {i} : loss {loss}')


v.save_session(model, optimizer)
model, optimizer = v.load_session()


for i in range(hm_epochs):

    output = v.propogate(input, model, len(target))
    loss = v.make_grads(output, target)

    # for param in model.params:
    #     print(param.grad)

    v.take_a_step(optimizer)

    print(f'epoch {i} : loss {loss}')
