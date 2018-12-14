sample_vector = [1] * 12
sample_vector_2 = [80] * 12
sample_sequence = [[sample_vector for e in range(4)], [sample_vector for e2 in range(4)], [sample_vector for e3 in range(4)]]
sample_sequence2 = [[sample_vector_2 for e in range(4)] for _ in range(14)]

input = sample_sequence
target = sample_sequence2





hm_epochs = 2
learning_rate = 0.001


hm_channels = 4
channel_size = 12
channel_storage_size = 10



import VanillaV2 as v

model = v.make_model(hm_channels,
                     channel_size,
                     channel_storage_size)

optimizer = v.make_optimizer(model, learning_rate)  # , which='rms', which='adam')


for i in range(hm_epochs):

    output = v.propogate(input, model, len(target))
    loss = v.make_grads(output, target)

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
