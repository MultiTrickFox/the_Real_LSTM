def generate_test_data(in_len, out_len, hm_channels, channel_size):
    import random
    sample_vector = [random.random()] * channel_size ; sample_vector_2 = [random.random()] * channel_size
    return [[sample_vector for e in range(hm_channels)] for _ in range(in_len)], \
           [[sample_vector_2 for e in range(hm_channels)] for _ in range(out_len)]

list = [[], []]
for _ in range(20):
    for e,ee in zip(list, generate_test_data(20, 21, 3, 15)): e.append(ee)
inputs, targets = list





network1 = (                # encoder
     tuple([12]),            # : intermediate state
     tuple([8, 12]),         # : global state alter
     tuple([8, 15]),         # : global decision
)

network2 = (                # decoder
     tuple([12, 12]),        # : intermediate state
     tuple([8, 10, 12]),     # : global state alter
     tuple([8, 10]),         # : global decision
)



hm_epochs = 3
learning_rate = 0.001


hm_channels = 3
channel_size = 15
storage_size = 12



import VanillaV2 as v


model = v.make_model(hm_channels, channel_size, storage_size,
                                        (network1, network2))
optimizer = v.make_optimizer(model, learning_rate, 'rms')



for i in range(hm_epochs):
    loss = 0

    for input, target in zip(inputs, targets):

        output = v.propogate(model, input, len(target))
        loss += v.make_grads(output, target)

    v.take_a_step(optimizer)

    print(f'epoch {i} : loss {loss}')


v.save_session(model, optimizer)
model, optimizer = v.load_session()


for i in range(hm_epochs):
    loss = 0

    for input, target in zip(inputs, targets):

        output = v.propogate(model, input, len(target))
        loss += v.make_grads(output, target)

    # for param in model.params:
    #     print(param.grad)

    # for name, param in zip(model.names, model.params):
    #     print(f'name : {name} , grad : {param.grad}')

    v.take_a_step(optimizer)

    print(f'epoch {i} : loss {loss}')
