def get_data(hm_channels, channel_size):
    import random
    def generate_test_data(in_len, out_len, hm_channels, channel_size):
        get_sample_vector = lambda : [random.random() for _ in range(channel_size)]
        return [[get_sample_vector() for e in range(hm_channels)] for _ in range(in_len)], \
               [[get_sample_vector() for e in range(hm_channels)] for _ in range(out_len)]

    list = [[], []]
    for _ in range(50):
        for e,ee in zip(list, generate_test_data(random.randint(40,60), random.randint(40,60), hm_channels, channel_size)): e.append(ee)
    return list
inputs, targets = get_data(3, 15)





hm_channels = 3
channel_size = 15
storage_size = 10


network1 = (                # encoder
     tuple([10]),            # : intermediate state
     tuple([8, 10]),         # : global state alter
     tuple([12]),             # : global decision
)

network2 = (                # decoder
     tuple([10, 10]),        # : intermediate state
     tuple([8, 9, 10]),      # : global state alter
     tuple([12]),             # : global decision
)





learning_rate = 0.01
hm_epochs = 10


import VanillaV2 as v

model = v.make_model(hm_channels,
                     channel_size,
                     storage_size,
                     (network1, network2)
                     )

optimizer = v.make_optimizer(model,
                             learning_rate,
                             'rms'
                             )





for i in range(1):

    for input, target in zip(inputs, targets):

        output = v.propogate(model, input, len(target))
        v.make_grads(output, target)

    v.take_a_step(optimizer)

    print(f'epoch {i} : ')


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

    print(f'epoch {i+1} : loss {loss}')
