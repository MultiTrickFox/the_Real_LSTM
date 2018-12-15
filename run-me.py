

hm_channels = 3
channel_size = 15
storage_size = 10


network1 = (               # encoder
     tuple([10]),           # : intermediate state
     tuple([8]),            # : global state alter
     tuple([12]),           # : global decision
)

network2 = (               # decoder
     tuple([10, 10]),       # : intermediate state
     tuple([8,   9]),       # : global state alter
     tuple([12]),           # : global decision
)




learning_rate = 0.01
hm_epochs = 20



import VanillaV2 as v

model = v.make_model(hm_channels,
                     channel_size,
                     storage_size,
                     (network1, network2)
                     )

data = v.make_data(hm_channels, channel_size,
                   min_seq_len=25,
                   max_seq_len=45,
                   hm_data=50   )


optimizer = v.make_optimizer(model,
                             learning_rate,
                             'rms'
                             )





for i in range(1):

    for input, target in data:

        output = v.propogate(model, input, len(target))
        v.make_grads(output, target)

        v.take_a_step(optimizer)

    print(f'epoch {i} : ')


v.save_session(model, optimizer)



model, optimizer = v.load_session()


for i in range(hm_epochs):
    loss = 0

    for input, target in data:

        output = v.propogate(model, input, len(target))
        loss += v.make_grads(output, target)

    # for param in model.params:
    #     print(param.grad)

    # for name, param in zip(model.names, model.params):
    #     print(f'name : {name} , grad : {param.grad}')

    v.take_a_step(optimizer)

    print(f'epoch {i} : loss {loss}')