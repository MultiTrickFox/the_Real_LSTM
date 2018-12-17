
from VanillaV2 import *


hm_channels  =  2
channel_size = 15
storage_size = 10



network1 = (               # encoder
     tuple([10]),           # : intermediate state
     tuple([8]),            # : global state alter
     tuple([8, 10]),        # : global decision
)

network2 = (               # decoder
     tuple([10, 10]),       # : intermediate state
     tuple([8,   9]),       # : global state alter
     tuple([8,  10]),       # : global decision
)



learning_rate = 0.01
hm_epochs = 2






model \
    = make_model(hm_channels,channel_size,
        storage_size, (network1, network2))

data \
    = make_data(hm_channels, channel_size,
    min_seq_len=25,
    max_seq_len=45,
    data_size=110)

optimizer \
    = make_optimizer(model,
                     learning_rate,
                     type='rms'
                     )



# basics



for i in range(1):

    for input, target in data:

        output = propogate(model, input, len(target))

        make_grads(output, target)
    take_step(optimizer)

    print(f'epoch {i} : ')

save_session(model, optimizer)



# details



model, optimizer = load_session()

for i in range(1):
    loss = 0

    for batch in data.batchify(50):

        for (input, target) in batch:

            output = propogate(model, input, target_length=len(target),
                                  dropout=0.1)
            loss += make_grads(output, target)

        take_step(optimizer)

    data.shuffle()

    print(f'> epoch {i} : {loss}')



# adv



data, dev, test = data.split(dev_ratio=0.1, test_ratio=0.2)


losses = ([], [], [])

for i in range(hm_epochs):
    loss = 0

    for batch in data.batchify(50):
        for (input, target) in batch:

            output = propogate(model, input, target_length=len(target),
                                    dropout=0.1)
            loss += make_grads(output, target)

        # for param in model.params:
        #     print(param.grad)

        # for name, param in zip(model.names, model.params):
        #     print(f'name : {name} , grad : {param.grad}')

        take_step(optimizer)

    data.shuffle()

    print(f'> epoch {i} : {loss}')
    losses[0].append(loss)



    for _,(set, name) in enumerate(zip((dev, test), ("dev", "test"))):
        loss = 0
        for (input, target) in set:

            output = propogate(model, input, target_length=len(target),
                                    dropout=0.0)
            loss += make_grads(output, target)

        optimizer.zero_grad()

        print(f'{name} loss : {loss}')
        losses[_+1].append(loss)





extra = False

if extra:



    # data splitting

    data, another, _ = data.split(dev_ratio=0.1, test_ratio=0.0)
    data, _, someset = data.split(dev_ratio=0.0, test_ratio=0.1)

    loss = 0
    for (input, target) in test + someset + another:

        output = propogate(model, input, target_length=len(target),
                                dropout=0.0)
        loss += make_grads(output, target)

    print('final test loss: ', loss)



    # visualization

    plot(losses)

    plot(losses[1])

    plot((losses[0], losses[2]))



