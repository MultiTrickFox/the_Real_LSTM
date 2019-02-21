from multiprocessing import Pool, cpu_count
from copy import deepcopy
from VanillaV2 import *



def run():



    loadSession = False



    data_size = 50
    batch_size = 10
    hm_epochs = 10
    learning_rate = 0.01
    dropout = 0.3


    hm_channels  = 3
    channel_size = 2
    storage_size = 5


    network1 = (               # encoder
         tuple([]),            # : intermediate state
         tuple([]),            # : global state alter
         tuple([4, 3]),        # : global decision
    )

    network2 = (               # decoder
         tuple([5,  5]),       # : intermediate state
         tuple([5,  5]),       # : global state alter
         tuple([4,  3, 2, 2]), # : global decision
    )



    model \
        = make_model(hm_channels,channel_size,
            storage_size, (network1, network2))

    data \
        = make_data(from_file='data*.pkl',
                    data_size=data_size)

    optimizer \
        = make_optimizer(model,
                         learning_rate,
                         type='rms'
                         )

    if loadSession:
        model_load, opt_load = load_session()
        if model_load: model = model_load
        if opt_load: optimizer = opt_load


    losses = []
    for _ in range(hm_epochs):
        epoch_loss = 0

        for batch in data.batchify(batch_size):

            model, loss = process_batch(model, batch, dropout)
            take_step(optimizer)

            epoch_loss += loss

        data.shuffle()
        print(f'\n epoch {_} completed, loss: {round(epoch_loss, 3)}')
        losses.append(epoch_loss)

    save_session(model, optimizer)
    plot(losses)



cpu_count = cpu_count()


def process_batch(model, batch, dropout):
    inner_model = model.model

    with Pool(cpu_count) as P:
        results = P.map_async(process_sample, [(deepcopy(inner_model), input, target, dropout) for (input, target) in batch])

        P.close()
        P.join()

    total_loss = 0

    for (gradient, loss) in results.get():
        set_grads(model, gradient)
        total_loss += loss

    print('/', end='', flush=True)
    return model, float(total_loss)


from The_Real_LSTM import propogate_model as forw_prop


def process_sample(data):
    model, input, target, dropout = data

    output = forw_prop(model, input, gen_iterations=len(target), dropout=dropout)
    loss = make_grads(output, target)
    grads = get_grads(model)

    return grads, loss



def get_grads(model):
    grads = []
    for _, network in enumerate(model):
        for __, module in enumerate(network):
            for ___, layer in enumerate(module):
                for ____, param in enumerate(layer.values()):
                    if param.grad is not None:
                        grads.append(param.grad.detach())
                    else: print(f'{_}.{__}.{___}.{____} : None grad.')

    return grads

def set_grads(model, grads):
    for _,param in enumerate(model.params):
        if param.grad is not None:
            param.grad += grads[_]
        else:
            param.grad = grads[_]





if __name__ == '__main__':
    run()
