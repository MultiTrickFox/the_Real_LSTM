from multiprocessing import Pool, cpu_count
from VanillaV2 import *



def run():


    loadSession = True


    hm_channels  = 2
    channel_size = 2
    storage_size = 2


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


    data_size     = 1_000
    batch_size    = 200
    hm_epochs     = 20
    learning_rate = 0.01
    dropout       = 0.3


    model \
        = make_model(hm_channels,channel_size,
            storage_size, (network1, network2))

    data \
        = make_data(from_file='samples_*.pkl',
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
        print(f'epoch {_} completed, loss: {round(epoch_loss, 3)}')
        losses.append(epoch_loss)

    save_session(model, optimizer)
    plot(losses)



cpu_count = cpu_count()



def process_batch(model, batch, dropout):

    with Pool(cpu_count) as P:
        results = P.map_async(process_sample, [(model.copy(), input, target, dropout) for (input, target) in batch])

        P.join()
        P.close()

    total_gradient = 0
    total_loss = 0

    for (gradient, loss) in results.get():
        total_gradient += numpy.array(gradient)
        total_loss += numpy.array(loss)

    for (grad, param) in zip(total_gradient, model.params):
        param.grad += grad

    return model, float(total_loss)


import numpy


def process_sample(model, input, target, dropout):

    output = propogate(model, input, len(target), dropout)
    loss = make_grads(output, target)
    grads = numpy.array([param.grad for param in model.params])

    return grads, loss





if __name__ == '__main__':
    run()
