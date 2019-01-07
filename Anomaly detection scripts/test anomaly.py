from models import load_model
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', choices = ['adam', 'sgd', 'adagrad'], default='adam')
parser.add_argument('--loss', choices = ['mean_squared_error', 'binary_crossentropy'], default='mean_squared_error')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_samples', type=int, default=50)
parser.add_argument('--result', default=os.path.join(curdir, 'result.png'))

def main(args):
    
    x_ww_npz = np.load("../x_ww_bw_50176_pre-processed.npz")
    x = x_ww_npz['arr_0']
    x_train = x[:int(len(x)*0.8)]
    x_test = x[int(len(x)*0.8):]
    
    x_tt_npz = np.load("../x_ttbar_bw_50176_pre-processed.npz")
    x_abnormal = x_tt_npz['arr_0']
    
    perm = np.random.permutation(args.test_samples)
    x_test = x_test[perm][:args.test_samples]
    x_abnormal = x_abnormal[perm][:args.test_samples]
    
    model_names = ['autoencoder', 'deep_autoencoder']
    
    for model_name in model_names:
        model = load_model(model_name)
        
        model.compile(optimizer=args.optimizer, loss=args.loss)

        model.fit(x=x_train,y=x_train, epochs=args.epochs, batch_size=args.batch_size)

        #test
        x_concat = np.concatenate([x_test, x_abnormal], axis=0)
        losses = []
        for x in x_concat:
            x = np.expand_dims(x, axis=0)
            loss = model.test_on_batch(x, x)
            losses.append(loss)

        #plot
        plt.plot(range(len(losses)), losses, linestyle='-', lw =1, label=model_name)

        del model
        
    #create graph
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('sample index')
    plt.ylabel('loss')
    plt.savefig(args.result)
    plt.clf()
               
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)