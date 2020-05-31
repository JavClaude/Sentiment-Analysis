import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import json

import torch
import progressbar

from utils import *
from model import LSTMModel

from sklearn.metrics import f1_score

def main(path_to_data: str,
         cache_dir: str, 
         texts_col: str,
         labels_col: str,
         n_classes: int,
         batch_size: int,
         batch_size_eval: int,
         min_lr: int,
         max_lr: int,
         n_epochs: int,
         cuda: int = 0):
    '''

    '''
    df = pd.read_csv(path_to_data)
    
    if os.path.isdir(cache_dir):
        logger.info('Cache dir found here {}'.format(cache_dir))
        pass
    else: 
        logger.info('Creating cache dir')
        os.mkdir(cache_dir)


    # Preprocess
    optimal_length = get_length(df, texts_col)
    X, vocab_size = encode_texts(df, texts_col, max_seq_length=optimal_length, return_vocab_size=True)

    y = get_labels(df, labels_col, n_classes)

    train_loader, test_loader = create_TorchLoaders(X, y, test_size=0.10, batch_size=batch_size, batch_size_eval=batch_size_eval)

    Model = LSTMModel(
        vocab_size=vocab_size,
        n_classes=n_classes
    )
    
    config_dict = {
        "vocab_size" : vocab_size,
        "n_classes"  : n_classes,
        "max_length" : optimal_length
    }

    if n_classes > 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.Adam(Model.parameters())

    ## Heuristic
    opt_cycle = ((((len(X)*(1-0.10))/batch_size)*n_epochs)*0.25)/2

    schedul = torch.optim.lr_scheduler.CyclicLR(optim, 
                                                min_lr, 
                                                max_lr, 
                                                step_size_up=opt_cycle, 
                                                step_size_down=opt_cycle,
                                                mode="exp_range",
                                                cycle_momentum=False)

    if cuda==1:
        Model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    metrics = {
        "training_loss" : [],
        "eval_loss" : [],
        "training_f1" : [],
        "eval_f1" : []
    }

    logger.info("Starting training for {} epochs".format(n_epochs))
    
    for epoch in range(n_epochs):
        Model.train()
        progress = progressbar.ProgressBar()
        for batch in progress(train_loader):
            batch = tuple(t for t in batch)

            inputs, labels = batch #unpacking
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            preds = Model(inputs)
            loss = criterion(preds, labels)

            ## Metrics computation
            metrics["training_loss"].append(loss.item())

            preds = preds.to("cpu").detach().numpy()
            preds = flat_pred(preds, 0.5)
            tmp_f1 = f1_score(labels.to("cpu").detach().numpy(), preds, average='macro')

            metrics["training_f1"].append(tmp_f1)

            ## Backward pass ##
            loss.backward()

            optim.step() #Gradient descent
            schedul.step()
            Model.zero_grad()
        
        logger.info("Epoch {} done with: training loss: {}\n training f1: {}".format(epoch, loss.item(), tmp_f1))
        
        ## Eval 
        progress = progressbar.ProgressBar()
        Model.eval()
        for batch in progress(test_loader):
            with torch.no_grad(): #computationaly efficient
                batch = tuple(t for t in batch)

                inputs, labels = batch
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)

                preds = Model(inputs)
                eval_loss = criterion(preds, labels)

                ## Eval metrics
                metrics["eval_loss"].append(eval_loss.item())

                preds = preds.to("cpu").detach().numpy()
                preds = flat_pred(preds, 0.5)
                tmp_f1 = f1_score(labels.to("cpu").detach().numpy(), preds, average='macro') ## detach 

                metrics["eval_f1"].append(tmp_f1)
        
        logger.info("Evaluation at iteration {} done: eval loss: {}\n eval f1: {}".format(epoch, eval_loss.item(), tmp_f1))


    ## Bring back model to cpu
    Model.cpu() 
    
    ## Get/Save param dict
    logger.info('Saving model in cache dir {}'.format(cache_dir))
    torch.save(Model.state_dict(), os.path.join(cache_dir,'state_dict.pt'))
    with open(os.path.join(cache_dir,'config_model.json'), 'w') as file:
        json.dump(config_dict, file)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="path to the data directory", type=str)
    parser.add_argument("--cache_dir", help="cache directory", type=str)
    parser.add_argument("--texts_col", help="name of the column containing textual data", type=str)
    parser.add_argument("--labels_col", help="name of the column containing labels", type=str)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_eval", type=int, default=64)
    parser.add_argument("--min_lr", type=float, default=0.00001)
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=0)

    args = parser.parse_args()

    main(
        path_to_data = args.data_path,
        cache_dir = args.cache_dir, 
        texts_col = args.texts_col,
        labels_col = args.labels_col,
        n_classes = args.n_classes,
        batch_size = args.batch_size,
        batch_size_eval=args.batch_size_eval,
        min_lr = args.min_lr,
        max_lr = args.max_lr,
        n_epochs = args.n_epochs,
        cuda = args.cuda
    )