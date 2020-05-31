import torch

class LSTMModel(torch.nn.Module):
    '''
    LSTM model Baseline

    parameters: 
    ----------

            - vocab_size = int, vocabulary size
            - embed_dim = int, embedding dim 
            - lstm_units = int, number of lstm units, 256 by default 
            - n_layers = int, number of lstm layers 
            - ffn = int, number of units in feed foward net 
            - n_classes = int, number of classes 
            - do_normalizarion = bool, if true apply layer normalization
            - pooling_strategy = str, pooling over the hidden dimension 
    '''
    def __init__(self,
                vocab_size: int,
                embed_dim: int = 300,
                lstm_units: int = 256, 
                n_layers: int = 2,
                ffn: int = 128, 
                n_classes: int = None,
                do_normalization: bool = True,
                pooling_strategy: str = 'avg',
                **kwargs):
        
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, lstm_units, num_layers = n_layers, batch_first = True, bidirectional = True)
        if do_normalization:
            self.LN = torch.nn.LayerNorm(lstm_units * 2)
        self.linear = torch.nn.Linear(lstm_units * 2, ffn)
        self.act = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(ffn, n_classes)

        self.do_norm = do_normalization
        self.pooling = pooling_strategy 

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        
        ## GlobalAvg pooling 
        if self.pooling == 'avg':
            x = torch.mean(x, axis = 1)

        if self.do_norm:
            x = self.LN(x)
        
        x = self.linear(x)
        x = self.act(x)

        x = self.linear2(x)

        return x

    
        