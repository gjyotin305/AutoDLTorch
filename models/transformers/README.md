## Transformers


### Positional Encoding:
It is used to inject the position information of each token in the input sequence. It uses sine and cosine functions of different frequencies to generate the positional encoding.

PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

i -> dimension
d_model -> embedding_size
pos -> position of the token

Each Dimension of the positional encoding corresponds to a sinusoid.The wavelengths form a geometric progression from 2pi to 10000*2pi.

### FeedForwardNetwork:
Here it is a simple three layer neural network with Dropout and ReLU activation function.

```bash
nn.Linear(d_model, d_hidden),
nn.ReLU(),
nn.Linear(d_hidden, d_hidden),
nn.Dropout(drop_prob),
nn.Linear(d_hidden, d_model)
```

