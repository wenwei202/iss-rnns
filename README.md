*To duplicate, please use the exact tensorflow versions as mentioned.*

# About
This is TensorFlow implementation for training sparse LSTMs and other Recurrent Neural Networks. 
Related paper is publised in ICLR 2018: [Learning Intrinsic Sparse Structures within Long Short-term Memory](https://www.microsoft.com/en-us/research/publication/learning-intrinsic-sparse-structures-within-long-short-term-memory/).
Both structurally sparse LSTMs and non-structurally sparse LSTMs are supported by the code.
The work on sparse CNNs is available [here](https://github.com/wenwei202/caffe/tree/scnn). Poster is [here](/Poster_Wen_ICLR2018.pdf).

We use L1-norm regularization to obtain non-structurally sparse LSTMs.
The effectiveness of L1-norm regularization is similar to connection pruning, which can significantly reduce parameters in LSTMs but the irregular pattern of non-zero weights may not be friendly for computation efficiency.

We use group Lasso regularization to obtain structurally sparse LSTMs.
It can both reduce parameters in models and obtain regular nonzero weights for fast computation.

We proposed Intrinsic Sparse Structures (ISS) in LSTMs. By removing one component of ISS, we can simultaneously remove one hidden state, one cell state, one forget gate, one input gate, one output gate and one input update. 
In this way, we get a regular LSTM but with hidden size reduced by one. The method of learning ISS is based on group Lasso regularization. The ISS approach is also extended to Recurrent Highway Networks to learn the number of units per layer.

# Examples
## Stacked LSTMs
Code in [ptb](/ptb) is stacked LSTMs for language modeling of Penn TreeBank dataset.
## Recurrent Highway Networks
Code in [rhns](/rhns) is ISS for Recurrent Highway Networks. ISS is proposed in LSTMs but can be easily extended to other recurrent neural networks like Recurrent Highway Networks.
## Attention model
Code in [bidaf](/bidaf) is an attention+LSTM model for Question Answering of SQuAD dataset.
