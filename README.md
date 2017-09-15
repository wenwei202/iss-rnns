# About
This is TensorFlow implementation for training sparse LSTMs. 
Related paper is in **Learning Intrinsic Sparse Structures within Long Short-term Memory**.
Both structurally sparse LSTMs and non-structurally sparse LSTMs are supported by the code.

We use L1-norm regularization to obtain non-structurally sparse LSTMs.
The effectiveness of L1-norm regularization is similar to connection pruning, which can significantly reduce parameters in LSTMs but the irregular pattern of non-zero weights may not be friendly for computation efficiency.

We use group Lasso regularization to obtain structurally sparse LSTMs.
It can both reduce parameters in models and obtain regular nonzero weights for fast computation.

We proposed Intrinsic Sparse Structures (ISS) in LSTMs. By removing one component of ISS, we can simultaneously remove one hidden state, one cell state, one forget gate, one input gate, one output gate and one input update. 
In this way, we get a regular LSTM but with hidden size reduced by one. The method of learning ISS is based on group Lasso regularization.
