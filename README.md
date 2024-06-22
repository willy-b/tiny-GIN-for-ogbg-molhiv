# Code for ogbg-molhiv leaderboard submission using GIN

(for https://web.archive.org/web/20240324173558/https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv )

Results so far: rocauc 0.7835 +/- 0.0125 (mean +/- sample std, n=10).
These were computed from deterministic results obtained running on CPU using the random seeds 0..9 inclusive (deterministic for same software version on same setup, CPU of L4 instance in Google Colab, see notebook link below).

Nothing impressive, just to practice GNNs and participate!
Other techniques, e.g. decision tree on molecular fingerprints are more efficient for this specific task but GNNs are very generalizable and flexible for all kinds of data!

## Steps to reproduce ogbg-molhiv leaderboard submission using GIN

1. Install dependencies (run `install_dependencies.sh` this comes with or commands below):

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-geometric # I'm using 2.5.3 right now
pip install ogb # I'm using 1.3.6 right now
```

2. Run this script `python main_gin.py` (I'm using python 3.10.12 but should be flexible)

## Leaderboard submission specifics

Hyperparameter values used:

- num_layers: 2

- hidden_dim: 64

- dropout: 0.5

- learning_rate: 0.001

- epochs: 50

- batch_size: 32

- weight_decay: 1e-6

Trained network from scratch on CPU of Google Colab L4 instance (used L4 GPU for speed in hyperparameter search, CPU for deterministic final results),
with the following random seeds and obtained the following results:

(it will give same results if using CPU of Google Colab L4 and same version of software; see Google Colab notebook link below; randomness affects the training process when training from scratch)

```
seed 0: 0.7923 valid, 0.7937 test
seed 1: 0.8084 valid, 0.7988 test
seed 2: 0.8106 valid, 0.7803 test
seed 3: 0.7909 valid, 0.7920 test
seed 4: 0.8027 valid, 0.7987 test
seed 5: 0.8053 valid, 0.7741 test
seed 6: 0.8070 valid, 0.7646 test
seed 7: 0.8047 valid, 0.7685 test
seed 8: 0.7888 valid, 0.7760 test
seed 9: 0.7987 valid, 0.7888 test
```

If you like Jupyter notebooks and/or Google Colab, you can check the results above in the following notebook, and/or copy the following notebook and run the commands to reproduce the result in their environment (please use CPU of L4 instance to reproduce randomness exactly):

https://colab.research.google.com/drive/11lx7DRuEhfdRGDu1Q_oWvILGQKrP2IsP?usp=sharing

Using `torch.mean()` and `torch.std()` to report the mean and unbiased sample standard deviation, one obtains:

```
>>> data = torch.tensor([0.7937, 0.7988, 0.7803, 0.7920, 0.7987, 0.7741, 0.7646, 0.7685, 0.7760, 0.7888])

>>> data.mean()
tensor(0.7835)

>>> data.std()
tensor(0.0125)
```

Note that the **test set performance is NEVER consulted or checked by the code in selecting the model**.
For each random seed during training the held out validation set performance is checked and the epoch with best validation set performance observed so far which did not have worse training performance than the previous best is kept as the result. If one preferred not to use the validation data to select the best model checkpoint, an additional subset of training data could be held out just to determine the best checkpoint to use, and it would not be expected to affect the test performance (selection of model checkpoint just needs to be done using data not used for training/gradient descent and which is not the test data upon which score is reported).

CSVs for validation and test set predictions vs ground truth will be generated as part of the script, if however you want examples for each of the seeds, I can provide upon request. A .pkl with the model weights will be generated at the end of the script which could be reused for inference. If you would like example weights, they are available upon request (or I can add them here if multiple people ask and would not want to generate on their CPU or in Google colab).

## Acknowledgements

Credit to Stanford XCS224W, they had a homework assignment using the ogbg-molhiv dataset which inspired this (the OGB leaderboard was not mentioned in and is not part of the course and this is NOT a copy paste of the homework - which used GCNConv instead of GIN for example).

Note, this uses the atom and not the edge features of the dataset. This is a work in progress and at time of writing I have been working on this for ~2 days, so haven't tried that yet. This same code will work if the dataset id is changed for ogbg-molpcba -- I am adding that next and attempting to pretrain on that to improve performance on this right now.