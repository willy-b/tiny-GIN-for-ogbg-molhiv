# Code for ogbg-molhiv leaderboard submission using GIN

(for https://web.archive.org/web/20240324173558/https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv )

Results so far: rocauc 0.7825 +/- 0.0121 (mean +/- sample std, n=10).
These were deterministic results obtained running on CPU using the random seeds 0..9 inclusive.

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

## Acknowledgements

Credit to Stanford XCS224W, they had a homework assignment using the ogbg-molhiv dataset which inspired this (the OGB leaderboard was not mentioned in and is not part of the course and this is NOT a copy paste of the homework - which used GCNConv instead of GIN for example).

Note, this uses the atom and not the edge features of the dataset. This is a work in progress and at time of writing I have been working on this for ~2 days, so haven't tried that yet. This same code will work if the dataset id is changed for ogbg-molpcba -- I am adding that next and attempting to pretrain on that to improve performance on this right now.