# OutOfEquilibrium-ProtSeq

## Getting started

Clone this repository on your local machine by running
```bash
git clone git@github.com:Bitbol-Lab/OutOfEquilibrium-ProtSeq.git
```
and move inside the root folder.

### 2 states
The 2states folder contains three scripts to generate sequences. First, ```Generate_sequences_eq_cluster.py``` allows to generate independent sequences at equilibrium. Second, ```Generate_sequences_VaryingTemperature.py``` allows to generate sequences with varying selection strength (temperature). Note that an equilibrium dataset is needed to generate sequences with varying temperature because an equilibrium sequence is taken at the beginning of the evolution. 
Finally, ```Generate_sequences_phylogeny_binaryvariables.py``` allows to generate sequences with phylogeny using a random ancestor at the root of the phylogeny.

The ```Analyse_data.py``` allows to infer contacts using mean-field DCA.

### Realistic model


