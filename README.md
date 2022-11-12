CraTENet
========

CraTENet is a multi-output deep neural network with multi-head self-attention for thermoelectric property prediction, 
based on the [CrabNet](https://github.com/anthony-wang/CrabNet) architecture. This repository contains code that can 
be used to reproduce the experiments described in the paper, including an implementation of the CraTENet model, using 
the Tensorflow and Keras frameworks. It also provides a means of obtaining the data required for the experiments.

<img src="resources/cratenet-arch.png" width="65%"/>

## Obtaining the Data

To reproduce the experiments in the paper, data from the Ricci et al. electronic transport database must be obtained and 
converted into a format that can be used with the Machine Learning models.

### Downloading from the Ricci et al. database

The full contents of the original Ricci et al. database can be downloaded from 
[https://doi.org/10.5061/dryad.gn001](https://doi.org/10.5061/dryad.gn001). At the time of this writing, the dataset on
the [Dryad](https://datadryad.org/stash) website (which hosts the data) is organized into a number of different files.
For the purposes of this project, we're only interested in the files `etransport_data_1.tar` and 
`etransport_data_2.tar`. These files must be downloaded, and their contents extracted. The contents of these archives 
are compressed .json files, one for each compound (identified by their Materials Project ID). The .tar archives contain 
thousands of compressed .json files, thus, it is perhaps best to extract a .tar file's contents into its own directory, 
for ease of use. 

### Extracting the <i>S</i> and <i>σ</i> Tensor Diagonals and Band Gap

Assuming that the Ricci et al. electronic transport database files have been downloaded and exist in two directories, 
`etransport_data_1/` and `etransport_data_2`, the following script can be used to extract the <i>S</i> and <i>σ</i> 
tensor diagonals (from which the target values will ultimately be derived):
```
python bin/extract_data_xyz.py --dir ./etransport_data_1 ./etransport_data_2 --out all_data_xyz.csv
```
The same can be done to extract the band gaps associated with each compound:
```
python bin/extract_data_gap.py --dir ./etransport_data_1 ./etransport_data_2 --out all_data_gap.csv
```

Alternatively, the extracted <i>S</i> and <i>σ</i> tensor diagonals can be downloaded directly:
```
python bin/fetch_data.py xyz
```
The `xyz` argument specifies that the tensor diagonals data should be downloaded. To download the extracted band gap 
data, use the `gap` argument instead:
```
python bin/fetch_data.py gap
```

