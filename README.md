CraTENet
========

CraTENet is a multi-output deep neural network with multi-head self-attention for thermoelectric property prediction, 
based on the [CrabNet](https://github.com/anthony-wang/CrabNet) architecture. This repository contains code that can 
be used to reproduce the experiments described in the paper, including an implementation of the CraTENet model, using 
the Tensorflow and Keras frameworks. It also provides a means of obtaining the data required for the experiments.

<img src="resources/cratenet-arch.png" width="60%"/>

## Obtaining the Training Data

To reproduce the experiments in the paper, data must be obtained from the Ricci et al. electronic transport database, 
and transformed into a format that the CraTENet model accepts. Although files containing the training data are provided 
for download, and can be immediately used with the model, the entire data pre-processing pipeline is described here for 
the sake of transparency and reproducibility.

### 1. Downloading the Ricci et al. database

The full contents of the original Ricci et al. database can be downloaded from 
[https://doi.org/10.5061/dryad.gn001](https://doi.org/10.5061/dryad.gn001). At the time of this writing, the dataset on
the [Dryad](https://datadryad.org/stash) website (which hosts the data) is organized into a number of different files.
For the purposes of this project, we're only interested in the files `etransport_data_1.tar` and 
`etransport_data_2.tar`. These files must be downloaded, and their contents extracted. The contents of these archives 
are compressed .json files, one for each compound (identified by their Materials Project ID). The .tar archives contain 
thousands of compressed .json files, thus, it is perhaps best to extract a .tar file's contents into its own directory, 
for ease of use.

_NOTE: It is not required that the Ricci et al. database data be downloaded. This can skipped. This information is 
provided for the sake of full reproducibility, should one wish to derive the training data from the original database._ 

### 2. Extracting the <i>S</i> and <i>σ</i> Tensor Diagonals and Band Gap

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

Alternatively, previously extracted <i>S</i> and <i>σ</i> tensor diagonals can be downloaded directly:
```
python bin/fetch_data.py xyz
```
The `xyz` argument specifies that the tensor diagonals data should be downloaded. To download the previously extracted 
band gap data, use the `gap` argument instead:
```
python bin/fetch_data.py gap
```

_NOTE: It is not required that these extracted datasets be obtained. This can skipped. This information is 
provided for the sake of full reproducibility, should one wish to derive the training data from the original database._ 

### 3. Computing the <i>S</i>, <i>σ</i> and <i>PF</i> Traces

Once the tensor diagonals have been extracted, the traces of the <i>S</i> and <i>σ</i> tensors, and the power factor 
(<i>PF</i>) trace, must be computed. These datasets can be created using the `all_data_xyz.csv` file.

For example, to create the Seebeck traces:
```
$ python bin/compute_traces.py seebeck --data all_data_xyz.csv.gz --out seebeck_traces.csv.gz
```
Similarly, the `cond` argument can be used (in place of the `seebeck` argument) to compute the electronic 
conductivity traces, and the `pf` argument can be used to compute the power factor traces.

Alternatively previously computed traces can be downloaded directly:
```
$ python bin/fetch_data.py seebeck_traces
```
The `cond_traces` argument can be used (in place of the `seebeck_traces` argument) to download previously computed 
electronic conductivity traces, and the `pf_traces` argument can be used to download previously computed power factor 
traces. 

_NOTE: It is not required that these trace datasets be obtained. This can skipped. This information is 
provided for the sake of full reproducibility, should one wish to derive the training data from the original database._
