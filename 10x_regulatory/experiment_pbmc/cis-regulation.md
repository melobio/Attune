#### cis-regulation of PCHi-C dataset
To verify the performance of Attune in cis-regulation inference, we follow the instruction of Ge Gao's work (GLUE) with settings of different distances.

##### Data
1. PCHi-C dataset is released in Javierre B.M. study. The downlink and preprocession can be found in https://github.com/gao-lab/GLUE/tree/master/data/hic/Javierre-2016.
2. 10k PBMC dataset from 10x platform is downloaded from offical website.

##### Comparison
The RegInf experiment in GLUE is worked well for performance evaluation (https://github.com/gao-lab/GLUE/tree/master/experiments/RegInf). Based on notebooks of GLUE, we change some codes in section of HVGs and genomic distance.