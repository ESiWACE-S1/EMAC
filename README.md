EMAC on Cartesius CPU-only:
===========================
First copy the file ``mh-linux-64__cartesius`` to ``messy_2.54.0/config/mh-linux-64``. Then do 
```
module load 2019

module load netCDF-Fortran/4.4.4-intel-2018b

./configure

make -j8
```

EMAC on Cartesius with CUDA:
============================
First copy the file ``mh-linux-64__cartesius`` to ``messy_2.54.0/config/mh-linux-64``. Clone the medina package and copy (or link) the source directory into messy-src/messy/util. Then run ```python f2c_alpha.py``` in the util directory and the cuda code gets generated.
Then do 
```
module load 2019

module load netCDF-Fortran/4.4.4-intel-2018b

module load CUDA/10.1.243

module unload compilerwrappers

./configure

make -j8
```
