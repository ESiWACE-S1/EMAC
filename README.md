EMAC on Cartesius:
===================
First copy the file ``mh-linux-64__cartesius`` to ``messy_2.54.0/config/mh-linux-64``. Then do 
```
module load 2019

module load netCDF-Fortran/4.4.4-intel-2018b

module load CUDA/10.1.243

./configure

make -j8
```
