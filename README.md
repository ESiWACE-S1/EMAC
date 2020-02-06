EMAC on Cartesius:
===================
First copy the file ``mh_linux64__cartesius`` to ``config/mh_linux64`` in the MESSy root directory. Then execute 
```
module load 2019

module load netCDF-Fortran/4.4.4-intel-2018b

./configure

make -j8
```
