# GrayScott with numba

Solve Gray-Scott equation with [numba](https://numba.readthedocs.io/en/stable/)

We use a numba installation with and without [SVML library](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml). SVML is Intel library provides a short vector math library, check if you are correctly install SVML

```bash
$ numba -s | grep SVML
__SVML Information__
SVML State, config.USING_SVML                 : True
SVML Library Loaded                           : True
llvmlite Using SVML Patched LLVM              : True
SVML Operational                              : True
```

## Conda environnement for Gray-scott resolution

```bash
conda create --name gs_env1 python=3.9 -y
conda activate gs_env1
conda install numpy -y 
pip install opencv-python
pip install  matplotlib
conda install numba -y 
conda install -c numba icc_rt -y
....
conda deactivate
```


# Performances

For 1000 images full HD (1920, 1080) an image is saved each 34 iteration (ie 34000 iterations)
I use 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz


## With @njit without SVML

```bash
step_frame=34
nb_frame=1000
CPU time= 439.44128702300003 s
(1000, 1920, 1080)
```

0.0129 second by iteration

## With @njit with SVML

```bash
step_frame=34
nb_frame=1000
CPU time= 352.219058588 s
(1000, 1920, 1080)
```

0.0103 second by iteration

