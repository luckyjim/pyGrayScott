# GrayScott with numba

Solve Gray-Scott equation with [numba](https://numba.readthedocs.io/en/stable/)

We use a numba installation with and without [SVML library](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml). SVML is Intel library provides a short vector math library.

```bash
$ numba -s | grep SVML
__SVML Information__
SVML State, config.USING_SVML                 : True
SVML Library Loaded                           : True
llvmlite Using SVML Patched LLVM              : True
SVML Operational                              : True
```

We can use :
* just add @njit decrator to naive implementation
* use [@stencil feature](https://numba.readthedocs.io/en/stable/user/stencil.html#using-the-stencil-decorator) to improve numpy approch




# Performances

For 1000 images full HD (1920, 1080) an image is saved each 34 iteration (ie 34000 iterations), see [ref C++ implementation](https://lappweb.in2p3.fr/~paubert/PERFORMANCE_WITH_STENCIL/5-4-1-4-5345.html)

I use 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz


## Naive implementation with @njit without SVML

```bash
step_frame=34
nb_frame=1000
CPU time= 439.44128702300003 s
(1000, 1920, 1080)
```

0.0129 second by iteration

## Naive implementation with @njit with SVML

```bash
step_frame=34
nb_frame=1000
CPU time= 352.219058588 s
(1000, 1920, 1080)
```

0.0103 second by iteration

## With @stencil feature

TO DO