# Descale

VapourSynth plugin to undo upscaling.

## Usage

The plugin itself only supports GrayS input.
The included python wrapper supports YUV, Gray, and RGB of every bitdepth.

```
descale.Debilinear(src:clip, width:int, height:int, src_left:float, src_top:float)

descale.Debicubic(src:clip, width:int, height:int, b:float, c:float, src_left:float, src_top:float)

descale.Delanczos(src:clip, width:int, height:int, taps:int, src_left:float, src_top:float)

descale.Despline16(src:clip, width:int, height:int, src_left:float, src_top:float)

descale.Despline36(src:clip, width:int, height:int, src_left:float, src_top:float)
```

## How does this work?

Resampling can be described as `A x = b`.

A is an n x m matrix with m being the input dimension and n the output dimension. x is the original vector with m elements, b is the vector after resampling with n elements. We want to solve this equation for x.

To do this, we extend the equation with the transpose of A: `A' A x = A' b`.

`A' A` is now a banded symmetrical m x m matrix and `A' b` is a vector with m elements.

This enables us to use LDLT decomposition on `A' A` to get `LD L' = A' A`. LD and L are both triangular matrices.

Then we solve `LD y = A' b` with forward substitution, and finally `L' x = y` with back substitution.

We now have the original vector `x`.


## Compilation

### Linux
```
g++ -std=c++11 -shared -fPIC -O2 descale.cpp -o libdescale.so
```

### Cross-compilation for Windows
```
x86_64-w64-mingw32-g++ -std=c++11 -shared -fPIC -O2 descale.cpp -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic -s -o libdescale.dll
```