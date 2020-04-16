# Descale

VapourSynth plugin to undo upscaling.

## Usage

The plugin itself only supports GrayS, RGBS, and YUV444PS input.
The included python wrapper supports YUV (every subsampling), Gray, and RGB of every bitdepth.

```
descale.Debilinear(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Debicubic(clip src, int width, int height, float b=0.0, float c=0.5, float src_left=0.0, float src_top=0.0)

descale.Delanczos(clip src, int width, int height, int taps=3, float src_left=0.0, float src_top=0.0)

descale.Despline16(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Despline36(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Despline64(clip src, int width, int height, float src_left=0.0, float src_top=0.0)
```

## How does this work?

Resampling can be described as `A x = b`.

A is an n x m matrix with `m` being the input dimension and `n` the output dimension. `x` is the original vector with `m` elements, `b` is the vector after resampling with `n` elements. We want to solve this equation for `x`.

To do this, we extend the equation with the transpose of A: `A' A x = A' b`.

`A' A` is now a banded symmetrical m x m matrix and `A' b` is a vector with `m` elements.

This enables us to use LDLT decomposition on `A' A` to get `LD L' = A' A`. LD and L are both triangular matrices.

Then we solve `LD y = A' b` with forward substitution, and finally `L' x = y` with back substitution.

We now have the original vector `x`.


## Compilation

### Linux

```
$ meson build
$ ninja -C build
```

### Cross-compilation for Windows
```
$ meson build --cross-file cross-mingw-x86_64.txt
$ ninja -C build
```
