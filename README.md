# Descale

Video/Image filter to undo upscaling.

Includes a VapourSynth and AviSynth+ plugin

## Usage

The VapourSynth plugin itself supports every constant input format. If the format is subsampled, left-aligned chroma planes are always assumed.
The included python wrapper, contrary to using the plugin directly, doesn't descale the chroma planes but scales them normally with `Spline36`.

```
descale.Debilinear(clip src, int width, int height, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Debicubic(clip src, int width, int height, float b=0.0, float c=0.5, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Delanczos(clip src, int width, int height, int taps=3, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline16(clip src, int width, int height, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline36(clip src, int width, int height, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline64(clip src, int width, int height, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Descale(clip src, int width, int height, str kernel, func custom_kernel, int taps=3, float b=0.0, float c=0.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, bool force=false, bool force_h=false, bool force_v=false, int opt=0)
```

The AviSynth+ plugin is used similarly, but without the `descale` namespace.
Custom kernels are only supported in the VapourSynth plugin.

### Custom kernels

```python
# Debilinear
core.descale.Descale(src, w, h, custom_kernel=lambda x: 1.0 - x, taps=1)

# Delanczos
import math
def sinc(x):
    return 1.0 if x == 0 else math.sin(x * math.pi) / (x * math.pi)
taps = 3
core.descale.Descale(src, w, h, custom_kernel=lambda x: sinc(x) * sinc(x / taps), taps=taps)

# You can also use the python wrapper instead of calling the plugin functions directly
import descale
descale.Decustom(src, w, h, lambda x: 1.0 - x, taps=1)
```

## How does this work?

Resampling can be described as `A x = b`.

A is an n x m matrix with `m` being the input dimension and `n` the output dimension. `x` is the original vector with `m` elements, `b` is the vector after resampling with `n` elements. We want to solve this equation for `x`.

To do this, we extend the equation with the transpose of A: `A' A x = A' b`.

`A' A` is now a banded symmetrical m x m matrix and `A' b` is a vector with `m` elements.

This enables us to use LDLT decomposition on `A' A` to get `LD L' = A' A`. `LD` and `L` are both triangular matrices.

Then we solve `LD y = A' b` with forward substitution, and finally `L' x = y` with back substitution.

We now have the original vector `x`.


## Compilation

By default only the VapourSynth plugin is compiled.
To build the AviSynth+ plugin, add `-Dlibtype=avisynth` or `-Dlibtype=both` to the meson command below.

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
