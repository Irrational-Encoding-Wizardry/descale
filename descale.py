from enum import IntEnum
from vapoursynth import core, GRAYS, RGBS, GRAY, YUV, RGB


class BorderHandling(IntEnum):
    MIRROR = 0
    ZERO = 1

class Opt(IntEnum):
    AUTO = 0
    NONE = 1
    AVX2 = 2

# If yuv444 is True chroma will be upscaled instead of downscaled (i.e. output is in 4:4:4 format)
# If gray is True the output will be grayscale (i.e. just the luma plane is processed and returned)
def Debilinear(src, width, height, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bilinear', border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Debicubic(src, width, height, b=0.0, c=0.5, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bicubic', b=b, c=c, border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Delanczos(src, width, height, taps=3, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='lanczos', taps=taps, border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline16(src, width, height, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline16', border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline36(src, width, height, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline36', border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline64(src, width, height, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline64', border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Decustom(src, width, height, custom_kernel, taps, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, custom_kernel=custom_kernel, taps=taps, border_handling=border_handling, yuv444=yuv444, gray=gray, chromaloc=chromaloc)


def Descale(src, width, height, kernel=None, custom_kernel=None, taps=None, b=None, c=None, border_handling=None, yuv444=False, gray=False, chromaloc=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    if src_cf == RGB and not gray:
        rgb = to_rgbs(src).descale.Descale(width, height, kernel, custom_kernel, taps, b, c, border_handling=border_handling)
        return rgb.resize.Point(format=src_f.id)

    y = to_grays(src).descale.Descale(width, height, kernel, custom_kernel, taps, b, c, border_handling=border_handling)
    y_f = core.query_video_format(GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError('Descale: The output dimension and the subsampling are incompatible.')

    uv_f = core.query_video_format(src_cf, src_st, src_bits, 0 if yuv444 else src_sw, 0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

    return core.std.ShufflePlanes([y,uv], [0,1,2], YUV)


# Helpers

def to_grays(src):
    return src.resize.Point(format=GRAYS)


def to_rgbs(src):
    return src.resize.Point(format=RGBS)


def get_plane(src, plane):
    return core.std.ShufflePlanes(src, plane, GRAY)
