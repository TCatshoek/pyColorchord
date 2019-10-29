import numpy as np
import math

# HSV to RGB conversion, http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


colormem = {}
def bin2color(bin, bins_p_octave):
    if (bin, bins_p_octave) in colormem.keys():
        return colormem[(bin, bins_p_octave)]
    else:
        color = hsv2rgb(((bin % bins_p_octave) / bins_p_octave) * 360, 1, 1)
        colormem[(bin, bins_p_octave)] = color
        return color


# Get frequency from piano key number https://en.wikipedia.org/wiki/Piano_key_frequencies
def getfreq(key_n):
    return 2**((key_n - 49) / 12) * 440


def foldfft(freqs, n_bins, n_per_octave):
    assert len(freqs) % n_bins == 0, f'n freqs {len(freqs)}, not divisable by {n_bins}'

    tmp = np.zeros(n_bins)

    for i in range(len(freqs) // n_per_octave):
        tmp[:] += freqs[i * n_per_octave: i * n_per_octave + n_per_octave]
    return tmp - 0.9 * np.min(tmp)