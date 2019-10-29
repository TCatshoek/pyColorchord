import numpy as np

pi2 = np.pi * 2
# e^jx = sin(x) + j*cos(x)
# Numpy dft for a single frequency
def do_dft(freq, signal, samplerate):
    freq_step = (len(signal) * freq) / samplerate

    frac_dists = np.array(range(len(signal))) / len(signal)
    points = signal * np.exp(-1j*pi2*(freq_step)*frac_dists)

    return np.mean(points)

# Actually we can do goertzel with non-int K yay
# https://www.dsprelated.com/showarticle/495.php
def do_dft_goertzel(freq, signal, samplerate):
    N = len(signal)
    f_step = samplerate / N

    k = freq / f_step

    alpha = 2 * np.pi * k / N
    beta = 2 * np.pi * k * (N-1) / N

    two_cos_a = 2*np.cos(alpha)
    a = np.cos(beta)
    b = -np.sin(beta)
    c = np.sin(alpha) * np.sin(beta) - np.cos(alpha) * np.cos(beta)
    d = np.sin(2 * np.pi * k)

    w1 = 0
    w2 = 0

    for sample in signal:
        w0 = sample + two_cos_a * w1 - w2
        w2 = w1
        w1 = w0

    return ((w1*a + w2*c) + 1j*(w1*b + w2*d)) / N


# # Test goertzel algorithm for faster dft https://www.embedded.com/the-goertzel-algorithm/
# # Even though this definitely won't be faster since it's pure python and not fast numpy C magic
# # This version only supports frequencies that are an integer multiple of the fundamental frequency
# # Which is no good for our purposes
# # https://gist.github.com/sebpiq/4128537
def do_dft_goertzel_int_k(freq, signal, samplerate):
    N = len(signal)

    f_step = samplerate / N
    f_step_normalized = 1.0 / N

    bin = int(np.floor(freq / f_step))

    f = bin * f_step_normalized

    w_real = 2 * np.cos(2 * np.pi * f)
    w_imag = np.sin(2 * np.pi * f)

    d1, d2 = 0.0, 0.0
    for sample in signal:
        y = sample + w_real * d1 - d2
        d2, d1 = d1, y

    #real = q1 - q2 * cos
    real = 0.5 * w_real * d1 - d2

    #imag = q2 * sin
    imag = w_imag * d1

    #mag = np.sqrt(real**2 + imag**2)
    mag = d2**2 + d1**2 - w_real * d1 * d2

    real /= N
    imag /= N
    mag /= N

    return real, imag, mag, f*samplerate


# Fast-ish vectorized dft for multiple given frequencies
def do_dfts(freqs, signal, samplerate):
    freqs = np.array(freqs)

    freq_steps = (len(signal) * freqs) / samplerate

    frac_dists = np.array(range(len(signal))) / len(signal)

    frac_steps = np.reshape(freq_steps, (-1, 1)).dot(np.reshape(frac_dists, (-1, 1)).T)

    points = signal * np.exp(-1j*pi2*frac_steps)

    return np.mean(points, axis=1)
