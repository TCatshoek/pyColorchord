import pyaudio
import time
import numpy as np
import pygame
import math
from scipy.ndimage import gaussian_filter1d

# Setup pygame
pygame.init()
window_w = 865
window_h = 512
screen = pygame.display.set_mode((window_w, window_h))

# instantiate PyAudio (1)
p = pyaudio.PyAudio()


pi2 = np.pi * 2
# e^jx = sin(x) + j*cos(x)
# Numpy dft for a single frequency
def do_dft(freq, signal, samplerate):
    freq_step = (len(signal) * freq) / samplerate

    frac_dists = np.array(range(len(signal))) / len(signal)
    points = signal * np.exp(-1j*pi2*(freq_step)*frac_dists)

    return np.mean(points)

# Fast-ish vectorized dft for multiple given frequencies
def do_dfts(freqs, signal, samplerate):
    freqs = np.array(freqs)

    freq_steps = (len(signal) * freqs) / samplerate

    frac_dists = np.array(range(len(signal))) / len(signal)

    frac_steps = np.reshape(freq_steps, (-1, 1)).dot(np.reshape(frac_dists, (-1, 1)).T)

    points = signal * np.exp(-1j*pi2*frac_steps)

    return np.mean(points, axis=1)

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

#TODO:
def foldfft(freqs, n_bins, n_per_octave):
    assert len(freqs) % n_bins == 0, f'n freqs {len(freqs)}, not divisable by {n_bins}'

    tmp = np.zeros(n_bins)

    for i in range(len(freqs) // n_per_octave):
        tmp[:] += freqs[i * n_per_octave: i * n_per_octave + n_per_octave]
    return tmp - 0.9 * np.min(tmp)


# Open pyaudio input stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True)

# start the stream
stream.start_stream()

# Piano key numbers
octaves = 9
#pianokeys = np.arange(25, (octaves * 12) + 1, 0.5)
pianokeys = np.arange(25, (octaves * 12) + 1, 0.5)
# Gather frequencies to DFT at
freqs = np.array([getfreq(key_n) for key_n in pianokeys])

# Memory to keep rolling average
n_keep = 3
avg_mem_idx = 0
avg_mem = np.zeros((n_keep, len(freqs)))

# Keep notes
actual_notes = np.zeros(12)

# MAIN LOOP
should_quit = False
while stream.is_active() and not should_quit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True

    start_time = time.time()

    # Read samples
    n_samples = 512

    samples = stream.read(n_samples)
    samples = np.frombuffer(samples, dtype=np.float32)

    # Apply windowing function
    samples = np.bartlett(len(samples)) * samples

    # Perform DFT
    # We cannot do FFT because we need the frequency bins to be chromatic
    dftime = time.time()
    dfts = np.abs(do_dfts(freqs, samples, 44100))
    print(f'DFT took {time.time() - dftime} s, { 1/ (time.time() - dftime)} Hz')

    # Taper first octave
    taper = np.ones(dfts.size)
    taper[0:48] = np.linspace(0, 1, 48)
    #taper[0:48] = 1 - np.flip(np.geomspace(0.0001, 1, 48))
    dfts = dfts * taper

    # Add dft result to rolling average
    avg_mem[avg_mem_idx, :] = dfts
    avg_mem_idx = (avg_mem_idx + 1) % n_keep
    avged_dfts = np.mean(avg_mem, axis=0)

    # Fold dft output
    notes = foldfft(avged_dfts, 24, 24)

    # Filter notes
    sigma = 1
    filtered_notes = gaussian_filter1d(notes, sigma, mode='wrap')

    # Find peaks
    peaks = []
    for idx, (prev, cur, next) in enumerate([filtered_notes[x : x + 3] for x in range(len(filtered_notes) - 4)]):
        if prev < cur and cur > next:
            peaks.append((idx, cur))

    # Only keep n larges peaks
    n_keep_peaks = 5
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[0:n_keep_peaks]

    # Put peaks in note bins
    for (idx, amplitude) in peaks:
        if idx in range(0, 24, 2):
           actual_notes[idx // 2] += amplitude
        else:
            prev = math.floor(idx / 2)
            next = math.ceil(idx / 2)
            actual_notes[prev] += 0.5 * amplitude
            actual_notes[next] += 0.5 * amplitude

    # Decay note bins
    actual_notes *= 0.8

    # Draw visualization
    screen.fill((0, 0, 0))

    # Draw folded dft
    per_note = window_w // len(notes)
    for i, note in enumerate(notes):
        #color = (255, 0, 0) if np.argmax(notes) == i else (255, 255, 255)
        color = bin2color(i, 24)
        pygame.draw.rect(screen, color, pygame.Rect(i * per_note + 1, 0, per_note - 1, note * 2000))

    # Draw filtered note distribution
    amp = 4000
    per_note = window_w // len(filtered_notes)
    # Calc line points
    coords = [(per_note * i + 0.5 * per_note, note * amp) for i, note in enumerate(filtered_notes)]
    # Loop around edges
    coords = [(-1.5 * per_note, filtered_notes[23] * amp)] + coords + [((len(filtered_notes) + 1.5) * per_note, filtered_notes[0] * amp)]
    pygame.draw.lines(screen, (255, 255, 255), False, coords)

    # Draw peak points
    for (idx, amplitude) in peaks:
        pygame.draw.circle(screen, bin2color(idx + 1, 24), (int(per_note * idx + 1.5 * per_note), int(amplitude * amp)), int(amplitude * 1000))

    # Draw complete dft
    per_note = window_w // len(avged_dfts)
    for i, note in enumerate(avged_dfts):
        color = bin2color(i, 24)
        pygame.draw.rect(screen, color, pygame.Rect(i * per_note + 1, window_h, per_note - 1, note * -2000))

    # Draw actual notes
    per_note = window_w // len(actual_notes)
    for i, note in enumerate(actual_notes):
        color = bin2color(i, 12)
        pygame.draw.rect(screen, color, pygame.Rect(i * per_note + 1, window_h - 100, per_note - 1, note * -2000))

    end_time = time.time()

    print('Total time', end_time - start_time,  1 / (end_time - start_time), 'Fps')

    pygame.display.flip()




# Cleanup
stream.stop_stream()
stream.close()
p.terminate()