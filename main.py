import pyaudio
import time
import numpy as np
import pygame
import math
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax

from dfts import do_dfts
from util import *

samplerate = 44100
n_samples = 1024

bin_width = samplerate / n_samples

# Setup pygame
pygame.init()
window_w = 865
window_h = 512
screen = pygame.display.set_mode((window_w, window_h))

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# Open pyaudio input stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=samplerate,
                input=True)

# start the stream
stream.start_stream()

# Piano key numbers
n_skipped = 1
octaves = 6
#pianokeys = np.arange(25, (octaves * 12) + 1, 0.5)
pianokeys = np.arange((12 * n_skipped) + 1, (12 * n_skipped) + 1 + (octaves * 12), 0.5)
# Gather frequencies to DFT at
freqs = np.array([getfreq(key_n) for key_n in pianokeys])

# ---- create scaling based on how often frequency bins overlap -----
# Need previous two octaves to get bins correct
prev_pianokeys = np.arange((12 * n_skipped) + 1 - 24, (12 * n_skipped) + 1, 0.5)
prev_freqs = np.array([getfreq(key_n) for key_n in prev_pianokeys])
allfreqs = np.concatenate((prev_freqs, freqs))

# Create scaling based on bin width
bins = []
for freq in allfreqs:
    halfbinw = bin_width / 2
    lower = freq - halfbinw
    upper = freq + halfbinw
    bins.append((lower, upper))

# Count the amount of bins a certain frequency is in
scale_count = np.zeros(len(freqs))
for idx, freq in enumerate(freqs):
    for (l, h) in bins:
        if freq > l and freq <= h:
            # How far from the sides are we?
            ldist = freq - l
            rdist = h - freq

            #normalize
            ldist_n = ldist / (ldist + rdist)
            rdist_n = rdist / (ldist + rdist)

            dist = min(ldist_n, rdist_n) * 2

            scale_count[idx] += dist

scaler = 1 - (scale_count / np.max(scale_count))


# Memory to keep rolling average
n_keep = 5
avg_mem_idx = 0
avg_mem = np.zeros((n_keep, len(freqs)))

# Keep notes
n_keep_notes = 1
actual_notes_idx = 0
actual_notes_mem = np.zeros((n_keep_notes, 12))

# MAIN LOOP
should_quit = False
while stream.is_active() and not should_quit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True

    start_time = time.time()

    # Read samples
    samples = stream.read(n_samples)
    samples = np.frombuffer(samples, dtype=np.float32)

    # Apply windowing function
    samples = np.blackman(len(samples)) * samples

    # Perform DFT
    # We cannot do FFT because we need the frequency bins to be chromatic
    dftime = time.time()
    dfts = np.abs(do_dfts(freqs, samples, samplerate))
    print(f'DFT took {time.time() - dftime} s, { 1/ (time.time() - dftime)} Hz')

    #Taper first octave
    taper = np.ones(dfts.size)
    taper[0:24] = np.linspace(0, 1, 24)
    #taper[0:48] = 1 - np.flip(np.geomspace(0.0001, 1, 48))
    dfts = dfts * scaler
    #dfts = dfts * taper

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

    # Include wraparound for first and last bin
    wrapback = [np.array([filtered_notes[-1], filtered_notes[0], filtered_notes[1]])]
    wrapfront = [np.array([filtered_notes[-2], filtered_notes[-1], filtered_notes[0]])]
    triplets = [filtered_notes[x: x + 3] for x in range(len(filtered_notes) - 2)]
    triplets = wrapback + triplets + wrapfront
    for idx, (prev, cur, next) in enumerate(triplets):
        if prev < cur and cur > next:
            peaks.append((idx, cur))

    # Only keep n largest peaks
    n_keep_peaks = 5
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[0:n_keep_peaks]

    # Put peaks in note bins
    for (idx, amplitude) in peaks:
        if idx in range(0, 24, 2):
            actual_notes_mem[actual_notes_idx, idx // 2] = amplitude
        else:
            prev = math.floor(idx / 2)
            next = math.ceil(idx / 2)
            actual_notes_mem[actual_notes_idx, prev] = 0.5 * amplitude
            actual_notes_mem[actual_notes_idx, next % 12] = 0.5 * amplitude

    # Decay note bins
    actual_notes_mem[actual_notes_idx, :] *= 0.8

    # # Try softmax to see if it helps isolate notes better?
    # actual_notes_mem[actual_notes_idx, :] = softmax(actual_notes_mem[actual_notes_idx])
    # actual_notes_mem[actual_notes_idx, :] -= np.min(actual_notes_mem[actual_notes_idx])
    actual_notes_idx = (actual_notes_idx + 1) % n_keep_notes
    actual_notes = np.mean(actual_notes_mem, axis=0) * 0.1

    # --------------Draw visualization-----------------
    screen.fill((0, 0, 0))

    # Draw folded dft
    per_note = window_w // len(notes)
    for i, note in enumerate(notes):
        color = (255, 255, 255) if np.argmax(notes) == i else bin2color(i, 24)
        #color = bin2color(i, 24)
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
        pygame.draw.circle(screen, bin2color(idx, 24), (int(per_note * idx + .5 * per_note), int(amplitude * amp)), int(amplitude * 1000))

    # Draw complete dft
    per_note = window_w // len(avged_dfts)
    for i, note in enumerate(avged_dfts):
        color = bin2color(i, 24)
        pygame.draw.rect(screen, color, pygame.Rect(i * per_note + 1, window_h, per_note - 1, note * -2000))

    # Draw actual notes
    per_note = window_w // len(actual_notes)
    for i, note in enumerate(actual_notes):
        color = bin2color(i, 12)
        pygame.draw.rect(screen, color, pygame.Rect(i * per_note + 1, window_h - 100, per_note - 1, note * -100000))

    end_time = time.time()

    print('Total time', end_time - start_time,  1 / (end_time - start_time), 'Fps')

    pygame.display.flip()


# Cleanup
stream.stop_stream()
stream.close()
p.terminate()