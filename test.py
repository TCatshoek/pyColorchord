import pyaudio
import time
import numpy as np
import pygame

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

# Get frequency from piano key number https://en.wikipedia.org/wiki/Piano_key_frequencies
def getfreq(key_n):
    return 2**((key_n - 49) / 12) * 440

#TODO:
def foldfft(freqs):
    tmp = np.zeros(12)
    for i in range(len(freqs) // 24):
        tmp[:] += np.mean(np.reshape(freqs[(i*24) : (i*24) + 24], (12,2)), axis=1)
    return tmp - 0.5 * np.mean(tmp)


# Open pyaudio input stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True)

# start the stream
stream.start_stream()

# MAIN LOOP
should_quit = False
while stream.is_active() and not should_quit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True

    # Read samples
    n_samples = 1024

    samples = stream.read(n_samples)
    samples = np.frombuffer(samples, dtype=np.float32)

    # Apply windowing function
    samples = np.bartlett(len(samples)) * samples

    # Gather frequencies to DFT at
    freqs = np.array([getfreq(key_n) for key_n in np.arange(0, 108, 0.5)])

    # Perform DFT
    # We cannot do FFT because we need the frequency bins to be chromatic
    dftime = time.time()
    dfts = np.abs(do_dfts(freqs, samples, 44100))
    print(f'DFT took {time.time() - dftime} s, { 1/ (time.time() - dftime)} Hz')

    # Draw visualization
    screen.fill((0, 0, 0))

    per_note = window_w // len(dfts)
    for i, note in enumerate(dfts):
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(i * per_note + 1, window_h, per_note - 1, note * -2000))


    pygame.display.flip()




# Cleanup
stream.stop_stream()
stream.close()
p.terminate()