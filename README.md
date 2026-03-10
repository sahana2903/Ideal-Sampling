# Reg.no : 212224060225
# Name   : Sahana Thasneem S.N

# Impulse, Natural, & Flat-top -Sampling

# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.

# Tools required
Computer with Google collab

# Program
# Impulse Sampling
```
# Impulse Sampling

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

fs = 100
t = np.arange(0, 1, 1/fs)

f = 5
signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(
    t_sampled,
    signal_sampled,
    linefmt='r-',
    markerfmt='ro',
    basefmt='r-',
    label='Sampled Signal (fs = 100 Hz)'
)
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))
# plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(
    t,
    reconstructed_signal,
    'r--',
    label='Reconstructed Signal (fs = 100 Hz)'
)
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```

# Natural Sampling
```
# Natural Sampling

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000                     # Sampling frequency (samples per second)
T = 1                         # Duration in seconds
t = np.arange(0, T, 1/fs)     # Time vector

fm = 5                        # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50               # Pulses per second
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]

# Interpolation - Zero-Order Hold (for visualization)
reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index + pulse_width] = sampled_signal[i]

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(
    t,
    reconstructed_signal,
    label='Reconstructed Message Signal',
    color='green'
)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

# Flat-top Sampling
```
# Flat-Top Sampling Simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Sampling parameters
fs = 1000          # Sampling frequency (Hz)
T = 1              # Duration (seconds)
t = np.arange(0, T, 1/fs)

# Message signal
fm = 5             # Message frequency (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse train parameters
pulse_rate = 50    # Pulses per second
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))

pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1

# Flat-top sampling
flat_top_signal = np.zeros_like(t)
sample_times = t[pulse_train_indices]
pulse_width_samples = int(fs / (2 * pulse_rate))  # Pulse width

for sample_time in sample_times:
    index = np.argmin(np.abs(t - sample_time))
    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

# Low-pass filter function
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

# Reconstruction
cutoff_freq = 2 * fm
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

# Plotting
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices],
         basefmt=" ", label='Ideal Sampling Instances')
plt.title('Ideal Sampling Instances')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, color='green',
         label=f'Reconstructed Signal (Cutoff = {cutoff_freq} Hz)')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

# Output Waveform
# Impulse Sampling
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/d8dbbffd-0cc9-4019-97b4-25f767982e5b" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/75406847-639d-47bf-8428-da33aab09603" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/d7c31ed9-27df-4a7f-9044-db72f3fc7327" />

# Natural Sampling
<img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/b77079a0-c726-451f-adae-df2e0d3756e7" />

# Flat-top Sampling
<img width="1398" height="990" alt="image" src="https://github.com/user-attachments/assets/ddb6d45f-3a91-4269-b32b-dd116ac127e2" />

# Results
Thus, the construction and reconstruction of Impulse, Natural, and Flat-top sampling were successfully implemented using Python, and the corresponding waveforms were obtained.

