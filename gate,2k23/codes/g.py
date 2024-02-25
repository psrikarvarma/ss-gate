import numpy as np
import matplotlib.pyplot as plt

def generate_bandlimited_pulse(duration, bandwidth, amplitude):
    t = np.linspace(-duration / 2, duration / 2, num=1000)  # Time array centered around 0

    # Generate a Gaussian-shaped pulse
    pulse = amplitude * np.exp(-(t ** 2) / (duration / 10))

    # Scale the waveform such that its peak value is A/2
    peak_value = np.max(np.abs(pulse))
    pulse /= peak_value  # Normalize the waveform
    pulse *= amplitude  # Scale the peak value to A/2

    return t, pulse

# Parameters
duration = 2   # Duration of the signal
bandwidth = 2  # Bandwidth of the pulse
amplitude = 1  # Original amplitude of the pulse

# Generate band-limited pulse
t, signal = generate_bandlimited_pulse(duration, bandwidth, amplitude)

# Plotting
plt.plot(t, signal)
plt.xlabel('f')
plt.ylabel('M(f)')
plt.grid(True)

# Set x-axis limits from -B to B
plt.xlim(-bandwidth / 2, bandwidth / 2)

# Set y-axis limits from -A/2 to A/2
plt.ylim(0, 10*amplitude/9)

# Set y-axis ticks and labels as multiples of amplitude A from 0 to A
plt.yticks(np.arange(0, amplitude + 0.1, step=0.2)*amplitude, labels=[f'{i:.2f}A' for i in np.arange(0, amplitude + 0.1, step=0.2)])

# Specify the number of ticks within the range from -B to B
num_ticks = 5
x_ticks = np.linspace(-bandwidth / 2, bandwidth / 2, num=num_ticks)
x_labels = [f'{i:.1f}B/2π' for i in np.linspace(-1, 1, num=num_ticks)]
plt.xticks(x_ticks, labels=x_labels)

plt.axhline(y=bandwidth/2,color='r',linestyle='--')
plt.savefig('../figs/ec,27.png')
plt.show()
