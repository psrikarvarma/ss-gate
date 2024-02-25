import numpy as np
import matplotlib.pyplot as plt

def generate_bandlimited_pulse(duration, bandwidth, amplitude, center):
    t = np.linspace(-duration / 2 + center, duration / 2 + center, num=1000)  # Time array centered around 'center'

    # Generate a Gaussian-shaped pulse
    pulse = amplitude * np.exp(-((t - center) ** 2) / (duration / 10))

    # Scale the waveform such that its peak value is A/2
    peak_value = np.max(np.abs(pulse))
    pulse /= peak_value  # Normalize the waveform
    pulse *= amplitude / 2  # Scale the peak value to A/2

    return t, pulse

# Parameters
duration = 2   # Duration of the signal
bandwidth = 2  # Bandwidth of the pulse
amplitude = 1  # Original amplitude of the pulse
separation = 20 * bandwidth  # Separation between the pulses
center1 = -5 * bandwidth
center2 = 5 * bandwidth

# Generate band-limited pulses
t1, signal1 = generate_bandlimited_pulse(duration, bandwidth, amplitude, center1)
t2, signal2 = generate_bandlimited_pulse(duration, bandwidth, amplitude, center2)

# Plotting
plt.plot(t1, signal1, color='blue')
plt.plot(t2, signal2, color='blue')
plt.xlabel('f')
plt.ylabel('Y(f)')
plt.grid(True)

# Set x-axis limits
plt.xlim(-6*bandwidth,6*bandwidth)

# Set y-axis limits from 0 to A
plt.ylim(0, amplitude)

# Set y-axis ticks and labels as multiples of amplitude A from 0 to A
plt.yticks(np.arange(0, amplitude + 0.1, step=0.2)*amplitude, labels=[f'{i:.2f}A' for i in np.arange(0, amplitude + 0.1, step=0.2)])

# Specify the number of ticks within the range from -6B to 6B
num_ticks = 9
x_ticks = np.linspace(-6*bandwidth, 6*bandwidth, num=num_ticks)
x_labels = [f'{int(i)}B/2π' for i in np.linspace(-12, 12, num=num_ticks)]
plt.xticks(x_ticks, labels=x_labels)

plt.axhline(y=bandwidth/4,color='r',linestyle='--')

plt.legend()
plt.savefig('../figs/ec,27(1).png')
plt.show()
