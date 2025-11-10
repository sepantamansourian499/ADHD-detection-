# ADHD-detection-


In this study, multichannel EEG signals were transformed into RGB image representations to capture frequency-band characteristics in a spatial–spectral format. For each EEG segment x(t), where 
ch denotes the electrode channel and 
𝑡
t represents the time samples, three parallel band-pass filters were applied to isolate distinct EEG rhythms corresponding to the following frequency ranges:

Red (R) channel: 4–8 Hz (theta band)

Green (G) channel: 8–12 Hz (alpha band)

Blue (B) channel: 12–40 Hz (beta/gamma bands)

Each sub-band signal was min–max normalized to the range 
[
0
,
1
]
[0,1] and stacked along the last dimension, producing an RGB image of size 
(
channels
×
samples
×
3
)
(channels×samples×3). This representation encodes temporal–spectral activity as color intensity variations across EEG channels, enabling convolutional neural networks (CNNs) or ResNet-based architectures to directly learn spatial–spectral features from two-dimensional inputs instead of raw one-dimensional waveforms.
