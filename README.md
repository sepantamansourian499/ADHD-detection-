# ADHD-detection-using EEG signals


### EEG-to-RGB Conversion

This project converts multichannel EEG signals into RGB image representations to capture frequency-band characteristics in a spatial–spectral format.  
For each EEG segment \( x_{ch}(t) \), where *ch* denotes the channel and *t* represents time samples, three parallel band-pass filters are applied to isolate characteristic EEG rhythms:

- **Red (R) channel:** 4–8 Hz (theta band)  
- **Green (G) channel:** 8–12 Hz (alpha band)  
- **Blue (B) channel:** 12–40 Hz (beta/gamma bands)

Each filtered sub-band signal is normalized to the range [0, 1] and stacked along the last dimension, forming an RGB image of size  
`(channels × samples × 3)`.

This representation encodes temporal–spectral activity as color intensity variations across EEG channels, allowing convolutional neural networks (CNNs) or ResNet backbones to directly extract spatial–spectral features from 2-D image inputs instead of raw 1-D waveforms.

