from __future__ import division
import numpy as np
import scipy
from scipy.signal import lfilter
import librosa
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn
# seaborn.set(style='ticks')
# import IPython.display

def import_audio(filepath):
    """
    filepath: string pointing to audio file on disk

    Simple function to quickly read audio data into numpy array
    
    returns: data = audio in samples across time
             data_l / _r = left and right channels seperated
             fs = sample rate
             t = time vector corresponding to data
    """ 
    # filepath = '/Users/harrison/Desktop/Bass Line.wav'
    data_s, fs = librosa.load(filepath, sr=44100, mono=False)

    # Seperate Left and Right Channels
    data_l = data_s[0,:]                # Left
    data_r = data_s[1,:]                # Right

    # Create a time vector for the audio
    t = np.linspace(0, (len(data_l)/fs), len(data_l))

    # Return all the goods
    return data_s, data_l, data_r, fs, t

def k_filter(x_t, fs, t):
    """
    x_t: audio data in samples across time
    fs: sample rate of x_t

    Performs standard 48khz K-Filtering as outlined in the ITU-R BS.1770-3 documentation.

    return: k-filtered data AND new 48khz fs
    """ 
    # Convert fs to 48khz to do K-Filtering
    if fs != 48000:
        x_t = librosa.resample(x_t, fs, 48000)
        t   = librosa.resample(t, fs, 48000)
        fs  = 48000

    # Hi-Shelf Boost of +4dB at 1681hz
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]

    # Create High-Pass roll off at 38hz
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    b2 = [1.0, -2.0, 1.0]

    # Filter in succession
    return lfilter(b2, a2, lfilter(b1, a1, x_t)), fs, t

def calc_loudness(filepath, measurement = 'momentary'):
    """
    filepath: audio to be analyzed
    measurement = Momentary, Short-Term, Integrated or Loudness-Range (LRA).  
                    These change the window size and overlap amount
    !!
    In the future this will be updated to reflect the findings in
    Pestana, Reiss, Barbosa (2013) where the time block is 280ms
    and the first stage of the K-Filtering reflects a 10dB boost
    instead of 4dB

    Only works for stereo audio.
    !!

    return: LUKS/LUFS
    """
    # Threshold constant
    abs_thresh = -70
    relative_adjustment = -10

    # Change win_size and overlap for type of analysis
    if measurement == 'momentary':
        win_size = 400
        overlap = 0
    elif measurement == 'short':
        win_size = 3000
        overlap = 0
    elif measurement == 'integrated':
        win_size = 400
        overlap = 75
    elif measurement == 'lra':
        win_size = 3000
        overlap = 66
        relative_adjustment = -20
        prc_low = 10
        prc_high = 95

    # Import Audio
    data_s, data_l, data_r, fs, t = import_audio(filepath)

    # K-filter
    data_l, fs_filt, t_filt = k_filter(data_l, fs, t)
    data_r, fs_filt, t_filt = k_filter(data_r, fs, t)

    # Buffer the signal matrix-style (input, block-size, hop-size)
    win_size = win_size*(fs_filt/1000)
    data_l = librosa.util.frame( data_l, win_size, win_size - (win_size*(overlap/100)) )
    data_r = librosa.util.frame( data_r, win_size, win_size - (win_size*(overlap/100)) )

    # Get the mean-square over each window
    z_l = np.mean(np.square(data_l), axis=0)
    z_r = np.mean(np.square(data_r), axis=0)

    # Sum the left and right channel, Convert to Loudness
    loudness_k = -0.691 + (10 * np.log10(np.add(z_l, z_r)))
    
    if measurement == 'momentary' or measurement == 'short':
        return loudness_k

    elif measurement == 'integrated' or measurement == 'lra':
        # Calculate the relative threshold
        sum_l  = 0
        sum_r  = 0
        Jg_len = len(loudness_k[loudness_k >= abs_thresh])
        Jg_idx = np.greater(loudness_k, abs_thresh)
        for i in xrange(len(Jg_idx)):
            if Jg_idx[i] == True:
                sum_l += z_l[i]
                sum_r += z_r[i]
        sum_Jg = (sum_l+sum_r)/Jg_len
        rel_thresh = (-0.691 + (10 * np.log10(sum_Jg))) + relative_adjustment

        # Break off and compute LRA
        if measurement == 'lra':
            # Include values greater than relative threshold.
            gated_loudness = loudness_k[loudness_k >= rel_thresh]
            n = len(gated_loudness)
            gated_loudness = np.sort(gated_loudness)

            # Get values at 10 and 95 percent
            perc_low = gated_loudness[round((n-1)*(prc_low/100))]
            perc_high = gated_loudness[round((n-1)*(prc_high/100))]

            # Computer LRA
            return (perc_high - perc_low)

        elif measurement == 'integrated':
            # Calculate the integrated loudness using relative threshold
            sum_l  = 0
            sum_r  = 0
            Jg_len = len(loudness_k[loudness_k >= rel_thresh])
            Jg_idx = np.greater(loudness_k, rel_thresh)
            for i in xrange(len(Jg_idx)):
                if Jg_idx[i] == True:
                    sum_l += z_l[i]
                    sum_r += z_r[i]
            sum_Jg = (sum_l+sum_r)/Jg_len
            
            # Return Integrated Loudness Value
            return -0.691 + (10 * np.log10(sum_Jg))











