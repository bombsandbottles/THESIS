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

def calc_rms(data, win_size):
    """
    data: audio as numpy array to be analyzed
    win_size: value in samples to create the blocks for analysis
    
    Simple RMS function that can accomodate stereo data.
    Used in calc_crest_factor

    return: RMS of signal
    """
    if len(data) == 2:
        # Seperate left and right channels
        data_l = data[0,:]               
        data_r = data[1,:]

        # Buffer up the data
        data_matrix_l = librosa.util.frame(data_l, win_size, win_size)
        data_matrix_r = librosa.util.frame(data_r, win_size, win_size)

        # Square and sum the left and right seperatley
        sum_l = np.sum(np.square(data_matrix_l), axis=0)
        sum_r = np.sum(np.square(data_matrix_r), axis=0)

        # Sum the left and right channels, take the mean, and sqrt
        return np.sqrt(np.divide(np.add(sum_l, sum_r), win_size*2))

    elif len(data) == 1:
        # Buffer up the data and perform root mean square
        data_matrix = librosa.util.frame(data, win_size, win_size)
        return np.sqrt(np.mean(np.square(data_matrix), axis=0))

def get_peaks_cf(data, win_size):
    """
    data: audio as numpy array to be analyzed
    win_size: value in samples to create the blocks for analysis
    
    Used in calc_crest_factor, this function returns an array of peak levels
    for each window.

    return: array of peak audio levels
    """
    if len(data) == 2:
        # Seperate left and right channels
        data_l = data[0,:]               
        data_r = data[1,:]

        # Buffer up the data
        data_matrix_l = librosa.util.frame(data_l, win_size, win_size)
        data_matrix_r = librosa.util.frame(data_r, win_size, win_size)

        # Get peaks for left and right channels
        peaks_l = np.amax(np.absolute(data_matrix_l), axis=0)
        peaks_r = np.amax(np.absolute(data_matrix_r), axis=0)
        return np.maximum(peaks_l, peaks_r)

    elif len(data) == 1:
        return np.amax(np.absolute(data_matrix), axis=0)

def calc_crest_factor(data, win_size, fs=44100):
    """
    data: audio as numpy array to be analyzed
    fs: sample rate of the data
    win_size: value in ms to create the blocks for analysis
    
    Given a window size for analysis (1s and 100ms typically for this research), 
    find the crest factor value of that windows as an indicator for dynamic range

    !!! calc_activity gets passed win_size in time still and not samples, this can 
    be done more elegantly !!!

    return: crest factor
    """
    # Buffer the signal matrix-style (input, block-size, hop-size)
    win_size_s = np.floor(win_size*(fs/1000))

    # Get the RMS level per window
    rms = calc_rms(data, win_size_s)

    # Get the peak audio level per window
    peaks = get_peaks_cf(data, win_size_s)

    # Figure out the active frames
    activity = calc_activity(data, win_size)

    # Calculate the Crest Factor per window
    return np.multiply(np.divide(peaks, rms), activity)

def calc_activity(data, win_size):
    """
    data: audio array in mono or stereo
    win_size: size in samples for the block analysis

    Utilyzing a Hysteresis Noise Gate, a time block is considered to be
    either active or inactive.  Hysteresis thresholds at -25 and -30 LUFS 
    are used to help prevent excessive switching of states.

    Concept pulled from  Mansbridge, Finn, and Reiss (2012)
        Implementation and Evaluation of Autonomous Multi-track Fader Control

    returns: an array of containings zeros and ones 
                                            zero = inactive
                                            one = active
    """
    # Define constants
    upper_thresh = -25
    lower_thresh = -30
    past_frame   = 0

    # Get our LUFS values
    LUFS = custom_LUFS(data, win_size)

    # Pre-Allocate Active Frames
    active_frames = np.zeros(len(LUFS))

    for i in xrange(len(LUFS)):
        # If above -25LUFS, 1 equal active frame
        if LUFS[i] > upper_thresh:
            active_frames[i] = 1

        # If the past frame was above -25LUFS and this ones above -30
        elif past_frame > upper_thresh and LUFS[i] > lower_thresh:
            active_frames[i] = 1
            
        # If the current frame is below the threshold, non-active
        elif LUFS[i] < upper_thresh:
            active_frames[i] = 0
        
        # The analyzed frame becomes the past frame
        past_frame = LUFS[i]

    return active_frames

def custom_LUFS(data, win_size, overlap=0):
    if len(data) == 2:
        # Seperate Left and Right Channels
        data_l = data[0,:]
        data_r = data[1,:]

        # K-filter
        data_l, fs_filt = temp_kfilter(data_l, fs=44100)
        data_r, fs_filt = temp_kfilter(data_r, fs=44100)

        # Buffer the signal matrix-style (input, block-size, hop-size)
        win_size = win_size*(fs_filt/1000)
        data_l = librosa.util.frame( data_l, win_size, win_size - (win_size*(overlap/100)) )
        data_r = librosa.util.frame( data_r, win_size, win_size - (win_size*(overlap/100)) )

        # Get the mean-square over each window
        z_l = np.mean(np.square(data_l), axis=0)
        z_r = np.mean(np.square(data_r), axis=0)

        # Sum the left and right channel, Convert to Loudness
        return -0.691 + (10 * np.log10(np.add(z_l, z_r)))

    elif len(data) == 1:
        # K-filter
        data, fs_filt = temp_kfilter(data, fs=44100)

        # Buffer the signal matrix-style (input, block-size, hop-size)
        win_size = win_size*(fs_filt/1000)
        data = librosa.util.frame( data, win_size, win_size - (win_size*(overlap/100)) )

        # Get the mean-square over each window
        z = np.mean(np.square(data), axis=0)

        # Sum the left and right channel, Convert to Loudness
        return -0.691 + (10 * np.log10(z))
    
def temp_kfilter(data, fs):
    """
    x_t: audio data in samples across time
    fs: sample rate of x_t

    TEMPORARY FUNCTION UNTIL THE LOUDNESS FUNCTION IS FIXED FOR ACTIVITY

    return: k-filtered data AND new 48khz fs
    """ 
    # Convert fs to 48khz to do K-Filtering
    if fs != 48000:
        data = librosa.resample(data, fs, 48000)
        fs  = 48000

    # Hi-Shelf Boost of +4dB at 1681hz
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]

    # Create High-Pass roll off at 38hz
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    b2 = [1.0, -2.0, 1.0]

    # Filter in succession
    return lfilter(b2, a2, lfilter(b1, a1, data)), fs


