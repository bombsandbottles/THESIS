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
    data, fs = librosa.load(filepath, sr=44100, mono=False)

    # Create a time vector for the audio
    if len(data) == 2:
        t = np.linspace(0, (len(data[0,:])/fs), len(data[0,:]))
    else:
        t = np.linspace(0, (len(data)/fs), len(data))
    
    # Return all the goods
    return data, fs, t

def k_filter(data, fs):
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

def calc_loudness(data, measurement = 'momentary', params = []):
    """
    data: audio in array form via librosa
    measurement = Momentary, Short-Term, Integrated or Loudness-Range (LRA).  
                    These change the window size and overlap amount
    param = a list of win_size and overlap for a custom loudness measurement

    !!
    In the future this will be updated to reflect the findings in
    Pestana, Reiss, Barbosa (2013) where the time block is 280ms
    and the first stage of the K-Filtering reflects a 10dB boost
    instead of 4dB
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
    elif measurement == 'custom':
        win_size = params[0]
        overlap = params[1]
    elif measurement == 'integrated':
        win_size = 400
        overlap = 75
    elif measurement == 'lra':
        win_size = 3000
        overlap = 66
        relative_adjustment = -20
        prc_low = 10
        prc_high = 95

    if len(data) == 2:
        # Seperate left and right channels
        data_l = data[0,:]               
        data_r = data[1,:]

        # K-filter
        data_l, fs_filt = k_filter(data_l, fs)
        data_r, fs_filt = k_filter(data_r, fs)

        # Buffer the signal matrix-style (input, block-size, hop-size)
        win_size = win_size*(fs_filt/1000)
        data_l = librosa.util.frame( data_l, win_size, win_size - (win_size*(overlap/100)) )
        data_r = librosa.util.frame( data_r, win_size, win_size - (win_size*(overlap/100)) )

        # Get the mean-square over each window
        z_l = np.mean(np.square(data_l), axis=0)
        z_r = np.mean(np.square(data_r), axis=0)

        # Sum the left and right channel, Convert to Loudness
        loudness_k = -0.691 + (10 * np.log10(np.add(z_l, z_r)))

        # Create Corresponding Time Vector
        loudness_t = create_time_vector(loudness_k, fs_filt, win_size - (win_size*(overlap/100)))
        
        if measurement == 'momentary' or measurement == 'short' or measurement == 'custom':
            return loudness_k, loudness_t

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

    # For Mono Data Now
    else:
        # K-filter
        data, fs_filt = k_filter(data, fs)

        # Buffer the signal matrix-style (input, block-size, hop-size)
        win_size = win_size*(fs_filt/1000)
        data = librosa.util.frame( data, win_size, win_size - (win_size*(overlap/100)) )

        # Get the mean-square over each window
        z = np.mean(np.square(data), axis=0)

        # Convert to Loudness
        loudness_k = -0.691 + (10 * np.log10(z))

        # Create Corresponding Time Vector
        loudness_t = create_time_vector(loudness_k, fs_filt, win_size - (win_size*(overlap/100)))
        
        if measurement == 'momentary' or measurement == 'short' or measurement == 'custom':
            return loudness_k, loudness_t

        elif measurement == 'integrated' or measurement == 'lra':
            # Calculate the relative threshold
            sum_m  = 0
            Jg_len = len(loudness_k[loudness_k >= abs_thresh])
            Jg_idx = np.greater(loudness_k, abs_thresh)
            for i in xrange(len(Jg_idx)):
                if Jg_idx[i] == True:
                    sum_m += z[i]
            sum_Jg = (sum_m)/Jg_len
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
                sum_m  = 0
                Jg_len = len(loudness_k[loudness_k >= rel_thresh])
                Jg_idx = np.greater(loudness_k, rel_thresh)
                for i in xrange(len(Jg_idx)):
                    if Jg_idx[i] == True:
                        sum_m += z[i]
                sum_Jg = (sum_m)/Jg_len
                
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

    else:
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

    else:
        data_matrix = librosa.util.frame(data, win_size, win_size)
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
    # Convert win_size from ms to samples
    win_size_s = np.floor(win_size*(fs/1000))

    # Get the RMS level per window
    rms = calc_rms(data, win_size_s)

    # Get the peak audio level per window
    peaks = get_peaks_cf(data, win_size_s)

    # Figure out the active frames
    activity = calc_activity(data, win_size)

    # Calculate the Crest Factor per window
    crest_factor = np.multiply(np.divide(peaks, rms), activity)
    
    # Create Time Vector
    crest_factor_t = create_time_vector(crest_factor, fs, win_size_s)
    
    return crest_factor, crest_factor_t

def calc_activity(data, win_size):
    """
    data: audio array in mono or stereo
    win_size: size in ms for the block analysis

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
    params = [win_size, 0]
    LUFS, LUFS_t = calc_loudness(data, 'custom', params)

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

def create_time_vector(data, fs_old, hop_size):
    """
    Creates a time vector to correspond to a windowed vector

    Parameters
    ----------
    data: audio array in mono or stereo

    hop_size: size in samples of the hop amount

    fs_old: The sample rate of the signal pre-windowing

    Returns
    -------
    A time vector corresponding to the windowed data
    """
    # Convert Sampling Rate based on Hop Size
    fs_new = fs_old/hop_size

    # Create Time Vector
    return np.linspace(0, (len(data)/fs_new), len(data))

def calc_spectral_centroid(data, win_size=2048):
    """
    Calculates the spectral centroid per stft window

    MIR toolbox says 50ms Window Time for All Spectral Features
    We're going to use 2048 samples to keep it power of 2 as 50ms at 44100 is 2000.
    Using 50 percent overlap as in MIR toolbox

    Parameters
    ----------
    data: audio array in mono

    win_size: analysis block size in samples

    Returns
    -------
    The spectral centroid for each window
    """
    # Compute an STFT but only keep the magnitudes
    X_matrix = np.abs(compute_stft(data, win_size))

    # Create hz vector and tile it for easy matrix multiplication
    fk = generate_fft_bins(win_size)
    fk_matrix = np.transpose(np.tile(fk, (X_matrix.shape[1], 1)))

    # Multiply each bin frequency by each magnitude, sum each window
    numerator = np.sum(np.multiply(fk_matrix, X_matrix), axis=0)

    # Sum the magnitudes for each window
    denominator = np.sum(X_matrix, axis=0)

    # Divide each windows results for the SC
    return np.divide(numerator, denominator)

def calc_spectral_spread(data, win_size=2048):
    """
    Calculates the spectral spread per stft window

    MIR toolbox says 50ms Window Time for All Spectral Features
    We're going to use 2048 samples to keep it power of 2 as 50ms at 44100 is 2000.
    Using 50 percent overlap as in MIR toolbox

    !!! THIS IS BROKEN AS SHIT RIGHT NOW??? !!!

    Parameters
    ----------
    data: audio array in mono

    win_size: analysis block size in samples

    Returns
    -------
    The spectral centroid for each window
    """
    # Compute an STFT but only keep the magnitudes (Square the mags?!?!?!)
    X_matrix = np.square(np.abs(compute_stft(data, win_size)))

    # Generate fk vector and tile it so each column is a window
    fk = generate_fft_bins(win_size)
    fk_matrix = np.transpose(np.tile(fk, (X_matrix.shape[1], 1)))

    # Calc sc and convert it into a matrix for easy calculations
    sc = calc_spectral_centroid(data)
    sc_matrix = np.tile(sc, (X_matrix.shape[0], 1))

    # Perform (fk-sc(m))^2
    numerator = np.square(np.subtract(fk_matrix, sc_matrix))
    # Multiply (fk-sc(m))^2 against X(m,k)
    numerator = np.multiply(numerator, X_matrix)
    # Sum the columns for each window
    numerator = np.sum(numerator, axis=0)

    # Perform sum(X(m,k)) to get denominator
    denominator = np.sum(X_matrix, axis=0)

    # Divide each windows results for the SS
    return np.sqrt(np.divide(numerator, denominator))

def force_mono(data, mode="arithmetic_mean"):
    """
    Forces a signal to mono...

    !!! Use this function in the calculation of spectral features? !!!

    Parameters
    ----------
    data: audio array

    mode: geometric_mean or sum, arithmetic_mean

    Returns
    -------
    An audio array that has now 1 dimension

    """
    if len(data) == 1:
        return data
    else:
        if mode == "arithmetic_mean":
            return np.mean(data, axis=0)
        elif mode == "geometric_mean":
            return scipy.stats.mstats.gmean(data, axis=0)
        elif mode == "sum":
            return np.sum(data, axis=0)
            
def compute_stft(data, win_size=2048, overlap=50, center=False):
    """
    Computes an STFT for a mono formatted signal

    !!! Unsure the exact difference of rfft and fft !!!

    Parameters
    ----------
    data: audio array in mono

    win_size: analysis block size in samples
    
    overlap: amount of overlap in percent
    
    center: a switch to pad the signal such that time 0 is at the center of the first window

    Returns
    -------
    An STFT matrix where each column is an FFT of a window

    """
    if len(data) == 2:
        print "Data needs to be mono!"
        print "Forcing data to mono..."
        data = force_mono(data)
        
    # Buffer the audio up
    audio_matrix = librosa.util.frame( data, win_size, win_size - (win_size*(overlap/100)) )

    # Create window
    window = np.hanning(win_size)

    # Window the audio frame by frame
    window_matrix = np.transpose(np.tile(window, (len(audio_matrix[0,:]), 1)))

    # Window the signal via dot multiplication
    windowed_audio = np.transpose(np.multiply(audio_matrix, window_matrix))

    # FFT
    return np.transpose(np.fft.rfft(windowed_audio))

def generate_fft_bins(win_size, fs=44100):
    """ 
    Given the fft window size, generate the hz value per bin
    
    Parameters
    ----------
    win_size: window size in samples

    fs: sample rate

    Returns
    -------
    A vector containing the hz value per bin (fk)

    """
    # Generate win_size points from 0hz-fs
    fk = np.linspace(0, fs, win_size)

    # Remove symmetry and cut to fs/2
    return fk[0:(len(fk)//2)+1]

def plot_stft(stft):
    # Lifted from Librosa Documentation to Plot STFT Quick/Easy
    D = librosa.logamplitude(np.abs(stft)**2, ref_power=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
        
def calc_zcr(data, win_size=2048):
    
    pass
