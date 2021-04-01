import random as rnd

rnd.seed(0)
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
from skimage.transform import resize
import os
os.environ['PYTHONHASHSEED']=str(0)
import numpy as np
from tqdm import tqdm
from scipy.fftpack import fft
from scipy.io import wavfile
from pyAudioAnalysis import audioAnalysis as AA
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import audioBasicIO
from AudioDataAugm import generator
from sklearn.model_selection import train_test_split 
from numpy import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def spectrogram(signal, sampling_rate, window, step, plot=False,
                show_progress=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (numOfShortTermWindows x num_fft)
    ARGUMENTS:
        signal:         the input signal samples
        sampling_rate:  the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        plot:           flag, 1 if results are to be ploted
        show_progress flag for showing progress using tqdm
    RETURNS:
    """
    window = int(window)
    step = int(step)
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    maximum = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (maximum - dc_offset)

    num_samples = len(signal)  # total number of signals
    count_fr = 0
    num_fft = int(window / 2)
    specgram = np.zeros((int((num_samples-step-window) / step) + 1, num_fft),
                        dtype=np.float64)
    for cur_p in tqdm(range(window, num_samples - step, step),
                      disable=not show_progress):
        count_fr += 1
        x = signal[cur_p:cur_p + window]
        X = abs(fft(x))
        X = X[0:num_fft]
        X = X / len(X)
        specgram[count_fr-1, :] = X

    freq_axis = [float((f + 1) * sampling_rate) / (2 * num_fft)
                 for f in range(specgram.shape[1])]
    time_axis = [float(t * step) / sampling_rate
                 for t in range(specgram.shape[0])]

    if plot:
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        fstep = int(num_fft / 5.0)
        frequency_ticks = range(0, int(num_fft) + fstep, fstep)
        frequency_tick_labels = \
            [str(sampling_rate / 2 -
                 int((f * sampling_rate) / (2 * num_fft)))
             for f in frequency_ticks]
        ax.set_yticks(frequency_ticks)
        ax.set_yticklabels(frequency_tick_labels)
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = \
            ['%.2f' % (float(t * step) / sampling_rate) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()
    #print(specgram.shape)
    return specgram, time_axis, freq_axis
    
def createSpecgramImage(signal, Fs, stWin, stStep):
    
    x = audioBasicIO.stereo_to_mono(signal)
    specgramOr, TimeAxis, FreqAxis = spectrogram(x, Fs, round(Fs * stWin), round(Fs * stStep), False) 
    """
    specgram_RGBa=cv2.resize(specgramOr, (250,250),interpolation = cv2.INTER_LINEAR)
    RGBa_image = Image.fromarray(np.uint8(plt.cm.jet(specgram_RGBa)*255))
    RGBimage = Image.fromarray(np.uint8(plt.cm.jet(specgram)*255))
    img=RGBimage.convert("RGB")
    """
    specgram=resize(specgramOr, (250,250))
    img = Image.fromarray(np.uint8(plt.cm.jet(specgram)*255), mode='RGB')
    
    #return np.asarray(img) , specgram , np.asarray(RGBa_image)
    return np.asarray(img)


def get_2sec_segment(fs, signal):

    #print('Signal Duration = {} seconds'.format(signal.shape[0] / fs))
    time_wav = np.arange(0, len(signal)) / fs

    signal_len = len(signal)
    segment_size_t = 2 # segment size in seconds
    segment_size = segment_size_t * fs  # segment size in samples

    # Break signal into list of segments in a single-line Python code
    segments = np.array([signal[x:x + segment_size] for x in np.arange(0, signal_len, segment_size)])
    
    segment_duration=0
    while(segment_duration<1):
        random_segment=segments[rnd.randint(0,len(segments)-1)]
        segment_duration=random_segment.shape[0] / fs
    #print('Segment\'s Duration = {} seconds'.format(segment_duration))
    return(random_segment)   
    
def multiplyDataViaAugmentationSNR(initial_signals, initial_labels, initial_Fs, SNR_list ,shuffle):

    new_signals=initial_signals
    new_labels=initial_labels
    new_Fs=initial_Fs
    size=len(initial_labels)
    
    for SNR in SNR_list:
        
        gen_SNR=generator(initial_signals, initial_labels, initial_Fs, size, shuffle , SNR)
        aug_signals,aug_labels,aug_fs=next(gen_SNR)
        new_signals=np.concatenate((new_signals,aug_signals),axis=0)
        new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
        new_labels=np.concatenate((new_labels,aug_labels),axis=0)
     
    #print("End of Data Augmentation")
    """
    for i,signal in enumerate(new_signals):
        segment=get_2sec_segment(new_Fs[i],signal)
        x = audioBasicIO.stereo_to_mono(np.int16(segment))
        specgram, TimeAxis, FreqAxis = spectrogram(x, new_Fs[i], round(new_Fs[i]* 0.01), round(new_Fs[i] * 0.01), False)
        image = resize(specgram, (250,250))
        data.append(image)    
            
    return [new_signals, data, new_labels, new_Fs]
    #specgram, TimeAxis, FreqAxis = sF.spectrogram(x, fs, round(fs * 0.004), round(fs * 0.002), True)
    """
    return [new_signals, new_labels, new_Fs]

"""   
def multiplyDataViaAugmentation(initial_signals, initial_labels, initial_Fs,shuffle):
           
    new_signals=initial_signals
    print(np.asarray(new_signals).shape)
    new_labels=initial_labels
    new_Fs=initial_Fs
    batch_size=len(initial_labels)
    print(batch_size)
    
    
    #Augmentation with Noise..
    SNR=random.randint(3,5)
    print(SNR)
    #gen=generator(initial_signals, initial_labels, initial_Fs, batch_size,True,SNR)
    gen_noise=generator(initial_signals, initial_labels, initial_Fs, batch_size, 
                        shuffle , noise_factor=SNR)
    aug_signals,aug_labels,aug_fs=next(gen_noise)
    print(np.asarray(aug_signals).shape) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation with Noise - complete")
    
    
    #Augmentation with Shifting Wavfile..
    gen_shift=generator(initial_signals, initial_labels, initial_Fs, batch_size, 
                        shuffle , noise_factor=0,
                        shift_max=1, shift_direction='both')
    aug_signals,aug_labels,aug_fs=next(gen_shift)
    #print(aug_signals) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)    
    print("Augmentation with Shifting Wavfile - complete")
    
    
    factor = np.random.uniform(low=0.9, high = 1.1)
    print(factor)
    #factor = 1.0  / length_change
    
    #Augmentation by changing pitch..
    gen_pitch=generator(initial_signals, initial_labels, initial_Fs, batch_size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=factor)                    
    aug_signals,aug_labels,aug_fs=next(gen_pitch)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing pitch - complete")
    
    #Augmentation by changing speed..
    #print("resample length_change = ",length_change)
    gen_speed=generator(initial_signals, initial_labels, initial_Fs, batch_size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=0,
                        speed_factor=factor)
    aug_signals,aug_labels,aug_fs=next(gen_speed)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing speed- complete")
    
    
    print("End of Data Augmentation with SNR : " + str(SNR))
    print("                       and factor : " + str(factor))

    return [new_signals, new_labels, new_Fs]
    
"""

def multiplyDataViaAugmentation(initial_signals, initial_labels, initial_Fs,shuffle):
           
    new_signals=initial_signals
    print(np.asarray(new_signals).shape)
    new_labels=initial_labels
    new_Fs=initial_Fs
    size=len(initial_labels)
    print(size)
    
    
    #Augmentation with Noise..
    #SNR=random.randint(3,5)
    #print(SNR)
    #gen=generator(initial_signals, initial_labels, initial_Fs, batch_size,True,SNR)
    gen_noise=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=-1)
    aug_signals,aug_labels,aug_fs=next(gen_noise)
    print(np.asarray(aug_signals).shape) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation with Noise - complete")
    
    
    #Augmentation with Shifting Wavfile..
    gen_shift=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=1, shift_direction='both')
    aug_signals,aug_labels,aug_fs=next(gen_shift)
    #print(aug_signals) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)

    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)    
    print("Augmentation with Shifting Wavfile - complete")
    
    
    #factor = np.random.uniform(low=0.9, high = 1.1)
    #print(factor)
    #factor = 1.0  / length_change
    
    #Augmentation by changing pitch..
    gen_pitch=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=-1)                    
    aug_signals,aug_labels,aug_fs=next(gen_pitch)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing pitch - complete")
    
    #Augmentation by changing speed..
    #print("resample length_change = ",length_change)
    gen_speed=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=0,
                        speed_factor=-1)
    aug_signals,aug_labels,aug_fs=next(gen_speed)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing speed- complete")

    
    
    print("End of Data Augmentation")

    return [new_signals, new_labels, new_Fs]
    
def DataAugmentation(initial_signals, initial_labels, initial_Fs,shuffle):
           
    new_signals=initial_signals
    print(np.asarray(new_signals).shape)
    new_labels=initial_labels
    new_Fs=initial_Fs
    size=len(initial_labels)
    print(size)
    
    """
    #Augmentation with Noise..
    SNR=[3,4,5]
    for snr in SNR:
	    #gen=generator(initial_signals, initial_labels, initial_Fs, batch_size,True, snr])
	    gen_noise=generator(initial_signals, initial_labels, initial_Fs, size, 
		                shuffle , noise_factor=snr)
	    aug_signals,aug_labels,aug_fs=next(gen_noise)
	    print(np.asarray(aug_signals).shape) 
	    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
	    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
	    new_labels=np.concatenate((new_labels,aug_labels),axis=0)

	    print(new_signals.shape)
    print("Augmentation with Noise - complete")
    
    """
    
    gen_noise=generator(initial_signals, initial_labels, initial_Fs, size, 
		                shuffle , noise_factor=-1)
    aug_signals,aug_labels,aug_fs=next(gen_noise)
    print(np.asarray(aug_signals).shape) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation with Noise - complete")
    
    
    #Augmentation with Shifting Wavfile..
    gen_shift=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=1, shift_direction='both')
    aug_signals,aug_labels,aug_fs=next(gen_shift)
    #print(aug_signals) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)    
    print("Augmentation with Shifting Wavfile - complete")
    
    
    #Augmentation with Shifting & Noise ..
    gen_shift=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=-1,
                        shift_max=1, shift_direction='both')
    aug_signals,aug_labels,aug_fs=next(gen_shift)
    
    #print(aug_signals) 
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)    
    print("Augmentation with Shifting & Noise - complete")
    

    #Augmentation by changing pitch..
    gen_pitch=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=-1)                    
    aug_signals,aug_labels,aug_fs=next(gen_pitch)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing pitch  - complete")
    
    #Augmentation by changing speed..
    #print("resample length_change = ",length_change)
    gen_speed=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=0,
                        shift_max=0, shift_direction=0,
                        pitch_factor=0,
                        speed_factor=-1)
    aug_signals,aug_labels,aug_fs=next(gen_speed)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing speed - complete")
    
    #Augmentation by changing pitch & noise..
    gen_pitch=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=-1,
                        shift_max=0, shift_direction=0,
                        pitch_factor=-1)                    
    aug_signals,aug_labels,aug_fs=next(gen_pitch)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing pitch & noise - complete")
    
    #Augmentation by changing speed and noise..
    #print("resample length_change = ",length_change)
    gen_speed=generator(initial_signals, initial_labels, initial_Fs, size, 
                        shuffle , noise_factor=-1,
                        shift_max=0, shift_direction=0,
                        pitch_factor=0,
                        speed_factor=-1)
    aug_signals,aug_labels,aug_fs=next(gen_speed)
    new_signals=np.concatenate((new_signals,aug_signals),axis=0)
    new_Fs=np.concatenate((new_Fs,aug_fs),axis=0)
    new_labels=np.concatenate((new_labels,aug_labels),axis=0)
    print(new_signals.shape)
    print("Augmentation by changing speed & noise - complete")   



    print("End of Data Augmentation")

    return [new_signals, new_labels, new_Fs]   


def dataAsImage(signals, fs):

    RGB=[]
    stWin=0.04
    stStep=0.02
    
    for i,signal in enumerate(signals):
        segment=get_2sec_segment(fs[i],signal)
        RGB.append(createSpecgramImage(segment, fs[i], stWin, stStep)) 
  
    return RGB
 
def extractFilesFromDirectory(data_dir):

    files=[]
    labels=[]
    
    emotion_dirs=[x[0] for x in os.walk(data_dir)]
    #print (emotion_dirs)
    print (emotion_dirs[1:])
    #print("-----")

    for i, emotion_dir in enumerate(emotion_dirs[1:]):
        emotion_files=[x[2] for x in os.walk(emotion_dir)]
        #print(emotion_dir)
        #print("-------")
        #print(emotion_files[0])
        
        for fname in emotion_files[0]: 
            files.append(fname)
            #print(fname)
            labels.append(i)
            
    signals=[]
    Fs=[]

    for i,fname in enumerate(files) :
        #print(emotion_dirs[labels_train[i]+1])
        fs, signal = wavfile.read(os.path.join(emotion_dirs[labels[i]+1], fname))
        signals.append(signal)
        Fs.append(fs)
    
    return signals, Fs , labels
    
"""
Prepares data for training.
"""
def data_preperation_as_CNN_input(X, Y):
  
	X=np.asarray(X)

	Y=np.asarray(Y)

	X=X.reshape((-1, 250, 250,3))

	X = X.astype('float32')
	
	#X /= 255

	Y = to_categorical(Y, 5)

	return X, Y
 
 
def init_dataSet():
    
    data_dir_train = '/media/gpu2/GpuTwo/georgia/EMOVO/EMOVOdataFinal/train'
    
    train_signals, train_Fs , labels_train=extractFilesFromDirectory(data_dir_train)                                        
        
    print ("Data Augmentation starts")
    #[train_signals, Y_train, train_Fs] = multiplyDataViaAugmentationSNR(train_signals, labels_train, train_Fs, [3,4,5], True)  
    #[train_signals, Y_train, train_Fs]=multiplyDataViaAugmentation(train_signals, labels_train, train_Fs, True) 
    [train_signals, Y_train, train_Fs]=DataAugmentation(train_signals, labels_train, train_Fs, True)   
    
    #Segmenting data on 2sec segments and producing its spegtrograms
    X_train=dataAsImage(train_signals, train_Fs)
    
    X_train, Y_train = data_preperation_as_CNN_input(X_train, Y_train)
    
    print(X_train.shape)
    print(Y_train.shape)
    
    data_dir_test = '/media/gpu2/GpuTwo/georgia/EMOVO/EMOVOdataFinal/test'
    
    """
    test_signals, test_Fs , Y_test=extractFilesFromDirectory(data_dir_test)
    
    X_test=dataAsImage(test_signals,test_Fs)
    
    X_test, Y_test = data_preperation_as_CNN_input(X_test, Y_test)
    """
    
    signals, Fs , Y =extractFilesFromDirectory(data_dir_test)
    
    X=dataAsImage(signals,Fs)
    
    X, Y = data_preperation_as_CNN_input(X, Y)
    
    X_test , X_val , Y_test , Y_val = train_test_split(X , Y , test_size = 0.5)
    print(X_test.shape)
    print(Y_test.shape)
    
    print(X_val.shape)
    print(Y_val.shape)
    
    #return X_train, Y_train, X_test, Y_test
    return X_train, Y_train, X_test, Y_test , X_val, Y_val
