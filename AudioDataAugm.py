#import random as rnd
import os

#rnd.seed(1)
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
os.environ['PYTHONHASHSEED']=str(0)

import numpy as np
from sklearn.utils import shuffle as s

import numpy as np
from scipy.io import wavfile
import IPython
import matplotlib.pyplot as plt
import cv2 as cv
from resizeimage import resizeimage
from skimage.transform import resize
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly
from pyAudioAnalysis import audioAnalysis as AA
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import audioBasicIO
from numpy import random

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "Augmentation Test"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return
    
def plots(file, fig_name):
    fs, signal = wavfile.read(file)
    time_wav = np.arange(0, len(signal)) / fs

    plotly.offline.iplot({ "data": [go.Scatter(x=time_wav, 
                                               y=signal[:, 0], 
                                               name='left channel'), 
                                    go.Scatter(x=time_wav, 
                                               y=signal[:, 1], 
                                               name='right channel')]})
    x = audioBasicIO.stereo_to_mono(signal)
    specgram, TimeAxis, FreqAxis = sF.spectrogram(x, fs, round(fs * 0.01),
                                                      round(fs * 0.01), False)
    image = resize(specgram, (250,250))
    print(image.shape)
    plt.imshow(image)
    print_figure(fig_name)
    plt.show()   

def getPower(clip):
    clip2 = clip.copy()
    clip2 = np.array(clip2) / (2.0**15)  # normalizing
    clip2 = clip2 **2
    return np.sum(clip2) / (len(clip2) * 1.0)


def addNoise(audio, snrTarget):
    if len(audio.shape)==2:
        noise = np.random.randn(len(audio),2)
    else:
        noise = np.random.randn(len(audio))
    sigPower = getPower(audio)
    noisePower = getPower(noise) 
    factor = (sigPower / noisePower ) / (10**(snrTarget / 10.0))  # noise Coefficient for target SNR

    # return np.int16( audio + noise * np.sqrt(factor) )
    return (audio + noise * np.sqrt(factor) )

"""
Shifting Time
The idea of shifting time is very simple. 
It just shift audio to left/right with a random second. 
If shifting audio to left (fast forward) with x seconds, 
first x seconds will mark as 0 (i.e. silence). 
If shifting audio to right (back forward) with x seconds, 
last x seconds will mark as 0 (i.e. silence).
"""
def shiftingTime(data, sampling_rate, shift_max, shift_direction):

    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    #print(shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0

    #print(augmented_data)
    return augmented_data


import librosa

def changingPitch(data, sampling_rate, pitch_factor):

    if len(data.shape)==2:
        left, right = np.transpose(data)[0], np.transpose(data)[1] #left and right channel
        #print("left : "+ str(left))
        #print("right : "+ str (right))
        augmented_left=librosa.effects.pitch_shift(left.astype(np.floating), sampling_rate, pitch_factor)
        augmented_right=librosa.effects.pitch_shift(right.astype(np.floating), sampling_rate, pitch_factor)
        #print("augmented_left : "+ str(augmented_left))
        #print("augmented_right : "+ str (augmented_right))
        augmented_data=np.column_stack((augmented_left,augmented_right))
    else:
        mono_channel = np.transpose(data)
        augmented_data=librosa.effects.pitch_shift(mono_channel.astype(np.floating), sampling_rate, pitch_factor)

    return augmented_data


def changingSpeed(data, speed_factor):
    augmented_data=[]
    if len(data.shape)==2:
        left, right = np.transpose(data)[0], np.transpose(data)[1] #left and right channel
        #print("left : "+ str(left))
        #print("right : "+ str (right))
        augmented_left=librosa.effects.time_stretch(left.astype(np.floating), speed_factor)
        augmented_right=librosa.effects.time_stretch(right.astype(np.floating), speed_factor)
        #print("augmented_left : "+ str(augmented_left))
        #print("augmented_right : "+ str (augmented_right))
        augmented_data=np.column_stack((augmented_left,augmented_right))
    else:
        mono_channel = np.transpose(data)
        augmented_data=librosa.effects.time_stretch(mono_channel.astype(np.floating), speed_factor)
    

    return augmented_data

def generator(data, labels,fs, size, shuffle , noise_factor=0, 
              shift_max=0, shift_direction=0,
              pitch_factor=0, 
              speed_factor=0):

    while True:
        if (shuffle):
            #sampling= rnd.choices(np.arange(len(labels)), k=size)
            sampling=random.permutation(np.arange(len(labels)))
        else:
            sampling=np.arange(size)
        
        batches=[]
        new_labels=[]
        new_fs=[]
        for i in sampling:
            batches.append(data[i])
            new_labels.append(labels[i])
            new_fs.append(fs[i])
        
        for j , batch in enumerate(batches):
            # Augmentation
            sample=batch
            if (noise_factor > 0):
                sample=addNoise(batch,noise_factor)
            elif (noise_factor == -1):
            	snr=random.randint(3,5)
            	#print(snr)
            	sample=addNoise(batch,snr)
            if (shift_max!=0):
                sample=shiftingTime(sample, fs[i], shift_max, shift_direction)
            if(pitch_factor > 0):
                sample=changingPitch(sample, fs[i], pitch_factor)
            elif (pitch_factor == -1):
            	factor=np.random.uniform(low=0.9, high = 1.1) 
            	#print(factor)
            	sample=changingPitch(sample, fs[i], factor)    
            if (speed_factor > 0):
                sample=changingSpeed(sample, speed_factor)
            elif (speed_factor == -1):
            	factor=np.random.uniform(low=0.9, high = 1.1) 
            	#print(factor)
            	sample=changingSpeed(sample, factor)   
            batches[j] = sample
       

        yield batches, new_labels, new_fs
        
