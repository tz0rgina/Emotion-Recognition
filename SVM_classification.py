import os

data_dir = 'C:/MachineLearningPractice/emovo/EMOVOdata'
emotion_dirs=[x[0] for x in os.walk(data_dir)]

print(emotion_dirs[1:])

from pyAudioAnalysis import audioTrainTest as aT

aT.extract_features_and_train(emotion_dirs[1:], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False, train_percentage=0.80)
aT.file_classification('C:/MachineLearningPractice/emovo/EMOVOdata/sadness/tri-f1-b1.wav',"svmSMtemp","svm")
