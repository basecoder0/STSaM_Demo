# STSaM_Demo
Speech Text Sentiment analysis Model

# Datasets used to train this model
SAVEE
https://www.kaggle.com/barelydedicated/savee-database

RAVDESS EMOTION DATA SET/SONG DATA SET
https://zenodo.org/record/1188976#.YLLSFLdKiUk


# Concept
My goal for this project is to create a Neural Network model that is able to generalize on a specific
mood/sentiment based on the speech and text input. The concept behind this idea is that through functional API I can
build a model that takes two distinct inputs, one for processing the text and the other for speech. Using a combination
of CNN, MLP, and RNN Neural Networks, I should be able to predict an emotion by concatenating the results from each
individually processed input.

# Data Preprocessing
Observing the image below, the a goal is to take audio input for the user and convert that into trainable data to be consumed
by the Neural Network. Using a the librosa library, it will allow me to convert sound waves into a Mel-Frequency Cepstral Coefficient (MFCC)
representation that can be used by the model. To get a decent understanding of what MFCC are, here is a link to the wiki page, that briefly goes over the 
concept [MFCC wiki](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).


- ##Add image

# Challenges
Some challenges I ran into when building this model, was first figuring out how exactly to process the sound wavs. What I mean by this is that I am not just
converting sound waves into MFCCS and thats it, nope some other factors to consider is sample rate, duration of the actual audio file itself, how many seconds
to offset the audio by before it starts reading the wav file, and resample type. While I myself wasn't exactly too sure on how to go about this process, I did take 
note from an already established project that I used as a baseline for this project. Here is a link to their [GitHub](https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer). Below you can see the setup I used for preprocessing my audio files

- ##Add image

# Model Architecture





