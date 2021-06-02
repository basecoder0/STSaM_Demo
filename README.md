# STSaM_Demo
Speech Text Sentiment analysis Model *(Proof of Concept)*

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
Observing the image below, the a goal is to take audio input from the user and convert that into trainable data to be consumed
by the Neural Network. Using a the librosa library, it will allow me to convert sound waves into a Mel-Frequency Cepstral Coefficient (MFCC)
representation that can be used by the model. To get a decent understanding of what MFCC are, here is a link to the wiki page, that briefly goes over the 
concept [MFCC wiki](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).

![Sound wave to MFCC](./images/Sound%20wav%20to%20mfcc.PNG)

Some challenges I ran into when building this model, was first figuring out how exactly to process the sound wavs. What I mean by this is that I am not just
converting sound waves into MFCCS and thats it, nope, some other factors to consider is sample rate, duration of the actual audio file itself, how many seconds
to offset the audio by before it starts reading the wav file, and resample type. While I myself wasn't exactly too sure on how to go about this process, I did take 
note from an already established project that I used as a baseline for this project. Here is a link to their [GitHub](https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer). In the image below you can see the setup I used for preprocessing my audio files, more details will be in the ipynb file

![Librosa arguments](./images/librosa_load.PNG)

Preprocessing text is a bit simpler, as I use a standard vectorization method (one-hot encoding) to transform the
text into a usable format that can be processed by the model. Currently I am training this model using the IMDB movie
reviews dataset, the features for the encoded string is currently at 10000. I could possibly use an encoding method with
smaller features but for the sake of testing this idea, this is the state for how the text should be transformed when being fed into the model.
Feel free to modify it as you see fit.

Also, and I may be stating the obvious here, but be sure the label classifications match the model input. Since this model is processing to different types of input MFCC/O.H.E features, make sure the 'Happy' text is paired with 'Happy' sound waves and the same goes for Sad and whatever other emotion. I have that process setup in the actual code so no worries.

# Model Architecture
![Model Architecture](./images/Model%20Architecture.PNG)

# Audio Layer (CNN) Processing
Convolutional layers are great at scanning and capturing interesting features on each
channel in a 2D space. 1D Convolutional layers are great at scanning 1D sequences of data based on a timeseries, in the
case of this project signals. Much like the 2D Convolutional layers, the 1D layers with 128 units, will sample the raw data (MFCC
features) with a shape of (216,1) using a kernel of size 5 as well as keeping the padding the ‘same’ to preserve the input
shape. 
-  I pad the input to maintain the frequency spectrum as different emotions can be represented by different
periods and amplitudes in a sound wave, since convolutional layers downsize a sample based on the kernel used, it would ultimately be removing the
context which can describe an emotion, so I want to preserve as much as we can. 

![Periods and Frequencies](./images/periods%20and%20frequencies.PNG)

-  I apply a Dropout layer, for the purpose of overfitting as well as a maxpooling layer to down sample our dataset by a factor of 8.

-  After extracting all the interesting features using the 1D Convolutional layers, I flatten the features into a
shape (,3456) tensor.

- Lastly the data is then processed through the Dense layers where the features are applied to an output
layer using softmax. 

# Experiments/Results

### Binary Classification {Happy, Sad}
![Binary Classification Results](./images/Binary%20Classification%20Results.PNG)
![Figure A Results](./images/Fig%20A%20results.PNG)
![Figure B Results](./images/Fig%20B%20results.PNG)

# Summary of Results
Based on what I’ve observed, for *Model/Fig A*, this model seems to have the most accurate prediction out of two for classifying 2 types of moods. Given that it does have the lowest test accuracy score, looking at the model accuracy and loss graph **Fig A1**, you can see that while there is a lot of noise, the generalization gap is quite small when compared to *Model/Fig B*. 

Running some live tests on Model B, while it is able to generalize on a mood, in my opinion it is not as confident as Model A. And even though the confusion matrix **Fig B2**  shows better prediction scores, you can see in the model accuracy and loss graph **Fig B1** that overfitting happens earlier around 25 epochs and the generalization gap is a bit larger than its counterpart, I feel this is why Model B does not perform as well as the scores indicate. 
While the Confusion Matrices were able to show how well these models generalized on the unseen test data, I believe the scores are weighted slightly higher due to the low amount of test samples

# Final Thoughts
Given that there aren't a lot of training samples for this prjoect, training this model was a slight challenge however it is still doable.
I was able to classify 2 types of emotions Happy and Sad, I did build another model for multi-classification purposes which 
classifies a 'Neutral' emotion as well but I choose to omit that from this write up, I will load .h5 file to this repository. I say these models,
are able to prove my initial concept for this project, and if I decide too in the future to pursue this idea further, I could be able to build a model that is 
able to classify more subtle emotions like ‘Calm’ and distinguish between more expressive emotions like ‘Disgust’ and ‘Surprised.


# Sources
- Speech Emotion Analyzer https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer 
- Direct access to your webcam and microphone inside Google Colab notebook https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_colab/ 
- Convert Speech to Text in Python https://www.thepythoncode.com/article/using-speechrecognition-to-convert-speech-to-text-python  
