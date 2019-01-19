# Emotion_recognition_CNN

## 1 Goal & research topic
Emotion detection and recognition technology has been a market that’s predicted to grow to $65 billion by 2023. This booming market is being fuelled by
advances in artificial intelligence, and big investment from organizations (across
a range of industries) keen to study consumer behaviour. It involves the analysis
of facial signals to determine internal emotions, including nuanced expression
and body language. In theory, this information can be used to then trigger
or change decision-making – or simply create a more emotive or personalized
experience.
To that end, we will attempt to predict facial emotions through Convolutional Neuron Networks (CNN).

## 2 Data-set
There exists different data sets available regard to emotion recognition, but in
this project we will be using AffectNet. The reason that we decided to use
this particular data set is that it was easier to access it and also all the images
were labeled. Whereas other data sets that we found had small number of
observations, some were not readily available for use and could only be accessible
after some time, or their data were not labeled.
AffectNet is the largest data set of facial expression. It consist of colour face
images with various resolutions. This data set has the following columns [4]:
• file path (sub directory and file name of the image)
• face x (x coordination of the location of the face in the image)
• face y (y coordination of the location of the face in the image)
• face width (width of the detected face in the image)
• face height (height of the detected face in the image)
• facial landmarks (coordination (x and y) of the 68 detected facial landmarks)
• expression (0 – neutral, 1- happy, 2- sad, 3- surprise, 4- fear, 5- disgust,
6- angry, 7- contempt, 8- none, 9- uncertain, 10- no-face)
• valence (valence value of the expression in interval [-1,+1] (for uncertain
and no-face categories the value is -2)
• arousal (arousal value of the expression in interval [-1,+1] (for Uncertain
and No-face categories the value is -2))
In total there are 1M images from which each class has a different number of
observations.

## 3 Related Work
Emotion recognition is used for a variety of reasons. Affectiva is an emotion
measurement technology company that grew out of MIT’s Media Lab. Affectiva has developed software to recognize human emotions based on facial cues
or physiological responses. This software has been used in commercial applications, in fact Affectiva has used emotion recognition to help advertisers and
content creators to sell their products more effectively.
Another major application has been in political polling. It has been
shown that significantly different responses to the candidates are measurable
using automated facial expression analysis and that these differences can predict
self-report candidate preference. Affectiva also made a Q-sensor that measures
the emotions of autistic children. This sensor, together with facial emotion
recognition techniques, has been used to develop CaptureMyEmotion, a mobile
application to support children affected by autism. In fact, many autistic children do not explicitly express their emotions or do so in a manner that widely
differs from standard forms of expression. Thanks to this Mobile Application
children affected by autism could better understand emotions.

## 4 Network architecture

### 4.1 Software
We stick with the recommended:
• Python
• Tensorflow

### 4.2 Network architecture
According to [1][2] a convolutional network performs better than a traditional
deep neural network for tasks like pattern recognition in images. We will stick
to the architecture of previous research, but also add some variation to see if
that improves our predictions.

### 4.2.1 Layers
How will the convolutional network look like?
• Input layer: An image including a face.
• M convolutional layers, applied dropout, max-pooling and ReLu.
• N fully connected layers.
• Final layer, a node for every class.

### 4.2.2 Convolutional layers
From the papers we saw that 2-4 layers perform well. We try to vary between
this. Also a 3x3 filter size has good practise. We will vary in the amount of
filters: 32 - 512. We also include zero-padding. The dropout percentage is
25-50% (against overfitting).
4.2.3 Activation function
ReLu has shown to speed up convergence [1, 2]. Also we suspect a challenge as
it comes to computational power while training the network, ReLu is known to
be fast.
### 4.2.4 Pooling Layer
Max-pooling, taking maximum with some dimension. Filter size 2x2.
### 4.2.5 Fully connected layers
Needed for flattening and classification. In [1][2] we saw layers of 2-4.

## 5 How will you evaluate the results?
The data set contains label data and, thus, they will be used to assess the
performance of the algorithm. Classification accuracy will be the first metric
for evaluation, i.e. the ratio of number of correct predictions to the total number
of input samples. Additional criteria will be recall, precision and F1-score.
An immediate derivation of the accuracy metric is the confusion matrix.
Through it, we will be able to see which emotions have been misclassified and
how. This will give us a better understanding of the training sample.
Lastly, we can use the M measure as proposed by Hand et al. [3] which is a
generalization approach that aggregates all pairs of classes based on the inherent
characteristics of the Area Under the Curve (AUC). The major advantage of
this method is that it is insensitive to class distribution and error costs.

## How to run
In order to run the project, certain versions of Python modules should be installed. 

We require `Python 3.6` which can be found here [here](https://www.python.org/downloads/)

After you've successfully installed `Python 3.6`, you should start installing the modules.

`pip3.6 install tensorflow==1.2.0`

`pip3.6 install keras==2.1.3`

`pip3.6 install `











