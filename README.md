# German Traffic Sign Classification Project
###### Self-driving Car Nanodegree at Udacity. 
---

Build a convolution neural network architecture capable of classifying german traffic signs. At a minimum, the architecture should be able to achieve 93% accuracy on the validation set. 

Project source: [CarND-Traffic-Sign-Classifier-Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

#### Data Set Summary & Exploration
#
The data set is split in 3 parts - Train, Valid and Test - each containing x and y values. Using **numpy** we can print out the size, shape and number of classes in the dataset :

| Label        | Value           
| ------------- |:-------------:
| Number of training examples     | 34799 |
| Number of testing examples      | 12630      |
| Image data shape | (32, 32, 3)      | 
| Number of classe | 43      | 

Distribution of images across 43 classes:

Train set
![](https://github.com/ismalakazel/carnd-traffic-sign-classifier/blob/master/assets/train_distribution.png)

Valid set
![](https://github.com/ismalakazel/carnd-traffic-sign-classifier/blob/master/assets/valid_distribution.png)

Test set
![](https://github.com/ismalakazel/carnd-traffic-sign-classifier/blob/master/assets/test_distribution.png)

#
#### Data Preprocessing
#
**grayscale:** For the most part the traffic signs are shapes and symbols so grayscaling is applied to reduce the completixy of the image to a single channel, taking into consideration that essential information would not be lost.

**normalization:** The Min-Max normalization technique was used to reduce the range values to 0-1. This steps is important to speed up training as the architecture is able to operate weights in a similar range value.

**mean subtraction:** This technique reduces the complexity even further by centering the pixels values.

History equalization was also experimented but accuracy results weren't much different with it.

Preprocessing pipeline
![](https://github.com/ismalakazel/carnd-traffic-sign-classifier/blob/master/assets/preprocess_pipeline.png)

A better explanation of this preprocessing pipeline can be found [here](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing).
#
#### Data Augmentation
#
As shown in the bar graphs, the dataset is uneven. Some classes have more examples than the others, which causes the model to not generalize so well during training and causing overfitting. In order to overcome this issue, new data was generated and concatenated to the existing data set.

For each image in classes with fewer than 750 images 15 new images were generate with the following augmentation process:
- Random rotation
- Random image perspective change
- Random contrast change

Train set
![](https://github.com/ismalakazel/carnd-traffic-sign-classifier/blob/master/assets/augmented_distribution.png)

#
#### Model Architecture
#
#
| Layer        | Description  |        
| ------------- |:-------------:
| Input     | 32x32x1 |
| **Convolution 5x5**      | 1x1 stride, valid padding, outputs 28x28x32    |
| Relu |
| **Convolution 5x5**      | 1x1 stride, valid padding, outputs 24x24x64      |
| Relu |
| Max pool      | 2x2 stride, same padding, outputs 12x12x64      |
| **Convolution 3x3**      | 1x1 stride, valid padding, outputs 10x10x96      |
| Relu |
| **Convolution 3x3**      | 1x1 stride, valid padding, outputs 8x8x128      |
| Relu |
| Max pool      | 2x2 stride, same padding, outputs 4x4x64      |
| **Dense Layer** | input 2048, ouputs 502 |
| **Dense Layer** | input 502, ouputs 502 |
| **Output Layer** | input 502, ouputs 43 |
|**Softmax**|  |
#
This model is an experiment that started from the LeNet architecture. With the original LeNet model plus the preprocessing pipeline it was possible to achieve validation accuracy around 94% and testing accuraccy of about 93%. From there on I tried to implement 

The AlexNet model was experimented during this project reaching arounce 93% accuracy, but the training time was too long, and I suspect that all the convolutional layers and maxpools made the image so small that image features weren't being capture properly.

So based on the LetNet I made an experiement by increasing the number of convolution layers to 4 appliying max pooling only on the second and fourth layers. I also added one more dense layer to the model. Though I studied different model architectures, this one was also about learning by intuition.

### 2.5 Model Training
#
#
| Parameter        | Description  |        
| ------------- |:-------------:
| Machine     | GPU instance on AWS |
| Training time     | 90 minutes |
| Number of epochs     | 120 |
| Batch size     | 128 |
| Learning rate | 0.0005 | 
| Optimizer | AdamOptmizer |
| Regularizer | L2 | 
| Regularizer penalty | 0.0005 | 
| Batch normalization |  | 
| Dropout conv layer | 0.8 keep probability |
| Dropout dense layer | 0.5 keep probability |

Training was done in a GPU instance to speed model training model fine-tuning. Testing on local machine can be frustrating unless it's equiped well for this kind of task. AWS on the other hand can be expensive and it's better used to perform final training and quick fine tunings.

Different Batch sizers are experimented. In order to accomodate a good ammount of images without losing to much speed during training, a batch 128 was used. 256 and 502 were some of the numbers experimented as well, however with little performance gained.

AdamOptimizer was used in thought the creation of this project. No other optimizer was tested.

When reading about overfitting issues I came accross this [paper](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) that mentions L2 regularizers as an additional boost to help dropout based neural networks. It proved to increase accuracy in the final training. 

Batch normalization layers was used in accordance to this [paper](https://arxiv.org/pdf/1502.03167.pdf). 

Bellow are the accuracy values reach with the model above:
#
| Accuracy        | Value  |        
| ------------- |:-------------:
| Valid |  1.0 |
| Train** |  0.99 |
| Test |  0.98 |

#
### 3. Conclusion
#
This project was a good way to get some intuition as well as getting more accostumed with deep learning terminologies. Going further I want to experiment with other models, trying different parameters and regularizations techniques. One take away is that aquiring good data and preparing it is good portion of extrating good predictions out of a deep neural network.
