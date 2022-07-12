# CIFAR-10-Classification-model

This is a classification model for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consisting of pictures of airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships and trucks.

In this project, I practised wrangling the dataset, and using a tensorflow to create a classification model.


Contents:

1. Wrangling Data

2. Displaying image

3. Creating model

4. optimizing model



Wrangling Data
---
I downloaded the data as a zip from [here](https://www.cs.toronto.edu/~kriz/cifar.html). The data came as a (10000 x 3072) tensor. Where each image was a vector tensor of size 3072. I used the first batch of images, and split them 80:20 into training and testing sets.


Displaying Image
---
I had issues viewing the images when I tried to reshape them into 32 x 32 size and plt.imshow()
![failed frogs](/failed%20frogs.png "Failed Frogs png")

 I found out I had to reshape the image using .reshape(3,32,32), then transpose with np.transpose(res, axes=[1, 2, 0]). This puts the image into a eligible format. I created a show_image function that allows me to view the image, which I later converted in to a function that converted my dataset into the other format.
 
 ![image format](/cifar-10%20tranpose%20images.png "Image format png")



Creating model using Tensorflow
---
I started off by dividing my training and testing data sets by 255, to alow them to range from 255. This will allow my model to perform more efficiently

I used a Sequential model with 2 layers with 128 nodes each, and a output layer with 10 nodes to represent each one of the possible categories

```
First version model
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_1 (Dense)            (None, 128)               393344    
                                                                 
 dense_2 (Dense)            (None, 128)               16512     
                                                                 
 dense_3 (Dense)            (None, 10)                1290      
                                                                 
=================================================================
Total params: 411,146
Trainable params: 411,146
Non-trainable params: 0
_________________________________________________________________
```

My model falied to fit, I got a ValueError: Failed to find data adapter ... etc. I fixed this my converting my Y training and testing sets from lists to numpy arrays

My first fitted model, was broken. The loss and acuracy did not change between epochs, and the accuracy was 0.10, which made sense statistically if the model was guessing completely randomly.

#### Improvement 2
To optimize my model, I change the activation function of the output layer into softmax. My model this time begins to overfit at around 8 epochs, and has a validation accuracy of approximately 0.40. This is a major improvement over my previous model, and It shows my model is working as intended, and learning from the training set.

`

#### Improvement 3
I decided to reformat my data into with a modified show_image funtion before feeding it into the model. I created a convert_dataset function that transformed every image in the dataset with the transformations shown in "Displaying Image". I found that feeding the image as (3,32,32) verses (1,3072) has absolutely no effect on the model. This model produced almost identical results as the previous model.

#### Improvement 4
I decided to improve the model itself, instead of using two dense layers I decided to build a model with multiple layers. 
```
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 64)        18496     
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dense_1 (Dense)            (None, 512)               819712    
                                                                 
 dense_2 (Dense)            (None, 128)               65664     
                                                                 
 dense_3 (Dense)            (None, 32)                4128      
                                                                 
 dense_4 (Dense)            (None, 10)                330       
                                                                 
 dense_5 (Dense)            (None, 10)                110       
                                                                 
=================================================================
Total params: 955,512
Trainable params: 955,512
Non-trainable params: 0
_________________________________________________________________
```
This time my model stil began to overfit at around 8-9 epochs, however the validation accuracy was slightly better at around 0.52

