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
I had issues viewing the images, I found out I had to reshape the image using .reshape(3,32,32), then transpose with np.transpose(res, axes=[1, 2, 0]). This puts the image into a eligible format. I created a show_image function that allows me to view the image, which I later converted in to a function that converted my dataset into the other format.


Creating model using Tensorflow
---
I started off by dividing my training and testing data sets by 255, to alow them to range from 255. This will allow my model to perform more efficiently

I used a Sequential model with 2 layers with 128 nodes each, and a output layer with 10 nodes to represent each one of the possible categories

My model falied to fit, I got a ValueError: Failed to find data adapter ... etc. I fixed this my converting my Y training and testing sets from lists to numpy arrays

My first fitted model, was broken. The loss and acuracy did not change between epochs, and the accuracy was 0.10, which made sense statistically if the model was guessing completely randomly.

#### Improvement 2
To optimize my model, I change the activation function of the output layer into softmax. My model this time begins to overfit at around 8 epochs, and has a validation accuracy of approximately 0.40. This is a major improvement over my previous model, and It shows my model is working as intended, and learning from the training set.

#### Improvement 3
I decided to reformat my data into with a modified show_image funtion before feeding it into the model. I created a convert_dataset function that transformed every image in the dataset with the transformations shown in "Displaying Image". I found that feeding the image as (3,32,32) verses (1,3072) has absolutely no effect on the model. This model produced almost identical results as the previous model.

#### Improvement 4
I decided to improve the model itself, instead of using two dense layers I decided to build a model with multiple layers. 



