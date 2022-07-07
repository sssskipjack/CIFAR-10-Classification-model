# CIFAR-10-Classification-model

This is a classification model for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consisting of pictures of airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships and trucks.

In this project, I practised wrangling the dataset, and using a tensorflow to create a classification model.


Contents:

1. Wrangling

2. Displaying image

3. creating model

4. optimizing model

5. predictions


Obtaining Data
---
I downloaded the data as a zip from [here](https://www.cs.toronto.edu/~kriz/cifar.html). The data came as a (10000 x 3072) tensor. Where each image was a vector tensor of size 3072. 


Viewing Image
---
To view the image I had to reshape the image using .reshape(3,32,32), then transpose with np.transpose(res, axes=[1, 2, 0]). This puts the image into a eligible format.



