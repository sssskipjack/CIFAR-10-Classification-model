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

# Optimizing model

#### Improvement 1
To optimize my model, I change the activation function of the output layer into softmax. My model this time begins to overfit at around 8 epochs, and has a validation accuracy of approximately 0.40. This is a major improvement over my previous model, and It shows my model is working as intended, and learning from the training set.

#### Improvement 2
I decided to reformat my data into with a modified show_image funtion before feeding it into the model. I created a convert_dataset function that transformed every image in the dataset with the transformations shown in "Displaying Image". I found that feeding the image as (3,32,32) verses (1,3072) has absolutely no effect on the model. This model produced almost identical results as the previous model.

#### Improvement 3
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
This time my model stil began to overfit at around 8-9 epochs

![Epochs graph](/50%20epoch%20test.png "epochs graph")

With these changes I was able to achieve a validation accuracy of 0.52, which is a improvement.


#### Optimization 5

For this optimization, Instead of using just batch 1 of the dataset, I used all 5 batches. 5 epochs was the optimal number, validation loss was the lowest, and validation accuracy was leveling out. After 5 epochs, my model began to overfit as validation loss rose 

![loss](/model%205%20accuracy%20and%20validation%20accuracy.png)
![Accuracy](/model%205%20validation%20loss%20and%20loss.png)

This time, I had a validation accuracy of 0.70, which is significantly higher. Simply by increasing the data set, I was able to improve my model by almost 20 percent


```Epoch 1/8
1250/1250 [==============================] - 136s 107ms/step - loss: 1.7186 - accuracy: 0.3540 - val_loss: 1.3694 - val_accuracy: 0.5045
Epoch 2/8
1250/1250 [==============================] - 126s 101ms/step - loss: 1.2325 - accuracy: 0.5583 - val_loss: 1.0975 - val_accuracy: 0.6093
Epoch 3/8
1250/1250 [==============================] - 141s 112ms/step - loss: 0.9999 - accuracy: 0.6462 - val_loss: 1.0169 - val_accuracy: 0.6408
Epoch 4/8
1250/1250 [==============================] - 131s 105ms/step - loss: 0.8488 - accuracy: 0.7036 - val_loss: 0.9426 - val_accuracy: 0.6758
Epoch 5/8
1250/1250 [==============================] - 141s 113ms/step - loss: 0.7315 - accuracy: 0.7471 - val_loss: 0.8589 - val_accuracy: 0.7076
Epoch 6/8
1250/1250 [==============================] - 135s 108ms/step - loss: 0.6214 - accuracy: 0.7822 - val_loss: 0.8646 - val_accuracy: 0.7065
Epoch 7/8
1250/1250 [==============================] - 136s 109ms/step - loss: 0.5158 - accuracy: 0.8204 - val_loss: 0.8595 - val_accuracy: 0.7219
Epoch 8/8
1250/1250 [==============================] - 138s 110ms/step - loss: 0.4341 - accuracy: 0.8497 - val_loss: 0.9820 - val_accuracy: 0.7148
```

Here are some of the images, and the categories my model put them in 
![predictions](/image%20predictions.png)
