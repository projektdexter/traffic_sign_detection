# traffic_sign_detection
traffic sign detection using ML

In the below code, I use ML tools to identify road traffic signs. This can be a good autonomous driving support for drivers with low to medium driving experience. 

### Traffics signs on roads
We know that every city has a variety of traffic signs that are deployed across the territory to inform drivers of the road conditions, locality, driving speed etc. Since the number of traffic signs have increased, new drivers will face difficulty in recognizing some of these traffic signs. Not only is this a problem for the driver since not following these instructions may amount to driving tickets, it is also unsafe for the surroundings. For example, drivers are careful when they see a school ahead road sign. If for some reason the driver is not able to identify these road signs, negligence on the road may occur in a place which is most dangerous.  

### ML to solve this problem?
The solution to the problem can be, machine learning. In the below code I show machine learning techniques which can be used with fairly good accuracy. 
Simple ML techniques from SKLearn will not return very high accuracy. Support Vector Classifier returns a lowly accuracy score of 7% while MPLClassifier returned a accuracy of 26%. 

### Neural Networks improve accuracy to 96.5%
The above results are not good although quite expected since these models are not very powerful when it comes to high dimensional data.
To improve our performance, I look at neural network models. I use 4 dense layers with ‘relu’ activation function and ‘HEUniform’ kernel initializer for back-propagation. The final layer uses softmax activation layer to predict the class. Finally, I use ‘sparse_categorical_crossentropy’ loss function and ‘Adamax’ optimizer to compile the model. 
The accuracy score for 5-layer ANN comes out to 96.5%.
### Convolutional layer imrpoves the accuracy to 99%

The final version uses convolution layer. In this version, I load the previous model with convolutional layer to further improve the accuracy score. 

### Dataset
For the purpose of this study, I use the [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) which has over 50k images of road signs in Germany along with their respective classes. Some of the images are shown below 

![00002_00002_00029](https://user-images.githubusercontent.com/114884444/203641247-cfeeea3e-e786-4a31-b741-e4e4808a3620.png)

__50 kmph speed limit__

![00014_00000_00001](https://user-images.githubusercontent.com/114884444/203641404-1f198487-70a7-4048-a255-e87d0e2133f8.png)

__Stop Sign__

![00020_00003_00024](https://user-images.githubusercontent.com/114884444/203641498-f173b7a0-b223-4ba5-b1fc-1d598392c1bc.png)

__Right Turn__

The dataset contains images of different size and therefore some level of preprocessing is done to convert the data into more manageble form. After preprocessing the data, we get a NxN matrix with rows representing the pixel density of an image.

## Results

We run the NN algorithm for 2 epochs.
```
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20)
```
## Output
```
Epoch 1/20
1226/1226 [==============================] - 17s 12ms/step - loss: 1.6335 - accuracy: 0.5791
Epoch 2/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.6847 - accuracy: 0.8086
Epoch 3/20
1226/1226 [==============================] - 17s 14ms/step - loss: 0.4803 - accuracy: 0.8636
Epoch 4/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.3898 - accuracy: 0.8873
Epoch 5/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.3312 - accuracy: 0.9022
Epoch 6/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.2937 - accuracy: 0.9145
Epoch 7/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.2589 - accuracy: 0.9245
Epoch 8/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.2310 - accuracy: 0.9313
Epoch 9/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.2045 - accuracy: 0.9397
Epoch 10/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.2028 - accuracy: 0.9396
Epoch 11/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.1809 - accuracy: 0.9460
Epoch 12/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1780 - accuracy: 0.9459
Epoch 13/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1601 - accuracy: 0.9515
Epoch 14/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.1544 - accuracy: 0.9527
Epoch 15/20
1226/1226 [==============================] - 15s 13ms/step - loss: 0.1466 - accuracy: 0.9555
Epoch 16/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1383 - accuracy: 0.9574
Epoch 17/20
1226/1226 [==============================] - 16s 13ms/step - loss: 0.1357 - accuracy: 0.9576
Epoch 18/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1212 - accuracy: 0.9627
Epoch 19/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1120 - accuracy: 0.9656
Epoch 20/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1127 - accuracy: 0.9656
```
