# Traffic sign detection using Machine Learning

In the below code, I use ML tools to identify road traffic signs. This can be a good autonomous driving support for drivers with low to medium driving experience. 

### Traffics signs on roads
We know that every city has a variety of traffic signs that are deployed across the territory to inform drivers of the road conditions, locality, driving speed etc. Since the number of traffic signs have increased, new drivers will face difficulty in recognizing some of these traffic signs. Not only is this a problem for the driver since not following these instructions may amount to driving tickets, it is also unsafe for the surroundings. For example, drivers are careful when they see a school ahead road sign. If for some reason the driver is not able to identify these road signs, negligence on the road may occur in a place which is most dangerous.  

### ML to solve this problem?
The solution to the problem can be, machine learning. In the below code I show machine learning techniques which can be used with fairly good accuracy. 
Simple ML techniques from SKLearn will not return very high accuracy. Support Vector Classifier returns a lowly accuracy score of 7% while MPLClassifier returned a accuracy of 26%. 

### Neural Networks improve accuracy to 96.5%
The above results are not good although quite expected since these models are not very powerful when it comes to high dimensional data.
To improve our performance, we look at neural network models. I use 4 dense layers with ‘relu’ activation function and ‘HEUniform’ kernel initializer for back-propagation. The final layer uses softmax activation layer to predict the class. Finally, I use ‘sparse_categorical_crossentropy’ loss function and ‘Adamax’ optimizer to compile the model. 
The accuracy score for 5-layer ANN comes out to 96.5%.
### Convolutional layer imrpoves the accuracy to >99%

In the final version, I load the previous model with convolutional layer to further improve the accuracy score. Adding ```Conv2D``` layer requires a lot of memory and makes the training slow. Nonetheless, we achieve a training accuracy of 99.75%.  

### Dataset
For the purpose of this study, I use the [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) which has over 50k images of road signs in Germany along with their respective classes. Some of the images are shown below 

![00002_00002_00029](https://user-images.githubusercontent.com/114884444/203641247-cfeeea3e-e786-4a31-b741-e4e4808a3620.png)

__50 kmph speed limit__

![00014_00000_00001](https://user-images.githubusercontent.com/114884444/203641404-1f198487-70a7-4048-a255-e87d0e2133f8.png)

__Stop Sign__

![00020_00003_00024](https://user-images.githubusercontent.com/114884444/203641498-f173b7a0-b223-4ba5-b1fc-1d598392c1bc.png)

__Right Turn__

The dataset contains images of different size and therefore some level of preprocessing is done to convert the data into more manageble form. After preprocessing the data, we get a NxN matrix with rows representing the pixel density of an image.

## Results ANN Model

We run the ANN algorithm for 20 epochs.
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

.
.
.
.
Epoch 19/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1120 - accuracy: 0.9656
Epoch 20/20
1226/1226 [==============================] - 15s 12ms/step - loss: 0.1127 - accuracy: 0.9656
```

## Results CNN Model
We run the CNN algorithm for 11 epochs.
```
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20)
```

### Output
```
Epoch 1/20
1226/1226 [==============================] - 140s 113ms/step - loss: 0.8862 - accuracy: 0.7881
Epoch 2/20
1226/1226 [==============================] - 133s 109ms/step - loss: 0.1812 - accuracy: 0.9549
Epoch 3/20
1226/1226 [==============================] - 133s 109ms/step - loss: 0.0977 - accuracy: 0.9752
.
.
.
Epoch 9/20
1226/1226 [==============================] - 133s 108ms/step - loss: 0.0226 - accuracy: 0.9938
Epoch 10/20
1226/1226 [==============================] - 135s 110ms/step - loss: 0.0168 - accuracy: 0.9957
Epoch 11/20
1226/1226 [==============================] - 133s 109ms/step - loss: 0.0122 - accuracy: 0.9970
```
