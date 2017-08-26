# Behaviour Cloning


| Note    | |
|:-----------|:-------------|
| **Source Code**  | https://github.com/aurangzaib/CarND-Behavioral-Cloning-P3  |
| **How to train**  | `cd implementation && python main.py`      |
| **How to test**  | `python drive.py model.h5`      |

The steps of the project are as follows:

-	Use the simulator to collect data of good driving behavior.

-	Build a convolution neural network (CNN) in Keras that predicts steering angles from images.

-	Train and validate the model with a training and validation set.

-	Test that the model successfully drives around the track without leaving the road.


---

### 1-	Data Collection & Augmentation:

#### i- Histogram Visualization:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/visualization.py`  |
| Method  | `Visualization.visualize_histogram`      |

![alt text](./documentation/steering-distribution-udacity.png)

![alt text](./documentation/steering-distribution-udacity-0-removed.png)

![alt text](./documentation/steering-distribution-augmented-0-removed.png)

![alt text](./documentation/steering-distribution-augmented-all-cameras-0-removed.png)

![alt text](./documentation/steering-distribution-augmented-all-cameras-flips-0-removed.png)

#### ii- Data Visualization:


| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/visualization.py`  |
| Method  | `Visualization.visualize_features`      |


![alt text](./documentation/data-exploration-1.png)

![alt text](./documentation/data-exploration-2.png)

![alt text](./documentation/data-exploration-3.png)

![alt text](./documentation/data-exploration-4.png)


#### iii- Cropping:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/visualization.py`  |
| Method  | `Visualization.visualize_roi`      |

![alt text](./documentation/ROI-1.png)

![alt text](./documentation/ROI-2.png)

![alt text](./documentation/ROI-3.png)

![alt text](./documentation/ROI-4.png)


### 2- Neural Network and Training Strategy:

#### i- Model Architecture

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/classifier.py`  |
| Method  | `Classifier.implement_classifier`      |

-	NVIDEA DNN architecture.
-	5 Convolution layers.
-	3 Fully Connected (Dense) layers.
-	1 Output layer.
- 	Normalization:
	-	To center the mean and standard deviation around 0.
-	Cropping:
	- To remove the areas of the image which are not useful for training the classifier e.g. sky etc.
-	Dropouts:
	- Used with Dense layers with keep probability of 0.5 to reduce the overfit.
- Backpropogation:
	- Adam Optimizer is used to update the weights and Mean Square Error (MSE) is used to keep track of the errors.


| Pipeline    |Filter  |  Kernel | Dropout  |  Output Dimension | Parameters |
|:-----------|:-------------|:-------------|:-------------|:-------------|:-------------|
| Normalization  |   |  |  |  |  0 | 
| Cropping  | 24|   |98.7  |160,320,3  |0  |
| Conv1  | 24  |5,5  |1.0  |65,320,3  |1824  |
| Conv2  | 36  |5,5  |1.0  |31,158,24  |21636  |
| Conv3  | 48  |5,5  |1.0  |14,77,36  |43248  |
| Conv4  | 64  |3,3  |1.0  |5,37,48  |27712  |
| Conv5  | 64 |3,3  |1.0  |1,33,64  |36928  |
| Flatten  |   |  |  |2112  | 0 |
| Dense1  |   |  |0.5  |100  |211300  |
| Dense2  |   |  |0.5  |50  |5050  |
| Dense2  |   |  |0.5  |10  | 510 |
| Output  |   |  |  |1  | 11 |