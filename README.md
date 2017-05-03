## Project: Traffic Sign Classification
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project we train a convolutional neural network to solve a classic problem in self-driving cars - identifying traffic signs from images. We built our model on top of LeNet5 architecture and trained it using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). We then test it using the testing data from the dataset as well as images from the internet that were shot unfiltered in the real world.

The code and data we used to achieve that can be found in this repository called *Traffic_Sign_Classifier.ipynb*

Dataset Exploration
---
The dataset is split into training data, validation data and testing data. Each example is composed of an traffic sign image and a label that describes the traffic sign class. The summary is as follows:
* training: 34799.0 - 67.13%
* validation: 4410 - 8.51%
* testing: 12630 - 24.36%
* total: 51839
* image dimensions: 32x32x3
* labels range: 0 - 42

The distribution of classes:
* Class 0: 270 - 0.52%
* Class 1: 2940 - 5.67%
* Class 2: 3000 - 5.79%
* Class 3: 1860 - 3.59%
* Class 4: 2640 - 5.09%
* Class 5: 2490 - 4.80%
* Class 6: 570 - 1.10%
* Class 7: 1890 - 3.65%
* Class 8: 1860 - 3.59%
* Class 9: 1950 - 3.76%
* Class 10: 2670 - 5.15%
* Class 11: 1740 - 3.36%
* Class 12: 2790 - 5.38%
* Class 13: 2880 - 5.56%
* Class 14: 1050 - 2.03%
* Class 15: 840 - 1.62%
* Class 16: 570 - 1.10%
* Class 17: 1470 - 2.84%
* Class 18: 1590 - 3.07%
* Class 19: 270 - 0.52%
* Class 20: 450 - 0.87%
* Class 21: 420 - 0.81%
* Class 22: 510 - 0.98%
* Class 23: 660 - 1.27%
* Class 24: 360 - 0.69%
* Class 25: 1980 - 3.82%
* Class 26: 780 - 1.50%
* Class 27: 300 - 0.58%
* Class 28: 690 - 1.33%
* Class 29: 360 - 0.69%
* Class 30: 600 - 1.16%
* Class 31: 1050 - 2.03%
* Class 32: 300 - 0.58%
* Class 33: 899 - 1.73%
* Class 34: 540 - 1.04%
* Class 35: 1590 - 3.07%
* Class 36: 510 - 0.98%
* Class 37: 270 - 0.52%
* Class 38: 2760 - 5.32%
* Class 39: 390 - 0.75%
* Class 40: 450 - 0.87%
* Class 41: 300 - 0.58%
* Class 42: 330 - 0.64%




### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
