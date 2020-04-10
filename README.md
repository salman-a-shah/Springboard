# Image Super Resolution using a Residual Network
SALMAN SHAH | SPRINGBOARD

This repository contains the data, code, and process used to complete my capstone project for Springboard—an intense Machine Learning Engineering course program. In this project, I built and put into production a residual neural network that increases the dimensions of an image 2x while maintaining good quality. 

Click to test the [deployed product](https://thephilosopher.pythonanywhere.com/).

The source for the deployed product is also open source. Click to see the [deployment repository](https://github.com/salman-a-shah/image-supersizer).

<p align="center">
  <img src="https://i.ibb.co/HTPXt1q/example-prediction.png">
</p>

### Project Structure
- dataset: A dataset built using image scrapers and python image libraries
- documents: Relevant documents for Springboard
- 1_data_wrangling: a notebook describing the data collection process of the project
- 2_machine_learning_prototype: a notebook explaining the viability of neural networks to solve the image super resolution problem
- 3_experimenting_with_PSNR_metric: a notebook experimenting with the PSNR metric
- 4_evolution_algorithm: a notebook for automating the hyperparameter search process in google colab
- 5_final_training_session: a notebook training the final model
- scraper, resizer, standardizer: Helper functions used to build the dataset

## Introduction 
A single image super resolution problem is a problem where one attempts to recover a higher quality version of a given image.

For example, given an image that is 400x300 pixels, the goal is to obtain an image that is 800x600 pixels (for 2x upscale) that maintains a quality that is equivalent to the given image, without making it appear pixelated or blurry.

## Inspiration
This project was inspired by [this paper](https://arxiv.org/abs/1802.08797) and [this implementation](https://github.com/idealo/image-super-resolution) of the paper's model.

This project, however, yields decent results without the dense components of the neural network.

## Data Collection
The dataset was built by scraping images from google images and standardizing the sizes of all images. The data collection process is straight forward—we need a set of original images of size `MxN` (which is used to compute the loss of the model) and a set of the same images downscaled to size `0.5Mx0.5N`. So I scraped 1300 images from google images, filtered through bad data to produce a set of 1087 images. Then I cropped those 1087 images to `600x600`. After that, I duplicated those images to another folder and downscaled them to `300x300`.

Details of the data collection process can be found in the [data wrangling notebook](https://github.com/salman-a-shah/Springboard/blob/master/notebooks/data_wrangling.ipynb).

The raw data is available for download [in this folder](https://drive.google.com/drive/folders/1ggEh_5B0rwm6NnrEHOPvIsTKgWJSOLbp?usp=sharing). 

## Model Architecture
The architecture used in this project is a residual neural network (a neural network that utilizes skip connections). A diagram depicting the model architecture is given below.

![alt text](https://i.ibb.co/tcvXb2f/Image-Supersizer-model-architecture.png)

## Loss Function and Metrics
Loss is computed by measuring pixel-to-pixel difference. Both mean squared error and mean absolute error were tested in this model. The referenced paper claims that mean absolute error is superior for this problem, but the difference was negligible in this case.

There were two metrics that were considered for this problem, the first being pixel-to-pixel accuracy and second being peak signal-to-noise ratio (PSNR). An interesting observation was that the PSNR model appeared to produce a much sharper image (too sharp at times), while the accuracy model appeared to sometimes produce a slightly blurry image. Overall, the PSNR seemed to do better in most cases.

Below are some comparisons between the input, accuracy model, PSNR model, and ground truth.

![alt_text](https://i.imgur.com/dZolJNK.jpg)

## More about the author
Visit my webpage at [salman-a-shah.github.io](https://salman-a-shah.github.io/).
