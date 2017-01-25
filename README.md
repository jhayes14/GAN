# GAN
Generate images via a Generative Adversarial Network (GAN)

Disclaimer: This was purely a learning exercise. I would not recommend using this over established results like DCGAN, additionally the training mechanisms used here have been advised against by DL experts.

## What is a GAN?

A GAN is a method for discovering and subsequently artificially generating the underlying distribution of a dataset; a method in the area of unsupervised representation learning. Most commonly it is applied to image generation tasks. 

A GAN combines two neural networks, called a Discriminator (D) and a Generator (G). Given a dataset, G takes as input random noise, and tries to produce something that resembles an item within the dataset. D takes as input both items within the real dataset and the artifical data produced by G, and tries to distinuish between the two. G and D are trained jointly. The important point is that G and D need to balance one another, neither can become too strong at their task with respect to the other. If G becomes very good at fooling D, this is usually because G has found a weakness in D's classification process which is not aligned with important features within the distribution. If D can easily tell artificial images from real ones, updating G's weights towards the right direction is a very very slow process, essentially G will not be able to learn from this process. 

## How does this GAN work?

I heavily borrowed from a number of other implementations [ [1](https://github.com/jacobgil/keras-dcgan) [2](https://github.com/skaae/torch-gan) [3](https://github.com/aleju/cat-generator) ]. However, with the other implementations I could not produce (decent) images on a single CPU in a short time frame, so I took a new approach to jointly train G and D, guaranteeing neither becomes too strong with respect to the other. Both G and D are DCNNs (deep convolutional neural networks), Batch Normalization is used for G but not for D, as in previous experiments [ [4](http://torch.ch/blog/2015/11/13/gan.html) ], I found Batch Normalization in D made D far too good at distinguishing artifical images from real images.

To ensure that neither G nor D become to good at their respective tasks, I first defined a margin of error, *e*, such that:

|(training loss of G) - (training loss of D)| < *e* , for each training batch.

This results in the lesser of the training of loss of G and D swapping at each successive training batch, resulting in neither becoming too powerful. In other words, if (training loss of G)<(training loss of D), then at the next batch, (training loss of D)<(training loss of G).

Training a GAN is extremely tough, a lot of care has to be paid to tuning the learning rate parameter (as well as other parameters), and takes a long time to get right.

## How to run

This uses Keras (for ML) and OpenCV (for image manipulation).

Run ```train.py``` with the additional flags:
- ```--path "A directory of jpg files used for training"```
- ```--batch_size int```
- ```--epochs int```
- ```--TYPE str``` "train" (for training) or "generate" (for when training is done and you want to create pretty pictures).
- ```--img_num int``` Used for number of images you want to generate when ```TYPE``` is "generate".

## Results

Here are some quick and dirty results after training on ~400 images of faces. Experiments were performed on a MacBook Air 1.4GHz Intel Core i5.

![After 0 minutes](https://github.com/jhayes14/GAN/blob/master/TEST.jpg)   

Initial noise produce by an untrained Generator.

![After 5 minutes](https://github.com/jhayes14/GAN/blob/master/Epoch_13_example.jpg)   

After training for 5-10 minutes.

![After 1 hour](https://github.com/jhayes14/GAN/blob/master/7.jpg)

After training for 45-60 minutes.



## Experiences

Overall this was a fun side-project. I used a very small training set of about 400 images and even with a single CPU machine was able to generate face like shapes within a few minutes, and more detailed faces withing a few hours. I imagine there are many ways to improve the training process to improve results. The limitation of 64x64 images means even after a long time images still look fairly distorted. I have now begun training on a dataset consisting of thousands of images, which will take substantially longer to train but will hopefully produce better results.

## I see VAE functionality in model.py, what gives?

This is something I would have liked to have implemented but didn't have time. Combining a Variational Autoencoder (VAE) with a GAN is popular as they seem to smooth out the rough edges produced by just a GAN. You can read about VAE's [here](https://arxiv.org/abs/1312.6114).

