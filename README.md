# Leveraging-Explainable-AI

Team Lead: [Rowan White]()  
Participants: [Aaron Barthwal](), [Khiem Nguyen](), [Laya Srinivas](), [Ram Gudur](), [Sanya Oak]()

## Poster
![Poster]()

## Introduction
Artificial intelligence is rapidly gaining traction in the world of technology as a powerful tool for data analysis, classification, and prediction. However, today’s AI typically relies on a ”black box” model: a human observer can’t extrapolate how the model makes its decisions through its operations alone. We rely on quantifying the model through its output. Explainable AI (XAI) is a hot topic in AI research today, endeavoring to make the inner workings of a ”black box” model clearer by interpreting the results

## Dataset
Our data is taken from IdenProf, an open‐source dataset collected by Moses Olafenwa and made publicly accessible on GitHub. IdenProf contains 11,000 images of workers from 10 classes of visually‐identifiable professions (ex: doc‐ tor, engineer, chef, pilot), sorted into training sets of 900 images per class, and testing sets of 200 images per class. The biased datasets are of three categories: watermark, uneven, and homogenous. In the watermark category, all chef and doctor images bear a watermark (a standard full‐image watermark for the former and a timestamp for the latter). In the uneven category, only the waiter and chef classes have 900 images, and the rest have 600. In the homogenous category, 6 of 10 datasets were manually pruned so that the images had more in common with each other.

## Our Model
Data Augmentation: Data augmentation artificially creates more diversity in a dataset by randomly altering images, whether by cropping, flipping, or rotating them. A dataset with more diversity teaches a model to generalize, and typically leads to better accuracy. We implemented TrivialAugment from PyTorch’s image classification sub‐library TorchVision to automate the process of transforming images at a random frequency and to random magnitudes.

Convolutional Neural Network: The image‐classification model takes image input in the form of tensors, long arrays of numbers representing the color values of each pixel. During the training process, the tensors are passed through layers of ”neurons”, which manipulate a number of weights to minimize the loss and maximize accuracy. Throughout this process, the data is simplified and compressed by pooling layers, which halve the dimensions of the image data going into the next layer. At the end of the process, the weights are used to predict the classification label of a test image, then produces class prediction probabilities, from which the greatest is selected as the result.

## SHAP and LIME
LIME (Local Interpretable Model‐Agnostic Explanations) and SHAP (Shapley Additive Explanations) are two XAI tools used for image classification, producing relevancy heat maps that demarcate the parts of an image which contributed the most to the classification result. In our research, we manually create sources of bias in an image‐classification dataset and train an image‐classification model on the flawed datasets. Then, using LIME and SHAP, we analyze the heat maps produced for any visual indicator of these data biases.

Using SHAP, we were able to see how different aspects affect what a machine learning model looks for when classifying images. Using the watermarked datasets, we were able to see that for the Chef and Doctor sets, the model picked up on the watermarks as a classifier for the images. The Chef set was marked with a diagonal watermark from bottom left to top right, as we see here, the SHAP values show a cross across the images due to the watermark and random horizontal flips. On the Doctor set, there was a watermark at the top of the im‐ age, which is seen with the SHAP values across the second SHAP column below.

![SHAP image]()

LIME (Local Interpretable Model‐agnostic Explanations) is a technique used for explaining the predictions of machine learning models by approximating the model’s decision boundary locally around a specific instance. LIME is model‐agnostic, meaning it can be applied to any machine learning model without needing knowledge of the model’s internal structure.

## Results
Leveraging the tools of explainable artificial intelligence, we were able to detect some forms of bias. Most notably, we were able to detect distinct patterns that imaging AI’s may pick up on such as when there is a common watermark used across a category of a dataset.
