# Dogs-vs-Cats-Image-Classification

This is a Convolutional Neural Network (CNN) which is written in python language. This CNN model is trained on few thousand images of cats and dogs, and later be able to predict if the given image is of a cat or a dog. This model achieves over 80% accuracy.

# Simple Image Classification using CNN — Deep Learning in python

The process of building a Convolutional Neural Network always involves four major steps. </br>
Step 1: Convolution </br>
Step 2: Pooling </br>
Step 3: Flattening </br>
Step 4: Full connection </br>


# Description

Cat Classifier was build using 2 layers Deep Neural Network. It was trained on 209 images and tested on 50 images.

Each image was a 64*64 RGB image. Input image was flattened to a vector. Correspnding vector was multiplied with weight matrix.

Weight matrix were randomly initalized and then added with the bias term. The result called linear unit. Then Relu of the linear unit was taken. 

Same process was repeated one more time, but this time instead of Relu, we used sigmoid activation function.

If the output was greater than 0.5 it was predicted as cat and if the output was less than 0.5 it was predicted as non-cat. 

Here Gradient Descent ws used to update the weights and biased terms.

Using these, Finally I have obtained traning accuracy of 98% and test accuracy of 80%.


# Dataset

Data set can be downloaded from [here](https://storage.googleapis.com/kaggle-competitions-data/kaggle/3362/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1545633181&Signature=lOxIjzrxpoFV4aS2e45gtKWtwfyWKU%2FW5OB0ZGmd9CtgrBlMJwB1KnrEZS8Y%2FoIRNFFEVM7pTXjW06EJLgLwesOQ5FFRlnqltVA1hGy9X1WhXe33qilAkD6YRSs5Ue4g%2F%2FCO6404sdCigo6Qk1oHTYaa1rL0XANGi3Y8DdOW31EKtutwsxqsL9LqikEtfrOWapwisa4JQPUgQpatktavg7KzPjImKlCr9SlZsAvbfDds0eSxWCW%2Bn92%2Bee%2Ff3e7b09zW5x%2FXI05W52yiHhEqxJ3LIU%2BWd%2FSbYnjW2VeArB9rSL4%2B2gjVxblluNsRIqvpQmy6xylDBEsSLrG%2F93RTug%3D%3D)
