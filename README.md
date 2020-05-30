# Face-Recognition-using-PCA-Deep-Learning

In this term project, I utilized data from the ORL and Yale B databases. ORL has labelled
images, while Yale B is not labelled. This labelling identifies if the some person occupies multiple images,
i.e. one person could have their picture taken from different angles; there unique identifier would accompany
all images containing this individual in a labelled dataset.
The interesting element regarding this labelling is that models utilizing CNNs need to use labelled

datasets. Otherwise, there does not exist a method to determine the accuracy of the predictions result-
ings from these models. As a result, we propose to use a network which uses the ORL dataset with labelled

images, and the PCA values from this dataset, to guide a neural network model to predict the labels of
Yale B dataset. The idea of this network is that labelling a dataset is difficult process, but by using the
information of PCA (which is low dimensional), we are able to predict the labels with improved accuracy.
In other words, we train the network with low-dimensional, correlated PCA data on our ’training dataset’
(ORL), to inform the labelling process of our ’testing dataset’ (Yale B).

For a low dimensional representation of the image using PCA. We create a CNN network which maps
the latent representation to the PCA value. Now we create another CNN network which takes the latent
representation of image and PCA as input. We feed both the inputs with the actual label of the image as
output of the CNN network.
By training the network, we show that the hidden layer learns all the information about the eigenvalues
in its hidden representation. The remaining information is received from the high dimensional images. So,
the network completely learns most essential features from the low dimensional PCA results, while learning
the remaining information from the high dimensional images (while feeding to the labels, i.e. the facial
recognition). We had hoped this would accelerate the learning rate of a network by a priori knowing the
cross correlations within a particular set of data.
