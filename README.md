# Machine learning CPP
by La Hai Nam
# Foreword
Thanks for [David Miller](https://www.millermattson.com/dave/?p=54) for the structure of the neural network code.
# Overview
The project in this repository was done for a college assignement, which I did with fellow students Viet Anh Kopietz and Minh David Nguyen. It is a machine learning programm which is currently able to learn the MNIST database and determine numbers. It was designed to study multithreading and how it impacts the machine learning speedwise and accuracy wise.
# Documentation (only in german)
For a brief documentation on how this projects was made refer to the documents folder. 
# Neural Network

For the project we use a neural network which consists of 4 layers: one inputlayer, 2 hiddenlayer and one ouputlayer.
The size of each layer is set from input to outputlayer: 784, 128, 128, 10 at default.
For the learning algorithm we use a classical feed-forward perceptron and backpropagation algorithm, with the use of the activation function sigmoid. For multithreading we use OpenMP. Our code also uses CXXXFLAGS -O0, -Os, -O2, -O3, -O2 -ftree-vectorize and -O2 -ftree-vectorize -ffast-math seperately to test their impact on training speed. The weights after training will be saved in the file weights.txt.
# Building the Code
(Keep in mind all coding and testing was done in Linux but hopefully should work in Windows as well)  
We have 2 folders which consist of the same neural network but  one with multithreading and one without. To learn and test the neural network proceed as follows:    
-  Go into desired folder (mpt_neural_networl_mit_omp == neural network with multithreading, mpt_neural_network_ohne_omp == neural network without multithreading,)
- Enter: "make train" to train the chosen neural network whereas a time will be shown for each CXXXFlag as it trains with every CXXFlag.
- Example: "make train"
OR
- Enter: "make x='any number' y='any number' benchmark" to test the training time of the chosen neural network with changed hiddenlayer size
 ("x" changes the size of the first hiddenlayer and "y" the size of the second hiddenlayer)
- Example: "make x=12 y=1280 benchmark"
- To test the Neural Network Enter: "make test" to train with default hiddenalyer size(128, 128) or enter "make x='any number' y='any number' test"
 Example: "make x=1280 y=10 test "() be sure to use the same hiddenlayer size which has been used to train the neural network.)
