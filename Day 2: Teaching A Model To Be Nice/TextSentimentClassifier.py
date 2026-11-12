# This module scans raw text files to build a vocabulary of all unique words
# and converts the sentences into word importance matricies using
# term frequency-inverse document frequency math (TF-IDF) 
# this will score every word by how often it appears and how important it is
from sklearn.feature_extraction.text import TfidfVectorizer

# A utility function that shuffles and splits the training data into two portions
# 1). A training set to teach the model
# 2). A testing set to see how the model generalizes to unseen data
from sklearn.model_selection import train_test_split

# The core pytorch library that handles multidimensional arrays called tensors
# Manages CPU and GPU memory allocation to provide fast mathmatical analysis for deep learning
import torch

# Pytorch's Neural Network module that provides the archetechtual building blocks
# to build neural networks so we dont need a ton of boilerplate code
import torch.nn as nn

# PyTorch's optimization module, contains algorithms like ADAM and SGD that take 
# the error gradeint computed during back progagation and updates the models internal
# weights to make it more accurate
import torch.optim as optim


#=== DEFINING THE DATA ===#

# A python list of raw strings we want our model to classify
texts = [
    "You're doing fantastic, keep it up!",
    "I hate talking to you, go away.",
    "That was a really helpful suggestion.",
    "You ruin everything you touch.",
    "Have a wonderful and productive day!",
    "You are completely useless."
]

# A parallel list of 1's and 0's representing the ground truth
# 1's represent nice 
# 0's represent mean
labels = [1, 0, 1, 0, 1, 0]


#=== TEXT VECTORIZATION WITH SCIKIT-LEARN ===#

# initializes the tfid vectorizer preparing to scan text, build a vocabulary
# and compute with math scores
vectorizer = TfidfVectorizer()

# This does 2 very important jobs in one line
# 1). fit: reads all the sentences, finds every unique word, and assigns each word a collum index
# 2). transfrom: converts every sentence into a row of numbers (sparse matrix is a matrix where many value are just 0) 
# representing how important each word is
X_sparse = vectorizer.fit_transform(texts)

#=== CONVERTING SCIKIT-LEARN DATA INTO PYTORCH TENSORS ===#


x_tensor = torch.tensor(
    # Because scikit-learn outputs sparse matricies and pytorch cannot read them directly
    # pytorch converts them into standard dense 
    X_sparse.toarray(), 

    # wraps the created numpy array into PyTorch tensors and forces all numbers to become
    # 32-bit floats, which are the standard data type nueral networks use                   
    dtype=torch.float32)


# This converts the list of labels into Pytorch Tensors
# The unsqueeze reshapes the label 1D arrays into a 2D collum vector
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

#=== DEFINING THE HYBRID NEURAL NETWORK ARCHETECHTURE ===#

# defines a standard neural network inheriting from the nn Module
class TFIDTextClassifier(nn.Module):

    #the setup function in which we difine the layers out model will use
    def __init__(self, input_dim):
        super(TFIDTextClassifier, self).__init__()
        self.sf1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()

        
        self.sf2 = nn.Linear(16, 1)

        # the sigmoid function forces all the outputs of the neural netowork to be between 1 and 0
        self.sigmoid = nn.Sigmoid()




