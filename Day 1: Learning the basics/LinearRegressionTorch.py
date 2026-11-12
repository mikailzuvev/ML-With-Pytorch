import torch #import the machine learning tools
import torch.nn as nn #this is the neural network generator in pytorch
import torch.optim as optim #this is the optimizer 

# Synthetic data generation so the code runs out-of-the-box
X = torch.randn(100, 1)
Y = 2 * X + 1 + torch.randn(100, 1) * 0.1

#this class inherits from nn Module, this lets pytorch know this class 
#contains weights that need tracking
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):

        #this initializes the nn.Module, so tracking works properly
        super(LinearRegressionModel, self).__init__()

        #instantiates a standard linear layer
        #creates a weight matrix (w) and a bias vector (b)
        self.linear = nn.Linear(input_dim, output_dim)

    #function passes inputed data through a linear regression model
    #y = wx + b
    def forward(self, x):
        return self.linear(x)


#create a model where you give one input dimension and one output dimension
#example of input_dim: house price, height, temperature
#example of output dimensions: house price, weight, icecream sales
model = LinearRegressionModel(input_dim=1, output_dim=1)

#sets up mean squared error as our loss metric
#Sum up the mean squared difference of the prediction and reality
#penalizes large misses
criterion = nn.MSELoss()

#configures stochastic gradient descent which minimizes the ML models 
#error by updating parameters with only one data point or small batch at a time
#rather than the whole database all at once, PERFORMANCE OPTIMIZER
optimizer = optim.SGD(

    # --- FIX 1: Added () to model.parameters() ---
    #tells the optimizer to look inside the model created and grab the random
    #weight and bias values and change only them
    model.parameters(), 

    #sets the learning rate, each time the optimizer makes a change
    #it will multiply it by the learning rate so it doesnt overshoot the curve
    #example: actual change = 10 * 0.01 = 0.1, this prevents the model from overshooting
    lr=0.1)


#--------------------------------------------------------------------#
#=========================== TRAINING LOOP ==========================#
#--------------------------------------------------------------------#

#to train a model we run a loop over multiple epochs (iterations of the dataset)
#inside the loop is where the actual learning happens
epochs = 1000
for epoch in range(epochs):

    #===CLEAR PAST MEMORIES ===#
    #wipes out gradient created by previous loop iterations
    #makes sure to remove the gradients pytorch adds by default
    #prevents massive error from previous calculations
    optimizer.zero_grad()


    #=== FORWARD PASS ===#
    #passes the input data X through the model
    #takes X multiplies it by the random weight and adds the random bias
    #saves the output into predictions
    predictions = model(X)


    #=== COMPUTE THE LOSS ===#
    #compares the prediction to the real value Y using mean squared error MSE
    #yields a single number that represents how wrong the model is
    #the lower the loss the better our fit line data is
    loss = criterion(predictions, Y)


    #=== BACKWARDS PASS / BACKPROPAGATION ===#
    # Triggers backpropagation, starting at the final loss number and finds the derivative
    # for every single parameter in model.parameters().
    # This step acts as a map pointing the way up the hill of error.
    # Identifies flaws and saves them to each parameter's unique .grad attribute.
    # .grad says: "If you want loss to go UP, change this parameter in this direction by (G)."
    loss.backward()


    #=== PARAMETER UPDATE ===#
    # Reads from the .grad attributes of the parameters to look at the values of (G).
    # Multiplies (G) by the learning rate (step size).
    # Subtracts that result from the parameter to nudge it in the exact OPPOSITE direction.
    # This physically shifts the weights and biases one step closer to reality.
    #examples for weight: New weight = Old Weight - (gradient value * learning rate)
    #example for bias: New Bias = Old Bias - (Gradient Value * learning rate)
    optimizer.step() 

    #=== STEPS TO ACTUALLY TRAIN AND RUN THE MODEL ===#
    # 1. Put the model in Evaluation Mode
    model.eval()

    # 2. Turn off the gradient engine to save memory and speed up processing
    with torch.no_grad():
        
        # 3. Create a new data point (e.g., X = 5.0)
        # Note: It must be a 2D float tensor to match the structure the model expects
        new_input = torch.tensor([[5.0]], dtype=torch.float32)
        
        # 4. Run the model!
        prediction = model(new_input)
        
        # 5. Extract the raw number from the tensor
        print(f"Prediction for X=5.0 is: {prediction.item():.4f}")