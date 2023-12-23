import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker

# Added for PyTorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader

#Added to Preprocess Pandas Data
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# Create PyTorch RegressorModel Class
class RegressorModel(nn.Module):
    
    def __init__(self, input_size):  
        """ 
        Initialise the NN model
          
        Arguments:
            - <input_size>: size of input to NN

        """
        self.input_size = input_size 
        self.output_size = 1

        # Non-Linear Regression Model
        super(RegressorModel, self).__init__()
        self.fc1 = nn.Linear(self.input_size,54, bias=True)
        self.fc2 = nn.Linear(54, 37, bias=True)
        self.fc3 = nn.Linear(37, 14, bias=True)
        self.fc4 = nn.Linear(14, self.output_size, bias=True)

        self.dropout = nn.Dropout(p=0.1)    # Include Dropout of 10 percent

    def forward(self, x):   # Called for every: ... = RegressorModel(x) - due to nn.Module
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x  
    

class Regressor():

    def __init__(self, x, nb_epoch = 10, size_batch = 16, learning_rate = 0.001):
        """ 
        Initialise the Regressor.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        # Store Parameters for Preprocessing
        self.label_binarizer = None
        self.x_mean_NaN = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # General Parameters
        self.nb_epoch = nb_epoch 
        self.batch_size = size_batch
        self.learning_rate = learning_rate
        input_size = x.shape[1] + 4     # +4 due to Data preprocessing with LabelBinarizer()

        # Regressor Model
        self.Model = RegressorModel(input_size)  


    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        x.reset_index(drop=True, inplace=True) # Important: otherwise you cannot concatenate dataframes with different indexing
        if(y is not None):
            y.reset_index(drop=True, inplace=True)

        if training:
            # Handle Textural Values using One-Hot Encoding
            if self.label_binarizer is None:
                self.label_binarizer = LabelBinarizer()
                one_hot_encoded = self.label_binarizer.fit_transform(x['ocean_proximity'])
            else:
                one_hot_encoded = self.label_binarizer.transform(x['ocean_proximity'])

            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=[f'ocean_proximity_{i}' for i in range(one_hot_encoded.shape[1])])
            x = pd.concat([x, one_hot_encoded_df], axis=1)
            x = x.drop('ocean_proximity', axis=1)   # Delete Middle Column

            # Handle NaN values, replacing them with mean of column
            if self.x_mean_NaN is None:
                self.x_mean_NaN = x.mean()
                x = x.fillna(self.x_mean_NaN)
            else:
                x = x.fillna(self.x_mean_NaN)

            # Normalize Numerical Values
            if self.x_std is None:
                self.x_mean = x.mean()
                self.x_std = x.std()
                x = (x-self.x_mean)/self.x_std           
                if(y is not None):
                    self.y_mean = y.mean()
                    self.y_std = y.std()
                    y = (y-self.y_mean)/self.y_std
        else:
            # Handle Textural Values using One-Hot Encoding
            one_hot_encoded = self.label_binarizer.transform(x['ocean_proximity']) 
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=[f'ocean_proximity_{i}' for i in range(one_hot_encoded.shape[1])])
            x = pd.concat([x, one_hot_encoded_df], join='inner',axis=1)
            x = x.drop('ocean_proximity', axis=1)   # Delete Middle Column
            
            # Handle NaN values, replacing them with mean of column
            x = x.fillna(self.x_mean)

            # # Normalize Numerical Values
            x = (x-self.x_mean)/self.x_std         
            if(y is not None):
                y = (y-self.y_mean)/self.y_std

        # Convert Pandas Dataframe Data into Torch Tensors 
        x = torch.tensor(x.values, dtype=torch.float32)
        if(y is not None):
            y = torch.tensor(y.values, dtype=torch.float32)
        return x, (y if isinstance(y, torch.Tensor) else None)

        
    def fit(self, x_train, y_train, x_val = None, y_val = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        # Define Training Parameters
        max_epochs = self.nb_epoch
        size_batch = self.batch_size
        learning_rate = self.learning_rate

        # Preprocess Data
        x, y = self._preprocessor(x_train, y = y_train, training = True) 
        # Prepare Data for Batching & Training
        dataset = torch.utils.data.TensorDataset(x,y)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=size_batch)     #Batch the data
        
        # Define Optimiser & Loss Function
        opt = optim.Adam(self.Model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
            
        dataset_len = len(x)
        losses = []
        losses_val = []
        for epoch in range(max_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:

                # Reset Gradients
                opt.zero_grad()            

                # Forward Pass
                y_hat = self.Model(inputs)
                
                # Compute Loss
                loss_value = criterion(y_hat,labels)
                running_loss += loss_value.item()*inputs.size(0)

                # Backward Pass (computing gradients)
                loss_value.backward()

                # Update Weights based on Computed Gradients
                opt.step()

            # Use this to store the loss of validation/test data at each epoch
            running_val_loss = 0.0
            if x_val is not None:
                x_test, y_test = self._preprocessor(x_val, y=y_val, training=False)
                self.Model.eval()
                with torch.no_grad():
                    predictions = self.Model(x_test)
                loss_value = criterion(predictions, y_test)
                losses_val.append(loss_value.item())
                running_val_loss = loss_value.item()
                print('epoch: ', epoch, 'test/validation loss: ', running_val_loss)
                self.Model.train()

            print('epoch: ', epoch, 'normalized loss: ', running_loss/dataset_len)

            losses.append(running_loss/dataset_len)

        return self, losses, losses_val, max_epochs

       
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        # Preprocess Data
        x, _ = self._preprocessor(x, training=False)

        # Ensure Model in Evaluation Mode
        self.Model.eval()

        # Forward Pass to get Predictions
        with torch.no_grad():
            predictions = self.Model(x)

        # Convert Torch Tensor into NumPy Array & Renormalise
        y_hat = (predictions.numpy()*self.y_std.values)+self.y_mean.values
 
        return y_hat          
        

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        # preprocess data
        x, y = self._preprocessor(x, y=y, training=False)

        # Ensure the model is in evaluation mode
        self.Model.eval()

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.Model(x)
        
        # Compute the mean squared error
        criterion = nn.MSELoss()
        loss_value = criterion(predictions, y)
        MSE_score = loss_value.item()

        # Implemented to calculate unnormalized RMSE (same as LabTS)
        # loss_unnnormalized_MSE = criterion(predictions*self.y_std.values+self.y_mean.values, y*self.y_std.values+self.y_mean.values)
        # RMSE = np.sqrt(loss_unnnormalized_MSE.item())
        # print("RMSE is: ", RMSE)
        
        return MSE_score


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(data):
    """
    Performs a hyper-parameter search for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x_train {pd.DataFrame} -- Raw input array of shape 
            (batch_size, input_size).
        - y_train {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        
    Returns:
        {dict} -- Dictionary containing the best hyperparameters found during the search.

    """
    output_label = "median_house_value"

    # Split data into training and validation ( = 0.6, 0.2 of original data)
    data_train, data_val = train_test_split(data, test_size=0.25, random_state=2)

    # Splitting Input & Output
    x_train = data_train.loc[:, data.columns != output_label]
    y_train = data_train.loc[:, [output_label]]

    x_val = data_val.loc[:, data.columns != output_label]
    y_val = data_val.loc[:, [output_label]]

    # Define Hyperparameter Search Grid
    param_grid = {
        'max_epochs': [ 300, 500, 700, 900, 1100, 1300],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    }

    min_error = np.inf
    best_params = {}        # Initialize dictionary of Best Parameters
    plot_values = []        # Initialize list of tuples for plotting

    for max_epochs in param_grid['max_epochs']:
        for lr in param_grid['learning_rate']:
            print(lr, max_epochs, flush=True)
            regressor = Regressor(x_train, nb_epoch=max_epochs, learning_rate=lr)

            regressor.fit(x_train, y_train)

            error = regressor.score(x_val, y_val)

            plot_values.append((lr, max_epochs, error))

            if error < min_error:
                min_error = error
                best_params = {'max_epochs': max_epochs, 'learning_rate': lr}
                best_params_list = [max_epochs, lr]

                
    # Extract the values from the tuples
    _, __, error_values = zip(*plot_values)

    Z = np.array(error_values).reshape(len(param_grid['max_epochs']), len(param_grid['learning_rate']))
    X = np.array(param_grid['learning_rate'])
    Y = np.array(param_grid['max_epochs'])
    X_log = np.log10(X)     # use log scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contourf(X_log, Y, Z, 100, cmap='viridis')
    # Set labels and title
    ax.set_xlabel('Learning Rate')
    ax.set_xticks(X_log) 
    ax.set_xticklabels(X) 
    ax.set_ylabel('Max Epochs')
    ax.set_zlabel('Error')
    ax.set_title('Hyperparameter Search Results')

    # Summarize Results
    print("Lowest Error of Hyperparameter training: %f using %s" % (min_error, best_params))  

    # Return Best Hyperparameters
    return best_params_list


def plot_best_model(x_train, y_train, x_test, y_test, best_params):
    
    # Training with best hyperparameters while calculating the test loss
    regressor = Regressor(x_train, nb_epoch=best_params[0], learning_rate=best_params[1])
    _, losses, losses_test, max_epoch = regressor.fit(x_train, y_train, x_val=x_test, y_val=y_test)

    # Plot Training vs. Testing Loss using best hyperparameters
    fig, (ax) = plt.subplots(1)
    fig.suptitle('Training Loss vs. Testing Loss using best hyperparameters', fontweight='bold')
    ax.plot(range(max_epoch), losses, label = 'Training Loss')
    ax.plot(range(max_epoch), losses_test, label = 'Testing Loss')
    ax.set_xlabel('Epoch',fontweight='bold')
    ax.set_ylabel('Normalized MSE',fontweight='bold')
    ax.legend(loc="best")

    error = regressor.score(x_test, y_test)
    print("\nRegressor normalized MSE using test set: {}\n".format(error))

    #plt.savefig("Training_vs_Testing_Loss_best_params.png")


def example_main():

    try:
        output_label = "median_house_value"
        # Use pandas to read CSV data as it contains various object types
        data = pd.read_csv("housing.csv") 

        # Split data into training, validation and testing 0.6, 0.2, (0.2)
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=2)

        # Splitting input and output
        x_train = data_train.loc[:, data.columns != output_label]
        y_train = data_train.loc[:, [output_label]]
        x_test = data_test.loc[:, data.columns != output_label]
        y_test = data_test.loc[:, [output_label]]

        try:
            regressor = Regressor(x_train,nb_epoch = 2)
            regressor.fit(x_train,y_train)
        except Exception:
            print("exception")

        # Hyperparameter Search
        #best_params = RegressorHyperParameterSearch(data_train)

        best_params = [900, 0.0001] # found from the search above

        # Training with best hyperparameters
        regressor = Regressor(x_train, nb_epoch=best_params[0], learning_rate=best_params[1])
        regressor.fit(x_train, y_train)

        # Error
        error = regressor.score(x_test, y_test)
                
        # print all the values
        print(best_params)
        print("\nRegressor normalized MSE using test set: {}\n".format(error))

        # Save the best model
        save_regressor(regressor)

        # Plot Training vs. Testing Loss using best hyperparameters
        #plot_best_model(x_train, y_train, x_test, y_test, best_params)


    except Exception as e:
        print("An error occurred: ", str(e))
        import traceback
        traceback.print_exc()

    plt.show()

if __name__ == "__main__":
    example_main()