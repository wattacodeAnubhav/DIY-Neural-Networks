import pandas as pd
import numpy as np

class DataPipeline:
    """
    Handles data ingestion, cleaning, and formatting.
    Prepares standard CSV data for the GlassBoxNetwork engine.
    """
    def __init__(self, filepath=None, dataframe=None):
        """Allows initializing with a filepath OR an existing Pandas DataFrame."""
        self.filepath = filepath
        self.df = dataframe
        
    def load_data(self):
        """Loads data from a CSV file into a Pandas DataFrame."""
        if self.filepath:
            self.df = pd.read_csv(self.filepath)
        return self.df
        
    def clean_data(self):
        """
        Cleans the dataset. 
        Instead of blindly imputing missing values with averages, we drop rows 
        with missing critical features (like square footage in housing data), 
        as predictions without those core metrics are logically flawed.
        """
        initial_rows = len(self.df)
        self.df = self.df.dropna() 
        final_rows = len(self.df)
        
        # Print a log so the student can see what happened
        print(f"Data Cleaned: Dropped {initial_rows - final_rows} rows with missing values.")
        return self.df
        
    def prepare_matrices(self, target_column):
        """
        Separates features (X) and targets (Y) and transposes them.
        Our NumPy neural network strictly expects the shape: (features, examples).
        """
        # Extract target (Y) and features (X)
        Y_raw = self.df[target_column].values
        X_raw = self.df.drop(columns=[target_column]).values
        
        # Transpose X to be (features, examples)
        X = X_raw.T 
        
        # Reshape Y to be (1, examples) instead of a flat array
        Y = Y_raw.reshape(1, -1)
        
        return X, Y
        
    def train_test_split(self, X, Y, test_ratio=0.2):
        """
        Splits the matrices into training and testing sets randomly.
        """
        m = X.shape[1] # Total number of examples
        
        # Create a random permutation of indices
        np.random.seed(42) # Fixed seed so students get reproducible results
        shuffled_indices = np.random.permutation(m)
        
        # Calculate where to slice the data
        test_size = int(m * test_ratio)
        
        # Slice the indices
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]
        
        # Split the actual data
        X_train = X[:, train_indices]
        Y_train = Y[:, train_indices]
        X_test = X[:, test_indices]
        Y_test = Y[:, test_indices]
        
        print(f"Data Split: {X_train.shape[1]} Training examples | {X_test.shape[1]} Testing examples")
        return X_train, Y_train, X_test, Y_test

# CHALLENGE: Can you add a method called `scale_features` that normalizes the X data 
# so all values fall between 0 and 1? (Hint: X = (X - X.min) / (X.max - X.min))