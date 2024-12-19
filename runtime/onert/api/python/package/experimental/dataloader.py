import numpy as np


class DataLoader:
    """
    A simple DataLoader to manage training and validation data.
    """

    def __init__(self, input_data_path, expected_data_path):
        """
        Initialize the DataLoader with input and expected data.

        Args:
            input_data_path (str): Path to the file containing input data (e.g., .npy).
            expected_data_path (str): Path to the file containing expected output data (e.g., .npy).
        """
        # Load input and expected data from files
        self.inputs = np.load(input_data_path)
        self.expecteds = np.load(expected_data_path)

        # Verify data consistency
        if len(self.inputs) != len(self.expecteds):
            raise ValueError("Input data and expected data must have the same length.")

    def split(self, validation_split):
        """
        Split the data into training and validation sets.

        Args:
            validation_split (float): Ratio of validation data. Must be between 0.0 and 1.0.

        Returns:
            tuple: Two DataLoader instances, one for training and one for validation.
        """
        if not (0.0 <= validation_split <= 1.0):
            raise ValueError("Validation split must be between 0.0 and 1.0.")

        split_index = int(len(self.inputs) * (1.0 - validation_split))
        train_inputs = self.inputs[:split_index]
        train_expecteds = self.expecteds[:split_index]
        val_inputs = self.inputs[split_index:]
        val_expecteds = self.expecteds[split_index:]

        train_loader = DataLoader.from_data(train_inputs, train_expecteds)
        val_loader = DataLoader.from_data(val_inputs, val_expecteds)
        return train_loader, val_loader

    def generate_batches(self, batch_size):
        """
        Generate batches of data.

        Args:
            batch_size (int): Number of samples per batch.

        Yields:
            tuple: A batch of inputs and expected outputs.
        """
        for i in range(0, len(self.inputs), batch_size):
            yield (
                self.inputs[i:i + batch_size],  # Batch of input data
                self.expecteds[i:i + batch_size],  # Batch of expected output data
            )

    @classmethod
    def from_data(cls, inputs, expecteds):
        """
        Create a DataLoader instance from raw data arrays.

        Args:
            inputs (np.ndarray): Input data array.
            expecteds (np.ndarray): Expected output data array.

        Returns:
            DataLoader: A new DataLoader instance.
        """
        loader = cls.__new__(cls)  # Bypass __init__
        loader.inputs = inputs
        loader.expecteds = expecteds
        return loader
