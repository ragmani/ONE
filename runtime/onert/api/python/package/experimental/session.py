import numpy as np

from ..native import libnnfw_api_pybind
from ..common.basesession import BaseSession


class ExperiemntalSession(BaseSession):
    """
    Class for training and inference using nnfw_session.
    """

    def __init__(self, nnpackage_path, backends="train"):
        """
        Initialize the experimental session.

        Args:
            nnpackage_path (str): Path to the nnpackage file or directory.
            backends (str): Backends to use, default is "train".
        """
        super().__init__(libnnfw_experimental.nnfw_session(nnpackage_path, backends))

    # TODO: Support optimizer feature
    def train(self,
              data_loader,
              batch_size,
              epochs,
              validation_split=0.0,
              checkpoint_path=None):
        """
        Train the model using the given data loader.

        Args:
            data_loader: A data loader providing input and expected data.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs to train.
            validation_split (float): Ratio of validation data. Default is 0.0 (no validation).
            checkpoint_path (str): Path to save or load the training checkpoint.
        """
        # Set training information
        train_info = self.session.train_get_traininfo()
        train_info.batch_size = batch_size
        self.session.train_set_traininfo(train_info)

        # Import checkpoint if provided
        if checkpoint_path:
            self.session.train_import_checkpoint(checkpoint_path)

        # Prepare session for training
        self.session.train_prepare()

        # Split data into training and validation
        train_data, val_data = data_loader.split(validation_split)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            train_loss = self._run_phase(train_data, train=True)
            print(f"Train Loss: {train_loss:.4f}")

            # Validation phase
            if validation_split > 0.0:
                val_loss = self._run_phase(val_data, train=False)
                print(f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if checkpoint_path:
                self.session.train_export_checkpoint(checkpoint_path)

    def _run_phase(self, data, train=True):
        """
        Run a training or validation phase.

        Args:
            data: Data generator providing input and expected data.
            train (bool): Whether to perform training or validation.

        Returns:
            float: Average loss for the phase.
        """
        total_loss = 0.0
        num_batches = 0

        for inputs, expecteds in data:
            # Set inputs
            for i, input_data in enumerate(inputs):
                self.session.train_set_input(
                    i, np.array(input_data, dtype=self.session.input_tensorinfo(i).dtype))

            # Set expected outputs
            for i, expected_data in enumerate(expecteds):
                self.session.train_set_expected(
                    i,
                    np.array(
                        expected_data, dtype=self.session.output_tensorinfo(i).dtype))

            # Run training or validation
            self.session.train(update_weights=train)

            # Accumulate loss
            batch_loss = sum(
                self.session.train_get_loss(i) for i in range(len(expecteds)))
            total_loss += batch_loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
