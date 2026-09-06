# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55

"""
Module defining the CommonClient class for federated learning client-side operations.

This module provides the CommonClient class that handles local training and evaluation
in federated learning scenarios for electric vehicle charging demand prediction.
It manages client-specific datasets, models, and training processes while supporting
parameter synchronization with the global model.

Key Features:
    - Local dataset management and DataLoader creation
    - Client-specific model initialization and training
    - Parameter synchronization with global model
    - Local evaluation and model localization
    - Support for auxiliary features and custom architectures
"""

# Standard library imports
from typing import Dict, Any, Callable, Optional, Generator

# Third-party imports
import torch
from torch.utils.data import DataLoader, Subset

# Local imports
from api.utils import CreateDataset, ASTGCNDataset


class CommonClient(object):
    """
    Client handler for federated learning on electric vehicle datasets.

    This class provides comprehensive functionality for client-side operations
    in federated learning scenarios. It manages local datasets, models, and
    training processes while supporting parameter synchronization with the
    global model maintained by the federated server.

    The client can perform local training, testing, model localization, and
    parameter updates. It supports both training and evaluation modes with
    configurable data splits and auxiliary features.

    Attributes:
        client_id (str): Unique identifier for the client.
        scaler (object): Scaler used to inverse-transform predictions.
        feat (np.ndarray): Feature array for this client [time, nodes].
        extra_feat (np.ndarray or None): Auxiliary features array.
        trainer (object): Trainer instance handling training and testing logic.
    """

    def __init__(
            self,
            client_id: str,
            data_dict: Dict[str, Any],
            scaler: Any,
            model_module: Callable[..., Any],
            trainer_module: Callable[..., Any],
            seq_l: int,
            pre_len: int,
            model_name: str,
            n_fea: int,
            batch_size: int,
            device: torch.device,
            save_path: str,
            support_rate: float = 1.0,
    ) -> None:
        """
        Initialize CommonClient by constructing dataset subsets, model, and trainer.

        This method sets up the complete client environment including data
        preprocessing, model initialization, dataset splitting, and trainer
        configuration. It creates support and test datasets based on the
        support_rate parameter for local training and evaluation.

        Args:
            client_id (str): Unique identifier for the client.
            data_dict (dict): Dictionary containing 'feat', 'extra_feat', and 'time'.
            scaler (object): Scaler fitted on training data for inverse transformation.
            model_module (callable): Class or factory function for model instantiation.
            trainer_module (callable): Class or factory function for trainer instantiation.
            seq_l (int): Input sequence length for time series prediction.
            pre_len (int): Prediction horizon length.
            model_name (str): Name of the model architecture to use.
            n_fea (int): Number of auxiliary feature channels.
            batch_size (int): Batch size for DataLoaders.
            device (torch.device): Torch device for computations (CPU/GPU).
            save_path (str): Path to save trained model parameters.
            support_rate (float): Fraction of local data to use for training (default: 1.0).
        """
        super(CommonClient, self).__init__()

        # Store client properties
        self.client_id = client_id
        self.scaler = scaler
        self.feat = data_dict['feat']  # Main feature data
        self.extra_feat = data_dict['extra_feat']  # Auxiliary features
        effective_seq_l = 48 if model_name == 'astgcn' else seq_l

        # Initialize model with client-specific parameters
        model = model_module(
            num_node=1,  # Single node for client (site or city)
            n_fea=n_fea,  # Number of auxiliary features
            model_name=model_name,  # Model architecture name
            seq_l=effective_seq_l,  # Input sequence length
            pre_len=pre_len,  # Prediction horizon
        )

        # Create full dataset and split into support and test sets
        if model_name == 'astgcn':
            dataset = ASTGCNDataset(
                self.feat,
                0,
                len(self.feat),
                pre_len,
                device,
            )
        else:
            dataset = CreateDataset(seq_l, pre_len, self.feat, self.extra_feat, device)  # Create complete dataset
        total_samples = len(dataset)  # Total number of samples
        support_count = int(total_samples * support_rate)  # Number of samples for training
        
        # Create index lists for dataset splitting
        train_indices = list(range(support_count))  # Training indices
        test_indices = list(range(support_count, total_samples))  # Test indices

        # Create dataset subsets for training and testing
        train_dataset = Subset(dataset, train_indices)  # Training subset
        test_dataset = Subset(dataset, test_indices)    # Testing subset

        # Create DataLoaders for local training and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Training loader with shuffling
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Test loader without shuffling

        # Determine if auxiliary features are present
        if self.extra_feat is None:
            extra_feat_tag = False  # No auxiliary features
        else:
            extra_feat_tag = True   # Auxiliary features present

        # Initialize trainer with data loaders and model
        self.trainer = trainer_module(
            train_loader=train_loader,      # Training data loader
            test_loader=test_loader,        # Test data loader
            extra_feat_tag=extra_feat_tag,  # Auxiliary features flag
            model=model,                    # Local model instance
            save_path=save_path,            # Model save path
            scaler=scaler,                  # Data scaler
            device=device,                  # Computation device
        )

    def train(
            self,
            now_epoch: int,
            local_epochs: int,
            save_model: bool = False,
    ) -> None:
        """
        Perform local training for a specified number of epochs.

        This method executes local training on the client's data for the
        specified number of epochs. It updates the trainer's epoch tracking
        and optionally saves the model weights after training.

        Args:
            now_epoch (int): Current federated round or global epoch number.
            local_epochs (int): Number of local epochs to train.
            save_model (bool): Whether to save model weights after training (default: False).
        """
        self.trainer.now_epoch = now_epoch      # Set current global epoch
        self.trainer.deploy_epoch = 1           # Set deployment epoch to 1
        self.trainer.training(epoch=local_epochs, save_model=save_model)  # Execute training

    def test(
            self,
            now_epoch: int,
            model_path: Optional[str] = None,
    ) -> None:
        """
        Evaluate the local model on the client's test dataset.

        This method performs evaluation of the local model on the client's
        test data. It can use either the current model state or load a
        specific model from a file path.

        Args:
            now_epoch (int): Current federated or local epoch number.
            model_path (str, optional): Path to a saved model for testing (default: None).
        """
        self.trainer.now_epoch = now_epoch  # Set current epoch
        self.trainer.test(model_path=model_path)  # Execute testing

    def localize(
            self,
            now_epoch: int,
            deploy_epochs: int,
            save_model: bool = False,
            model_path: Optional[str] = None,
    ) -> None:
        """
        Perform iterative local training and testing for model localization.

        This method implements a localization strategy where the model is
        trained for one epoch and then tested, repeating this process for
        the specified number of deployment epochs. This is useful for
        fine-tuning the model on local data.

        Args:
            now_epoch (int): Current federated round or global epoch number.
            deploy_epochs (int): Number of deployment epochs to run.
            save_model (bool): Whether to save model weights after each epoch (default: False).
            model_path (str, optional): Path to a saved model for testing (default: None).
        """
        self.trainer.now_epoch = now_epoch  # Set current global epoch
        
        # Iterate through deployment epochs
        for deploy_epoch in range(1, deploy_epochs + 1):
            self.trainer.deploy_epoch = deploy_epoch  # Set current deployment epoch
            self.trainer.training(epoch=1, save_model=save_model)  # Train for one epoch
            self.trainer.test(model_path=model_path)  # Test the model

    def refresh(self, model: Any) -> None:
        """
        Update the local model parameters from a global model.

        This method synchronizes the local model parameters with those
        from a global model (typically from the federated server).
        It copies parameter values from the global model to the local model.

        Args:
            model (object): Global model instance with .model.parameters() method.
        """
        # Iterate through model parameters and copy from global to local
        for w_local, w_global in zip(
                self.trainer.ev_model.model.parameters(),  # Local model parameters
                model.model.parameters()                   # Global model parameters
        ):
            w_local.data.copy_(w_global.data)  # Copy parameter data

    def get_model(self) -> Generator[torch.Tensor, None, None]:
        """
        Retrieve the local model's current parameters.

        This method provides access to the current parameters of the local
        model. It returns a generator that yields torch.Tensor parameters,
        which can be used for parameter aggregation in federated learning.

        Returns:
            generator: Iterable of torch.Tensor parameters from the local model.
        """
        return self.trainer.ev_model.model.parameters()  # Return model parameters
