# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55

"""
Module defining the CommonServer class for federated learning server-side operations.

This module provides the CommonServer class that orchestrates federated learning
across multiple clients for electric vehicle charging demand prediction. It manages
client coordination, model aggregation, and global training procedures.

Key Features:
    - Client management for training and evaluation
    - Model parameter aggregation using FedAvg
    - Global training coordination across multiple rounds
    - Model localization support
    - Evaluation orchestration across clients
"""

# Standard library imports
from typing import List, Any, Generator

# Third-party imports
import torch
from torch.autograd import Variable


class CommonServer(object):
    """
    Server handler for federated learning aggregation and evaluation.

    This class provides comprehensive functionality for orchestrating federated
    learning across multiple clients. It manages training and evaluation clients,
    aggregates local model updates using specified strategies, and coordinates
    global training and localization procedures.

    The server implements the central coordination logic for federated learning,
    including parameter aggregation, client synchronization, and evaluation
    management across distributed clients.

    Attributes:
        train_clients (list): List of client instances for training.
        eval_clients (list): List of client instances for evaluation.
        aggregation (str): Aggregation strategy; e.g., 'fedavg'.
        model (object): Global model instance to be updated.
    """

    def __init__(
            self,
            train_clients: List[Any],
            eval_clients: List[Any],
            model: Any,
            aggregation: str = 'fedavg',
    ) -> None:
        """
        Initialize CommonServer with clients and aggregation settings.

        This method sets up the federated learning server with training and
        evaluation clients, a global model, and the aggregation strategy.
        The server will coordinate federated learning across the specified
        clients using the given aggregation method.

        Args:
            train_clients (list): List of client instances participating in training.
            eval_clients (list): List of client instances reserved for evaluation.
            model (object): Global model instance to aggregate parameters into.
            aggregation (str): Aggregation method to use; default is 'fedavg'.
        """
        super(CommonServer, self).__init__()
        self.train_clients = train_clients  # Training client instances
        self.eval_clients = eval_clients    # Evaluation client instances
        self.aggregation = aggregation      # Aggregation strategy
        self.model = model                  # Global model instance

    def train(self, global_epochs: int, local_epochs: int) -> None:
        """
        Conduct federated training over multiple global rounds.

        This method implements the core federated learning training loop.
        Each global round consists of: (1) distributing the global model to
        training clients, (2) performing local training on each client,
        (3) aggregating local model updates, and (4) evaluating the updated
        global model on evaluation clients.

        Args:
            global_epochs (int): Number of global federated rounds to execute.
            local_epochs (int): Number of local epochs to train per client per round.
        """
        # Iterate through global federated rounds
        for global_epoch in range(1, global_epochs + 1):
            local_models = []  # Store local model parameters from all training clients
            
            # Collect local model parameters from each training client
            for train_c in self.train_clients:
                train_c.refresh(self.model)  # Update client with current global model
                train_c.train(now_epoch=global_epoch, local_epochs=local_epochs, save_model=False)  # Local training
                local_models.append(train_c.get_model())  # Collect local model parameters

            # Aggregate local models into the global model
            self.aggregate(local_models)  # Perform parameter aggregation

            # Evaluate updated global model on evaluation clients
            for eval_c in self.eval_clients:
                eval_c.refresh(self.model)  # Update evaluation client with new global model
                eval_c.test(now_epoch=global_epoch)  # Perform evaluation

    def localize(self, now_epoch: int, deploy_epochs: int) -> None:
        """
        Perform model localization by iterative fine-tuning on eval_clients.

        This method implements model localization by fine-tuning the global
        model on evaluation clients through iterative training and testing.
        This is useful for adapting the global model to specific local
        conditions or for deployment preparation.

        Args:
            now_epoch (int): Current federated round or global epoch number.
            deploy_epochs (int): Number of local deployment epochs to run on each client.
        """
        # Perform localization on each evaluation client
        for eval_c in self.eval_clients:
            eval_c.refresh(self.model)  # Update client with current global model
            eval_c.localize(now_epoch=now_epoch, deploy_epochs=deploy_epochs)  # Execute localization

    def aggregate(self, local_models: List[Generator[torch.Tensor, None, None]]) -> None:
        """
        Aggregate multiple local model parameter lists into the global model.

        This method implements Federated Averaging (FedAvg) by computing the
        weighted average of local model parameters across all clients. It
        handles parameter initialization, None value handling, and proper
        averaging to update the global model.

        Args:
            local_models (list): List of parameter generators from training clients.
        """
        num_clients = len(local_models)  # Number of participating clients
        
        # Initialize global parameters to zero before averaging
        for idx, local_m in enumerate(local_models):
            for w_global, w_local in zip(self.model.model.parameters(), local_m):
                # Zero-initialize global parameters on first client iteration
                if idx == 0:
                    zero_tensor = Variable(torch.zeros_like(w_global))  # Create zero tensor
                    w_global.data.copy_(zero_tensor.data)  # Initialize to zero
                
                # Handle None local weights by zeroing them
                if w_local is None:
                    w_local = Variable(torch.zeros_like(w_global))  # Replace None with zeros
                
                # Accumulate weighted average (FedAvg)
                w_global.data.add_(w_local.data / num_clients)  # Add weighted contribution
