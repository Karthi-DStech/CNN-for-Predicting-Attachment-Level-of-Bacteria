import os
import sys


from options.train_options import TrainOptions
from model.cnn import CNN
from utils import visualiser
import time
from data.topographies import BiologicalObservation
import torch
from call_methods import make_network, make_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run() -> None:

    opt = TrainOptions().parse()
    train_dataloader, test_dataloader = make_dataset(opt)
    model = make_network(opt.network_name, opt)
    print(len(train_dataloader))
    print(len(test_dataloader))
    # Variable to store best loss for saving best model
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    # Object to save model performance and configuration
    visualise = visualiser.Visualizer(opt)
    # Frequency to print epochs
    print_freq = 1

    
    for epoch in range(opt.n_epochs):
        # Save loss
        train_running_loss = 0.0
        test_running_loss = 0.0
        # Train the model
        for data in train_dataloader:
            model.set_input(data)
            train_running_loss += model.train_step()
        train_loss = train_running_loss / len(train_dataloader)
        visualise.log_performance(
            loss=train_loss, 
            epoch=epoch,
            is_train=True,
            print_freq=print_freq
            )
        # Test the model
        with torch.no_grad():
            for data in test_dataloader:
                model.set_input(data)
                test_running_loss += model.test_step()
            test_loss = test_running_loss / len(test_dataloader)
        visualise.log_performance(
            loss=test_loss, 
            epoch=epoch,
            is_train=False,
            print_freq=print_freq
            )
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
            model.save_model()
    
    if opt.is_eval:    
        # Load best model
        model.load_model()

        # Save best model performance
        # Compute other performance evaluation metrics and save them
        # Training data
        eval_metrics = {}
        for data in train_dataloader:
            model.set_input(data)
            # capture all the evaluation metrics using a dictionary
            eval_step = model.eval_step()
            for key in eval_step:
                if key in eval_metrics:
                    eval_metrics[key] += eval_step[key]
                else:
                    eval_metrics[key] = eval_step[key]
        for key in eval_metrics:
            eval_metrics[key] /= len(train_dataloader)
        visualise.log_best_performance(
                loss=best_train_loss, 
                epoch=epoch,
                metrics = eval_metrics,
                is_train=True
                )
        # Test data
        eval_metrics = {}
        for data in test_dataloader:
            model.set_input(data)
            # capture all the evaluation metrics using a dictionary
            eval_step = model.eval_step()
            for key in eval_step:
                if key in eval_metrics:
                    eval_metrics[key] += eval_step[key]
                else:
                    eval_metrics[key] = eval_step[key]
        for key in eval_metrics:
            eval_metrics[key] /= len(test_dataloader)
        visualise.log_best_performance(
                loss=best_test_loss, 
                epoch=epoch,
                metrics = eval_metrics,
                is_train=False
                )     
        




if __name__ == "__main__":
    run()