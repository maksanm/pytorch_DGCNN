from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
import numpy as np

class ResultParser:
    def __init__(self, training_results: str):
        self.dataset, self.batch_size, self.results =  self._load_from_txt(training_results)

    def _load_from_txt(self, training_results: str):
        '''
        Parses txt output of run_DGCNN.sh cript to extract training informations.
        Returns dataset name and list of train and test accuracies for every fold and iteration.
        '''
        dataset = None
        bsz = None
        data = {}

        # Initialize variables to store the current fold and accuracy lists
        current_fold = None
        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses=[]

        # Read the file and process it line by line
        with open(training_results, 'r') as file:
            for line in file:
                
                # Get dataset name
                if dataset == None:
                    dataset_match = re.search(r"data='([^']*)'", line)
                    if dataset_match:
                        dataset = dataset_match.group(1)

                # Get batch size
                if bsz == None:
                    bsz_match = re.search(r"batch_size=(\d+)", line)
                    if bsz_match:
                        bsz = bsz_match.group(1)

                # Check for the Namespace line to get the fold number
                fold_match = re.search(r'Namespace\(.*fold=(\d+),.*\)', line)
                if fold_match:
                    # If we have already collected accuracies for a previous fold, save them
                    if current_fold is not None:
                        data[current_fold] = {
                            'train_accuracy': train_accuracies,
                            'test_accuracy': test_accuracies,
                            'train_loss': train_losses,
                            'test_loss': test_losses
                        }
                    # Reset for the new fold
                    current_fold = int(fold_match.group(1))
                    train_accuracies = []
                    test_accuracies = []
                    train_losses = []
                    test_losses = []
                
                # Use regex to find the train accuracy in lines starting with 'average training of epoch'
                train_match_acc = re.search(r'average training of epoch \d+:.*acc ([\d\.]+)', line)
                if train_match_acc:
                    train_accuracies.append(float(train_match_acc.group(1)))
                
                # train loss
                train_match_loss = re.search(r'average training of epoch \d+:.*loss ([\d\.]+)', line)
                if train_match_loss:
                    train_losses.append(float(train_match_loss.group(1)))
                
                # Use regex to find the test accuracy in lines starting with 'average test of epoch'
                test_match_acc = re.search(r'average test of epoch \d+:.*acc ([\d\.]+)', line)
                if test_match_acc:
                    test_accuracies.append(float(test_match_acc.group(1)))

                # test loss
                test_match_loss = re.search(r'average test of epoch \d+:.*loss ([\d\.]+)', line)
                if test_match_loss:
                    test_losses.append(float(test_match_loss.group(1)))

                # Save the last fold data
                if current_fold is not None:
                    data[current_fold] = {
                        'train_accuracy': train_accuracies,
                        'test_accuracy': test_accuracies,
                        'train_loss': train_losses,
                        'test_loss': test_losses
                    }

        return dataset, bsz, data
    
    def last_epoch_acc(self, split: str):
        '''
        split can be 'train' or 'test'
        Returns a list of last epoch accuracy for every fold
        '''
        result = []
        for _, results in self.results.items():
            result.append(results[f"{split}_accuracy"][-1])
        
        return result
    
    def best_epoch_acc(self, split: str):
        '''
        Returns a list of best epoch accuracy for every fold
        '''
        result = []
        for _, results in self.results.items():
            result.append(max(results[f"{split}_accuracy"]))
        
        return result
    
    def avg_last_epoch_acc(self, split: str):
        '''
        Avg accuracy from last epoch of every fold
        '''
        last_epoch_acc = self.last_epoch_acc(split)
        return np.mean(last_epoch_acc)
    
    def avg_best_epoch_acc(self, split: str):
        '''
        Avg accuracy from best epoch of every fold
        '''
        best_epoch_acc = self.best_epoch_acc(split)
        return np.mean(best_epoch_acc)
    
    def get_accuracy_loss_arrays(self, split: str, metric: str):
        '''
        metric can be 'accuracy' or 'loss'
        Returns {number of folds} pairs of arrays x and y for plotting, containing accuracy or loss.
        '''
        x = []
        y = []

        # Extracting x and y values
        for _, results in self.results.items():
            x.append(range(len(results[f"{split}_{metric}"])))
            y.append(results[f"{split}_{metric}"])

        return x, y
    
    def plot_accuracy_loss(self, split: str, metric: str):
        '''
        metric can be 'accuracy' or 'loss'
        Plots training accuracies for every fold
        '''
        
        x, y = self.get_accuracy_loss_arrays(split, metric)

        # Plotting all folds on the same plot with different colors and legend
        plt.figure(figsize=(10, 6))
        for i in range(len(x)):
            plt.plot(x[i], y[i], label=f"Fold {i+1}")

        # Calculating and plotting average accuracy
        avg_accuracy = [np.mean([y_fold[i] for y_fold in y if len(y_fold) > i]) for i in range(len(y[0]))]

        plt.plot(range(len(avg_accuracy)), avg_accuracy, label="Average", linestyle='--', color='black')

        # Adding titles, labels, and legend
        plt.title(f"Accuracy per Fold and Average Accuracy for {split} set")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plt.show()

    def plot_train_test_acc_loss(self, metric: str):
        '''
        Plots training accuracies for every fold
        '''
        
        _, y = self.get_accuracy_loss_arrays("train", metric)
        avg_accuracy_train = [np.mean([y_fold[i] for y_fold in y if len(y_fold) > i]) for i in range(len(y[0]))]
        _, y = self.get_accuracy_loss_arrays("test", metric)
        avg_accuracy_test = [np.mean([y_fold[i] for y_fold in y if len(y_fold) > i]) for i in range(len(y[0]))]

        plt.plot(range(len(avg_accuracy_train)), avg_accuracy_train, label="train", linestyle='--', color='blue')
        plt.plot(range(len(avg_accuracy_test)), avg_accuracy_test, label="test", linestyle='--', color='green')

        plt.title(f"Comparison of average {metric} for train and test set")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()



        plt.show()
        
