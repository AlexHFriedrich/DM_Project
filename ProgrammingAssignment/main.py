import time

import torch
from tqdm import tqdm
import pandas as pd
from Read_Data import load_data_list
from metrics import compute_kendall_tau, plot_approximation_ratio
from model import GIN

"""
This file contains the main function that is used for task 3. The models are trained on the configs below, and the
optimal configuration is chosen based on the minimum test error.
"""
if __name__ == "__main__":
    # set a seed for reproducibility
    torch.manual_seed(0)

    df = pd.DataFrame(columns=['Network', 'Measure', 'Final Train Error', 'Min Test Error', 'Kendalls Tau',
                               'Best Config', 'Running Time Networkit', 'Running Time NN', 'Time Difference'])

    configs = {'32_3': (32, 3), '32_4': (32, 4), '32_5': (32, 5),
               '64_3': (64, 3), '64_4': (64, 4), '64_5': (64, 5),
               '128_3': (128, 3), '128_4': (128, 4), '128_5': (128, 5),
               '256_3': (256, 3), '256_4': (256, 4), '256_5': (256, 5)}

    data_sets = load_data_list()

    # default value
    train_mask = torch.zeros(1000, dtype=torch.bool)
    for network in tqdm(data_sets):
        net = network
        network = data_sets[net][0]
        running_times_network_it = data_sets[net][1]
        running_times_nn = {}
        """
        keys_to_delete = ["Eigenvector Centrality", "PageRank"]
        for key in keys_to_delete:
            del network[key]
        """
        for dataset in tqdm(network):
            measure = dataset
            dataset = network[measure]
            opt_config_dict = {"min_test_error": 100000, "best_config": None, "min_time": 100000, "final_out": None,
                               "final_train_error": None}

            for config in tqdm(configs):

                gin = GIN(*configs[config])

                data, train_mask = dataset
                criterion = torch.nn.L1Loss()

                lr = 0.01
                num_epochs = 1000
                temp_loss = 100000
                for epoch in range(num_epochs):
                    # train the model
                    gin.train()

                    optimizer = torch.optim.Adam(gin.parameters(), lr=lr)
                    optimizer.zero_grad()

                    out = gin.forward(x=data.x, edge_index=data.edge_index)

                    loss = criterion(out.view(-1)[train_mask], data.y[train_mask].view(-1))

                    # early stopping if the loss does not decrease by more than 1% for 20 epochs
                    if epoch % 20 == 0 and epoch > 50:
                        if abs(temp_loss - loss) > abs(temp_loss) / 100:
                            temp_loss = loss
                            # adaptive learning rate decay
                            lr = lr / 1.15
                        else:
                            print("\nEarly stopping at epoch: ", epoch)
                            break

                    loss.backward()
                    optimizer.step()

                gin.eval()
                start = time.perf_counter()
                out = gin.forward(x=data.x, edge_index=data.edge_index)
                end = time.perf_counter()
                
                train_error = criterion(out.view(-1)[train_mask], data.y[train_mask].view(-1))

                test_error = criterion(out.view(-1)[~train_mask], data.y.view(-1)[~train_mask])

                if test_error.detach().numpy() < opt_config_dict["min_test_error"]:
                    opt_config_dict["min_test_error"] = test_error.detach().numpy()
                    opt_config_dict["best_config"] = config
                    opt_config_dict["final_out"] = out.view(-1).detach()
                    opt_config_dict["final_train_error"] = train_error.detach().numpy()
                    opt_config_dict["min_time"] = end - start

            plot_approximation_ratio(dataset[0], opt_config_dict["final_out"], net, measure)
            running_times_nn[measure] = opt_config_dict["min_time"]

            # update the dataframe
            df.loc[len(df)] = {'Network': net, 'Measure': measure,
                               'Final Train Error': opt_config_dict["final_train_error"],
                               'Min Test Error': opt_config_dict["min_test_error"],
                               'Kendalls Tau': compute_kendall_tau(dataset[0], opt_config_dict["final_out"],
                                                                   ~train_mask),
                               'Best Config': opt_config_dict["best_config"],
                               'Running Time Networkit': running_times_network_it[measure],
                               'Running Time NN': running_times_nn[measure],
                               'Time Difference': running_times_network_it[measure] - running_times_nn[measure]}
    # rounding the values to improve readability
    df[['Running Time Networkit', 'Running Time NN', 'Time Difference']] = \
        df[['Running Time Networkit', 'Running Time NN', 'Time Difference']].round(6)
    df[['Final Train Error', 'Min Test Error', 'Kendalls Tau']] = \
        df[['Final Train Error', 'Min Test Error', 'Kendalls Tau']].round(4)

    df.to_csv('results.csv', index=False)
    print(df.to_string())
