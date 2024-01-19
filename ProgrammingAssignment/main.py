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
    for network in tqdm(data_sets):
        net = network
        network = data_sets[net][0]
        running_times_network_it = data_sets[net][1]
        running_times_nn = {}
        for dataset in tqdm(network):
            measure = dataset
            dataset = network[measure]

            opt_config_dict = {"min_test_error": 100000, "best_config": None, "min_time": 100000, "final_out": None,
                               "final_train_error": None}

            for config in tqdm(configs):

                gin = GIN(*configs[config])

                data, train_mask = dataset

                # train the model
                gin.train()
                optimizer = torch.optim.Adam(gin.parameters(), lr=0.001)
                criterion = torch.nn.L1Loss()

                num_epochs = 500
                for epoch in range(num_epochs + 1):
                    optimizer.zero_grad()

                    out = gin.forward(x=data.x, edge_index=data.edge_index)

                    loss = criterion(out.view(-1)[train_mask], data.y[train_mask].view(-1))

                    loss.backward()
                    optimizer.step()

                gin.eval()
                out = gin.forward(x=data.x, edge_index=data.edge_index)

                train_error = criterion(out.view(-1)[train_mask], data.y[train_mask].view(-1))
                start = time.perf_counter()
                test_error = criterion(out.view(-1)[~train_mask], data.y.view(-1)[~train_mask])
                end = time.perf_counter()

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
                               'Kendalls Tau': compute_kendall_tau(dataset[0], opt_config_dict["final_out"]),
                               'Best Config': opt_config_dict["best_config"],
                               'Running Time Networkit': running_times_network_it[measure],
                               'Running Time NN': running_times_nn[measure],
                               'Time Difference': running_times_network_it[measure] - running_times_nn[measure]}

    df.to_csv('results.csv', index=False)
