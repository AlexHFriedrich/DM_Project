import time

import torch

from ProgrammingAssignment.Read_Data import load_data_list
from ProgrammingAssignment.metrics import compute_kendall_tau, plot_approximation_ratio
from ProgrammingAssignment.model import GIN

configs = {'32_3': (32, 3), '32_4': (32, 4), '32_5': (32, 5),
           '64_3': (64, 3), '64_4': (64, 4), '64_5': (64, 5),
           '128_3': (128, 3), '128_4': (128, 4), '128_5': (128, 5),
           '256_3': (256, 3), '256_4': (256, 4), '256_5': (256, 5)}


data_sets = load_data_list()
for network in data_sets:
    net = network
    network = data_sets[net][0]
    running_times_network_it = data_sets[net][1]
    running_times_nn = {}
    for dataset in network:
        measure = dataset
        dataset = network[measure]
        min_test_error = 100000
        best_config = None
        min_time = 100000
        final_out = None
        for config in configs:

            gin = GIN(*configs[config])

            data, train_mask = dataset
            # train the model
            gin.train()
            optimizer = torch.optim.Adam(gin.parameters(), lr=0.001)
            criterion = torch.nn.L1Loss()

            num_epochs = 100
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
            test_error = criterion(out.view(-1), data.y.view(-1))
            end = time.perf_counter()

            if test_error < min_test_error:
                final_train_error = train_error
                min_test_error = test_error
                best_config = config
                min_time = end - start
                final_out = out.view(-1)

            running_times_nn[measure] = min_time

        # print network and measure name
        print(f'Network: {net}')
        print(f'Measure: {measure}')
        print(f'Best config: {best_config}')
        print(f'Min test error: {min_test_error}')
        plot_approximation_ratio(dataset[0], final_out, measure)
        compute_kendall_tau(dataset[0], final_out)

    # print the running times of networkit against those of the nn model for each measure
    for measure in running_times_network_it:
        print(f'Network: {net}')
        print(f'Measure: {measure}')
        print(f'Running time of networkit: {running_times_network_it[measure]}')
        print(f'Running time of nn: {running_times_nn[measure]}')
        # print the time difference between networkit and the nn model for each measure
        print(f'Time difference: {running_times_network_it[measure] - running_times_nn[measure]}')
