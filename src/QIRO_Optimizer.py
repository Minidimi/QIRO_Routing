from src.util.test_helpers import *

"""
Provides functions to perform the full QIRO process for the TSP and CVRP
"""

def perform_qiro_cvrp(num_nodes, seeds, demands, df=None, exp_type= 'sa', reps=1, logging = False, instance_file='tsp_instances.csv', run_name='default', samples=1000, debug=False, correlations_best=False):
    """
    Performs QIRO for CVRP instances with a problem-specific update step

    Parameters:
        num_nodes: Array of the number of nodes for which tests should be performed
        seeds: Random seeds for the optimization
        demands: Array of demands for the instances; Here, the same demands are used for the instances
        df: pd.DataFrame where the results are saved; If None, a new DataFrame is created
        exp_type: Quantum algorithm used for QIRO; Options: Simulated annealing 'sa', QAOA 'qaoa', VQE 'vqe' or exact solver 'numpy'
        reps: Number of layers for VQE or QAOA
        logging: Determines whether information should be displayed during runs
        instance_file: File where the instances are located; Instances should have a seed, an adjacency_matrix and a num_node column
        run_name: Name of the run under which the test data should be saved
        samples: Number of shots when measuring a state
        debug: If True, the data is not saved to a file
        correlations_best: Determines if only the best samples should be used to compute correlations instead of all samples
    """
    folder = "./"
    tsp_instances = pd.read_csv(os.path.join(folder, instance_file))
    tsp_instances['adjacency_matrix'] = tsp_instances['adjacency_matrix'].apply(lambda x: np.array(ast.literal_eval(x)))
    tsp_instances = tsp_instances.loc[tsp_instances['num_nodes']==num_nodes]

    if df is None:
        df = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])
    
    df_single = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])

    for seed in seeds:
        print('Starting test run with', num_nodes, 'nodes and seed', seed)
        for ind in tsp_instances.index:
            #prepare DataFrame
            instance = tsp_instances['seed'][ind]
            df.loc[len(df), 'n'] = num_nodes
            df.loc[len(df) - 1, 'instance'] = instance
            df.loc[len(df) - 1, 'seed'] = seed
            df_single.loc[len(df_single), 'n'] = num_nodes
            df_single.loc[len(df_single) - 1, 'instance'] = instance
            df_single.loc[len(df_single) - 1, 'seed'] = seed
            first_step = True

            betas = []
            gammas = []
            energies = []
            best_energies = []
            orders = []
            
            adj_matrix = tsp_instances['adjacency_matrix'][ind]
            G = nx.from_numpy_array(adj_matrix)
            np.random.seed(seed)
            capacity = 10

            K = int(np.ceil(sum(demands) / float(capacity)))

            ord = np.ones(K * num_nodes) * -1
            
            #Initialize the routes such that the depot is always visited first
            for i in range(K):
                ord[i * num_nodes] = 0

            problem = CVRP(G, K, demands, capacity=capacity, order=ord)
            if exp_type == 'vqe':
                expval = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'numpy':
                expval = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'sa':
                expval = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
            else:
                expval = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            
            qiro = QIRO_CVRP(nc=1, expectation_values=expval)

            while -1 in ord:
                #If only one node is left to be placed, the problem is trivially solvable
                if sum(ord == -1) <= 1:
                    break
                n = sum([k < 0 for k in ord])
                start_time = time.time()

                # optimize the correlations
                qiro.expectation_values.optimize()

                #Get the result from the quantum algorithms without QIRO
                if first_step:
                    result_single = np.fromiter(qiro.expectation_values.state, dtype=int)
                    energy_single = qiro.expectation_values.best_energy
                    if logging:
                        print('Result from one step quantum algorithm:', result_single, 'Energy', energy_single)
                    ord_single = np.ones(K * num_nodes) * -1
                    for i in range(K):
                        ord_single[i * num_nodes] = 0

                    ord_single_start = copy.deepcopy(ord_single)
                    
                    for k in range(K):
                        for i in range(1, num_nodes):
                            start_index = (i - 1) * num_nodes + get_starting_index(ord_single_start, num_nodes, k)
                            feasible = True
                            node = -1
                            for j in range(num_nodes):
                                if result_single[start_index + j] == 1:
                                    if node != -1:
                                        feasible = False
                                    node = j
                            if node != 0 and (node in ord_single or node == -1):
                                feasible = False
                            ord_single[i + k * num_nodes] = node
                    
                    #Check cost feasibility
                    for k in range(K):
                        cost = 0
                        for i in range(num_nodes):
                            ind = i + num_nodes * k
                            node = int(ord_single[ind])
                            cost += demands[node]
                        if cost > capacity:
                            feasible = False

                    if not feasible:
                        print('Infeasible solution was found by simulated annealing')
                        distance_single = -1
                    else:
                        distance_single = get_distance_cvrp(G, ord_single, K)
                    if logging:
                        print('Simulated Annealing in the first step found solution', ord_single, 'with distance', distance_single)

                    first_step = False

                end_time = time.time()
                if logging:
                    print('Total QAOA process took', end_time - start_time, 'seconds')

                # sorts correlations in decreasing order
                sorted_correlation_dict = sorted(
                    qiro.expectation_values.expect_val_dict.items(),
                    key=lambda item: (item[1], np.random.rand()),
                    reverse=False,
                )
                
                #Remove two-point correlations since only single-point correlations are used for this update
                remove = []
                for i in range(len(sorted_correlation_dict)):
                    if len(sorted_correlation_dict[i][0]) != 1:
                        remove.append(sorted_correlation_dict[i])
                
                for key in remove:
                    sorted_correlation_dict.remove(key)

                i = 0
                max_expect_val_sign = -1
                max_expect_val_location = []

                #Search for the best correlation
                while len(max_expect_val_location) != 1 or max_expect_val_location[0] >= get_starting_index(ord, num_nodes, K):
                    max_expect_val_location, max_expect_val = sorted_correlation_dict[
                                        i
                                    ]
                    max_expect_val_location = [
                        qiro.problem.position_translater[idx]
                        for idx in max_expect_val_location
                    ]
                    max_expect_val_sign = np.sign(max_expect_val).astype(int)
                    i += 1

                if logging:
                    print('Maximum expectation value location:', max_expect_val_location[0])

                #Find node, time step and index corresponding to the correlation index
                l = max_expect_val_location[0]

                k = 0
                while l >= get_starting_index(ord, num_nodes, k):
                    k += 1
                k -= 1

                l = l - get_starting_index(ord, num_nodes, k)

                t = int(l / num_nodes)
                t_shift = 0
                for i in range(num_nodes):
                    if i > t + t_shift:
                        break
                    if ord[k * num_nodes + i] >= 0:
                        t_shift += 1

                i = l - t * num_nodes
                loc = k * num_nodes + t + t_shift

                #Fix node
                ord[loc] = i
                
                energy = qiro.expectation_values.best_energy

                
                best_energy, best_state = qiro.expectation_values.get_best_energy()

                #Define reduced problem
                problem = CVRP(G, K, demands, capacity=capacity, order=ord)
                if exp_type == 'vqe':
                    qiro.expectation_values = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'numpy':
                    qiro.expectation_values = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'sa':
                    qiro.expectation_values = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
                else:
                    qiro.expectation_values = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                    
                energies.append(energy)
                best_energies.append(best_energy)
                orders.append(np.copy(ord))
                if logging:
                    print('Fixed node', i, 'at step', loc, 'with correlation', max_expect_val, '. Energy is', energy, '. Best energy is', best_energy)
                    print(ord)
                
                #Checks if all nodes have already been visited and stops QIRO in that case
                contains_all = True
                for i in range(num_nodes):
                    if i not in ord:
                        contains_all = False

                if contains_all:
                    for i in range(len(ord)):
                        if ord[i] == -1:
                            ord[i] = 0
                    print('Fixed remaining steps to 0')
                    print(ord)
                    break
            
            #Fill the missing node
            if not contains_all:
                missing_node = 1

                while missing_node in ord:
                    missing_node += 1
                if missing_node >= num_nodes:
                    missing_node = 0
                
                ord[ord == -1] = missing_node
                if logging:
                        print('Fixed last node node', missing_node)
                        print(ord)

            #Check for feasibility
            feasible = True
            for i in range(num_nodes):
                if not i in ord:
                    feasible = False

            unique, counts = np.unique(ord, return_counts=True)
            duplicates = dict(zip(unique, counts))
            for i in range(1, num_nodes):
                if not i in duplicates or duplicates[i] != 1:
                    feasible = False

            for k in range(K):
                cost = 0
                for i in range(num_nodes):
                    ind = i + num_nodes * k
                    node = int(ord[ind])
                    cost += demands[node]
                if cost > capacity:
                    feasible = False

            df.loc[len(df) - 1, 'betas'] = betas
            df.loc[len(df) - 1, 'gammas'] = gammas
            df.loc[len(df) - 1, 'energies'] = energies
            df.loc[len(df) - 1, 'best_energies'] = best_energies
            df.loc[len(df) - 1, 'orders'] = orders
            df.loc[len(df) - 1, 'solution'] = ord
            if feasible:
                distance = get_distance_cvrp(G, ord, K)
            else:
                distance = -1
            df.loc[len(df) - 1, 'distance'] = distance

            w = nx.to_numpy_array(G)
            N = len(ord)
            best_distance = solve_cvrp_or(adj_matrix, demands, np.ones(K) * capacity)
            df.loc[len(df) - 1, 'best_distance'] = best_distance
            df.loc[len(df) - 1, 'lr'] = best_distance / distance

            df_single.loc[len(df_single) - 1, 'orders'] = result_single
            df_single.loc[len(df_single) - 1, 'solution'] = ord_single
            df_single.loc[len(df_single) - 1, 'energies'] = energy_single
            df_single.loc[len(df_single) - 1, 'distance'] = distance_single
            df_single.loc[len(df_single) - 1, 'best_distance'] = best_distance
            df_single.loc[len(df_single) - 1, 'lr'] = best_distance / distance_single
    if not debug:
        df_single.to_csv(os.path.join('./','results/results_cvrp_' + run_name + '_' + str(num_nodes) + '_single_step' + '.csv'))
        df.to_csv(os.path.join('./','results/results_cvrp_' + run_name + '_' + str(num_nodes) + '.csv'))

def perform_qiro_tsp(num_nodes, seeds, df=None, exp_type= 'single', reps=1, logging = False, instance_file='tsp_instances.csv', run_name='default', samples=1000, debug=False, correlations_best=False):
    """
    Performs QIRO for TSP instances with a problem-specific update step

    Parameters:
        num_nodes: Array of the number of nodes for which tests should be performed
        seeds: Random seeds for the optimization
        df: pd.DataFrame where the results are saved; If None, a new DataFrame is created
        exp_type: Quantum algorithm used for QIRO; Options: Simulated annealing 'sa', QAOA 'qaoa', VQE 'vqe' or exact solver 'numpy'
        reps: Number of layers for VQE or QAOA
        logging: Determines whether information should be displayed during runs
        instance_file: File where the instances are located; Instances should have a seed, an adjacency_matrix and a num_node column
        run_name: Name of the run under which the test data should be saved
        samples: Number of shots when measuring a state
        debug: If True, the data is not saved to a file
        correlations_best: Determines if only the best samples should be used to compute correlations instead of all samples
    """
    folder = "./"
    tsp_instances = pd.read_csv(os.path.join(folder, instance_file))
    tsp_instances['adjacency_matrix'] = tsp_instances['adjacency_matrix'].apply(lambda x: np.array(ast.literal_eval(x)))
    tsp_instances = tsp_instances.loc[tsp_instances['num_nodes']==num_nodes]

    if df is None:
        df = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])
    
    df_single = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])

    for seed in seeds:
        print('Starting test run with', num_nodes, 'nodes and seed', seed)
        for ind in tsp_instances.index:
            #Prepare DataFrames
            instance = tsp_instances['seed'][ind]
            df.loc[len(df), 'n'] = num_nodes
            df.loc[len(df) - 1, 'instance'] = instance
            df.loc[len(df) - 1, 'seed'] = seed
            df_single.loc[len(df_single), 'n'] = num_nodes
            df_single.loc[len(df_single) - 1, 'instance'] = instance
            df_single.loc[len(df_single) - 1, 'seed'] = seed
            first_step = True

            betas = []
            gammas = []
            energies = []
            best_energies = []
            orders = []
            
            adj_matrix = tsp_instances['adjacency_matrix'][ind]
            G = nx.from_numpy_array(adj_matrix)

            np.random.seed(seed)
            ord = np.ones(num_nodes) * -1
            ord[0] = 0
            n = sum([k < 0 for k in ord])

            problem = TSP(G, ord)
            if exp_type == 'vqe':
                expval = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'numpy':
                expval = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'sa':
                expval = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
            else:
                expval = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            
            qiro = QIRO_TSP(nc=1, expectation_values=expval)

            #Continue QIRO while there is more than one unfixed TSP step
            while -1 in ord:
                if sum(ord == -1) <= 1:
                    break
                n = sum([k < 0 for k in ord])
                start_time = time.time()
                # optimize the correlations
                qiro.expectation_values.optimize()

                #Determine the solution from the quantum algorithm without QIRO
                if first_step:
                    result_single = np.fromiter(qiro.expectation_values.state, dtype=int)
                    energy_single = qiro.expectation_values.best_energy
                    ord_single = np.ones(num_nodes) * -1
                    ord_single[0] = 0
                    feasible = True
                    for i in range(1, len(ord)):
                        start_index = (i - 1) * n
                        node = -1
                        for j in range(n):
                            if result_single[start_index + j] == 1:
                                if node != -1:
                                    feasible = False
                                node = j + 1
                        if node in ord_single or node == -1:
                            feasible = False
                        ord_single[i] = node
                    
                    if -1 in ord_single:
                        feasible = False

                    if not feasible:
                        print('Infeasible solution was found by simulated annealing')
                        distance_single = -1
                    else:
                        distance_single = get_distance(G, ord_single)
                    if logging:
                        print('Simulated Annealing in the first step found solution', ord_single, 'with distance', distance_single)

                    first_step = False

                end_time = time.time()
                if logging:
                    print('Total QAOA process took', end_time - start_time, 'seconds')

                # sorts correlations in decreasing order
                sorted_correlation_dict = sorted(
                    qiro.expectation_values.expect_val_dict.items(),
                    key=lambda item: (item[1], np.random.rand()),
                    reverse=False,
                )

                #Remove two-point correlations since this update step only uses single-point correlations
                remove = []
                for i in range(len(sorted_correlation_dict)):
                    if len(sorted_correlation_dict[i][0]) != 1:
                        remove.append(sorted_correlation_dict[i])
                
                for key in remove:
                    sorted_correlation_dict.remove(key)

                i = 0
                max_expect_val_sign = -1
                max_expect_val_location = []

                #Find index of best correlation
                while len(max_expect_val_location) != 1:
                    max_expect_val_location, max_expect_val = sorted_correlation_dict[
                                        i
                                    ]
                    max_expect_val_location = [
                        qiro.problem.position_translater[idx]
                        for idx in max_expect_val_location
                    ]
                    max_expect_val_sign = np.sign(max_expect_val).astype(int)
                    i += 1

                if logging:
                    print('Maximum expectation value location:', max_expect_val_location[0])

                #Determine node and time step corresponding to the index of the chosen correlation
                l = max_expect_val_location[0]
                t = int(l / n)
                i = l - t * n

                k = 0
                while k <= i:
                    if k in ord:
                        i += 1
                    k += 1

                for position in range(len(ord)):
                    if ord[position] == -1:
                        t -= 1
                    if t < 0:
                        t = position
                        break
                
                #Fix node and TSP step
                ord[t] = i

                energy = qiro.expectation_values.best_energy

                best_energy, best_state = qiro.expectation_values.get_best_energy()

                #Update reduced problem
                problem = TSP(G, ord)
                
                if exp_type == 'vqe':
                    qiro.expectation_values = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'numpy':
                    qiro.expectation_values = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'sa':
                    qiro.expectation_values = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
                else:
                    qiro.expectation_values = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                    
                energies.append(energy)
                best_energies.append(best_energy)
                orders.append(np.copy(ord))
                if logging:
                    print('Fixed node', i, 'at step', t, 'with correlation', max_expect_val, '. Energy is', energy, '. Best energy is', best_energy)
                    print(ord)
            
            #Fix trivial remaining node after QIRO is finished
            missing_node = 0

            while missing_node in ord:
                missing_node += 1
            
            ord[ord == -1] = missing_node
            if logging:
                    print('Fixed last node node', missing_node)
                    print(ord)

            df.loc[len(df) - 1, 'betas'] = betas
            df.loc[len(df) - 1, 'gammas'] = gammas
            df.loc[len(df) - 1, 'energies'] = energies
            df.loc[len(df) - 1, 'best_energies'] = best_energies
            df.loc[len(df) - 1, 'orders'] = orders
            df.loc[len(df) - 1, 'solution'] = ord
            distance = get_distance(G, ord)
            df.loc[len(df) - 1, 'distance'] = distance

            w = nx.to_numpy_array(G)
            N = len(ord)
            best_distance, _ = solve_tsp_or(adj_matrix)
            df.loc[len(df) - 1, 'best_distance'] = best_distance
            df.loc[len(df) - 1, 'lr'] = best_distance / distance

            df_single.loc[len(df_single) - 1, 'orders'] = result_single
            df_single.loc[len(df_single) - 1, 'solution'] = ord_single
            df_single.loc[len(df_single) - 1, 'energies'] = energy_single
            df_single.loc[len(df_single) - 1, 'distance'] = distance_single
            df_single.loc[len(df_single) - 1, 'best_distance'] = best_distance
            df_single.loc[len(df_single) - 1, 'lr'] = best_distance / distance_single
    if not debug:
        df_single.to_csv(os.path.join('./','results/results_' + run_name + '_' + str(num_nodes) + '_single_step' + '.csv'))
        df.to_csv(os.path.join('./','results/results_' + run_name + '_' + str(num_nodes) + '.csv'))

def perform_qiro_tsp_qubit_level(num_nodes, seeds, df=None, exp_type= 'single', reps=1, logging = False, instance_file='tsp_instances.csv', run_name='default', samples=1000, debug=False, correlations_best=False):
    """
    Performs QIRO for TSP instances with a qubit-level update step

    Parameters:
        num_nodes: Array of the number of nodes for which tests should be performed
        seeds: Random seeds for the optimization
        df: pd.DataFrame where the results are saved; If None, a new DataFrame is created
        exp_type: Quantum algorithm used for QIRO; Options: Simulated annealing 'sa', QAOA 'qaoa', VQE 'vqe' or exact solver 'numpy'
        reps: Number of layers for VQE or QAOA
        logging: Determines whether information should be displayed during runs
        instance_file: File where the instances are located; Instances should have a seed, an adjacency_matrix and a num_node column
        run_name: Name of the run under which the test data should be saved
        samples: Number of shots when measuring a state
        debug: If True, the data is not saved to a file
        correlations_best: Determines if only the best samples should be used to compute correlations instead of all samples
    """
    folder = "./"
    tsp_instances = pd.read_csv(os.path.join(folder, instance_file))
    tsp_instances['adjacency_matrix'] = tsp_instances['adjacency_matrix'].apply(lambda x: np.array(ast.literal_eval(x)))
    tsp_instances = tsp_instances.loc[tsp_instances['num_nodes']==num_nodes]

    if df is None:
        df = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])
    
    
    df_single = pd.DataFrame(columns=['instance', 'n', 'seed', 'distance', 'best_distance', 'lr', 'solution', 'betas', 'gammas', 'energies', 'best_energies', 'orders'])

    for seed in seeds:
        print('Starting test run with', num_nodes, 'nodes and seed', seed)
        for ind in tsp_instances.index:
            instance = tsp_instances['seed'][ind]
            df.loc[len(df), 'n'] = num_nodes
            df.loc[len(df) - 1, 'instance'] = instance
            df.loc[len(df) - 1, 'seed'] = seed
            df_single.loc[len(df_single), 'n'] = num_nodes
            df_single.loc[len(df_single) - 1, 'instance'] = instance
            df_single.loc[len(df_single) - 1, 'seed'] = seed
            first_step = True

            betas = []
            gammas = []
            energies = []
            best_energies = []
            orders = []
            
            adj_matrix = tsp_instances['adjacency_matrix'][ind]
            G = nx.from_numpy_array(adj_matrix)

            ord = np.ones(num_nodes) * -1
            ord[0] = 0
            n = sum([k < 0 for k in ord])
            fixed = np.ones((num_nodes - 1) * (num_nodes - 1)) * -1

            problem = QubitTSP(G, ord, fixed=fixed)
            if exp_type == 'vqe':
                expval = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'numpy':
                expval = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            elif exp_type == 'sa':
                expval = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
            else:
                expval = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
            
            qiro = QIRO_TSP(nc=1, expectation_values=expval)

            # Continue QIRO while there are qubits that have not been fixed
            while sum(fixed != -1) != ((num_nodes - 1) * (num_nodes - 1)):
                if sum(fixed != -1) >= ((num_nodes - 1) * (num_nodes - 1)) - 2:
                    break
                n = sum([k < 0 for k in ord])
                start_time = time.time()

                # optimize the correlations
                qiro.expectation_values.optimize()

                # Determine solution from the quantum algorithm without QIRO
                if first_step:
                    result_single = np.fromiter(qiro.expectation_values.state, dtype=int)
                    energy_single = qiro.expectation_values.best_energy
                    ord_single = np.ones(num_nodes) * -1
                    ord_single[0] = 0
                    feasible = True
                    for i in range(1, len(ord)):
                        start_index = (i - 1) * n
                        node = -1
                        for j in range(n):
                            if result_single[start_index + j] == 1:
                                if node != -1:
                                    feasible = False
                                node = j + 1
                        if node in ord_single or node == -1:
                            feasible = False
                        ord_single[i] = node

                    if -1 in ord_single:
                        feasible = False

                    if not feasible:
                        print('Infeasible solution was found by simulated annealing')
                        distance_single = -1
                    else:
                        distance_single = get_distance(G, ord_single)
                    if logging:
                        print('Simulated Annealing in the first step found solution', ord_single, 'with distance', distance_single)

                    first_step = False

                end_time = time.time()
                if logging:
                    print('Total QAOA process took', end_time - start_time, 'seconds')

                # Sorts correlations in decreasing order; Here correlations with a high absolute value are used
                sorted_correlation_dict = sorted(
                    qiro.expectation_values.expect_val_dict.items(),
                    key=lambda item: (abs(item[1]), np.random.rand()),
                    reverse=True,
                )

                # Removes two-point correlations since only single-point correlations are used here
                remove = []
                for i in range(len(sorted_correlation_dict)):
                    if len(sorted_correlation_dict[i][0]) != 1:
                        remove.append(sorted_correlation_dict[i])
                
                for key in remove:
                    sorted_correlation_dict.remove(key)

                i = 0
                max_expect_val_sign = -1
                max_expect_val_location = []
                while len(max_expect_val_location) != 1:
                    max_expect_val_location, max_expect_val = sorted_correlation_dict[
                                        i
                                    ]
                    max_expect_val_location = [
                        qiro.problem.position_translater[idx]
                        for idx in max_expect_val_location
                    ]
                    max_expect_val_sign = np.sign(max_expect_val).astype(int)
                    i += 1


                if logging:
                    print('Maximum expectation value location:', max_expect_val_location[0])

                #If the correlation is negative, the qubit is lekely to be 1, otherwise it is set to 0
                if max_expect_val_sign < 0:
                    fixed[problem.index_mapper[max_expect_val_location[0]]] = 1
                else:
                    fixed[problem.index_mapper[max_expect_val_location[0]]] = 0


                energy = qiro.expectation_values.best_energy
                
                best_energy = 0

                if logging:
                    print('Fixed qubit', problem.index_mapper[max_expect_val_location[0]], 'with correlation', max_expect_val, '. Energy is', energy, '. Best energy is', best_energy)
                    print(fixed)

                #Updates QUBO problem
                problem = QubitTSP(G, ord, fixed=fixed)

                if exp_type == 'vqe':
                    qiro.expectation_values = VQEExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'numpy':
                    qiro.expectation_values = NumpyExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                elif exp_type == 'sa':
                    qiro.expectation_values = SAExpectationValues(problem, logging=logging, num_reads=samples, correlations_best=correlations_best)
                else:
                    qiro.expectation_values = MultiLayerQAOAExpectationValues(problem, reps=reps, seed=seed, logging=logging, correlations_best=correlations_best)
                    
                energies.append(energy)
                best_energies.append(best_energy)
                orders.append(np.copy(ord))
            
            #Fixes the last qubit
            qiro.expectation_values.optimize()
            last_result = np.fromiter(qiro.expectation_values.state, dtype=int)
            index_total = 0
            for i in last_result:
                while fixed[index_total] != -1:
                    index_total += 1
                fixed[index_total] = i
                index_total += 1

            if logging:
                    print('Fixed last qubits with assignment', fixed)

            #Evaluates result from the final qubit assignments
            result_qubit = []
            for i in range((num_nodes - 1) * (num_nodes - 1)):
                result_qubit.append(fixed[i])
            
            feasible = True
            for i in range(1, len(ord)):
                start_index = (i - 1) * n
                
                node = -1
                for j in range(n):
                    if result_qubit[start_index + j] == 1:
                        if node != -1:
                            feasible = False
                        node = j + 1
                if node in ord or node == -1:
                    feasible = False
                ord[i] = node

            if -1 in ord:
                feasible = False
            
            for i in range(len(ord)):
                for j in range(len(ord)):
                    if i != j and ord[i] == ord[j]:
                        feasible = False

            if not feasible:
                print('Infeasible solution was found by QIRO')
                distance = -1
            else:
                distance = get_distance(G, ord)
            if logging:
                print('QIRO found solution', ord, 'with distance', distance)


            df.loc[len(df) - 1, 'betas'] = betas
            df.loc[len(df) - 1, 'gammas'] = gammas
            df.loc[len(df) - 1, 'energies'] = energies
            df.loc[len(df) - 1, 'best_energies'] = best_energies
            df.loc[len(df) - 1, 'orders'] = orders
            df.loc[len(df) - 1, 'solution'] = ord
            df.loc[len(df) - 1, 'distance'] = distance

            w = nx.to_numpy_array(G)
            N = len(ord)
            best_distance, _ = solve_tsp_or(adj_matrix)
            df.loc[len(df) - 1, 'best_distance'] = best_distance
            df.loc[len(df) - 1, 'lr'] = best_distance / distance

            
            df_single.loc[len(df_single) - 1, 'orders'] = result_single
            df_single.loc[len(df_single) - 1, 'solution'] = ord_single
            df_single.loc[len(df_single) - 1, 'energies'] = energy_single
            df_single.loc[len(df_single) - 1, 'distance'] = distance_single
            df_single.loc[len(df_single) - 1, 'best_distance'] = best_distance
            df_single.loc[len(df_single) - 1, 'lr'] = best_distance / distance_single
    if not debug:
        df_single.to_csv(os.path.join('./','results/results_' + run_name + '_' + str(num_nodes) + '_single_step' + '.csv'))
        df.to_csv(os.path.join('./','results/results_' + run_name + '_' + str(num_nodes) + '.csv'))