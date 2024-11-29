"""
Set of util functions, specifically providing exact solvers for the TSP and CVRP using Google OR tools and other functions for evaluating results.
"""
import networkx as nx
import numpy as np
import random
from src.Generate_TSP import TSP, QubitTSP
from src.Generate_CVRP import CVRP
from QIRO_Routing import QIRO_TSP, QIRO_CVRP
from src.MultiLayerQAOA import MultiLayerQAOAExpectationValues
from src.VQEExpectationValues import VQEExpectationValues
from src.NumpyExpectationValues import NumpyExpectationValues
from src.SAExpectationValues import SAExpectationValues
from itertools import permutations
import seaborn as sns
import pandas as pd
import os
import ast
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy

def get_distance(graph, order):
    """
    Calculates the TSP distance for a given solution

    Parameters:
        graph: nx.Graph of the TSP instance
        order: Permutation of nodes defining the proposed solution
    """
    l = len(order)
    weigths = nx.get_edge_attributes(graph, 'weight')
    result = 0
    for i in range(l):
        ind0, ind1 = (order[i], order[(i + 1) % l])
        edge = (ind0, ind1) if ind0 < ind1 else (ind1, ind0)
        result += weigths[edge]
    return result

def get_distance_cvrp(graph, order, K):
    """
    Calculates the CVRP distance for a given solution

    Parameters:
        graph: nx.Graph of the TSP instance
        order: Permutation of nodes defining the proposed solution
        K: Number of routes in the solution
    """
    weigths = nx.get_edge_attributes(graph, 'weight')
    l = int(len(order) / K)
    result = 0
    for k in range(K):
        for i in range(l):
            ind0, ind1 = (order[k * l + i], order[k * l + (i + 1) % l])
            edge = (ind0, ind1) if ind0 < ind1 else (ind1, ind0)
            if edge in weigths:
                result += weigths[edge]
    return result

def get_best_state(problem):
    """
    Brute force approach for getting the best energy state

    Parameters:
        problem: Problem containing the underlying QUBO matrix
    """
    best_energy = np.inf
    best_state = ''
    num_qubits = problem.matrix.shape[0] - 1
    for i in range(2**(num_qubits)):
        b = format(i, f'#0{num_qubits + 3}b')[2:]
        s = np.array([int(digit, 2) for digit in b])
        energy = s @ problem.matrix @ s.T
        if energy < best_energy:
            best_energy = energy
            best_state = b
    return best_energy, best_state

def get_starting_index(ord, n, k):
    """
    Helper function for the TSP returning the index at what point in the encoding a timestep starts

    Parameters:
        ord: Previously fixed nodes from QIRO
        n: Number of nodes in the TSP

    """
    reduced_order = ord[:n * k]
    number_set = sum([k != -1 for k in reduced_order])
    return k * n * n - number_set * n

def evaluate_cvrp_solution(solution, ord, num_nodes, K):
    """
    Helper function for the CVRP to get the permutation of visited nodes from an assignment of variables.

    Parameters:
        solution: Array of binary variables in the QUBO encoding of the CVRP
        ord: Array of previously fixed nodes from QIRO 
        num_nodes: Number of nodes in the CVRP
        K: Number of routes in the CVRP
    """
    ord_copy = copy.deepcopy(ord)
    for l in range(len(solution)):
        if l >= get_starting_index(ord, num_nodes, K):
            break
        if solution[l] == 1:
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
            loc = int(k * num_nodes + t + t_shift)
            ord_copy[loc] = i
    return ord_copy

def evaluate_tsp_solution(solution, ord):
    """
    Helper function for the TSP to get the permutation of visited nodes from an assignment of variables.

    Parameters:
        solution: Array of binary variables in the QUBO encoding of the TSP
        ord: Array of previously fixed nodes from QIRO 
    """
    ord_copy = copy.deepcopy(ord)
    n = sum([k < 0 for k in ord])
    for l in range(len(solution)):
        if solution[l] == 1:
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
            ord_copy[t] = i
    return ord_copy


def solve_tsp_or(adj_matrix):
    """
    Helper function computing a TSP result with the Google OR Tools (https://developers.google.com/optimization/routing/vrp).

    Parameters:
        adj_matrix: Adjacency matrix of the TSP graph
    """
    manager = pywrapcp.RoutingIndexManager(len(adj_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return adj_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    return route_distance, plan_output

def solve_cvrp_or(adj_matrix, demands, capacities):
    """
    Helper function computing a CVRP result with the Google OR Tools (https://developers.google.com/optimization/routing/vrp).

    Parameters:
        adj_matrix: Adjacency matrix of the CVRP graph
        demands: Demands of the nodes in the CVRP graph
        capacities: Array of vehicle capacities for each route
    """
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(adj_matrix), len(capacities), 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return adj_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return print_solution_cvrp(demands, len(capacities), manager, routing, solution)
    else:
        return -1

def print_solution_cvrp(demands, n_vehicles, manager, routing, solution):
    """Prints solution on console."""
    #print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    #output = []
    for vehicle_id in range(n_vehicles):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += demands[node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        #output.append(index)
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        #print(plan_output)
        total_distance += route_distance
        total_load += route_load
        print(plan_output)
    print(f"Total distance of all routes: {total_distance}m")
    #print(f"Total load of all routes: {total_load}")
    return total_distance