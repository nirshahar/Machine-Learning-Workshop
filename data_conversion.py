from pysmiles import read_smiles
from stellargraph import StellarGraph
import logging
import numpy as np
import networkx

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

NON_AGG_FILENAME = "control_non_aggregates"
AGG_FILENAME = "agg"


l = 22
p = 0
element_to_vec = {}
edge_to_vec = {0: np.concatenate((np.array([1]), np.zeros(26))), 1: np.concatenate((np.array([0, 1]), np.zeros(25))), 1.5: np.concatenate((np.array([0, 0, 1]), np.zeros(24))), 2: np.concatenate((np.array([0, 0, 0, 1]), np.zeros(23))), 3: np.concatenate((np.array([0, 0, 0, 0, 1]), np.zeros(22)))}


# def create_features2(graph):
# 	global p
# 	for node in graph:
# 		element_string = graph.nodes[node]["element"]
# 		if element_string not in element_to_vec:
# 			new_one_hot_vector = np.array([0 for i in range(l)])
# 			new_one_hot_vector[p] = 1
# 			p += 1
# 			element_to_vec.update({element_string: new_one_hot_vector})
#
# 		zero_bond = 0
# 		one_bond = 0
# 		one_half_bond = 0
# 		two_bond = 0
# 		three_bond = 0
# 		for adj_node in graph.adj[node]:
# 			edge_type = graph.adj[node][adj_node]['order']
# 			if edge_type == 0:
# 				zero_bond += 1
# 			elif edge_type == 1:
# 				one_bond += 1
# 			elif edge_type == 1.5:
# 				one_half_bond += 1
# 			elif edge_type == 2:
# 				two_bond += 1
# 			elif edge_type == 3:
# 				three_bond += 1
# 			else:
# 				raise Exception("bond type not understood: " + str(edge_type))
#
# 		bonds = np.array([zero_bond, one_bond, one_half_bond, two_bond, three_bond])
#
# 		graph.nodes[node]["element"] = np.concatenate((np.array([0]), bonds, element_to_vec[element_string]))
#
# 	new_graph = networkx.create_empty_copy(graph)
# 	edges = graph.edges.data("order")
# 	for i, (v, u, bond) in enumerate(edges):
# 		edge_name = "edge_node" + str(i)
#
# 		new_graph.add_node(edge_name)
#
# 		new_graph.add_edge(v, edge_name)
# 		new_graph.add_edge(edge_name, u)
#
# 		new_graph.nodes[edge_name]["element"] = np.concatenate((np.array([1]), edge_to_vec[bond]))
#
# 	return new_graph

def create_features(graph):
	global p
	for node in graph:

		element_string = graph.nodes[node]["element"]
		if element_string not in element_to_vec:
			new_one_hot_vector = np.array([0 for i in range(l)])
			new_one_hot_vector[p] = 1
			p += 1
			element_to_vec.update({element_string: new_one_hot_vector})

		zero_bond = 0
		one_bond = 0
		one_half_bond = 0
		two_bond = 0
		three_bond = 0
		for adj_node in graph.adj[node]:
			edge_type = graph.adj[node][adj_node]['order']
			if edge_type == 0:
				zero_bond += 1
			elif edge_type == 1:
				one_bond += 1
			elif edge_type == 1.5:
				one_half_bond += 1
			elif edge_type == 2:
				two_bond += 1
			elif edge_type == 3:
				three_bond += 1
			else:
				raise Exception("bond type not understood: " + str(edge_type))

		bonds = np.array([zero_bond, one_bond, one_half_bond, two_bond, three_bond])

		extra_node_data = np.array([graph.nodes[node]['charge'], float(graph.nodes[node]['aromatic']), graph.nodes[node]['hcount']])

		features = np.concatenate((extra_node_data, bonds, element_to_vec[element_string]))

		graph.nodes[node]["features"] = features

	return graph


def get_non_agg_generator(non_agg_filename=NON_AGG_FILENAME, data_count=None):
	cnt = 0
	with open(non_agg_filename + ".txt", "r") as non_agg_file:
		non_agg = non_agg_file.readline()
		while non_agg != "" and (data_count is None or cnt < data_count):
			parsed_non_agg = non_agg.strip().split()[1].replace("se", "Se")

			graph = read_smiles(parsed_non_agg)
			graph = create_features(graph)

			stellar_graph = StellarGraph.from_networkx(graph=graph, node_features="features")

			yield stellar_graph

			cnt += 1
			non_agg = non_agg_file.readline()


def get_non_agg_list(non_agg_filename=NON_AGG_FILENAME, data_count=None):
	non_agg_list = []

	with open(non_agg_filename + ".txt", "r") as non_agg_file:
		non_aggs = non_agg_file.readlines()

		data_count = len(non_aggs) if data_count is None else data_count
		for non_agg in non_aggs[:data_count]:
			parsed_non_agg = non_agg.strip().split()[1].replace("se", "Se")

			graph = read_smiles(parsed_non_agg)
			graph = create_features(graph)

			sg_non_agg = StellarGraph.from_networkx(graph=graph, node_features="features")

			non_agg_list.append(sg_non_agg)

	return non_agg_list


def get_agg_generator(agg_filename=AGG_FILENAME, data_count=None):
	cnt = 0
	with open(agg_filename + ".txt", "r") as agg_file:
		agg = agg_file.readline()
		while agg != "" and (data_count is None or cnt < data_count):
			parsed_agg = agg.strip().split()[0]

			graph = read_smiles(parsed_agg)
			graph = create_features(graph)

			yield StellarGraph.from_networkx(graph=graph, node_features="features")

			cnt += 1
			agg = agg_file.readline()



def get_agg_list(agg_filename=AGG_FILENAME, data_count=None):
	agg_list = []

	with open(agg_filename + ".txt", "r") as agg_file:
		aggs = agg_file.readlines()
		data_count = len(aggs) if data_count is None else data_count

		for agg in aggs[:data_count]:
			parsed_agg = agg.strip().split()[0]

			graph = read_smiles(parsed_agg)
			graph = create_features(graph)

			sg_agg = StellarGraph.from_networkx(graph=graph, node_features="features")

			agg_list.append(sg_agg)

	return agg_list



if __name__ == "__main__":
	for x in get_non_agg_generator():
		break

