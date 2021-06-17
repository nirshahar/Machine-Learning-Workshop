from pysmiles import *
from stellargraph import StellarGraph
import logging
import numpy as np
from networkx import Graph

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

NON_AGG_FILEPATH = "control_non_aggregates"
AGG_FILEPATH = "agg"

l = 23
p = 0
element_to_vec = {}
edge_to_vec = {0: np.concatenate((np.array([1]), np.zeros(26))), 1: np.concatenate((np.array([0, 1]), np.zeros(25))),
			   1.5: np.concatenate((np.array([0, 0, 1]), np.zeros(24))),
			   2: np.concatenate((np.array([0, 0, 0, 1]), np.zeros(23))),
			   3: np.concatenate((np.array([0, 0, 0, 0, 1]), np.zeros(22)))}


def parse_smiles(
		smiles: str,
		explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	graph = read_smiles(
		smiles,
		explicit_hydrogen=explicit_hydrogen,
		zero_order_bonds=zero_order_bonds,
		reinterpret_aromatic=reinterpret_aromatic
	)

	graph = create_features(
		graph,
		explicit_hydrogen=explicit_hydrogen,
		compute_valence=compute_valence,
		valence_respect_hcount=valence_respect_hcount,
		valence_respect_bond_order=valence_respect_bond_order,
		valence_max_bond_order=valence_max_bond_order
	)

	return graph


def create_features(
		graph: Graph,
		explicit_hydrogen: bool = False,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	if compute_valence:
		fill_valence(
			graph,
			respect_hcount=valence_respect_hcount,
			respect_bond_order=valence_respect_bond_order,
			max_bond_order=valence_max_bond_order
		)

	global p
	for node in graph:

		element_string = graph.nodes[node]["element"]
		if element_string not in element_to_vec:
			new_one_hot_vector = np.array([0 for _ in range(l)])
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

		extra_node_data = np.array([graph.nodes[node]['charge'], float(graph.nodes[node]['aromatic'])])

		features = np.concatenate((extra_node_data, bonds, element_to_vec[element_string]))

		if not explicit_hydrogen:
			features = np.concatenate((features, np.array([graph.nodes[node]['hcount']])))

		graph.nodes[node]['features'] = features

	return graph


def get_data(
		is_agg: bool,
		is_generator: bool = False,
		filepath: str = None,
		data_count: int = None,
		explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	if is_agg:
		filepath = filepath if filepath is not None else AGG_FILEPATH
		if is_generator:
			return get_agg_generator(
				agg_filepath=filepath,
				data_count=data_count,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

		else:
			return get_agg_list(
				agg_filepath=filepath,
				data_count=data_count,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

	else:
		filepath = filepath if filepath is not None else NON_AGG_FILEPATH
		if is_generator:
			return get_non_agg_generator(
				non_agg_filepath=filepath,
				data_count=data_count,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

		else:
			return get_non_agg_list(
				non_agg_filepath=filepath,
				data_count=data_count,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)


def get_non_agg_generator(
		non_agg_filepath: str = NON_AGG_FILEPATH,
		data_count: int = None,
		explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	cnt = 0
	with open(non_agg_filepath + ".txt", "r") as non_agg_file:
		non_agg = non_agg_file.readline()

		while non_agg != "" and (data_count is None or cnt < data_count):
			parsed_non_agg = non_agg.strip().split()[1].replace("se", "Se")

			graph = parse_smiles(
				parsed_non_agg,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

			sg_non_agg = StellarGraph.from_networkx(graph=graph, node_features="features")

			yield sg_non_agg

			non_agg = non_agg_file.readline()
			cnt += 1


def get_non_agg_list(
		non_agg_filepath: str = NON_AGG_FILEPATH,
		data_count: int = None,
		explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	non_agg_list = []

	with open(non_agg_filepath + ".txt", "r") as non_agg_file:
		non_aggs = non_agg_file.readlines()

		data_count = len(non_aggs) if data_count is None else data_count
		for non_agg in non_aggs[:data_count]:
			parsed_non_agg = non_agg.strip().split()[1].replace("se", "Se")

			graph = parse_smiles(
				parsed_non_agg,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

			sg_non_agg = StellarGraph.from_networkx(graph=graph, node_features="features")

			non_agg_list.append(sg_non_agg)

	return non_agg_list


def get_agg_generator(
		agg_filepath: str = AGG_FILEPATH,
		data_count: int = None,
		explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3
):

	cnt = 0
	with open(agg_filepath + ".txt", "r") as agg_file:
		agg = agg_file.readline()
		while agg != "" and (data_count is None or cnt < data_count):
			parsed_agg = agg.strip().split()[0]

			graph = parse_smiles(
				parsed_agg,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

			yield StellarGraph.from_networkx(graph=graph, node_features="features")

			cnt += 1
			agg = agg_file.readline()


def get_agg_list(
		agg_filepath=AGG_FILEPATH,
		data_count=None,
		explicit_hydrogen=False,
		zero_order_bonds=True,
		reinterpret_aromatic=True,
		compute_valence=False,
		valence_respect_hcount=True,
		valence_respect_bond_order=True,
		valence_max_bond_order=3
):

	agg_list = []

	with open(agg_filepath + ".txt", "r") as agg_file:
		aggs = agg_file.readlines()
		data_count = len(aggs) if data_count is None else data_count

		for agg in aggs[:data_count]:
			parsed_agg = agg.strip().split()[0]

			graph = parse_smiles(
				parsed_agg,
				explicit_hydrogen=explicit_hydrogen,
				zero_order_bonds=zero_order_bonds,
				reinterpret_aromatic=reinterpret_aromatic,
				compute_valence=compute_valence,
				valence_respect_hcount=valence_respect_hcount,
				valence_respect_bond_order=valence_respect_bond_order,
				valence_max_bond_order=valence_max_bond_order
			)

			sg_agg = StellarGraph.from_networkx(graph=graph, node_features="features")

			agg_list.append(sg_agg)

	return agg_list


if __name__ == "__main__":
	print(get_data(True, is_generator=False, data_count=100))
