from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pysmiles
import networkx as nx
from networkx import Graph


def parse_smiles(smile,explicit_hydrogen: bool = False,
		zero_order_bonds: bool = True,
		reinterpret_aromatic: bool = True,
		compute_valence: bool = False,
		valence_respect_hcount: bool = True,
		valence_respect_bond_order: bool = True,
		valence_max_bond_order: int = 3):
	mol = Chem.MolFromSmiles(smile)
	if mol is None:
		print("Cannot parse smile: " + str(smile))

	graph = Graph()

	for atom in mol.GetAtoms():
		graph.add_node(atom.GetIdx())

	for bond in mol.GetBonds():
		graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())

	return graph

# smiles = "CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2.[Cl-]"
# smiles2 = "OCCn2c(=N)n(CCOc1ccc(Cl)cc1Cl)c3ccccc23"
#
# mol = Chem.MolFromSmiles(smiles, sanitize=False)
# Draw.ShowMol(mol)
# plt.show()
# 
#
# rdkit_graph = parse_smiles(smiles)
# nx.draw_networkx(rdkit_graph)
# plt.show()
#
#
# graph = pysmiles.read_smiles(smiles)
# nx.draw_networkx(graph)
# plt.show()
#
# edges_to_remove = []
#
# for node in graph:
# 	for adj_node in graph.adj[node]:
# 		edge_type = graph.adj[node][adj_node]['order']
#
# 		if edge_type == 0:
# 			print("aaaaaa")
# 			if (adj_node, node) not in edges_to_remove:
# 				edges_to_remove.append((node, adj_node))
#
# for a, b in edges_to_remove:
# 	graph.remove_edge(a, b)
#
# nx.draw_networkx(graph)
# plt.show()

#
# import logging
# logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning
#
# with open("agg.txt", "r") as f:
# 	for unparsed_smiles in f.readlines():
# 		smiles = unparsed_smiles.strip().split()[0]
# 		#smiles = unparsed_smiles.strip().split()[1].replace("se", "Se")
#
# 		mol = Chem.MolFromSmiles(smiles)
# 		graph = pysmiles.read_smiles(smiles)
#
# 		for a, b in zip(mol.GetAtoms(), graph.nodes):
# 			if a.GetNumImplicitHs() != graph.nodes[b]['hcount']:
# 				print(a.GetNumImplicitHs(), graph.nodes[b]['hcount'])
# 				print(smiles)
#
# print("done!")

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt

from IPython.display import SVG
IPythonConsole.ipython_useSVG=True


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetNumImplicitHs() + 1)
    return mol


mol = Chem.MolFromSmiles("OCCn2c(=N)n(CCOc1ccc(Cl)cc1Cl)c3ccccc23")
mol = mol_with_atom_index(mol)
mc = Chem.Mol(mol.ToBinary())

Draw.ShowMol(mc)
plt.show()

drawer = rdMolDraw2D.MolDraw2DSVG(450, 200)
drawer.DrawMolecule(mc)
drawer.FinishDrawing()

svg = drawer.GetDrawingText()
Draw.display(SVG(svg.replace('svg:','')))