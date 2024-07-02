import matplotlib.pyplot as plt
import networkx as nx

# Define the layers and their connections
layers = [
    ("Input", "Conv1"),
    ("Conv1", "Layer1_Block1_Conv1"),
    ("Layer1_Block1_Conv1", "Layer1_Block1_BN1"),
    ("Layer1_Block1_BN1", "Layer1_Block1_Conv2"),
    ("Layer1_Block1_Conv2", "Layer1_Block1_BN2"),
    ("Layer1_Block1_BN2", "Layer1_Block2_Conv1"),
    ("Layer1_Block2_Conv1", "Layer1_Block2_BN1"),
    ("Layer1_Block2_BN1", "Layer1_Block2_Conv2"),
    ("Layer1_Block2_Conv2", "Layer1_Block2_BN2"),
    ("Layer1_Block2_BN2", "Layer2_Block1_Conv1"),
    ("Layer2_Block1_Conv1", "Layer2_Block1_BN1"),
    ("Layer2_Block1_BN1", "Layer2_Block1_Conv2"),
    ("Layer2_Block1_Conv2", "Layer2_Block1_BN2"),
    ("Layer2_Block1_BN2", "Layer2_Block2_Conv1"),
    ("Layer2_Block2_Conv1", "Layer2_Block2_BN1"),
    ("Layer2_Block2_BN1", "Layer2_Block2_Conv2"),
    ("Layer2_Block2_Conv2", "Layer2_Block2_BN2"),
    ("Layer2_Block2_BN2", "Layer3_Block1_Conv1"),
    ("Layer3_Block1_Conv1", "Layer3_Block1_BN1"),
    ("Layer3_Block1_BN1", "Layer3_Block1_Conv2"),
    ("Layer3_Block1_Conv2", "Layer3_Block1_BN2"),
    ("Layer3_Block1_BN2", "Layer3_Block2_Conv1"),
    ("Layer3_Block2_Conv1", "Layer3_Block2_BN1"),
    ("Layer3_Block2_BN1", "Layer3_Block2_Conv2"),
    ("Layer3_Block2_Conv2", "Layer3_Block2_BN2"),
    ("Layer3_Block2_BN2", "Layer4_Block1_Conv1"),
    ("Layer4_Block1_Conv1", "Layer4_Block1_BN1"),
    ("Layer4_Block1_BN1", "Layer4_Block1_Conv2"),
    ("Layer4_Block1_Conv2", "Layer4_Block1_BN2"),
    ("Layer4_Block1_BN2", "Layer4_Block2_Conv1"),
    ("Layer4_Block2_Conv1", "Layer4_Block2_BN1"),
    ("Layer4_Block2_BN1", "Layer4_Block2_Conv2"),
    ("Layer4_Block2_Conv2", "Layer4_Block2_BN2"),
    ("Layer4_Block2_BN2", "FC"),
    ("FC", "Output"),
]

# Create the graph
G = nx.DiGraph()
G.add_edges_from(layers)

# Define positions for nodes
pos = {
    "Input": (0, 5),
    "Conv1": (1, 5),
    "Layer1_Block1_Conv1": (2, 7),
    "Layer1_Block1_BN1": (3, 7),
    "Layer1_Block1_Conv2": (4, 7),
    "Layer1_Block1_BN2": (5, 7),
    "Layer1_Block2_Conv1": (2, 3),
    "Layer1_Block2_BN1": (3, 3),
    "Layer1_Block2_Conv2": (4, 3),
    "Layer1_Block2_BN2": (5, 3),
    "Layer2_Block1_Conv1": (6, 6),
    "Layer2_Block1_BN1": (7, 6),
    "Layer2_Block1_Conv2": (8, 6),
    "Layer2_Block1_BN2": (9, 6),
    "Layer2_Block2_Conv1": (6, 4),
    "Layer2_Block2_BN1": (7, 4),
    "Layer2_Block2_Conv2": (8, 4),
    "Layer2_Block2_BN2": (9, 4),
    "Layer3_Block1_Conv1": (10, 6),
    "Layer3_Block1_BN1": (11, 6),
    "Layer3_Block1_Conv2": (12, 6),
    "Layer3_Block1_BN2": (13, 6),
    "Layer3_Block2_Conv1": (10, 4),
    "Layer3_Block2_BN1": (11, 4),
    "Layer3_Block2_Conv2": (12, 4),
    "Layer3_Block2_BN2": (13, 4),
    "Layer4_Block1_Conv1": (14, 6),
    "Layer4_Block1_BN1": (15, 6),
    "Layer4_Block1_Conv2": (16, 6),
    "Layer4_Block1_BN2": (17, 6),
    "Layer4_Block2_Conv1": (14, 4),
    "Layer4_Block2_BN1": (15, 4),
    "Layer4_Block2_Conv2": (16, 4),
    "Layer4_Block2_BN2": (17, 4),
    "FC": (18, 5),
    "Output": (19, 5),
}

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
plt.title("ResNet Architecture Visualization")
plt.show()