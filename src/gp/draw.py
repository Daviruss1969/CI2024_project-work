from graphviz import Digraph
from IPython.display import Image, display

import gp

def draw_node(node: gp.Node)-> Digraph:
    """Return the graph of a given Node to be rendered"""
    # Create graph
    graph = Digraph(format='png')
    graph.attr('node', shape="circle")

    # Create a graph node
    node_id = str(id(node))
    graph.node(node_id, label=str(node))

    # Add successors recursively
    for _, child in enumerate(node.successors):
        draw_node_worker(child, graph, node_id)

    img_bytes = graph.pipe(format="png")
    img = Image(img_bytes)
    display(img)

    return graph


def draw_node_worker(node: gp.Node, graph: Digraph, parent: str)-> None:
    """Recursive worker of the draw node function, do not call it"""
    # Create a graph node
    node_id = str(id(node))
    graph.node(node_id, label=str(node))

    # Link to parent
    graph.edge(parent, node_id)

    # Add successors recursively
    for _, child in enumerate(node.successors):
        draw_node_worker(child, graph, parent=node_id)