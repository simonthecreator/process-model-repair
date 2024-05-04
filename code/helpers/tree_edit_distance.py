from zss import simple_distance, Node
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py import convert_to_process_tree

def pt_to_zss_tree(tree: ProcessTree):
    node_name = tree.operator.value if tree.operator else tree.label
    root_node = Node(node_name)
    for child in tree.children:
        root_node.addkid(pt_to_zss_tree(child))
    return (root_node)

def get_edit_distance(node1: Node, node2: Node):
    return simple_distance(node1, node2)

def editdist_zss_tree_petri(zss_tree: Node, pn: PetriNet, im: Marking, fm: Marking):
    pt = convert_to_process_tree(pn, im, fm)
    zss_tree_from_pt = pt_to_zss_tree(pt)
    return get_edit_distance(zss_tree, zss_tree_from_pt)