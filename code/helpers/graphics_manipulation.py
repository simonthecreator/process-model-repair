import pydotplus
from pydotplus.graphviz import Dot, Node
from pm4py.visualization.petri_net import visualizer
from pm4py.objects.petri_net.obj import PetriNet, Marking
from IPython.display import Image
import tempfile
from graphviz import Digraph
from typing import Union

def pn_to_dot(pn: PetriNet) -> Dot:
    """PetriNet object to Dot which is a editable graphical representation"""
    gviz = visualizer.apply(pn)
    return gviz_to_dot(gviz)

def gviz_to_dot(gviz) -> Dot:
    gviz_dot = gviz.source
    dot = pydotplus.graph_from_dot_data(gviz_dot)
    return dot

def transform_petri_to_dot_if_it_is_not_already(pn_new: Union[PetriNet, Digraph]) -> Dot:
    if type(pn_new) is PetriNet:
        dot_new = pn_to_dot(pn_new)
    else: # Digraph is already Dot
        dot_new = gviz_to_dot(pn_new)
    return dot_new

def get_node_by_name(dot: Dot, name: str) -> Node:
    for node in dot.get_node_list():
        if node.get_name() == name:
            return node

def set_fillcolor_transition_by_label(pn: Union[PetriNet, Digraph], label: str, color_code = '#33C4FF'):
    """Set fillcolor of transition that is identified by label. Default color is light blue."""

    dot = transform_petri_to_dot_if_it_is_not_already(pn)

    for node in dot.get_node_list():

        node_label = node.get_label()

        if node_label:

            # node we are looking for
            if label == node_label:
                node.set_fillcolor(color_code)

            elif 'tau' in node_label:
                # transitions that are in the original net already to black
                node.set_fillcolor('#000000')
                # remove label from silent transitions to make net replayable here too
                node.set_label('')

        else: # assume that objects with no label are silent transitions
            # transitions that are in the original net already to black
            node.set_fillcolor('#000000')
            # remove label from silent transitions to make net replayable here too
            node.set_label('')            

    return dot

def mark_new_parts_of_pn(pn_new: Union[PetriNet, Digraph], pn_old, color_code = '#33C4FF', mark_inserted_places = False):
    """Color parts that are in pn_new but not in pn_old with color specified in color_code.

    Args:
        pn_new (PetriNet): new petri net with added parts
        pn_old (PetriNet): old (original) petri net
        color_code (str, optional): _description_. Defaults to '#33C4FF' (light blue).

    Returns:
        Dot: Graphical representation for illustration only, not for replay, etc.
    """
    dot_new = transform_petri_to_dot_if_it_is_not_already(pn_new)

    dot_old = pn_to_dot(pn_old)

    dot_old_node_labels = [node.get_label() for node in dot_old.get_node_list()]

    if mark_inserted_places:
        # iterate over edges to change fillcolor for places that have been inserted
        # places themselves have no label and cannot be identified in the Dot object via the name
        # because the name is randomly given in graphviz objects
        for e in dot_new.get_edge_list():

            # get source and destination of each edge
            src_name = e.get_source()
            dst_name = e.get_destination()
            src_node = get_node_by_name(dot_new, src_name)
            dst_node = get_node_by_name(dot_new, dst_name)

            if '_sub' in dst_node.get_label():
                # change fillcolor of source place, if the destination transition is part of a sub-process that has been added in 'avoid-flower' style
                src_node.set_fillcolor(color_code)

    for node in dot_new.get_node_list():

        label = node.get_label()
        if label:
            # ignore places that are in old dot
            if label == '\"⬤\"' or label == '\"■\"':
                continue

        if label not in dot_old_node_labels:
            #print("not in old labels")
            node.set_fillcolor(color_code)

            # remove '_sub' which has been added to distinguish new transitions from old that have same label
            if '_sub' in label:
                node.set_label(label[1:-5])

        if label:
            if 'τ' in label:
                # remove label from silent transitions to make net replayable (graphically). The net that is created here will not be used for replay
                node.set_label('')
            elif any([keyword in label for keyword in ['tau', 'skip']]):
                # transitions that are in the original net already to black
                node.set_fillcolor('#000000')
                # remove label from silent transitions to make net replayable here too
                node.set_label('')

    return dot_new

def update_process_tree_node_labels(pt_dot: Dot) -> Dot:
    for node in pt_dot.get_node_list():
        label = node.get_label()
        if label == 'xor':
            node.set_label('×')
        if label == 'seq':
            node.set_label('➔')
        if label == 'and':
            node.set_label('∧')
        if label == "\"xor loop\"":
            node.set_label('⥀')
        if label == 'or':
            node.set_label('∨')
    return pt_dot

# The following function is used to add labels to places in petri_nets for grpahical representation
# Source: Beispiel 49 in https://python.hotexamples.com/de/examples/graphviz/Digraph/attr/python-digraph-attr-method-examples.html?utm_content=cmp-true
def graphviz_visualization(net,
                           initial_marking=None,
                           final_marking=None,
                           image_format="png",
                           decorations=None,
                           debug=False,
                           set_rankdir=None):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if decorations is None:
        decorations = {}

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(net.name,
                  filename=filename.name,
                  engine='dot',
                  graph_attr={'bgcolor': 'white'})
    if set_rankdir:
        viz.graph_attr['rankdir'] = set_rankdir
    else:
        viz.graph_attr['rankdir'] = 'LR'

    # transitions
    viz.attr('node', shape='box')
    # add transitions, in order by their (unique) name, to avoid undeterminism in the visualization
    trans_sort_list = sorted(list(net.transitions),
                             key=lambda x:
                             (x.label
                              if x.label is not None else "tau", x.name))
    for t in trans_sort_list:
        decorations[t] = {'label': t.label,
                          'color': 'white'}
        if t.label is not None:
            if t in decorations and "label" in decorations[
                    t] and "color" in decorations[t]:
                viz.node(str(id(t)),
                         decorations[t]["label"],
                         style='filled',
                         fillcolor=decorations[t]["color"],
                         border='1')
            else:
                viz.node(str(id(t)), str(t.label))
        else:
            if debug:
                viz.node(str(id(t)), str(t.name))
            elif t in decorations and "color" in decorations[
                    t] and "label" in decorations[t]:
                viz.node(str(id(t)),
                         decorations[t]["label"],
                         style='filled',
                         fillcolor=decorations[t]["color"],
                         fontsize='8')
            else:
                viz.node(str(id(t)), "", style='filled', fillcolor="black")

    # places
    viz.attr('node', shape='circle', fixedsize='true', width='0.75')
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted(
        [x for x in list(net.places) if x in initial_marking],
        key=lambda x: x.name)
    places_sort_list_fm = sorted([
        x for x in list(net.places)
        if x in final_marking and not x in initial_marking
    ],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted([
        x for x in list(net.places)
        if x not in initial_marking and x not in final_marking
    ],
                                        key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm
    
    counter = 1
    for p in places_sort_list:
        decorations[p] = {'label': 'p'+str(counter),
                          'color': 'white'}
        if p in initial_marking:
            viz.node(str(id(p)),
                     '⬤',
                     style='filled',
                     fillcolor="white",
                     fontsize='26')
        elif p in final_marking:
            viz.node(str(id(p)), '■', style='filled', fontsize='26', fillcolor="white", shape='doublecircle')
        else:
            if debug:
                viz.node(str(id(p)), str(p.name))
            else:
                if p in decorations and "color" in decorations[
                        p] and "label" in decorations[p]:
                    viz.node(str(id(p)),
                             decorations[p]["label"],
                             style='filled',
                             fillcolor=decorations[p]["color"],
                             fontsize='14')
                else:
                    viz.node(str(id(p)), "")
                
                counter += 1

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs),
                            key=lambda x: (x.source.name, x.target.name))
    for a in arcs_sort_list:
        if a in decorations and "label" in decorations[
                a] and "penwidth" in decorations[a]:
            viz.edge(str(id(a.source)),
                     str(id(a.target)),
                     label=decorations[a]["label"],
                     penwidth=decorations[a]["penwidth"])
        elif a in decorations and "color" in decorations[a]:
            viz.edge(str(id(a.source)),
                     str(id(a.target)),
                     color=decorations[a]["color"])
        else:
            viz.edge(str(id(a.source)), str(id(a.target)))
    viz.attr(overlap='false')
    viz.attr(fontsize='11')
    viz.attr(bgcolor='white')

    viz.format = image_format

    return viz

def print_png(dot: Dot):
    """Print image as png. Can be used with display() in a Jupyter Notebook"""
    return Image(dot.create_png())