from collections import Counter
from . import helpers

def log_moves_to_subtrace_list(log_move_locations_per_trace: dict, source_place_name: str):
    """Transform list of moves to list of activities (names)

    Args:
        log_move_locations_per_trace (dict): dict with location as key and another dict with trace_id as key and list of moves as value
        source_place_name (str): name of place to list moves for (alignments for this place)

    Returns:
        list: list of activity names (instead of whole log move)
    """
    # get list of moves for each trace
    subtraces = list(log_move_locations_per_trace[source_place_name].values())
    output_list = list()
    for subtrace in subtraces:
        # append activity name for each log move
        output_list.append([move[0] for move in subtrace])
    return output_list

def subtraces_list_to_dict(subtraces_list):
  subtraces_len_dict = {}
  for elem in subtraces_list:
      len_elem = len(elem)
      if len_elem not in subtraces_len_dict.keys():
          subtraces_len_dict[len_elem] = list()
      subtraces_len_dict[len_elem].append(elem)

  return subtraces_len_dict

def lists_share_pct_elements(l1: list, l2: list, pct: float = 0.5):
  # get the intersection of the two lists
  intersection = list(set(l1) & set(l2))
  # get the percentage of the intersection
  intersection_pct_1 = len(intersection) / len(l1)
  intersection_pct_2 = len(intersection) / len(l2)
  return intersection_pct_1 >= pct and intersection_pct_2 >= pct

def cluster_subtraces(subtr: list):
    """Cluster traces together that share a certain percentage of mutual elements

    Args:
        subtr (list): list of subtraces to be clustered

    Returns:
        list: list of clustered subtraces
    """
    indices_clustered = set()  # Set to store the indices that have already been clustered
    return_list = []  # List to store the final clustered subtr lists

    for i, elem in enumerate(subtr):
        if i not in indices_clustered:
            subtr_cluster = [elem]  # Start a new cluster with the current element
            for j, sub_elem in enumerate(subtr[i:], start=i):
                if i != j and j not in indices_clustered:
                    elem_flattened = helpers.flatten_list(elem)
                    if lists_share_pct_elements(elem_flattened, sub_elem):  # Check if the elements share a certain condition
                        subtr_cluster.append(sub_elem)  # Add the sub_elem to the current cluster
                        indices_clustered.add(j)  # Mark the index as clustered
            indices_clustered.add(i)  # Mark the current index as clustered
            if subtr_cluster not in return_list:
                return_list.append(subtr_cluster)  # Add the cluster to the final return list
    return return_list

def align_subtraces(subtraces: list):
    """decompose and cluster subtraces

    Args:
        subtraces (list): list of subtraces

    Returns:
        list: list of aligned (i.e. decomposed and clustered) subtraces (list of lists)
    """
    subtraces_len_dict = subtraces_list_to_dict(subtraces_list=subtraces)
    for i in range(0, 100).__reversed__():
        if i in subtraces_len_dict.keys():
            for beta in subtraces_len_dict[i]:
                beta_str = ','.join(beta)
                # iterate over all subtraces that come after (and are shorter than) beta
                for k in range(0, len(subtraces)):
                    beta_1 = subtraces[k]
                    beta_1_str = ','.join(beta_1)
                    # if the shorter subtrace is a subset of the longer subtrace
                    if (beta_1_str in beta_str) and (beta_1_str!=beta_str) and not lists_share_pct_elements(beta, beta_1):
        
                        # remove the longer subtrace from the list
                        subtraces.remove(beta)
                        # add the decomposed subtrace to the list
                        components = beta_str.split(beta_1_str)
                        prefix = components[0].strip(',')
                        suffix = components[1].strip(',')
                        subtraces.append(beta_1)
                        # add the prefix and suffix to the list
                        if prefix != '':
                            subtraces.append(prefix.split(','))
                        if suffix != '':
                            subtraces.append(suffix.split(','))
                        break # if we find a match, we can stop looking for more matches
            subtraces.sort(key=len, reverse=True)
            subtraces_len_dict = subtraces_list_to_dict(subtraces_list=subtraces)
    subtraces = helpers.deduplicate_list_of_lists(subtraces)
    #print(f"align_subtraces: {subtraces}")
    subtraces = cluster_subtraces(subtraces)
    return subtraces

######## Helper functions ########

def get_element_counts_list_of_lists(list_of_lists):
    # flatten the list of lists
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    # get the counts of each element
    element_counts = Counter(flattened_list)
    return element_counts

def count_element_in_list_of_lists(element, list_of_lists):
    element_counts = get_element_counts_list_of_lists(list_of_lists)
    return element_counts[element]

def find_most_frequent_element(list_of_lists):

    element_counts = get_element_counts_list_of_lists(list_of_lists)
    
    # Find the most frequent element(s)
    most_frequent_elements = element_counts.most_common(1)
    
    return most_frequent_elements[0][0]

def find_common_elements(lists):
    common_elements = set(lists[0])  # Start with the elements from the first list
    
    # Perform set intersection with the remaining lists
    for lst in lists[1:]:
        common_elements = common_elements.intersection(lst)
    
    return list(common_elements)

######## End Helper functions ########

def log_moves_to_sublogs_dict(log_move_locations_per_trace: dict):
    """Aligns (decompose and cluster) subtraces for each location (i.e. each key in the dict

    Args:
        log_move_locations_per_trace (dict): dict with location as key and another dict with trace_id as key and list of moves as value

    Returns:
        dict: aligned subtraces for each location
    """
    sublogs_dict = {}
    for loc in log_move_locations_per_trace.keys():
        # get list of subtraces for location
        subtraces_list = log_moves_to_subtrace_list(log_move_locations_per_trace, loc)
        # sort by length to check longer subtraces first. They are more likely to be decomposed 
        subtraces_list.sort(key=len, reverse=True)
        subtr = align_subtraces(subtraces_list)
        sublogs_dict[loc] = subtr
    return sublogs_dict

def group_into_sublogs(sublogs_dict: dict):
    """Find mutual places among subtraces and group them into sublogs (i.e. subtraces that share a certain place)

    Args:
        sublogs_dict (dict): aligned subtrace per location

    Returns:
        dict: grouped subtraces per location
    """
    sublogs_new = dict()
    while len(sublogs_dict) > 0:
        
        # Find the most frequent place among all subtraces
        all_markings = list()
        for key in sublogs_dict.keys():
            places_list = eval(key)
            places_list_without_marking = [p_wo_marking.split(':')[0] for p_wo_marking in places_list]
            places_list_without_marking
            all_markings.append(places_list_without_marking)
        most_frequent_place = find_most_frequent_element(all_markings)

        # Find all subtraces that contain the most frequent place
        loc_to_be_added_to_sublogs = list()
        subtraces_to_be_added_to_sublogs = list()
        # Iterate over all subtraces
        for key, value in sublogs_dict.items():
            if most_frequent_place in key:
                loc_as_list = eval(key)
                loc_to_be_added_to_sublogs.append(loc_as_list)
                subtraces_to_be_added_to_sublogs.append(value)
        # Remove the subtrace from the sublogs_dict
        for loc in loc_to_be_added_to_sublogs:
            del sublogs_dict[str(loc)]
        # Find the common places among the subtraces
        common_places = find_common_elements(loc_to_be_added_to_sublogs)
        # Add the subtraces to the sublogs
        sublogs_new[str(common_places)] = helpers.flatten_list(subtraces_to_be_added_to_sublogs)
    return sublogs_new

def pick_relevant_locations(sublogs_dict, location_list_list):
    """Locations (which are here the keys of sublogs_dict), can consist of multiple places.
       This function keeps only places in locations that are most frequent among the whole sublogs.
       If there are multiple places that are equally frequent, then all of them are kept.
       Refers to chapter `5.5. Improving the placement of a subprocess` in Paper *Model repair — aligning process models to reality* by Fahland and van der Aalst

    Args:
        sublogs_dict (dict): _description_
        location_list_list (list): _description_

    Returns:
        dict: sublogs_dict with (possibly) edited locations (keys)
    """
    # set to add edited locs to bc we can't edit a dict while iterating over it
    locs_edited = set()
    # iterate over all sublogs
    for loc_str in sublogs_dict.keys():
        # convert loc_str to list
        loc_list = eval(loc_str)

        # Find the most frequent place among all subtraces
        loc_count = [count_element_in_list_of_lists(loc, location_list_list) for loc in loc_list]
        if not loc_count:
            # if loc_count is an empty list
            continue
        loc_count_max = max(loc_count)
        
        # if the current place in the location is not (one of) the most frequent, then remove it from the location (i.e. reduce the location)
        # add places that are not most frequent to list `to_be_deleted`
        to_be_deleted = list()
        for loc_c in loc_count:
            if loc_c != loc_count_max:
                to_be_deleted.append(loc_list[loc_count.index(loc_c)])         

        # delete not-most-frequent from loc_list
        for elem in to_be_deleted:
            if elem in loc_list:
                loc_list.remove(elem)

        # add tuple of loc_list (which might have been edited) and the original loc_str       
        locs_edited.add((str(loc_list), loc_str))

    for loc_str_edited, loc_unedited in locs_edited:
        # if loc has been edited, update the dict key
        if loc_str_edited != loc_unedited:
            # overwrite the old key with the new key
            sublogs_dict[loc_str_edited] = sublogs_dict.pop(loc_unedited)
    return sublogs_dict