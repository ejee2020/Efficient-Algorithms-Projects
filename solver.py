import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import random
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""


def dijkstra(G, source, target): #source and target should be vertices(nodes)
    return nx.dijkstra_path_length(G, source, target)


def build_a_graph(adjacency_matrix) : #adjacency_matrix: given input
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    return G


def build_a_dictionary(G, list_of_locations): #G: graph, list_of_locations: given input
    dictionary = {}
    for i in range(len(list_of_locations)):
        for j in range(len(list_of_locations)):
            if i == j:
                dictionary[(list_of_locations[i], list_of_locations[j])] = 0
            else:
                dictionary[(list_of_locations[i], list_of_locations[j])] = dijkstra(G, list_of_locations[i], list_of_locations[j])
    return dictionary


def k_cluster(vertices, dists, k):
  centers = []
  c_1=vertices[random.randint(0,len(vertices)-1)]
  centers.append(c_1)
  while len(centers)<k:
    min_dists={}
    for v in vertices:
      if v in centers:
        continue
      c_dists={}
      for c in centers:
        c_dists[(v,c)]=dists[(v,c)]
      min_dists[v]=min(c_dists.values())
    farthest_v = [k for k, v in min_dists.items() if v == max(min_dists.values())][0]
    centers.append(farthest_v)
  return centers


def build_clusters(dists, list_of_houses, centers):
  clusters={}
  for h in list_of_houses:
    c_dists={}
    for c in centers:
      c_dists[c]=dists[(h,c)]
    closest_c = [k for k, v in c_dists.items() if v == min(c_dists.values())][0]
    if closest_c in clusters:
      clusters[closest_c].append(h)
    else:
      clusters[closest_c]=[h]
  return clusters


def build_complete_graph(starting_location, centers, dists):
    G = nx.Graph()
    vertices = centers.copy()
    vertices.append(starting_location)
    G.add_nodes_from(vertices)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            G.add_edge(vertices[i], vertices[j], weight=dists[(vertices[i], vertices[j])])
    return G


def appr_algorithm(G, starting_point):
    # Find a minimum spanning tree T of G
    T=nx.minimum_spanning_tree(G, weight='weight')

    dfs = nx.dfs_preorder_nodes(T, starting_point);
    listnode = [];
    for item in dfs:
        listnode += [item]

    # Create the hamiltonian tour
    L =nx.Graph()
    L.add_nodes_from(G.nodes(data=True))

    cost = 0
    weight = nx.get_edge_attributes(G,'weight')
    for index, item in enumerate(listnode):
        if index < len(G) - 1:
            L.add_edge(item, listnode[index+1])
            cost += G[item][listnode[index+1]]['weight']
        else:
            L.add_edge(item, listnode[0])
            cost += G[item][listnode[0]]['weight']
    listnode.append(starting_point)
    result = (cost, T, L, listnode)
    return result


def build_car_path(listnode, G):
    car_path=[listnode[0]]
    for i in range(len(listnode)-1):
        shortest_path=nx.shortest_path(G,listnode[i],listnode[i+1])
        car_path+=shortest_path[1:]
    return car_path


def tsp(G, starting_vertex):
    # build a minimum spanning tree
    MSTree = minimum_spanning_tree(G)
#    print("MSTree: ", MSTree)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
#    print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
#    print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G, starting_vertex)

#    print("Eulerian tour: ", eulerian_tour)

    current = eulerian_tour[0]
    path = [current]
    visited={}
    for v in eulerian_tour:
        visited[v]=False
    visited[current]=True
    for v in eulerian_tour:
        if not visited[v]:
            path.append(v)
            visited[v]=True
    path.append(path[0])
    return path
'''
    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]['weight']
            current = v

    path.append(path[0])

'''
#    print("Result path: ", path)
#    print("Result length of the path: ", length)


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)



class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v]['weight'], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u]['weight'] < length:
                length = G[v][u]['weight']
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)


def find_eulerian_tour(MatchedMSTree, G, starting_vertex):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = starting_vertex
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w
    EP=[start_vertex]+EP
    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST


def get_optimal_map(density):
    map1={50: 0.2, 100: 0.5, 200: 0.6}
    map2={50: 0.4, 100: 0.75, 200: 0.5}
    map3={50: 0.8, 100: 0.6, 200: 0.8}
    map4={50: 0.8, 100: 0.75, 200: 0.7}
    if density<0.25:
        return map1
    if density>=0.25 and density<0.5:
        return map2
    if density>=0.5 and density<0.75:
        return map3
    else:
        return map4


def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    indexes_of_locations = convert_locations_to_indices(list_of_locations, list_of_locations)
    indexes_of_houses = convert_locations_to_indices(list_of_homes, list_of_locations)
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    distances = build_a_dictionary(G, indexes_of_locations)
    length=len(list_of_locations)
    if length<=50:
        size=50
    if length>50 and length<=100:
        size=100
    if length>100 and length<=200:
        size=200
    k=get_optimal_map(nx.density(G))[size]*length
    centers = k_cluster(indexes_of_locations, distances, k)
    clusters = build_clusters(distances, indexes_of_houses, centers)
    clusters_invert={}
    for k in clusters:
        houses=clusters[k]
        for h in houses:
            clusters_invert[h]=k
    index_of_start = convert_locations_to_indices([starting_car_location], list_of_locations)[0]
    useful_centers = list(clusters.keys())
    complete_G = build_complete_graph(index_of_start, useful_centers, distances)
#    result = appr_algorithm(complete_G, index_of_start)
    path = tsp(complete_G, index_of_start)
    car_path = build_car_path(path, G)
    for location in car_path:
        if location in indexes_of_houses and location not in clusters:
            center=clusters_invert[location]
            clusters[center].remove(location)
            if len(clusters[center])==0:
                del clusters[center]
            clusters[location]=[location]
    drop_offs=clusters
    return car_path, drop_offs

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
