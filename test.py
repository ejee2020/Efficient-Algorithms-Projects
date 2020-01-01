from student_utils import *
import utils
import random

def dijkstra(G, source, target): #source and target should be vertices(nodes)
    return nx.dijkstra_path_length(G, source, target)

def build_a_graph(adjacency_matrix) : #adjacency_matrix: given input
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    return G

def build_a_dictionary(G, list_of_locations): #G: graph, list_of_locations: given input
    dictionary = {}
    for i in range(len(list_of_locations)):
        for j in range(len(list_of_locations)):
            if (i == j):
                dictionary[(list_of_locations[i], list_of_locations[j])] = 0
            else:
                dictionary[(list_of_locations[i], list_of_locations[j])] = dijkstra(G, list_of_locations[i], list_of_locations[j])
    return dictionary

def k_cluster(vertices, dists, k):
  centers=[]
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
    print("MSTree: ", MSTree)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
    print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
    print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G, starting_vertex)
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


if __name__ == "__main__":
    input_data=utils.read_file("inputs/24_200.in")
    number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix=data_parser(input_data)
    indexes_of_locations=convert_locations_to_indices(list_of_locations,list_of_locations)
    indexes_of_houses=convert_locations_to_indices(list_of_houses,list_of_locations)
#    print("The number of locations: ",number_of_locations)
#    print("The number of houses: ",number_of_houses)
#    print("The list of locations: ", list_of_locations)
#    print("The list of houses: ", list_of_houses)
    G,message=adjacency_matrix_to_graph(adjacency_matrix)
    print("The density of the graph is: ",nx.density(G))
    distances=build_a_dictionary(G,indexes_of_locations)
    centers=k_cluster(indexes_of_locations,distances,160)
    clusters=build_clusters(distances,indexes_of_houses,centers)
#    print("The clusters are: ", clusters)
#    print("The number of clusters are: ", len(clusters))
    index_of_start=convert_locations_to_indices([starting_location],list_of_locations)[0]
    useful_centers=list(clusters.keys())
    complete_G=build_complete_graph(index_of_start,useful_centers,distances)
#    result = appr_algorithm(complete_G, index_of_start)
#    print("The starting_location is: ", starting_location)
#    listnode=result[-1]
#    car_path=build_car_path(listnode, G)
    path=tsp(complete_G, index_of_start)
    car_path=build_car_path(path, G)
    print("The car path is: ", car_path)
    cost, message=cost_of_solution(G, car_path, clusters)
    print(message)