import collections

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = collections.defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance

def dijkstra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes: 
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path

def build(width, height, obs):
    g = Graph()

    for y in range(height):
        for x in range(width):
            g.add_node((x, y))

    for y in range(height):
        for x in range(width):
            if not obs[x, y]:
                if x < width - 1 and not obs[x + 1, y]:
                    g.add_edge((x, y), (x + 1, y), 1)
                if y < height - 1 and not obs[x, y + 1]:
                    g.add_edge((x, y), (x, y + 1), 1)
    return g

def get_dist_map(width, height, obs, tx, ty):
    g = build(width, height, obs)
    visited, path = dijkstra(g, (tx, ty))
    return visited
