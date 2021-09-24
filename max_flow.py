from queue import Queue, PriorityQueue
from typing import Optional, Union, List, Tuple, Dict, Set, NewType
import numpy as np


def maxflow(
            arcs: List[Tuple[int, int, Union[int, float]]],
            v: int,
            flow: Optional[List[Tuple[int, int, Union[int, float]]]] = None,
            find_mincut: bool = True
) -> Union[Tuple[int, np.ndarray], Tuple[int, np.ndarray, List[int]]]:
    """
    Computes the maximum flow of the graph.

    Parameters
    ----------
    arcs: List[Tuple[int, int, Union[int, float]]]
        numpy array [v,3] - [[node1, node2, capacity]]
    v: int
        number of nodes (vertices)
    flow: Optional[np.ndarray]
        correct sub-optimal (optional) flow [v,3] - [[node1, node2, flow]]
    find_mincut: bool = False
        bool value that determines whether mincut should be returned

    Returns the value of the maximum flow, the flow as the edge list graph
    and (optionally) the minimum cut.
    """

    Node = NewType('Node', int)
    Num = NewType('Num', Union[int, float])

    class ArcCapacity:
        """
        Container for storing the node's arc.

        Stores the arc's head node, its capacity and reference to paired arc.
        """

        def __init__(self, capacity: Num) -> None:
            self.capacity: Num = capacity
            self.paired_arc: Optional[ArcCapacity] = None


    class NodeArcs:
        """Container for storing the graph adjacency list."""

        def __init__(self) -> None:
            self.arcs: Dict[int, ArcCapacity] = dict()

        # get: arc[node] -> capacity
        def __getitem__(self, node: Node) -> Optional[int]:
            arc = self.arcs.get(node)
            if arc is not None:
                return arc.capacity
            return None

        # set: arc[node] = capacity
        def __setitem__(self, node: Node, capacity: int) -> None:
            arc = self.arcs.get(node)
            if arc is not None:
                arc.capacity = capacity
                return
            raise ValueError("Atempt to set capacity of non-existent arc")

        def get_arc(self, node: Node) -> Optional[ArcCapacity]:
            return self.arcs.get(node)

        def append(self, node: Node, capacity: int) -> None:
            self.arcs[node] = ArcCapacity(capacity)

        def __str__(self) -> str:
            if len(self.arcs) == 0:
                return "[]"

            string: str = "["
            for node, arc in self.arcs.items():
                string += f'({node}: {arc.capacity}), '
            return string[:-2] + "]"


    def translate_from_arcs() -> None:
        """Transforms the edge list graph representation to the adjacency list."""

        for arc in arcs:
            sought_arc: Optional[ArcCapacity] = c[arc[0]].get_arc(arc[1])

            if sought_arc is None:
                c[arc[0]].append(arc[1], arc[2])
                sought_arc = c[arc[0]].get_arc(arc[1])

                c[arc[1]].append(arc[0], 0.0)
                paired_arc = c[arc[1]].get_arc(arc[0])

                sought_arc.paired_arc = paired_arc
                paired_arc.paired_arc = sought_arc
            else:
                sought_arc.capacity = arc[2]


    def residual_from_flow() -> None:
        """Computes the resideal graph from the given flow and returns the flow value."""
        
        for arc in flow:
            sought_arc: ArcCapacity = c[arc[0]].get_arc(arc[1])
            if sought_arc.capacity - arc[2] < 0 :
                print("BADBADBAD", arc[0], arc[1], sought_arc.capacity, arc[2])
            sought_arc.capacity -= arc[2]
            sought_arc.paired_arc.capacity += arc[2]

            if arc[1] == sink:
                x[sink] += arc[2]


    _M: int = len(arcs) + v
    _gb_counter: int = _M - 1

    def global_relabeling() -> None:
        """Invokes global relabeling each M calls."""

        def height_as_distance(from_node: Node):
            nodes_to_check: 'Queue[Node]' = Queue()
            nodes_to_check.put(from_node)

            while not nodes_to_check.empty():
                node = nodes_to_check.get()

                for arcnode, arc in c[node].arcs.items():
                    if arc.paired_arc.capacity > 0 and not nodes_checked[arcnode]:
                        h[arcnode] = h[node] + 1
                        nodes_checked[arcnode] = True
                        nodes_to_check.put(arcnode)


        nonlocal _gb_counter

        _gb_counter += 1
        if _gb_counter == _M:
            _gb_counter = 0

            nodes_checked: List[bool] = [True] + [False for _ in range(v - 2)] + [True]
            height_as_distance(sink)
            height_as_distance(source)
            # print(h)


    def mincut() -> List[Node]:
        """Returns the minimum cut of the graph."""

        mincut_nodes: Set[Node] = set([source])
        nodes_to_check: 'Queue[Node]' = Queue()
        nodes_to_check.put(source)

        while not nodes_to_check.empty():
            node = nodes_to_check.get()

            for arcnode, arc in c[node].arcs.items():
                if arc.capacity > 0 and arcnode not in mincut_nodes:
                    mincut_nodes.add(arcnode)
                    nodes_to_check.put(arcnode)

        return list(mincut_nodes)


    def push(node: Node, arcnode: Node, arc: ArcCapacity) -> None:
        """The maximum flow push operation."""

        delta = min(x[node], arc.capacity)
        x[node] -= delta
        x[arcnode] += delta
        arc.capacity -= delta
        arc.paired_arc.capacity += delta

        # print(f"push: {node} --{delta}-> {arcnode}")


    source: Node = 0
    sink: Node = v - 1

    # high level optimization
    node_queue: 'PriorityQueue[Node]' = PriorityQueue()
    q: List[bool] = [False for _ in range(v)] # is node in queue

    def push_node_to_queue(node: Node) -> None:
        if not q[node]:
            node_queue.put((h[node], node))
            q[node] = True

    def pop_node_from_queue() -> Node:
        node = node_queue.get()[1]
        q[node] = False
        return node


    # height function
    h: List[int] = [v] + [2 * v for _ in range(v - 2)] + [0]

    # excess function
    x: List[int] = [0.0 for _ in range(v)]

    # residual capacity function
    c: List[NodeArcs] = [NodeArcs() for _ in range(v)]
    translate_from_arcs()

    if flow is not None:
        residual_from_flow()
        # print("\nFLOW:", x[sink])


    # initialization
    for arcnode, arc in c[source].arcs.items():
        x[arcnode] = arc.capacity
        c[arcnode][source] = arc.capacity
        arc.capacity = 0.0

        # fisrt nodes to queue
        node_queue.put((0, arcnode))
        q[arcnode] = True

    global_relabeling()


    # main
    while not node_queue.empty(): # while there are some active nodes
        node = pop_node_from_queue()
        if node in (source, sink):
            continue

        # print(f"\nACTIVE: {node}")

        while x[node] != 0.0:
            height = 2 * v # min height for relabeling
            # print("\n", x[node], c[node])

            for arcnode, arc in c[node].arcs.items():
                if arc.capacity != 0: # if the arc is admissible
                    # pushing
                    if h[node] == h[arcnode] + 1:
                        push(node, arcnode, arc)
                        push_node_to_queue(arcnode)
                        global_relabeling()
                        if x[node] == 0.0:
                            break
                    # finding relabeling height
                    else:
                        # print(f"height update: n {arc.node} h {h[arc.node]}")
                        height = min(height, h[arcnode])

            #relabeling
            if x[node] != 0.0:
                # print(f"relabel (ex: {x[node]}): {h[node]} -> {height + 1}")#
                h[node] = height + 1
                global_relabeling()


    f = [[0, 0, 0.0] for i in range(len(arcs))]
    for i in range(len(f)):
        f[i][0] = arcs[i][0]
        f[i][1] = arcs[i][1]
        f[i][2] = arcs[i][2] - c[arcs[i][0]][arcs[i][1]]
        if f[i][2] < 0:
            f[i][2] = 0.0

    # print("\nThe residual graph: node -> arc(node, capacity)")
    # for i in range(v):
    #     print(i, "->", c[i])
    # print("\nMAXFLOW:", x[sink])
    # print(f, end="\n\n")
    # print("Mincut:", mincut())

    if find_mincut:
        return x[sink], f, mincut()
    else:
        return x[sink], f


if __name__ == "__main__":
    val, flow, mincut = maxflow([
        [0, 1, 1], [0, 2, 12.1], [1, 2, 2.1], [2, 1, 0],
        [1, 3, 10.1], [3, 2, 5.1], [2, 4, 14], [4, 3, 7],
        [4, 5, 2.1], [3, 5, 15.1],
        ], 6)

    val, flow, mincut = maxflow([
        [0, 1, 16.1], [0, 2, 13.1], [1, 2, 10.1], [2, 1, 4.2],
        [1, 3, 12.1], [3, 2, 9.2], [2, 4, 14.1], [4, 3, 7.4],
        [4, 5, 4.1], [3, 5, 20.4],
        ], 6, flow)
    
    val, flow, mincut = maxflow([
        [0, 1, 16.1], [0, 2, 13.1], [1, 2, 10.1], [2, 1, 4.2],
        [1, 3, 12.1], [3, 2, 9.2], [2, 4, 14.1], [4, 3, 7.4],
        [4, 5, 4.1], [3, 5, 20.4],
        ], 6, flow)

    maxflow(np.array([
        [0, 1, 3], [1, 2, 4], [2, 5, 2], [0, 3, 5],
        [3, 5, 7], [0, 4, 2], [4, 3, 3], [4, 5, 1]
        ]), 6)

    maxflow(np.array([
        [0, 1, 3], [0, 2, 3], [2, 1, 10], [1, 4, 2],
        [2, 4, 1], [4, 6, 2], [4, 3, 1], [0, 3, 4],
        [3, 5, 5], [4, 5, 1], [5, 6, 5]
        ]), 7)

    maxflow(np.array([
        [0, 1, 10], [0, 2, 8], [2, 1, 4], [1, 2, 5],
        [2, 4, 10], [1, 3, 5], [3, 2, 7], [4, 3, 10],
        [3, 4, 6], [3, 5, 3], [4, 5, 14]
        ]), 6)
