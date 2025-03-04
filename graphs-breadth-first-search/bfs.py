from typing import Dict, List, Optional
from collections import deque

GraphType = Dict[str, List[str]]

def bfs_fifo(graph: GraphType, start: str, end: Optional[str] = None) -> bool:
    queue = deque([start])
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)
        print(node, queue)

        if node == end:
            return True
        
        for neighbor in graph.get(node, []):  
            if neighbor not in visited:
                queue.append(neighbor)

    return False

if __name__ == "__main__":
    graph = {
        "A": ["B", "C", "D"],
        "B": ["A", "H", "E", "D"],
        "C": ["A", "F"],
        "D": ["B", "A", "F"],
        "E": ["B", "F"],
        "F": ["C", "D", "E", "G"],
        "G": ["F"],
        "H": ["B"],
    }

    print("\nIterative BFS (FIFO) starting from A:")
    bfs_fifo(graph, "A")

    print("\nIterative BFS (FIFO) starting from A and ending at F:")
    bfs_fifo(graph, "A", "F")
