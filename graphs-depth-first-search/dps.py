#!/usr/bin/env python

from typing import Dict, List, Optional

GraphType = Dict[str, List[str]]

def dfs_filo(graph: GraphType, start: str, end: Optional[str] = None) -> bool:
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        print(node, stack)

        if node == end:
            return True
        
        for neighbor in reversed(graph.get(node, [])):  
            if neighbor not in visited:
                stack.append(neighbor)

    return False

def is_cyclic(graph: GraphType, start: str, end: Optional[str] = None) -> bool:
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            print(node)
            return True

        visited.add(node)
        
        for neighbor in reversed(graph.get(node, [])):  
            if neighbor not in visited:
                stack.append(neighbor)

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

    print("\nIterative DFS (FILO) starting from A:")
    dfs_filo(graph, "A")

    print("\nIterative DFS (FILO) starting from A and ending at F:")
    dfs_filo(graph, "A", "F")
    
    print(f"\nIs the graph cyclic?")
    print(is_cyclic(graph, "A"))
