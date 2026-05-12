"""
Advanced Data Structures in Python

This module implements several advanced data structures including:
- Binary Search Tree (BST)
- Hash Map with separate chaining
- Priority Queue (Min-Heap)
- Graph with adjacency list representation
"""

from typing import Any, List, Optional, Dict, Tuple
from collections import deque
import heapq


class BinarySearchTree:
    """Binary Search Tree implementation with O(log n) average case operations."""

    class Node:
        def __init__(self, value: Any):
            self.value = value
            self.left: Optional['BinarySearchTree.Node'] = None
            self.right: Optional['BinarySearchTree.Node'] = None

    def __init__(self):
        self.root: Optional[BinarySearchTree.Node] = None
        self.size = 0

    def insert(self, value: Any) -> None:
        """Insert a value into the BST. O(log n) average case."""
        if self.root is None:
            self.root = self.Node(value)
            self.size += 1
            return

        self._insert_recursive(self.root, value)

    def _insert_recursive(self, node: Node, value: Any) -> None:
        if value < node.value:
            if node.left is None:
                node.left = self.Node(value)
                self.size += 1
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = self.Node(value)
                self.size += 1
            else:
                self._insert_recursive(node.right, value)

    def search(self, value: Any) -> bool:
        """Search for a value in the BST. O(log n) average case."""
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node: Optional[Node], value: Any) -> bool:
        if node is None:
            return False
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self) -> List[Any]:
        """Return values in sorted order. O(n) time."""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: Optional[Node], result: List[Any]) -> None:
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)


class HashMap:
    """Hash Map implementation with separate chaining for collision resolution."""

    def __init__(self, capacity: int = 16, load_factor: float = 0.75):
        self.capacity = capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(capacity)]

    def _hash(self, key: Any) -> int:
        """Compute hash value for a key."""
        return hash(key) % self.capacity

    def put(self, key: Any, value: Any) -> None:
        """Insert or update a key-value pair. O(1) average case."""
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.size += 1

        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value for a key. O(1) average case."""
        index = self._hash(key)
        bucket = self.buckets[index]

        for k, v in bucket:
            if k == key:
                return v
        return None

    def remove(self, key: Any) -> bool:
        """Remove a key-value pair. O(1) average case."""
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def _resize(self) -> None:
        """Double the capacity and rehash all entries."""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)


class PriorityQueue:
    """Priority Queue implementation using min-heap. O(log n) for push/pop."""

    def __init__(self):
        self.heap: List[Tuple[int, Any]] = []

    def push(self, priority: int, item: Any) -> None:
        """Add an item with priority. O(log n)."""
        heapq.heappush(self.heap, (priority, item))

    def pop(self) -> Optional[Tuple[int, Any]]:
        """Remove and return the highest priority item. O(log n)."""
        if not self.heap:
            return None
        return heapq.heappop(self.heap)

    def peek(self) -> Optional[Tuple[int, Any]]:
        """Return the highest priority item without removing it. O(1)."""
        if not self.heap:
            return None
        return self.heap[0]

    def is_empty(self) -> bool:
        """Check if the priority queue is empty."""
        return len(self.heap) == 0


class Graph:
    """Graph implementation using adjacency list with BFS and DFS."""

    def __init__(self, directed: bool = False):
        self.adjacency_list: Dict[Any, List[Any]] = {}
        self.directed = directed

    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph."""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, vertex1: Any, vertex2: Any) -> None:
        """Add an edge between two vertices."""
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        self.adjacency_list[vertex1].append(vertex2)
        if not self.directed:
            self.adjacency_list[vertex2].append(vertex1)

    def bfs(self, start: Any) -> List[Any]:
        """Breadth-First Search traversal. O(V + E) time."""
        if start not in self.adjacency_list:
            return []

        visited = set()
        queue = deque([start])
        result = []

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)

                for neighbor in self.adjacency_list[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return result

    def dfs(self, start: Any) -> List[Any]:
        """Depth-First Search traversal. O(V + E) time."""
        if start not in self.adjacency_list:
            return []

        visited = set()
        result = []

        def dfs_recursive(vertex: Any):
            if vertex in visited:
                return
            visited.add(vertex)
            result.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                dfs_recursive(neighbor)

        dfs_recursive(start)
        return result

    def shortest_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """Find shortest path using BFS. O(V + E) time."""
        if start not in self.adjacency_list or end not in self.adjacency_list:
            return None

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            vertex, path = queue.popleft()

            if vertex == end:
                return path

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


def main():
    """Demonstrate the data structures."""
    # Binary Search Tree
    bst = BinarySearchTree()
    for value in [5, 3, 7, 1, 4, 6, 8]:
        bst.insert(value)
    print("BST inorder:", bst.inorder_traversal())
    print("BST search 4:", bst.search(4))

    # Hash Map
    hm = HashMap()
    for i in range(10):
        hm.put(f"key{i}", f"value{i}")
    print("HashMap get key5:", hm.get("key5"))

    # Priority Queue
    pq = PriorityQueue()
    for priority, item in [(3, "low"), (1, "high"), (2, "medium")]:
        pq.push(priority, item)
    print("Priority Queue pop:", pq.pop())

    # Graph
    g = Graph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("A", "C")
    print("Graph BFS:", g.bfs("A"))
    print("Graph DFS:", g.dfs("A"))
    print("Graph shortest path A->C:", g.shortest_path("A", "C"))


if __name__ == "__main__":
    main()
