/**
 * Advanced C++ Data Structures and Algorithms
 * 
 * This module demonstrates modern C++ features including:
 * - Smart pointers and memory management
 * - Template metaprogramming
 * - STL containers and algorithms
 * - Lambda expressions and functional programming
 */

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <string>
#include <queue>
#include <stack>

// ============================================================================
// SMART POINTERS AND MEMORY MANAGEMENT
// ============================================================================

template<typename T>
class SmartPointer {
private:
    std::unique_ptr<T> ptr;
    size_t ref_count;

public:
    explicit SmartPointer(T* p) : ptr(p), ref_count(1) {}
    
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr.get(); }
    
    size_t getRefCount() const { return ref_count; }
};

// ============================================================================
// TEMPLATE DATA STRUCTURES
// ============================================================================

template<typename T>
class BinarySearchTree {
private:
    struct Node {
        T data;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        Node(const T& val) : data(val), left(nullptr), right(nullptr) {}
    };
    
    std::unique_ptr<Node> root;
    
    void insertHelper(std::unique_ptr<Node>& node, const T& value) {
        if (!node) {
            node = std::make_unique<Node>(value);
        } else if (value < node->data) {
            insertHelper(node->left, value);
        } else {
            insertHelper(node->right, value);
        }
    }
    
    bool searchHelper(const std::unique_ptr<Node>& node, const T& value) const {
        if (!node) return false;
        if (value == node->data) return true;
        if (value < node->data) return searchHelper(node->left, value);
        return searchHelper(node->right, value);
    }
    
    void inorderHelper(const std::unique_ptr<Node>& node, std::vector<T>& result) const {
        if (!node) return;
        inorderHelper(node->left, result);
        result.push_back(node->data);
        inorderHelper(node->right, result);
    }
    
public:
    void insert(const T& value) {
        insertHelper(root, value);
    }
    
    bool search(const T& value) const {
        return searchHelper(root, value);
    }
    
    std::vector<T> inorder() const {
        std::vector<T> result;
        inorderHelper(root, result);
        return result;
    }
};

// ============================================================================
// HASH MAP IMPLEMENTATION
// ============================================================================

template<typename K, typename V>
class HashMap {
private:
    struct Bucket {
        K key;
        V value;
        std::shared_ptr<Bucket> next;
        
        Bucket(const K& k, const V& v) : key(k), value(v), next(nullptr) {}
    };
    
    std::vector<std::shared_ptr<Bucket>> buckets;
    size_t capacity;
    size_t size;
    
    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % capacity;
    }
    
    void resize() {
        auto old_buckets = std::move(buckets);
        capacity *= 2;
        buckets.resize(capacity);
        size = 0;
        
        for (const auto& bucket : old_buckets) {
            auto current = bucket;
            while (current) {
                put(current->key, current->value);
                current = current->next;
            }
        }
    }
    
public:
    HashMap(size_t initial_capacity = 16) : capacity(initial_capacity), size(0) {
        buckets.resize(capacity);
    }
    
    void put(const K& key, const V& value) {
        if (size > capacity * 0.75) {
            resize();
        }
        
        size_t index = hash(key);
        auto current = buckets[index];
        
        while (current) {
            if (current->key == key) {
                current->value = value;
                return;
            }
            current = current->next;
        }
        
        auto new_bucket = std::make_shared<Bucket>(key, value);
        new_bucket->next = buckets[index];
        buckets[index] = new_bucket;
        size++;
    }
    
    bool get(const K& key, V& value) const {
        size_t index = hash(key);
        auto current = buckets[index];
        
        while (current) {
            if (current->key == key) {
                value = current->value;
                return true;
            }
            current = current->next;
        }
        
        return false;
    }
    
    bool remove(const K& key) {
        size_t index = hash(key);
        auto& current = buckets[index];
        
        if (!current) return false;
        
        if (current->key == key) {
            current = current->next;
            size--;
            return true;
        }
        
        while (current->next) {
            if (current->next->key == key) {
                current->next = current->next->next;
                size--;
                return true;
            }
            current = current->next;
        }
        
        return false;
    }
    
    size_t getSize() const { return size; }
};

// ============================================================================
// GRAPH ALGORITHMS
// ============================================================================

template<typename T>
class Graph {
private:
    std::unordered_map<T, std::vector<T>> adjacency_list;
    
public:
    void addEdge(const T& from, const T& to) {
        adjacency_list[from].push_back(to);
        adjacency_list[to]; // Ensure 'to' exists in the map
    }
    
    std::vector<T> bfs(const T& start) const {
        std::vector<T> result;
        std::queue<T> queue;
        std::unordered_set<T> visited;
        
        queue.push(start);
        visited.insert(start);
        
        while (!queue.empty()) {
            T current = queue.front();
            queue.pop();
            result.push_back(current);
            
            auto it = adjacency_list.find(current);
            if (it != adjacency_list.end()) {
                for (const auto& neighbor : it->second) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        return result;
    }
    
    std::vector<T> dfs(const T& start) const {
        std::vector<T> result;
        std::stack<T> stack;
        std::unordered_set<T> visited;
        
        stack.push(start);
        
        while (!stack.empty()) {
            T current = stack.top();
            stack.pop();
            
            if (visited.find(current) == visited.end()) {
                visited.insert(current);
                result.push_back(current);
                
                auto it = adjacency_list.find(current);
                if (it != adjacency_list.end()) {
                    for (const auto& neighbor : it->second) {
                        if (visited.find(neighbor) == visited.end()) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
        
        return result;
    }
};

// ============================================================================
// LAMBDA EXPRESSIONS AND FUNCTIONAL PROGRAMMING
// ============================================================================

class FunctionalUtils {
public:
    template<typename T, typename F>
    static std::vector<T> filter(const std::vector<T>& vec, F predicate) {
        std::vector<T> result;
        std::copy_if(vec.begin(), vec.end(), std::back_inserter(result), predicate);
        return result;
    }
    
    template<typename T, typename F>
    static auto map(const std::vector<T>& vec, F transform) {
        using ResultType = decltype(transform(vec[0]));
        std::vector<ResultType> result;
        result.reserve(vec.size());
        
        for (const auto& item : vec) {
            result.push_back(transform(item));
        }
        
        return result;
    }
    
    template<typename T, typename F>
    static T reduce(const std::vector<T>& vec, T initial, F accumulator) {
        T result = initial;
        for (const auto& item : vec) {
            result = accumulator(result, item);
        }
        return result;
    }
};

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

int main() {
    std::cout << "=== C++ Data Structures and Algorithms Demo ===" << std::endl;
    
    // Binary Search Tree
    std::cout << "\n--- Binary Search Tree ---" << std::endl;
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.insert(60);
    bst.insert(80);
    
    auto inorder = bst.inorder();
    std::cout << "Inorder traversal: ";
    for (int val : inorder) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Search 40: " << (bst.search(40) ? "Found" : "Not found") << std::endl;
    std::cout << "Search 100: " << (bst.search(100) ? "Found" : "Not found") << std::endl;
    
    // Hash Map
    std::cout << "\n--- Hash Map ---" << std::endl;
    HashMap<std::string, int> hashmap;
    hashmap.put("apple", 5);
    hashmap.put("banana", 3);
    hashmap.put("orange", 7);
    
    int value;
    if (hashmap.get("apple", value)) {
        std::cout << "apple: " << value << std::endl;
    }
    
    hashmap.put("apple", 10);
    if (hashmap.get("apple", value)) {
        std::cout << "apple (updated): " << value << std::endl;
    }
    
    std::cout << "Hash map size: " << hashmap.getSize() << std::endl;
    
    // Graph
    std::cout << "\n--- Graph Algorithms ---" << std::endl;
    Graph<int> graph;
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 4);
    graph.addEdge(3, 5);
    graph.addEdge(4, 5);
    
    auto bfs_result = graph.bfs(0);
    std::cout << "BFS from 0: ";
    for (int node : bfs_result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    
    auto dfs_result = graph.dfs(0);
    std::cout << "DFS from 0: ";
    for (int node : dfs_result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    
    // Functional Programming
    std::cout << "\n--- Functional Programming ---" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto even_numbers = FunctionalUtils::filter(numbers, [](int n) {
        return n % 2 == 0;
    });
    
    std::cout << "Even numbers: ";
    for (int n : even_numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto squared = FunctionalUtils::map(numbers, [](int n) {
        return n * n;
    });
    
    std::cout << "Squared: ";
    for (int n : squared) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto sum = FunctionalUtils::reduce(numbers, 0, [](int acc, int n) {
        return acc + n;
    });
    
    std::cout << "Sum: " << sum << std::endl;
    
    return 0;
}
