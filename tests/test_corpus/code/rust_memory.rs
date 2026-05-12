// Advanced Memory Management in Rust
//
// This module demonstrates Rust's ownership system and memory management:
// - Ownership and borrowing rules
// - Smart pointers (Box, Rc, Arc, Weak)
// - Memory safety guarantees
// - Zero-cost abstractions
// - Unsafe Rust when needed

use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};

// ============================================================================
// OWNERSHIP AND BORROWING
// ============================================================================

/// Demonstrates ownership transfer
fn ownership_demo() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved, no longer valid
                 // println!("{}", s1); // This would cause a compile error
    println!("s2: {}", s2);
}

/// Demonstrates borrowing
fn borrowing_demo() {
    let s1 = String::from("hello");

    // Immutable borrow - multiple borrows allowed
    let len1 = calculate_length(&s1);
    let len2 = calculate_length(&s1);
    println!("Length: {} (both borrows valid)", len1);

    // Mutable borrow - only one allowed at a time
    let mut s2 = String::from("hello");
    change(&mut s2);
    println!("Modified: {}", s2);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}

// ============================================================================
// SMART POINTERS
// ============================================================================

/// Box<T> - Heap allocation with single ownership
fn box_demo() {
    // Box for heap allocation
    let b = Box::new(5);
    println!("Box value: {}", *b);

    // Box for recursive data structures
    #[derive(Debug)]
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }

    use List::{Cons, Nil};

    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("Recursive list: {:?}", list);
}

/// Rc<T> - Reference counting for multiple ownership
fn rc_demo() {
    #[derive(Debug)]
    struct Node {
        value: i32,
        next: Option<Rc<Node>>,
    }

    let a = Rc::new(Node {
        value: 5,
        next: None,
    });
    println!("a strong count: {}", Rc::strong_count(&a));

    let b = Node {
        value: 10,
        next: Some(Rc::clone(&a)),
    };
    let c = Node {
        value: 15,
        next: Some(Rc::clone(&a)),
    };

    println!("a strong count after b and c: {}", Rc::strong_count(&a));
    println!("b: {:?}, c: {:?}", b, c);
}

/// Arc<T> - Thread-safe reference counting
fn arc_demo() {
    use std::thread;

    let data = Arc::new(Mutex::new(vec![1, 2, 3, 4, 5]));
    let mut handles = vec![];

    for _ in 0..10 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let mut num = data.lock().unwrap();
            num.push(*num.last().unwrap() + 1);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Arc result: {:?}", *data.lock().unwrap());
}

/// Weak<T> - Non-owning reference to avoid reference cycles
fn weak_demo() {
    #[derive(Debug)]
    struct Node {
        value: i32,
        parent: RefCell<Weak<Node>>,
        children: RefCell<Vec<Rc<Node>>>,
    }

    let leaf = Rc::new(Node {
        value: 3,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![]),
    });

    {
        let branch = Rc::new(Node {
            value: 5,
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(vec![Rc::clone(&leaf)]),
        });

        *leaf.parent.borrow_mut() = Rc::downgrade(&branch);

        println!(
            "branch strong: {}, weak: {}",
            Rc::strong_count(&branch),
            Rc::weak_count(&branch)
        );
        println!(
            "leaf strong: {}, weak: {}",
            Rc::strong_count(&leaf),
            Rc::weak_count(&leaf)
        );
    }

    println!("leaf parent: {:?}", leaf.parent.borrow().upgrade());
}

// ============================================================================
// MEMORY SAFETY PATTERNS
// ============================================================================

/// Safe string manipulation without buffer overflows
fn safe_string_ops() {
    let mut s = String::from("hello");

    // Safe push - automatically resizes
    s.push('!');
    s.push_str(" world");

    // Safe indexing - returns Option
    let first_char = s.chars().next();
    println!("First char: {:?}", first_char);

    // Safe slicing - checks UTF-8 boundaries
    let slice = s.get(0..5);
    println!("Slice: {:?}", slice);
}

/// Safe vector operations
fn safe_vector_ops() {
    let mut v = vec![1, 2, 3];

    // Safe push - automatically resizes
    v.push(4);

    // Safe access - returns Option
    let third = v.get(2);
    println!("Third element: {:?}", third);

    // Safe iteration
    for (i, &item) in v.iter().enumerate() {
        println!("Index {}: {}", i, item);
    }
}

/// Custom smart pointer with Drop trait
struct CustomSmartPointer {
    data: String,
}

impl CustomSmartPointer {
    fn new(data: String) -> Self {
        println!("Creating CustomSmartPointer with data: {}", data);
        CustomSmartPointer { data }
    }
}

impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("Dropping CustomSmartPointer with data: {}", self.data);
    }
}

fn drop_demo() {
    let _c = CustomSmartPointer::new(String::from("some data"));
    let _d = CustomSmartPointer::new(String::from("other data"));
    println!("CustomSmartPointers created");
}

// ============================================================================
// UNSAFE RUST (when necessary)
// ============================================================================

/// Unsafe block for FFI or low-level operations
unsafe fn unsafe_demo() {
    // Dereferencing raw pointer
    let mut num = 5;
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;

    unsafe {
        println!("r1 is: {}", *r1);
        *r2 = 10;
        println!("r2 is: {}", *r2);
    }

    // Calling unsafe function
    dangerous_function();
}

unsafe fn dangerous_function() {
    println!("This is an unsafe function");
}

/// Safe wrapper around unsafe code
fn safe_wrapper() {
    let mut v = vec![1, 2, 3, 4, 5];

    let (a, b) = safe_split_at_mut(&mut v, 2);
    println!("a: {:?}, b: {:?}", a, b);
}

fn safe_split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();

    assert!(mid <= len);

    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

// ============================================================================
// ZERO-COST ABSTRACTIONS
// ============================================================================

/// Generic function with zero runtime cost
fn generic_demo<T: std::fmt::Display>(item: T) {
    println!("Item: {}", item);
}

/// Iterator chain - optimized to zero-cost loop
fn iterator_demo() {
    let numbers = vec![1, 2, 3, 4, 5];

    let sum: i32 = numbers
        .iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum();

    println!("Sum of squares of even numbers: {}", sum);
}

/// Const generics - compile-time parameters
fn const_generic_demo() {
    let array = [1, 2, 3];
    let sum = sum_array(&array);
    println!("Sum: {}", sum);
}

fn sum_array<const N: usize>(arr: &[i32; N]) -> i32 {
    arr.iter().sum()
}

// ============================================================================
// PRACTICAL EXAMPLES
// ============================================================================

/// Thread-safe cache using Arc and Mutex
struct Cache {
    data: Arc<Mutex<std::collections::HashMap<String, String>>>,
}

impl Cache {
    fn new() -> Self {
        Cache {
            data: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    fn get(&self, key: &str) -> Option<String> {
        let data = self.data.lock().unwrap();
        data.get(key).cloned()
    }

    fn set(&self, key: String, value: String) {
        let mut data = self.data.lock().unwrap();
        data.insert(key, value);
    }

    fn clone(&self) -> Self {
        Cache {
            data: Arc::clone(&self.data),
        }
    }
}

/// Memory pool for efficient allocation
struct MemoryPool<T> {
    pool: Vec<Box<T>>,
}

impl<T> MemoryPool<T> {
    fn new() -> Self {
        MemoryPool { pool: Vec::new() }
    }

    fn allocate(&mut self, value: T) -> *mut T {
        let boxed = Box::new(value);
        let ptr = Box::leak(boxed) as *mut T;
        self.pool.push(unsafe { Box::from_raw(ptr) });
        ptr
    }

    fn deallocate(&mut self, ptr: *mut T) {
        if let Some(pos) = self.pool.iter().position(|b| b.as_ref() as *const T == ptr) {
            self.pool.remove(pos);
        }
    }
}

impl<T> Drop for MemoryPool<T> {
    fn drop(&mut self) {
        self.pool.clear();
    }
}

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

fn main() {
    println!("=== Rust Memory Management Demo ===\n");

    println!("1. Ownership:");
    ownership_demo();

    println!("\n2. Borrowing:");
    borrowing_demo();

    println!("\n3. Box:");
    box_demo();

    println!("\n4. Rc:");
    rc_demo();

    println!("\n5. Arc:");
    arc_demo();

    println!("\n6. Weak:");
    weak_demo();

    println!("\n7. Safe string operations:");
    safe_string_ops();

    println!("\n8. Safe vector operations:");
    safe_vector_ops();

    println!("\n9. Drop trait:");
    drop_demo();

    println!("\n10. Unsafe:");
    unsafe_demo();

    println!("\n11. Safe wrapper:");
    safe_wrapper();

    println!("\n12. Generics:");
    generic_demo(42);
    generic_demo("hello");

    println!("\n13. Iterators:");
    iterator_demo();

    println!("\n14. Const generics:");
    const_generic_demo();

    println!("\n15. Thread-safe cache:");
    let cache = Cache::new();
    cache.set("key1".to_string(), "value1".to_string());
    println!("Cache get: {:?}", cache.get("key1"));

    println!("\n=== Demo Complete ===");
}
