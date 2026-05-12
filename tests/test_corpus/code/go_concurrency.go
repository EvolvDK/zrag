// Advanced Concurrency in Go
//
// This module demonstrates Go's concurrency model:
// - Goroutines and channels
// - Select statements
// - Mutex and sync primitives
// - Context for cancellation
// - Worker pools
// - Rate limiting

package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// GOROUTINES AND CHANNELS
// ============================================================================

// Basic goroutine example
func goroutineDemo() {
	fmt.Println("1. Goroutines:")

	// Launch multiple goroutines
	for i := 0; i < 5; i++ {
		go func(n int) {
			fmt.Printf("  Goroutine %d\n", n)
		}(i)
	}

	// Wait for goroutines to finish
	time.Sleep(time.Millisecond * 100)
}

// Buffered channels
func bufferedChannelDemo() {
	fmt.Println("\n2. Buffered Channels:")

	ch := make(chan int, 3)

	// Producer
	go func() {
		for i := 0; i < 5; i++ {
			ch <- i
			fmt.Printf("  Sent: %d\n", i)
		}
		close(ch)
	}()

	// Consumer
	for val := range ch {
		fmt.Printf("  Received: %d\n", val)
	}
}

// Unbuffered channels (synchronous)
func unbufferedChannelDemo() {
	fmt.Println("\n3. Unbuffered Channels:")

	ch := make(chan string)

	// Goroutine that sends
	go func() {
		time.Sleep(time.Millisecond * 100)
		ch <- "hello"
		fmt.Println("  Sent: hello")
	}()

	// Main goroutine receives
	msg := <-ch
	fmt.Printf("  Received: %s\n", msg)
}

// ============================================================================
// SELECT STATEMENTS
// ============================================================================

// Select for multiple channels
func selectDemo() {
	fmt.Println("\n4. Select:")

	ch1 := make(chan string)
	ch2 := make(chan string)

	go func() {
		time.Sleep(time.Millisecond * 100)
		ch1 <- "one"
	}()

	go func() {
		time.Sleep(time.Millisecond * 200)
		ch2 <- "two"
	}()

	for i := 0; i < 2; i++ {
		select {
		case msg1 := <-ch1:
			fmt.Printf("  Received from ch1: %s\n", msg1)
		case msg2 := <-ch2:
			fmt.Printf("  Received from ch2: %s\n", msg2)
		}
	}
}

// Select with timeout
func selectTimeoutDemo() {
	fmt.Println("\n5. Select with Timeout:")

	ch := make(chan string)

	go func() {
		time.Sleep(time.Second * 2)
		ch <- "delayed"
	}()

	select {
	case msg := <-ch:
		fmt.Printf("  Received: %s\n", msg)
	case <-time.After(time.Second):
		fmt.Println("  Timeout!")
	}
}

// ============================================================================
// SYNC PRIMITIVES
// ============================================================================

// Mutex for mutual exclusion
func mutexDemo() {
	fmt.Println("\n6. Mutex:")

	var counter int
	var mu sync.Mutex

	// Launch multiple goroutines that increment counter
	for i := 0; i < 10; i++ {
		go func() {
			mu.Lock()
			counter++
			mu.Unlock()
		}()
	}

	// Wait for all goroutines
	time.Sleep(time.Millisecond * 100)
	mu.Lock()
	fmt.Printf("  Counter: %d\n", counter)
	mu.Unlock()
}

// WaitGroup for synchronization
func waitGroupDemo() {
	fmt.Println("\n7. WaitGroup:")

	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			fmt.Printf("  Worker %d\n", n)
			time.Sleep(time.Millisecond * 50)
		}(i)
	}

	wg.Wait()
	fmt.Println("  All workers done")
}

// Atomic operations
func atomicDemo() {
	fmt.Println("\n8. Atomic:")

	var counter int64

	// Launch multiple goroutines that increment counter atomically
	for i := 0; i < 100; i++ {
		go func() {
			atomic.AddInt64(&counter, 1)
		}()
	}

	time.Sleep(time.Millisecond * 100)
	fmt.Printf("  Counter: %d\n", atomic.LoadInt64(&counter))
}

// Once for single execution
func onceDemo() {
	fmt.Println("\n9. Once:")

	var once sync.Once
	var result string

	for i := 0; i < 5; i++ {
		go func(n int) {
			once.Do(func() {
				result = "initialized"
				fmt.Printf("  Initialization by goroutine %d\n", n)
			})
			fmt.Printf("  Goroutine %d sees: %s\n", n, result)
		}(i)
	}

	time.Sleep(time.Millisecond * 100)
}

// ============================================================================
// CONTEXT FOR CANCELLATION
// ============================================================================

// Context cancellation
func contextCancelDemo() {
	fmt.Println("\n10. Context Cancellation:")

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("  Worker cancelled")
				return
			default:
				fmt.Println("  Working...")
				time.Sleep(time.Millisecond * 100)
			}
		}
	}()

	time.Sleep(time.Millisecond * 300)
	cancel()
	time.Sleep(time.Millisecond * 100)
}

// Context with timeout
func contextTimeoutDemo() {
	fmt.Println("\n11. Context Timeout:")

	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*500)
	defer cancel()

	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("  Worker timed out")
				return
			default:
				fmt.Println("  Working...")
				time.Sleep(time.Millisecond * 100)
			}
		}
	}()

	time.Sleep(time.Second)
}

// Context with deadline
func contextDeadlineDemo() {
	fmt.Println("\n12. Context Deadline:")

	deadline := time.Now().Add(time.Millisecond * 500)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("  Worker reached deadline")
				return
			default:
				fmt.Println("  Working...")
				time.Sleep(time.Millisecond * 100)
			}
		}
	}()

	time.Sleep(time.Second)
}

// ============================================================================
// WORKER POOLS
// ============================================================================

// Simple worker pool
func workerPoolDemo() {
	fmt.Println("\n13. Worker Pool:")

	jobs := make(chan int, 10)
	results := make(chan int, 10)

	// Start workers
	for w := 1; w <= 3; w++ {
		go worker(w, jobs, results)
	}

	// Send jobs
	for j := 1; j <= 5; j++ {
		jobs <- j
		fmt.Printf("  Sent job %d\n", j)
	}
	close(jobs)

	// Collect results
	for a := 1; a <= 5; a++ {
		result := <-results
		fmt.Printf("  Received result %d\n", result)
	}
}

func worker(id int, jobs <-chan int, results chan<- int) {
	for j := range jobs {
		fmt.Printf("  Worker %d processing job %d\n", id, j)
		time.Sleep(time.Millisecond * 100)
		results <- j * 2
	}
}

// Worker pool with context
func workerPoolWithContextDemo() {
	fmt.Println("\n14. Worker Pool with Context:")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	jobs := make(chan int, 10)
	results := make(chan int, 10)

	// Start workers
	for w := 1; w <= 3; w++ {
		go workerWithContext(ctx, w, jobs, results)
	}

	// Send jobs
	for j := 1; j <= 5; j++ {
		jobs <- j
		fmt.Printf("  Sent job %d\n", j)
	}
	close(jobs)

	// Collect results
	for a := 1; a <= 5; a++ {
		result := <-results
		fmt.Printf("  Received result %d\n", result)
	}
}

func workerWithContext(ctx context.Context, id int, jobs <-chan int, results chan<- int) {
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("  Worker %d shutting down\n", id)
			return
		case j, ok := <-jobs:
			if !ok {
				return
			}
			fmt.Printf("  Worker %d processing job %d\n", id, j)
			time.Sleep(time.Millisecond * 100)
			results <- j * 2
		}
	}
}

// ============================================================================
// RATE LIMITING
// ============================================================================

// Rate limiting with ticker
func rateLimitDemo() {
	fmt.Println("\n15. Rate Limiting:")

	// Rate limiter: allow 1 request per 100ms
	limiter := time.NewTicker(time.Millisecond * 100)
	defer limiter.Stop()

	// Simulate incoming requests
	requests := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		requests <- i
	}
	close(requests)

	for req := range requests {
		<-limiter.C
		fmt.Printf("  Request %d processed\n", req)
	}
}

// Burst rate limiting
func burstRateLimitDemo() {
	fmt.Println("\n16. Burst Rate Limiting:")

	// Allow up to 3 bursts, then 1 per 200ms
	limiter := time.NewTicker(time.Millisecond * 200)
	burstLimiter := make(chan time.Time, 3)

	// Pre-fill burst limiter
	for i := 0; i < 3; i++ {
		burstLimiter <- time.Now()
	}

	// Refill burst limiter
	go func() {
		for t := range limiter.C {
			select {
			case burstLimiter <- t:
			default:
			}
		}
	}()

	// Simulate incoming requests
	requests := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		requests <- i
	}
	close(requests)

	for req := range requests {
		<-burstLimiter
		fmt.Printf("  Request %d processed\n", req)
	}
}

// ============================================================================
// PRACTICAL EXAMPLES
// ============================================================================

// Concurrent web scraper
type Result struct {
	URL    string
	Status int
	Body   string
}

func scrapeURL(url string) Result {
	time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
	return Result{
		URL:    url,
		Status: 200,
		Body:   fmt.Sprintf("Content from %s", url),
	}
}

func concurrentScrapeDemo() {
	fmt.Println("\n17. Concurrent Web Scraping:")

	urls := []string{
		"https://example.com/1",
		"https://example.com/2",
		"https://example.com/3",
	}

	results := make(chan Result, len(urls))

	// Launch scrapers
	for _, url := range urls {
		go func(u string) {
			results <- scrapeURL(u)
		}(url)
	}

	// Collect results
	for i := 0; i < len(urls); i++ {
		result := <-results
		fmt.Printf("  %s: %d\n", result.URL, result.Status)
	}
}

// Fan-out, fan-in pattern
func fanOutFanInDemo() {
	fmt.Println("\n18. Fan-Out, Fan-In:")

	// Fan-out: distribute work to multiple workers
	input := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		input <- i
	}
	close(input)

	// Create multiple workers
	worker1 := workerFunc("worker1", input)
	worker2 := workerFunc("worker2", input)

	// Fan-in: collect results
	output := make(chan int, 20)
	go func() {
		for result := range worker1 {
			output <- result
		}
	}()
	go func() {
		for result := range worker2 {
			output <- result
		}
	}()

	// Collect all results
	for i := 0; i < 10; i++ {
		fmt.Printf("  Result: %d\n", <-output)
	}
}

func workerFunc(name string, input <-chan int) <-chan int {
	output := make(chan int)
	go func() {
		for n := range input {
			result := n * n
			fmt.Printf("  %s: %d -> %d\n", name, n, result)
			output <- result
		}
		close(output)
	}()
	return output
}

// Pipeline pattern
func pipelineDemo() {
	fmt.Println("\n19. Pipeline:")

	// Stage 1: Generate numbers
	numbers := generate(1, 10)

	// Stage 2: Square numbers
	squared := square(numbers)

	// Stage 3: Filter even numbers
	filtered := filter(squared, func(n int) bool {
		return n%2 == 0
	})

	// Consume results
	for n := range filtered {
		fmt.Printf("  Result: %d\n", n)
	}
}

func generate(start, end int) <-chan int {
	out := make(chan int)
	go func() {
		for i := start; i <= end; i++ {
			out <- i
		}
		close(out)
	}()
	return out
}

func square(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for n := range in {
			out <- n * n
		}
		close(out)
	}()
	return out
}

func filter(in <-chan int, predicate func(int) bool) <-chan int {
	out := make(chan int)
	go func() {
		for n := range in {
			if predicate(n) {
				out <- n
			}
		}
		close(out)
	}()
	return out
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	fmt.Println("=== Go Concurrency Demo ===\n")

	goroutineDemo()
	bufferedChannelDemo()
	unbufferedChannelDemo()
	selectDemo()
	selectTimeoutDemo()
	mutexDemo()
	waitGroupDemo()
	atomicDemo()
	onceDemo()
	contextCancelDemo()
	contextTimeoutDemo()
	contextDeadlineDemo()
	workerPoolDemo()
	workerPoolWithContextDemo()
	rateLimitDemo()
	burstRateLimitDemo()
	concurrentScrapeDemo()
	fanOutFanInDemo()
	pipelineDemo()

	fmt.Println("\n=== Demo Complete ===")
}
