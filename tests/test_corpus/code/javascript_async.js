/**
 * Advanced Async Programming in JavaScript
 *
 * This module demonstrates modern async patterns including:
 * - Promises and async/await
 * - Parallel and sequential execution
 * - Error handling patterns
 * - Cancellation and timeouts
 * - Rate limiting and debouncing
 */

// ============================================================================
// PROMISE UTILITIES
// ============================================================================

/**
 * Retry a promise-based operation with exponential backoff
 * @param {Function} fn - Function that returns a Promise
 * @param {number} maxRetries - Maximum number of retry attempts
 * @param {number} delay - Initial delay in milliseconds
 * @returns {Promise} Resolved when operation succeeds
 */
async function retryWithBackoff(fn, maxRetries = 3, delay = 1000) {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries) {
        const backoffDelay = delay * Math.pow(2, attempt);
        console.log(`Attempt ${attempt + 1} failed, retrying in ${backoffDelay}ms`);
        await new Promise(resolve => setTimeout(resolve, backoffDelay));
      }
    }
  }

  throw lastError;
}

/**
 * Execute promises in parallel with concurrency limit
 * @param {Array<Function>} tasks - Array of functions returning Promises
 * @param {number} concurrency - Maximum concurrent executions
 * @returns {Promise<Array>} Results from all tasks
 */
async function parallelWithLimit(tasks, concurrency = 5) {
  const results = new Array(tasks.length);
  const executing = new Set();

  for (let i = 0; i < tasks.length; i++) {
    const promise = tasks[i]().then(result => {
      results[i] = result;
      executing.delete(promise);
    });

    executing.add(promise);

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

/**
 * Execute promises sequentially, passing results to next task
 * @param {Array<Function>} tasks - Array of functions returning Promises
 * @param {*} initialValue - Initial value for the chain
 * @returns {Promise<*>} Final result
 */
async function sequential(tasks, initialValue) {
  return tasks.reduce(async (acc, task) => {
    const result = await acc;
    return task(result);
  }, Promise.resolve(initialValue));
}

// ============================================================================
// ERROR HANDLING PATTERNS
// ============================================================================

/**
 * Safe async execution with fallback value
 * @param {Function} fn - Async function to execute
 * @param {*} fallback - Fallback value if function fails
 * @returns {Promise<*>} Result or fallback
 */
async function safeAsync(fn, fallback = null) {
  try {
    return await fn();
  } catch (error) {
    console.error('Async error:', error.message);
    return fallback;
  }
}

/**
 * Execute multiple promises and return all results, even if some fail
 * @param {Array<Promise>} promises - Array of promises
 * @returns {Promise<Array>} Array of {status, value/reason} objects
 */
async function allSettled(promises) {
  return Promise.allSettled(promises);
}

/**
 * Race with timeout - reject if promise doesn't resolve in time
 * @param {Promise} promise - Promise to race
 * @param {number} timeout - Timeout in milliseconds
 * @param {string} message - Timeout error message
 * @returns {Promise} Promise result or timeout error
 */
async function withTimeout(promise, timeout, message = 'Operation timed out') {
  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(message)), timeout)
  );

  return Promise.race([promise, timeoutPromise]);
}

// ============================================================================
// CANCELLATION AND CONTROL
// ============================================================================

/**
 * Create a cancellable promise
 * @param {Function} executor - Promise executor function
 * @returns {Object} Object with promise and cancel method
 */
function cancellable(executor) {
  let cancelled = false;
  let rejectFn;

  const promise = new Promise((resolve, reject) => {
    rejectFn = reject;
    executor(
      value => {
        if (!cancelled) resolve(value);
      },
      error => {
        if (!cancelled) reject(error);
      }
    );
  });

  return {
    promise,
    cancel: () => {
      cancelled = true;
      rejectFn(new Error('Operation cancelled'));
    }
  };
}

/**
 * Debounce an async function
 * @param {Function} fn - Async function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
function debounceAsync(fn, delay = 300) {
  let timeoutId;
  let pendingPromise = null;

  return async function(...args) {
    clearTimeout(timeoutId);

    if (pendingPromise) {
      // Cancel previous pending operation
      pendingPromise.cancel();
    }

    const { promise, cancel } = cancellable((resolve, reject) => {
      timeoutId = setTimeout(async () => {
        try {
          resolve(await fn(...args));
        } catch (error) {
          reject(error);
        }
      }, delay);
    });

    pendingPromise = { promise, cancel };
    return promise;
  };
}

/**
 * Rate limit async function calls
 * @param {Function} fn - Async function to rate limit
 * @param {number} limit - Maximum calls per interval
 * @param {number} interval - Interval in milliseconds
 * @returns {Function} Rate-limited function
 */
function rateLimit(fn, limit = 5, interval = 1000) {
  const queue = [];
  let running = 0;
  let lastReset = Date.now();

  return async function(...args) {
    return new Promise((resolve, reject) => {
      const execute = async () => {
        running++;
        try {
          const result = await fn(...args);
          resolve(result);
        } catch (error) {
          reject(error);
        } finally {
          running--;
          processQueue();
        }
      };

      queue.push(execute);
      processQueue();
    });
  };

  function processQueue() {
    const now = Date.now();
    if (now - lastReset >= interval) {
      lastReset = now;
      running = 0;
    }

    while (queue.length > 0 && running < limit) {
      const task = queue.shift();
      task();
    }
  }
}

// ============================================================================
// PRACTICAL EXAMPLES
// ============================================================================

/**
 * Fetch data with retry and timeout
 * @param {string} url - URL to fetch
 * @param {Object} options - Fetch options
 * @returns {Promise<Response>} Fetch response
 */
async function fetchWithRetry(url, options = {}) {
  return retryWithBackoff(
    () => withTimeout(
      fetch(url, options),
      10000,
      'Fetch request timed out'
    ),
    3,
    1000
  );
}

/**
 * Batch API requests with concurrency control
 * @param {Array<string>} urls - Array of URLs to fetch
 * @param {number} concurrency - Max concurrent requests
 * @returns {Promise<Array>} Array of responses
 */
async function batchFetch(urls, concurrency = 5) {
  const tasks = urls.map(url => () => fetchWithRetry(url));
  return parallelWithLimit(tasks, concurrency);
}

/**
 * Process data pipeline with error handling
 * @param {Array} data - Input data
 * @param {Array<Function>} processors - Array of processing functions
 * @returns {Promise<Array>} Processed data
 */
async function processDataPipeline(data, processors) {
  const results = await Promise.all(
    data.map(async item => {
      try {
        return await sequential(
          processors.map(p => (d) => p(d)),
          item
        );
      } catch (error) {
        console.error('Pipeline error:', error.message);
        return null;
      }
    })
  );

  return results.filter(r => r !== null);
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

async function demonstrateAsyncPatterns() {
  console.log('=== Async Programming Demo ===\n');

  // Retry with backoff
  console.log('1. Retry with backoff:');
  let attempts = 0;
  const result = await retryWithBackoff(async () => {
    attempts++;
    if (attempts < 3) {
      throw new Error('Not yet');
    }
    return 'Success!';
  });
  console.log('Result:', result, '\n');

  // Parallel execution with limit
  console.log('2. Parallel with limit:');
  const tasks = Array.from({ length: 10 }, (_, i) =>
    () => new Promise(resolve => setTimeout(() => resolve(i), 100))
  );
  const parallelResults = await parallelWithLimit(tasks, 3);
  console.log('Results:', parallelResults, '\n');

  // Sequential execution
  console.log('3. Sequential execution:');
  const sequentialTasks = [
    (x) => Promise.resolve(x + 1),
    (x) => Promise.resolve(x * 2),
    (x) => Promise.resolve(x - 3)
  ];
  const sequentialResult = await sequential(sequentialTasks, 5);
  console.log('Result:', sequentialResult, '\n');

  // Timeout
  console.log('4. Timeout:');
  try {
    await withTimeout(
      new Promise(resolve => setTimeout(() => resolve('Done'), 2000)),
      1000,
      'Should timeout'
    );
  } catch (error) {
    console.log('Error:', error.message, '\n');
  }

  // Debounce
  console.log('5. Debounce:');
  const debouncedFn = debounceAsync(async (x) => {
    console.log('Processing:', x);
    return x * 2;
  }, 500);

  debouncedFn(1);
  debouncedFn(2);
  debouncedFn(3);
  await new Promise(resolve => setTimeout(resolve, 600));
  console.log('Debounced complete\n');

  console.log('=== Demo Complete ===');
}

// Run demonstration if this file is executed directly
if (require.main === module) {
  demonstrateAsyncPatterns().catch(console.error);
}

module.exports = {
  retryWithBackoff,
  parallelWithLimit,
  sequential,
  safeAsync,
  allSettled,
  withTimeout,
  cancellable,
  debounceAsync,
  rateLimit,
  fetchWithRetry,
  batchFetch,
  processDataPipeline
};
