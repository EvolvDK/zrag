/**
 * Advanced React Hooks Patterns
 *
 * This module demonstrates advanced React hooks patterns:
 * - Custom hooks for reusable logic
 * - Performance optimization hooks
 * - State management patterns
 * - Side effect management
 * - Context and composition
 */

import { useState, useEffect, useCallback, useMemo, useRef, useContext, createContext, useReducer, useLayoutEffect, useImperativeHandle, forwardRef } from 'react';

// ============================================================================
// CUSTOM HOOKS
// ============================================================================

/**
 * useLocalStorage - Persist state in localStorage
 * @param key - Storage key
 * @param initialValue - Initial value
 * @returns [value, setValue] tuple
 */
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('Error reading localStorage:', error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error('Error setting localStorage:', error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
}

/**
 * useDebounce - Debounce a value
 * @param value - Value to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced value
 */
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * useThrottle - Throttle a value
 * @param value - Value to throttle
 * @param limit - Time limit in milliseconds
 * @returns Throttled value
 */
function useThrottle<T>(value: T, limit: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastRan = useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
}

/**
 * usePrevious - Get previous value
 * @param value - Current value
 * @returns Previous value
 */
function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

/**
 * useToggle - Toggle boolean state
 * @param initialValue - Initial value
 * @returns [value, toggle] tuple
 */
function useToggle(initialValue: boolean = false): [boolean, () => void] {
  const [value, setValue] = useState(initialValue);
  const toggle = useCallback(() => setValue(v => !v), []);
  return [value, toggle];
}

/**
 * useArray - Array state management
 * @param initialArray - Initial array
 * @returns Array operations
 */
function useArray<T>(initialArray: T[] = []) {
  const [array, setArray] = useState<T[]>(initialArray);

  const push = useCallback((element: T) => {
    setArray(prev => [...prev, element]);
  }, []);

  const filter = useCallback((callback: (item: T) => boolean) => {
    setArray(prev => prev.filter(callback));
  }, []);

  const update = useCallback((index: number, newElement: T) => {
    setArray(prev => [
      ...prev.slice(0, index),
      newElement,
      ...prev.slice(index + 1)
    ]);
  }, []);

  const remove = useCallback((index: number) => {
    setArray(prev => [
      ...prev.slice(0, index),
      ...prev.slice(index + 1)
    ]);
  }, []);

  const clear = useCallback(() => setArray([]), []);

  return { array, setArray, push, filter, update, remove, clear };
}

// ============================================================================
// PERFORMANCE OPTIMIZATION
// ============================================================================

/**
 * useWindowSize - Track window size with debouncing
 * @returns Window dimensions
 */
function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
}

/**
 * useScrollPosition - Track scroll position
 * @returns Scroll position
 */
function useScrollPosition() {
  const [scrollPosition, setScrollPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleScroll = () => {
      setScrollPosition({
        x: window.scrollX,
        y: window.scrollY,
      });
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return scrollPosition;
}

/**
 * useMediaQuery - Responsive design hook
 * @param query - Media query string
 * @returns Boolean indicating if query matches
 */
function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    if (media.matches !== matches) {
      setMatches(media.matches);
    }

    const listener = () => setMatches(media.matches);
    media.addEventListener('change', listener);
    return () => media.removeEventListener('change', listener);
  }, [matches, query]);

  return matches;
}

// ============================================================================
// ASYNC OPERATIONS
// ============================================================================

/**
 * useAsync - Async operation with loading and error states
 * @param asyncFunction - Async function to execute
 * @returns Async operation state
 */
function useAsync<T>(
  asyncFunction: () => Promise<T>,
  immediate = true
) {
  const [status, setStatus] = useState<'idle' | 'pending' | 'success' | 'error'>('idle');
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(async () => {
    setStatus('pending');
    setValue(null);
    setError(null);

    try {
      const response = await asyncFunction();
      setValue(response);
      setStatus('success');
    } catch (error) {
      setError(error as Error);
      setStatus('error');
    }
  }, [asyncFunction]);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  return { execute, status, value, error };
}

/**
 * useFetch - Fetch data with loading and error states
 * @param url - URL to fetch
 * @param options - Fetch options
 * @returns Fetch state
 */
function useFetch<T>(url: string, options?: RequestInit) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const abortController = new AbortController();

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(url, {
          ...options,
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const json = await response.json();
        setData(json);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err as Error);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => {
      abortController.abort();
    };
  }, [url, options]);

  return { data, loading, error };
}

// ============================================================================
// FORM HANDLING
// ============================================================================

/**
 * useForm - Form state management
 * @param initialValues - Initial form values
 * @param validate - Validation function
 * @returns Form state and handlers
 */
function useForm<T extends Record<string, any>>(
  initialValues: T,
  validate?: (values: T) => Record<string, string>
) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setValues(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleBlur = useCallback((event: React.FocusEvent<HTMLInputElement>) => {
    const { name } = event.target;
    setTouched(prev => ({ ...prev, [name]: true }));

    if (validate) {
      const validationErrors = validate(values);
      setErrors(validationErrors);
    }
  }, [values, validate]);

  const handleSubmit = useCallback((callback: (values: T) => void) => {
    return (event: React.FormEvent) => {
      event.preventDefault();

      if (validate) {
        const validationErrors = validate(values);
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length === 0) {
          callback(values);
        }
      } else {
        callback(values);
      }
    };
  }, [values, validate]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  }, [initialValues]);

  return {
    values,
    errors,
    touched,
    handleChange,
    handleBlur,
    handleSubmit,
    reset,
  };
}

// ============================================================================
// CONTEXT PATTERNS
// ============================================================================

/**
 * Theme context
 */
type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

/**
 * useTheme - Access theme context
 * @returns Theme context
 */
function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

/**
 * ThemeProvider component
 */
function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light');

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  }, []);

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// ============================================================================
// REDUCER PATTERNS
// ============================================================================

/**
 * Counter reducer
 */
type CounterAction =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'incrementBy'; payload: number }
  | { type: 'reset' };

interface CounterState {
  count: number;
}

function counterReducer(state: CounterState, action: CounterAction): CounterState {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    case 'incrementBy':
      return { count: state.count + action.payload };
    case 'reset':
      return { count: 0 };
    default:
      return state;
  }
}

/**
 * useCounter - Counter with reducer
 * @returns Counter state and actions
 */
function useCounter(initialCount = 0) {
  const [state, dispatch] = useReducer(counterReducer, { count: initialCount });

  const increment = useCallback(() => dispatch({ type: 'increment' }), []);
  const decrement = useCallback(() => dispatch({ type: 'decrement' }), []);
  const incrementBy = useCallback((n: number) => dispatch({ type: 'incrementBy', payload: n }), []);
  const reset = useCallback(() => dispatch({ type: 'reset' }), []);

  return {
    count: state.count,
    increment,
    decrement,
    incrementBy,
    reset,
  };
}

// ============================================================================
// REFS AND IMPERATIVE HANDLES
// ============================================================================

/**
 * useInterval - Interval with cleanup
 * @param callback - Callback function
 * @param delay - Delay in milliseconds
 */
function useInterval(callback: () => void, delay: number | null) {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const id = setInterval(() => savedCallback.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

/**
 * useClickOutside - Detect clicks outside element
 * @param ref - Element ref
 * @param handler - Click handler
 */
function useClickOutside(ref: React.RefObject<HTMLElement>, handler: () => void) {
  useEffect(() => {
    const listener = (event: MouseEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return;
      }
      handler();
    };

    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}

// ============================================================================
// EXAMPLE COMPONENTS
// ============================================================================

/**
 * Search component with debouncing
 */
function SearchComponent() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery) {
      console.log('Searching for:', debouncedQuery);
    }
  }, [debouncedQuery]);

  return (
    <input
      type="text"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Search..."
    />
  );
}

/**
 * Counter component with reducer
 */
function CounterComponent() {
  const { count, increment, decrement, reset } = useCounter(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
      <button onClick={decrement}>Decrement</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}

/**
 * Form component with validation
 */
function FormComponent() {
  const { values, errors, touched, handleChange, handleBlur, handleSubmit, reset } = useForm(
    { email: '', password: '' },
    (values) => {
      const errors: Record<string, string> = {};

      if (!values.email) {
        errors.email = 'Email is required';
      } else if (!/\S+@\S+\.\S+/.test(values.email)) {
        errors.email = 'Email is invalid';
      }

      if (!values.password) {
        errors.password = 'Password is required';
      } else if (values.password.length < 6) {
        errors.password = 'Password must be at least 6 characters';
      }

      return errors;
    }
  );

  const onSubmit = (data: typeof values) => {
    console.log('Form submitted:', data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label>Email:</label>
        <input
          type="email"
          name="email"
          value={values.email}
          onChange={handleChange}
          onBlur={handleBlur}
        />
        {touched.email && errors.email && <span>{errors.email}</span>}
      </div>
      <div>
        <label>Password:</label>
        <input
          type="password"
          name="password"
          value={values.password}
          onChange={handleChange}
          onBlur={handleBlur}
        />
        {touched.password && errors.password && <span>{errors.password}</span>}
      </div>
      <button type="submit">Submit</button>
      <button type="button" onClick={reset}>Reset</button>
    </form>
  );
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  useLocalStorage,
  useDebounce,
  useThrottle,
  usePrevious,
  useToggle,
  useArray,
  useWindowSize,
  useScrollPosition,
  useMediaQuery,
  useAsync,
  useFetch,
  useForm,
  useTheme,
  ThemeProvider,
  useCounter,
  useInterval,
  useClickOutside,
  SearchComponent,
  CounterComponent,
  FormComponent,
};
