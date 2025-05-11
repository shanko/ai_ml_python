import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

def sequential_brute_force():
    start_time = time.time()
    for i in range(1, 10):
        for j in range(i + 2, 10): 
            if (i * j) == 15 and abs(j - i) == 2:
                return f'{i} & {j}'
    end_time = time.time()
    print(f"Sequential brute force method: {end_time - start_time}")
    return None

def parallelized_brute_force():
    with ThreadPoolExecutor(max_workers=5) as executor:
        result = list(executor.map(_check_parity, [(i,j) for i in range(1,10) for j in range(i + 2,10)]))
        return next(filter(lambda x: x is not None, result))

def _check_parity(p):
    return (p[0] * p[1]) == 15 and abs(p[1] - p[0]) == 2 if len(p) == 2 else False

def algebraic_method():
    for i in range(1,10):
        for j in range(i+2,10): 
            if (i *j )==15 and abs(j-i)==2:
                return f'{i} & {j}'
    
# Using benchmark library to compare the speed
import timeit

def sequential_brute_force_benchmark():
    start_time = time.time()
    for _ in range(100):
        sequential_brute_force()
    end_time = time.time()
    print(f"Sequential brute force method: {(end_time - start_time) * 1000000}ms")
    
def parallelized_brute_force_benchmark():
    num_iterations = 100
    start_time = time.time()
    list(map(_check_parity, [(i,j) for i in range(1,10) for j in range(i + 2,10)]))
    end_time = time.time()
    print(f"Parallelized brute force method: {(end_time - start_time) * 1000000} ms")

def algebraic_method_benchmark():
    num_iterations = 100
    start_time = time.time()
    list(map(lambda p : (p[0] *p[1]) ==15 and abs(p[1]-p[0])==2 if len(p)==2 else False,[(i,j) for i in range(1,10) for j in range(i + 2,10)]))
    end_time = time.time()
    print(f"Algebraic method: {(end_time - start_time) * 1000000}ms")

def benchmark():
    num_iterations = 100
    sequential_brute_force_benchmark()
    parallelized_brute_force_benchmark()
    algebraic_method_benchmark()

if __name__ == "__main__":
    result1=sequential_brute_force()
    result2=parallelized_brute_force()
    result3=algebraic_method() 
    print(f"Sequential brute force method result: {result1}")
    print(f"Parallelized brute force method result: {result2}")
    print(f"Algebraic method result: {result3}")
    print('------')
    benchmark()
