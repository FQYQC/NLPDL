import time
import itertools

def sol1(list_of_lists):
    flattened_list = []
    for x in list_of_lists:
        for y in x:
            flattened_list.append(y)
    return flattened_list

def sol2(list_of_lists):
    flattened_list = sum(list_of_lists, [])
    return flattened_list

def sol3(list_of_lists):
    flattened_list = [y for x in list_of_lists for y in x]
    return flattened_list

def sol4(list_of_lists):
    flattened_list = list(itertools.chain(*list_of_lists))
    return flattened_list

if __name__ == "__main__":
    list_of_lists = [list(range(1000)) for _ in range(1000)]
    for i in range(1, 5):
        tot_time = 0
        for _ in range(10):
            start = time.time()
            eval(f"sol{i}(list_of_lists)")
            end = time.time()
            tot_time += end - start
        print(f"Solution {i} took {tot_time / 10.} seconds")
       
       
# Solution 1 took 0.0214430570602417 seconds
# Solution 2 took 1.9747170209884644 seconds
# Solution 3 took 0.010944032669067382 seconds
# Solution 4 took 0.006340336799621582 seconds