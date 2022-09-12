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
    list_of_lists = [list(range(1000))] * 1000
    for i in range(1, 5):
        start_time = time.perf_counter()
        locals()['sol' + str(i)](list_of_lists)
        end_time = time.perf_counter()
        print('Solution ' + str(i) + ' took ' + str(end_time - start_time) + ' seconds')


# Solution 1 took 0.02140442800009623 seconds
# Solution 2 took 0.8658737830000973 seconds
# Solution 3 took 0.01031219600008626 seconds
# Solution 4 took 0.00475121300041792 seconds