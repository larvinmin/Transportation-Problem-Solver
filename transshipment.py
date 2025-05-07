

"""
  O D
O . .
D . .
"""

def min_cost_method(matrix, init_supply, init_demand):
    allocation_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    blacklist_rows = set()
    blacklist_cols = set()
    supply = list(init_supply)
    demand = list(init_demand)

    while sum(supply) > 0 and sum(demand) > 0:
        min_val = float('inf')
        min_ij = (-1,-1)
        for i in range(len(matrix)):
            if i in blacklist_rows:
                continue
            for j in range(len(matrix[i])):
                if j in blacklist_cols:
                    continue
                if matrix[i][j] < min_val:
                    min_val = matrix[i][j]
                    min_ij = (i,j)
        if min_ij == (-1, -1):
             print("could not find valid min cell")
             break
        (i, j) = min_ij
        allocated = min(supply[i], demand[j])
        supply[i] = supply[i] - allocated
        demand[j] = demand[j] - allocated
        allocation_matrix[i][j] = allocated
        if supply[i] == 0:
            blacklist_rows.add(i)
        if demand[j] == 0:
            blacklist_cols.add(j)
    return allocation_matrix

def northwest_corner_method(supply, demand):
    n, m = len(supply), len(demand)
    allocation = [[0]*m for _ in range(n)]
    i = j = 0
    s = supply[:]
    d = demand[:]
    basis_tracker = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]

    while i < n and j < m:
        allocated = min(s[i], d[j])
        allocation[i][j] = allocated
        basis_tracker[i][j] = 1

        s[i] -= allocated
        d[j] -= allocated
        if s[i] == 0:
            i += 1
        else:
            j += 1
    return allocation, basis_tracker

def fix_degeneracy(basis_tracker, matrix):
    rows, cols = len(basis_tracker), len(basis_tracker[0])

    for i in range(rows):
        if sum(basis_tracker[i]) == 0:
            j_min = min(range(cols), key=lambda j: matrix[i][j])
            basis_tracker[i][j_min] = 1

    for j in range(cols):
        col_sum = sum(basis_tracker[i][j] for i in range(rows))
        if col_sum == 0:
            i_min = min(range(rows), key=lambda i: matrix[i][j])
            basis_tracker[i_min][j] = 1

    return basis_tracker


def calculate_uv(matrix, basis_tracker):
    u_sol = [None]*len(matrix)
    v_sol = [None]*len(matrix[0])
    # set u_0 to 0, find first V's to get started
    u_sol[0] = 0

    changed = True
    while changed:
        changed = False
        for i in range(len(u_sol)):
            if u_sol[i] is not None:
                for j in range (len(basis_tracker[i])):
                    if basis_tracker[i][j] != 0 and v_sol[j] is None:
                        v_sol[j] = matrix[i][j] - u_sol[i]
                        changed = True

        for j in range(len(v_sol)):
            if v_sol[j] is not None:
                for i in range(len(basis_tracker)):
                    if basis_tracker[i][j] != 0 and u_sol[i] is None:
                        u_sol[i] = matrix[i][j] - v_sol[j]
                        changed = True

    return u_sol, v_sol


def calculate_w(matrix, u_sol, v_sol):
    w_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            w_matrix[i][j] = u_sol[i] + v_sol[j] - matrix[i][j]
    return w_matrix


def loop(matrix, start, path=None, visited=None, last_dir=None):
    rows, cols = len(matrix), len(matrix[0])
    if path is None:
        path = [start]
    if visited is None:
        visited = {start}
    x, y = path[-1]

    # found start and loop
    if (x, y) == start and len(path) >= 4:
        return path
    # find direction
    if last_dir == 'horizontal':
        directions = [(1, 0), (-1, 0)] 
        next_dir = 'vertical'
    else:
        directions = [(0, 1), (0, -1)]
        next_dir = 'horizontal'

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # go in direction
        while 0 <= nx < rows and 0 <= ny < cols:
            # Only consider it if it's the start or basic
            if (nx, ny) == start or matrix[nx][ny] != 0:
                if (nx, ny) not in visited or (nx, ny) == start:
                    path.append((nx, ny))
                    visited.add((nx, ny))
                    dir_label = 'vertical' if dx else 'horizontal'
                    result = loop(
                        matrix, start, path, visited, next_dir
                    )
                    if result:
                        return result
                    path.pop()
                    visited.remove((nx, ny))

                # pass through if needed
                nx += dx
                ny += dy
                continue
                # break
            # ignore zero
            nx += dx
            ny += dy
    return None
            



def transporation_simplex(matrix, supply, demand):
    # make supply and demand extended for transshipment points
    extended_supply = list(supply)
    extended_demand = list(demand)
    total_supply = sum(supply)
    for d in demand:
        extended_supply.append(0)
    for s in supply: 
        extended_demand.insert(0, 0)
    for i in range(len(demand)):
        extended_supply[i] += 50
    for i in range(len(demand)): 
        extended_demand[i] += 50

    print("supply", extended_supply)
    print("demand", extended_demand)

    # use min cost to create basic feasible solution
    bfs = min_cost_method(matrix, extended_supply, extended_demand)
    basis_tracker = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(bfs)):
        for j in range(len(bfs[i])):
            if bfs[i][j] != 0: 
                basis_tracker[i][j] = 1

    # bfs,basis_tracker = northwest_corner_method(extended_supply, extended_demand)

    for row in bfs:
        print(" ".join(f"{val:3}" for val in row))
    
    # fix degeneracy in basis_tracker
    basis_tracker = fix_degeneracy(basis_tracker,matrix)
    print("basis tracker intitial:")
    for row in basis_tracker:
        print(" ".join(f"{val:3}" for val in row))


    cost = 0
    for i in range(len(bfs)):
        for j in range(len(bfs[i])):
            cost += bfs[i][j] * matrix[i][j]
    print("initial cost:", cost)

    iteration_count = 0
    while(True):
        iteration_count += 1
        # find all u and v, assuming one is 0 
        u_sol,v_sol = calculate_uv(matrix, basis_tracker)
        print("u vector: ", u_sol), print("v vector: ", v_sol)

        # find all w_ij given u and v
        w_vec = calculate_w(matrix, u_sol, v_sol)
        # for row in w_vec:
        #     print(" ".join(f"{val:3}" for val in row))

        # check if all negative or positive, find largest
        largest = -100000
        entering = (-1,-1)
        for i in range(len(w_vec)):
            for j in range(len(w_vec[i])):
                if basis_tracker[i][j] == 0 and w_vec[i][j] > largest:
                    largest = w_vec[i][j]
                    entering = (i,j)

        print("largest w:", largest)
        
        if entering == (-1, -1):
            print("Optimal solution found.")
            return bfs

        
        # check for optimality
        if largest <= 0:
            print("we are done")
            print("Number of iterations:", iteration_count-1)
            return bfs
            # break
            
        # build a loop
        enter_i,enter_j = entering
        print("entering: ", entering)
        # for row in basis_tracker:
        #     print(" ".join(f"{val:3}" for val in row))

        found_loop = loop(basis_tracker, (enter_i, enter_j), [(enter_i, enter_j)], set([(enter_i, enter_j)]), 'horizontal')
        found_loop.pop()
        print("loop vertices: ", found_loop)

        # find odd min and subtract it from everything in loop
        min_in_loop = 1000000
        leave_i,leave_j = 0,0
        for iter in range(len(found_loop)):
            i,j = found_loop[iter]
            if iter % 2 == 1:
                if bfs[i][j] < min_in_loop:
                    min_in_loop = bfs[i][j]
                    leave_i,leave_j = i,j
        
        basis_tracker[enter_i][enter_j] = 1
        basis_tracker[leave_i][leave_j] = 0

        # subtract step
        for iter in range(len(found_loop)):
            i,j = found_loop[iter]
            if iter % 2 == 1:
                bfs[i][j] -= min_in_loop
            else:
                bfs[i][j] += min_in_loop

        print("solution after iteration: ")
        for row in bfs:
            print(" ".join(f"{val:3}" for val in row))
        # print("basis tracker after iteration: ")
        # for row in basis_tracker:
        #     print(" ".join(f"{val:3}" for val in row))

        cost = 0
        for i in range(len(bfs)):
            for j in range(len(bfs[i])):
                cost += bfs[i][j] * matrix[i][j]
        print("current cost:", cost)

        basis_count = 0
        for i in range(len(basis_tracker)):
            for j in range(len(basis_tracker[i])):
                if basis_tracker[i][j] != 0:
                    basis_count += 1
        if basis_count != 19:
            print("oh no")
            break

        print("\nnext iteration:")
    

def visualize(matrix):
    import pydot
    graph = pydot.Dot(graph_type="digraph", rankdir="LR")

    # Add nodes
    for i in range(10):
        label = f"S{i}" if i < 5 else f"D{i}"
        color = "lightblue" if i < 5 else "lightgreen"
        graph.add_node(pydot.Node(label, style="filled", fillcolor=color))

    # Add edges for flow
    for i in range(10):
        for j in range(10):
            qty = matrix[i][j]
            if qty > 0:
                src = f"S{i}" if i < 5 else f"D{i}"
                dst = f"S{j}" if j < 5 else f"D{j}"
                graph.add_edge(pydot.Edge(src, dst, label=str(qty)))

    graph.write_png("transport_flow.png")








if __name__ == "__main__":
    import time
    supply = [10, 12, 5, 15, 8]
    demand = [8, 11, 13, 10, 8]

    matrix = [
        [0, 1, 2, 3, 4, 7, 6, 5, 4, 3],
        [1, 0, 1, 2, 3, 6, 5, 4, 3, 2],
        [2, 1, 0, 1, 2, 4, 3, 2, 1, 2],
        [3, 2, 1, 0, 1, 8, 7, 4, 3, 6],
        [4, 3, 2, 1, 0, 2, 1, 3, 4, 2],
        [7, 6, 4, 8, 2, 0, 4, 3, 2, 1],
        [6, 5, 3, 7, 1, 4, 0, 4, 3, 2],
        [5, 4, 2, 4, 3, 3, 4, 0, 4, 3],
        [4, 3, 1, 3, 4, 2, 3, 4, 0, 4],
        [3, 2, 2, 6, 2, 1, 2, 3, 4, 0]
    ]
    start = time.time()
    solution = transporation_simplex(matrix, supply, demand)
    end = time.time()
    length = end - start
    print(length)

    visualize(solution)



