using PyCall
nx = pyimport("networkx")

# Window size
window_x = 100
window_y = 100

# defining snake default position
snake_position = [1, 1]
 
# defining first 4 blocks of snake body
snake_body = [[1, 1]]

# fruit position
fruit_position = [rand(2:(window_x//50)),
                  rand(2:(window_y//50))]
fruit_spawn = true

# setting default snake direction towards
# right
direction = "RIGHT"
change_to = direction

function solve()
    global snake_body, snake_position, fruit_position, fruit_spawn, direction, window_x, window_y
    height = window_y//50
    width  = window_x//50
    ## In the hierachy of step from A* search: 
        #make the graph 
        #input to solving_HCP 
        #return the result
    
    l_direct = ["UP", "DOWN", "LEFT", "RIGHT"]
    stack = []
    for d in l_direct
        new_pos = false 
        if d == "UP" && direction != "DOWN"
            new_pos = [snake_position[1], snake_position[2]-1]
        end
        if d == "DOWN" && direction!= "UP"
            new_pos = [snake_position[1], snake_position[2]+1]
        end
        if d == "LEFT" && direction != "RIGHT"
            new_pos = [snake_position[1]-1, snake_position[2]]
        end
        if d == "RIGHT" && direction != "LEFT"
            new_pos = [snake_position[1]+1, snake_position[2]]
        end
        if new_pos != false && 1 <= new_pos[1] <= width && 1 <= new_pos[2] <= height && new_pos ∉ snake_body
            insert!(stack, 1, [d, 
                        abs(fruit_position[1] - new_pos[1]) + 
                        abs(fruit_position[2] - new_pos[2]), new_pos])
        end
    end
    stack = sort!(stack, by = x -> x[2])
    #println(stack)
    #println(fruit_position)
    #Creating new graph for each direction 
    for (d,s, new_pos) in stack
        G = nx.Graph()
        G.add_nodes_from(1:width*height)
        # Make a fully connected graph out of the board 
        for i=1:width-1 
            for j=1:height-1 
                G.add_edge(i+j*width, i+1+j*width)
                G.add_edge(i+j*width, i+(j+1)*width)
            end
        end
        
        for j=1:height-1 
            G.add_edge(width + j*width, width + (j+1)*width)
        end
        
        for i=1:width - 1
            G.add_edge(i + height*width, i + 1 + height*width)
        end
        

        #Removing edge conneting snake with neighbor squares. 
        for (i,pos) in enumerate(snake_body) 
            if 1<i<length(snake_body) 
                neighbors = [[pos[1]+1, pos[2]], [pos[1]-1, pos[2]], [pos[1], pos[2]+1], [pos[1], pos[2]-1]]
                for ne in neighbors
                    if ne!=snake_body[i+1] && ne!=snake_body[i-1] && 1 <= ne[1] <= width && 1 <= ne[2] <= height 
                        try
                            G.remove_edge(pos[1] + pos[2]*width, ne[1] + ne[2]*width)
                        catch e 
                        end
                    end
                end
            end
        end
        

        #Modifying the graph according to the new direction 
        neighbors = [[snake_position[1]+1, snake_position[2]], [snake_position[1]-1, snake_position[2]], 
                        [snake_position[1], snake_position[2]+1], [snake_position[1], snake_position[2]-1]]
        for ne in neighbors 
            if length(snake_body) > 1 
                if ne ∉ snake_body && ne != new_pos && 1 <= ne[1] <= width && 1 <= ne[2] <= height
                    G.remove_edge(snake_position[1] + snake_position[2]*width, ne[1] + ne[2]*width)
                end
            end
        end

        qubo_matrix = generate_qubo(G)
        #res = solve_BF(qubo_matrix)            
        #res = PT.solve_PT(G)
        
        if res==false 
            continue
        end
        
        path_pos = []
        for k in res 
            append!(path_pos, [k%width*50 + 25, k//width*50 + 25])
        end

        new_game(d)
        return path_pos
    end           
end

function generate_qubo(G)
    n = G.order()
    varsDict = Dict()
    orderDict = Dict()
    index = 1
    for i=1:n
        for j=1:n 
            varsDict[(i,j)] = index 
            orderDict[index] = (i,j)
            index += 1
        end
    end
    # initialize Q
    Q = Dict()
    for i=1:n*n
        for j=1:n*n 
            Q[i,j] = 0
        end
    end
    # p_1
    for i=1:n
        for iprime=1:n
            index = varsDict[(i,iprime)]
            Q[index,index] -= 2
        end
        for iprime1=1:n
            for iprime2=1:n
                index1 = varsDict[(i,iprime1)]
                index2 = varsDict[(i,iprime2)]
                Q[index1 , index2] += 1
            end
        end
    end
    # p_2
    for iprime =1:n 
        for i =1:n
            index = varsDict[(i,iprime)] 
            Q[index,index] -= 2
        end
        for i1 = 1:n
            for i2 = 1:n
                index1 = varsDict[(i1,iprime)]
                index2 = varsDict[(i2,iprime)]
                Q[index1 , index2] += 1
            end
        end
    end
    
    # h
    for i_1 = 1:n
        for i_2 = 1:n
            if (i_2 ∉ G.neighbors(i_1)) && (i_1 != i_2)
                for j = 1:n-1
                    index_1 = varsDict[i_1,j]
                    index_2 = varsDict[i_2,j+1]
                    Q[index_1, index_2] += 1
                end
                index_1 = varsDict[i_1, n]
                index_2 = varsDict[i_2, 1]
                Q[index_1 ,index_2] += 1
            end
        end
    end
    
    # Making Q uppertriangular for i in range(n*n):
    for i = 1:n*n
        for j = 1:n*n
            if (i > j) && (Q[i,j]!=0)
                Q[j,i] += Q[i,j] 
                Q[i,j] = 0
            end
        end
    end
    J = [[0 for j=1:n*n] for i=1:n*n]
    for i=1:n*n
        for j = 1:n*n
            J[i][j] = Q[i,j]
        end
    end
    return J,n*n 
end

function generate_path(ans, orderDict)
    res = []
    for i in ans
        if ans[i] == 1 
            append!(res, orderDict[int(i)])
        end
    end
    res = sort!(res, by = x -> x[2])
    path = [i for (i,j) in res]
    return path
end

function check(ans, Q, G) 
    E = 0 
    for (i,j) in Q
        E += Q[i,j]*ans[string(i)]*ans[string(j)]
        #E += Q[i,j]*ans[i]*ans[j]
    end
    if E != -2*length(G.nodes)
        return false
    end
    return true
end

function new_game(change_to)
    global snake_body, snake_position, fruit_position, fruit_spawn, direction

    direction = change_to
 
    # Moving the snake
    if direction == "UP"
        snake_position[2] -= 1
    elseif direction == "DOWN"
        snake_position[2] += 1
    elseif direction == "LEFT"
        snake_position[1] -= 1
    else
        snake_position[1] += 1
    end

    # Snake body growing mechanism
    # if fruits and snakes collide then the snake's length will be incremented by one
    insert!(snake_body, 1, snake_position)
    if snake_position[1] == fruit_position[1] && snake_position[2] == fruit_position[2]
        fruit_spawn = false
    else
        pop!(snake_body)
    end
         
    if not fruit_spawn
        choices = []
        for i=1:window_x//50 
            for j=1:window_y//50
                if [i,j] not in snake_body 
                    append!(choice, [i,j])
                end
            end
        fruit_position = rand(choices)
        end
         
    fruit_spawn = true
    end
end


while true
    path = solve()
    if length(snake_body) == window_x//50 * window_y//50 - 1
        #If the snake and the fruit fill the entire board then halt the game  
        print("Victory")
        break
    end
end

println(solve())

ssolve()

solve()


