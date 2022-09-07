import Pkg
Pkg.add("LightGraphs")
Pkg.add("Parameters")
Pkg.add("LinearAlgebra")
#Pkg.add("MKL")
Pkg.add("CUDA")
Pkg.add("Bits")
Pkg.add("DocStringExtensions")

using LightGraphs
using Parameters

using LinearAlgebra#, MKL
using CUDA
using Bits
using DocStringExtensions

struct Spectrum{T <: Real}
    energies::Array{T}
    states::Array{Int, 2}
end

function _energy_kernel(J, energies, σ)
    T = eltype(J)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    L = size(σ, 1)
    for j ∈ idx:stride:length(energies)
        for i=1:L if tstbit(j, i) @inbounds σ[i, j] = 1 end end
        en = zero(T)
        for k=1:L
            @inbounds en += J[k, k] * σ[k, j]
            for l=k+1:L @inbounds en += σ[k, j] * J[k, l] * σ[l, j] end
        end
        energies[j] = en
    end
    return
end

"""
$(TYPEDSIGNATURES)
Computes the low energy spectrum for the Ising model.
"""
function exhaustive_search(J::Array{<:Real, 2}; num_states::Int=1)
    T = eltype(J)
    L = size(J, 1)
    N = 2 ^ L

    J_d = CUDA.CuArray(J)
    energies = CUDA.zeros(T, N)
    σ = CUDA.fill(Int32(-1), L, N)

    th = 2 ^ 10
    bl = cld(N, th)

    @cuda threads=th blocks=bl _energy_kernel(J_d, energies, σ)

    perm = sortperm(energies)[1:num_states]

    Spectrum{T}(Array(energies[perm]), Array(σ[:, perm]))
end

# Window size
window_x = 100
window_y = 100

# defining snake default position
snake_position = [1, 1]
 
# defining first 4 blocks of snake body
snake_body = [[1, 1]]

# fruit position
fruit_position = [rand(1:(window_x//50)),
                  rand(1:(window_y//50))]
fruit_spawn = true

# setting default snake direction towards
# right
direction = "RIGHT"
change_to = direction

function solve()
    global snake_body, snake_position, fruit_position, fruit_spawn, direction, window_x, window_y
    height = Int32(window_y//50)
    width  = Int32(window_x//50)
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
        G = SimpleGraph(width*height)
        # Make a fully connected graph out of the board 
        for i=1:width-1 
            for j=1:height-1
                add_edge!(G, i+(j-1)*width, i+1+(j-1)*width)
                add_edge!(G, i+(j-1)*width, i+j*width)
            end
        end
        
        for j=1:height-1 
            add_edge!(G, width + (j-1)*width, width + j*width)
        end
        
        for i=1:width - 1
            add_edge!(G, i + (height-1)*width, i + 1 + (height-1)*width)
        end
        

        #Removing edge conneting snake with neighbor squares. 
        for (i,pos) in enumerate(snake_body) 
            if 1<i<length(snake_body) 
                neighbors = [[pos[1]+1, pos[2]], [pos[1]-1, pos[2]], [pos[1], pos[2]+1], [pos[1], pos[2]-1]]
                for ne in neighbors
                    if ne!=snake_body[i+1] && ne!=snake_body[i-1] && 1 <= ne[1] <= width && 1 <= ne[2] <= height 
                        try
                            rem_edge!(G, pos[1] + (pos[2]-1)*width, ne[1] + (ne[2]-1)*width)
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
                    rem_edge!(G, snake_position[1] + (snake_position[2]-1)*width, ne[1] + (ne[2]-1)*width)
                end
            end
        end
        
        ising_matrix, orderDict, size, Q = generate_qubo(G)
        res = exhaustive_search(ising_matrix)
        state = res.states
        ans = [0 for i=1:size]
        for (i,s) in enumerate(state[:,1])
            if s == 1
                ans[i] = s
            end
        end
        
        if check(ans, Q, G) == false 
            continue
        end
        
        path_pos = []
        for k in generate_path(ans, orderDict) 
            append!(path_pos, [k%width*50 + 25, k//width*50 + 25])
        end

        new_game(d)
        return path_pos
    end           
end

function generate_qubo(G)
    n = length(collect(vertices(G)))
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
            if (i_2 ∉ neighbors(G, i_1)) && (i_1 != i_2)
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
    J = Array{Float64}(undef,n*n, n*n)
    for i=1:n*n
        for j = i+1:n*n
            J[i,j] = 1/4*(Q[i,j] + Q[j,i])
        end
        for j = 1:i-1
            J[i,j] = 0.0
        end
        J[i,i] = 1/4*sum(Q[i,j] + Q[j,i] for j=1:n*n)
        
    end
    return J, orderDict, n*n, Q
end

function generate_path(ans, orderDict)
    res = []
    for (i,s) in enumerate(ans)
        if s == 1 
		append!(res, [orderDict[i]])
        end
    end
    res = sort!(res, by = x -> x[2])
    path = [i for (i,j) in res]
    return path
end

function check(ans, Q, G) 
    E = 0 
    for (i,j) in keys(Q)
       # E += Q[i,j]*ans[string(i)]*ans[string(j)]
        E += Q[i,j]*ans[i]*ans[j]
    end
    if E != -2*length(collect(vertices(G)))
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
         
    if fruit_spawn==false
        choices = []
        for i=1:window_x//50 
            for j=1:window_y//50
                if [i,j] ∉ snake_body 
			append!(choices,[[i,j]])
                end
            end
        fruit_position = rand(choices)
        end
         
    fruit_spawn = true
    end
end

@time begin
    while true
        path = solve()
        if length(snake_body) == window_x//50 * window_y//50 - 1
        #If the snake and the fruit fill the entire board then halt the game  
            print("Victory")
            break
        end
    end
end


