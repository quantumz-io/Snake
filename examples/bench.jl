
using Snake
using CUDA

T = Float64

max_states = 100
L = 16
J = rand(T, L, L)

@time sp = exhaustive_search(J, num_states=max_states)
@time sp = exhaustive_search(J, num_states=max_states)

nothing
