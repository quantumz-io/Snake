
using Snake

max_states = 100
L = 16
J = rand(L, L)

@time sp = brute_force(J, num_states=max_states)
@time sp = brute_force(J, num_states=max_states)

nothing
