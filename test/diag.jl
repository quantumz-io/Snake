using Snake
using LinearAlgebra

L = 16
num_states = 1

@testset "Correct results when no interaction" begin
    for T ∈ (Float16, Float32, Float64)
        h = diagm(rand(T, L))

        sp = exhaustive_search(h, num_states=num_states)

        @test eltype(sp.energies[1]) == T
        @test sp.energies[1] ≈ -sum(h)
        @test vec(sp.states) == fill(-1, L)
    end
end
