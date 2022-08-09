
using Snake
using CUDA
using LinearAlgebra

L = 16


@testset "No interaction" begin

    for T ∈ (Float16, Float32, Float64)
        h = diagm(rand(T, L))
        sp = exhaustive_search(h, num_states=1)

        @test sp.energies[1] ≈ -sum(h)
        @test vec(sp.states) == fill(-1, L)
    end
end
