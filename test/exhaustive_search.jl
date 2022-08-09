using Snake
using LinearAlgebra

num_states = 1

@testset "Correct results when no interaction" begin
    for T ∈ (Float16, Float32, Float64)
        for L ∈ 2 .^ collect(1:4)
            h = diagm(rand(T, L))

            sp = exhaustive_search(h, num_states=num_states)

            @test eltype(sp.energies[1]) == T
            @test sp.energies[1] ≈ -sum(h)
            @test vec(sp.states) == fill(-1, L)
        end
    end
end

@testset "Correct results for 1D chain with no biases" begin
    for T ∈ (Float16, Float32, Float64)
        for L ∈ 2 .^ collect(1:4)
            J = zeros(T, L, L)
            for i ∈ 1:L-1 J[i, i+1] = one(T) end

            sp = exhaustive_search(J, num_states=num_states)

            @test eltype(sp.energies[1]) == T
            @test sp.energies[1] ≈ -sum(J)
        end
    end
end
