export exhaustive_search

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
