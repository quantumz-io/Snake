using Snake
using Test

my_tests = []

push!(my_tests,
    "exhaustive_search.jl",
)

for my_test ∈ my_tests
    include(my_test)
end
