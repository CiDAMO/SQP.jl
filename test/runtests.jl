using SQP

using Test

using NLPModels
using LinearAlgebra

function test()
  @testset "teste 1" begin
    f1(x) = (x[1] - 2x[3]*(1 - x[2]))^4 + (x[2] - x[1])^2
    x1 = [1.0, 1.0, 1.0]
    c1(x) = [x[1] * x[3], x[1] + x[3]^3 - x[2] * x[1]]
    nlp1 = ADNLPModel(f1, x1, c = c1, ucon = [1.0, 2.0], lcon = [1.0, 2.0])
    sol1 = with_logger(NullLogger()) do
      sqp(nlp1)
    end
    @test sol1.primal_feas < 1e-8
    @test sol1.dual_feas < 1e-8
    @test sol1.elapsed_time < 30
    @test sol1.iter < 1000
  end
end

test()
