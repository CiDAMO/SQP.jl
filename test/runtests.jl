using SQP

using Test

using NLPModels
using LinearAlgebra

function test()
  @testset "teste que falha" begin
    f1(x) = -x[1]*x[2]*x[3]
    x1 = [1.0, 1.0, 1.0]
    c1(x) = [x[1]*x[2] + 2*x[1]*x[3] + 2*x[2]*x[3] - 12]
    nlp1 = ADNLPModel(f1, x1, c = c1, ucon = [0], lcon = [0])
    sol1 = with_logger(NullLogger()) do
      sqp(nlp1)
    end
    @test sol1.primal_feas <= 1e-8
    @test sol1.dual_feas <= 1e-8
    @test sol1.elapsed_time < 30
    @test sol1.iter < 1000
  end
end

test()
