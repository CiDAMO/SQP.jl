using SQP

using JSOSolvers, LinearAlgebra, Logging, SparseArrays, Test

const jsosolvers_folder = joinpath(dirname(pathof(JSOSolvers)), "..", "test", "solvers")
include(joinpath(jsosolvers_folder, "unconstrained.jl"))

using NLPModels

function test()
  # unconstrained tests from JSOSolvers
  # test_unconstrained_solver(sqp)

  @testset "Small equality constrained problems" begin
    for (x0, m, f, c, sol) in [([1.0; 2.0], 1,
                                x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
                                x -> [4x[1] + 6x[2] - 10],
                                ones(2)
                               ),
                               ([-1.2; 1.0], 1,
                                x -> (x[1] - 1)^2,
                                x -> [10 * (x[2] - x[1]^2)],
                                ones(2)
                               ),
                               ([-1.2; 1.0], 1,
                                x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
                                x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
                                ones(2)
                               ),
                               ([2.0; 1.0], 2,
                                x -> -x[1],
                                x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 12],
                                [4.0; 3.0]
                               )
                              ]
      nlp = ADNLPModel(f, x0, c=c, lcon=zeros(m), ucon=zeros(m))
      output = with_logger(NullLogger()) do
        sqp(nlp)
      end

      @test isapprox(output.solution, sol, rtol=1e-6)
      @test output.primal_feas < 1e-6
      @test output.dual_feas < 1e-6
      @test output.status == :first_order
    end
  end

end

test()
