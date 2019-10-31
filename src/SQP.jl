module SQP


export sqp


using LinearAlgebra
using Krylov
using NLPModels
using SolverTools
using JSOSolvers


function sqp(nlp :: AbstractNLPModel;
             atol :: Real = 1e-8,
             rtol :: Real = 1e-8,
             max_eval :: Int = -1,
             cons_tol :: Real = 1e-8,
             max_iter :: Int = 1000,
             max_time :: Float64 = 30.0,
             relax_param :: Float64 = 0.8,
             trust_region_radius :: Float64 = 3.0,
             μ0 :: Float64 = 1.0)


    if bound_constrained(nlp)
        return tron(nlp, x = copy(nlp.meta.x0), max_time = max_time, atol = atol, rtol = rtol, max_eval = max_eval)
    end

    if unconstrained(nlp)
        return trunk(nlp, x = copy(nlp.meta.x0), max_time = max_time, nm_itmax = max_iter, atol = atol, rtol = rtol, max_eval = max_eval)
    end

    if equality_constrained(nlp) == false || has_bounds(nlp) == true
        error("This SQP implementation only works for equality cons.aints and unbounded variables")
    end

    iter = 1
    start_time = time()
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    x = copy(nlp.meta.x0)
    y = copy(nlp.meta.y0)
    fx = obj(nlp, x)
    cx = cons(nlp, x) - nlp.meta.ucon
    gx = grad(nlp, x)
    A = jac(nlp, x)
    W = Symmetric(hess(nlp, x, y = y), :L)
    Z = nullspace(Matrix(A))
    norm_cx = norm(cx)
    ρ = 0.0
    μ = μ0
    last_accepted_μ = μ0 # for the first iteration
    last_accepted_norm_c = norm_cx
    last_rejected = false # the information from the last two steps is needed for the μ update
    last_but_one_rejected = false

    exitflag = :unknow
    dual = A'*y - gx
    normdual = norm(dual)
    tol = atol + rtol * normdual
    success = normdual < tol && norm_cx < cons_tol
    if success
        exitflag = :first_order
    end
    Δt = time() - start_time
    tired = Δt > max_time || iter > max_iter || neval_obj(nlp) > max_eval > 0
    if tired
         if Δt > max_time
            exitflag = :max_time
        elseif iter > max_iter
            exitflag = :max_iter
        else
            exitflag = :max_eval
        end
    end

    @info log_header([:iter, :time, :dual, :normcx, :radius, :ratio], [Int, Float64, Float64, Float64, Float64, Float64])
    @info log_row(Any[iter, Δt, normdual, norm_cx, trust_region_radius, ρ])

    while !(success || tired)
        v = lsmr(A, -cx, radius = relax_param * trust_region_radius)[1]
        ZWZ = Z' * W * Z
        ZWv = Z' * (W*v + gx)
        u = cg(ZWZ, -ZWv, radius = sqrt(trust_region_radius^2 - norm(v)^2))[1]
        d = v + Z * u

        next_x = x + d
        next_f = obj(nlp, next_x)
        next_c = cons(nlp, next_x) - nlp.meta.ucon
        next_norm_c = norm(next_c)
        vpred = norm_cx - norm(A*v + cx)
        upred = 0.5 * (u'*ZWZ*u)[1] + dot(ZWv,u)
        μ_bar = 0.1 + upred / vpred # auxiliary variable for μ update

        μ_plus = max(μ, μ_bar) # another auxiliary variable
        if μ_plus > μ && μ_plus < 5*μ && μ > last_accepted_μ && norm_cx > 0.2*last_accepted_norm_c && (last_rejected || last_but_one_rejected)
            μ_plus = min(5*μ, μ_plus + 25*(μ_plus - last_accepted_μ))
        end
        if μ_plus == μ && norm(v) < relax_param * trust_region_radius/10 && norm_cx < 1e4 * tol
            μ_plus = max(μ0, μ_bar, norm(y))
        end

        ared = fx - next_f + μ_plus * (norm_cx - next_norm_c) + 1e-32 #1e-32 to avoid NaN
        pred = -0.5 * d'*W*d - dot(gx,d) + μ_plus * vpred + 1e-32
        ρ = ared / pred

        if ρ > 0
            x = next_x
            fx = next_f
            cx = next_c
            norm_cx = next_norm_c
            if norm_cx <= 1e-2
                relax_param = 0.2
            end
            A = jac(nlp, x)
            gx = grad(nlp, x)
            y = lsmr(A', gx)[1]
            W = Symmetric(hess(nlp, x, y = y), :L)
            Z = nullspace(Matrix(A))
            last_but_one_rejected = last_rejected
            last_rejected = false
            last_accepted_μ = μ
            last_accepted_norm_c = norm_cx
            if ρ >= 0.75
              trust_region_radius = trust_region_radius * 1.5
            end
        else
            trust_region_radius = trust_region_radius / 1.5
            last_rejected = true
        end
        μ = μ_plus

        dual = A'*y - gx
        normdual = norm(dual)
        success = normdual < tol && norm_cx < cons_tol
        if success
            exitflag = :first_order
        end
        Δt = time() - start_time
        iter += 1
        tired = Δt > max_time || iter > max_iter || neval_obj(nlp) > max_eval > 0
        if tired
            if Δt > max_time
                exitflag = :max_time
            elseif iter > max_iter
                exitflag = :max_iter
            else
                exitflag = :max_eval
            end
        end

        @info log_row(Any[iter, Δt, normdual, norm_cx, trust_region_radius, ρ])

    end

    cx = cx + nlp.meta.ucon

    return GenericExecutionStats(exitflag, nlp, solution = x, objective = fx, dual_feas = normdual, primal_feas = norm_cx, iter = iter, elapsed_time = Δt)

end


end # module
