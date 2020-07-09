module SQP

export sqp

using LinearAlgebra
using Krylov
using NLPModels
using SolverTools
using JSOSolvers

function sqp(nlp :: AbstractNLPModel;
    	     x :: AbstractVector = copy(nlp.meta.x0),
             atol :: Real = sqrt(eps(eltype(x))),
             rtol :: Real = sqrt(eps(eltype(x))),
             max_eval :: Int = -1,
             max_iter :: Int = 1000,
             max_time :: Float64 = 30.0,
             relax_param :: Float64 = 0.6,
             trust_reg :: Float64 = 1.0,
             μ0 :: Float64 = 1.0)

    if bound_constrained(nlp)
        return tron(nlp, x = copy(nlp.meta.x0), max_time = max_time, atol = atol, rtol = rtol, max_eval = max_eval)
    end

    if unconstrained(nlp)
        return trunk(nlp, x = copy(nlp.meta.x0), max_time = max_time, nm_itmax = max_iter, atol = atol, rtol = rtol, max_eval = max_eval)
    end

    if equality_constrained(nlp) == false || has_bounds(nlp) == true
        error("This SQP implementation only works for equality constraints and unbounded variables")
    end

    iter = 0
    start_time = time()
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    fx = obj(nlp, x)
    cx = cons(nlp, x) - nlp.meta.ucon
    gx = grad(nlp, x)
    A = jac(nlp, x)
    LSMR_dual = lsmr(A', gx)
    y = LSMR_dual[1]
    maior_y = norm(y, Inf)
    W = Symmetric(hess(nlp, x, y), :L)
    Z = nullspace(Matrix(A))
    norm_cx = norm(cx)
    tr = TrustRegion(trust_reg)
    ρ = 0.0
    μ = μ0

    exitflag = :unknow
    dual = A'*y - gx
    normdual = norm(dual)
    success = normdual <  atol + rtol * normdual && norm_cx < atol + rtol * norm_cx
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
    @info log_row(Any[iter, Δt, normdual, norm_cx, tr.radius, ρ])

    while !(success || tired)
        LSMR = lsmr(A, -cx, radius = relax_param * tr.radius, atol = atol, rtol = rtol, itmax = max(2 * n, 50))
        v = LSMR[1]
	    ZWZ = Z' * W * Z
        ZWv = Z' * (W*v + gx)
        CG = cg(ZWZ, -ZWv, radius = sqrt(tr.radius^2 - norm(v)^2), atol = atol, rtol= 0.0, itmax = max(2 * n, 50))
	    u = CG[1]
	    Zu = Z * u
        d = v + Zu
	    norm_v =  norm(v)

        next_x = x + d
        next_f = obj(nlp, next_x)
        next_c = cons(nlp, next_x) - nlp.meta.ucon
        next_norm_c = norm(next_c, 1)
        vpred = norm_cx - norm(A*v + cx)
        D = dot(d, gx)
        dWd = (d'*W*d)[1]
        dpred = 0.5 * dWd + D

        μ_bar = maior_y + 0.1
        if(dWd >= 0)
            if(μ > μ_bar)
                μ_plus = μ
            else
                μ_plus = μ_bar
            end
        else
            μ_plus = μ_bar - dWd / norm_cx
        end

        μ = max(μ_plus, dpred / (0.7 * vpred) + 0.1)

        ϕx = fx + μ * norm_cx
        ϕn = next_f + μ * next_norm_c
        Δm =  μ * vpred - dpred
	    μ_norm_cx = μ * norm_cx

        ared, pred = aredpred(nlp, ϕn, ϕx, Δm, next_x, d, D - μ_norm_cx)
        ρ = ared / pred
        set_property!(tr, :ratio, ρ)

        normv = norm(v)
        normu = norm(Zu)

        if acceptable(tr)
	        x = next_x
	        fx = next_f
	        cx = next_c
	        norm_cx = next_norm_c
	        A = jac(nlp, x)
	        gx = grad(nlp, x)
            LSMR_dual = lsmr(A', gx)
	        y = LSMR_dual[1]
	        W = Symmetric(hess(nlp, x, y), :L)
	        Z = nullspace(Matrix(A)) 
	    elseif (normv <= 0.1 * normu) && (normv <= 0.8 * relax_param * tr.radius)
            LSMR_aux = lsmr(A, next_c, atol = atol, rtol = rtol)
		    w = LSMR_aux[1]
		    d_soc = d + w
	        next_x = x + d_soc
	        next_f = obj(nlp, next_x)
	        next_cx = cons(nlp, next_x) - nlp.meta.ucon
		    next_norm_c = norm(next_cx)
   		    ϕx = fx + μ * norm_cx
            ϕn = next_f + μ * next_norm_c
		    D = dot(d_soc, gx)
		    Δm =  μ * (norm_cx - norm(A*d_soc + cx)) - 0.5 * (d_soc'*W*d_soc)[1] - D
		
		    ared, pred = aredpred(nlp, ϕn, ϕx, Δm, next_x, d, D - μ_norm_cx)
            ρ = ared / pred
            set_property!(tr, :ratio, ρ)
		
		    if acceptable(tr)
			    x = next_x
                fx = next_f
                cx = next_c
                norm_cx = next_norm_c
			    A = jac(nlp, x)
                gx = grad(nlp, x)
                LSMR_dual = lsmr(A', gx)
                y = LSMR_dual[1]
         	    W = Symmetric(hess(nlp, x, y), :L)
       	        Z = nullspace(Matrix(A))
            end
        end
		
        update!(tr, norm(d, 2))

        dual = A'*y - gx
        normdual = norm(dual)
        success = normdual < atol + rtol * normdual && norm_cx < atol + rtol * norm_cx
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

        @info log_row(Any[iter, Δt, normdual, norm_cx, tr.radius, ρ])
    end

    cx = cx + nlp.meta.ucon

    return GenericExecutionStats(exitflag, nlp, solution = x, objective = fx, dual_feas = normdual, primal_feas = norm_cx, iter = iter, elapsed_time = Δt)
end

end # module
