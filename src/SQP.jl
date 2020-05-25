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
             max_iter :: Int = 25,
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
    y = copy(nlp.meta.y0)
    fx = obj(nlp, x)
    cx = cons(nlp, x) - nlp.meta.ucon
    gx = grad(nlp, x)
    A = jac(nlp, x)
    W = Symmetric(hess(nlp, x, y), :L)
    Z = nullspace(Matrix(A))
    norm_cx = norm(cx)
    tr = TrustRegion(trust_reg)
    ρ = 0.0
    μ = μ0
    last_accepted_μ = μ0 # for the first iteration
    last_accepted_norm_c = norm_cx
    last_rejected = false # the information from the last two steps is needed for the μ update
    last_but_one_rejected = false

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
        v = lsmr(A, -cx, radius = relax_param * tr.radius, atol = atol, rtol = rtol, itmax = max(2 * n, 50))[1]
        ZWZ = Z' * W * Z
        ZWv = Z' * (W*v + gx)
        u = cg(ZWZ, -ZWv, radius = sqrt(tr.radius^2 - norm(v)^2), atol = atol, rtol = 0.0, itmax = max(2 * n, 50))[1]
	Zu = Z * u
        d = v + Zu

        next_x = x + d
        next_f = obj(nlp, next_x)
        next_c = cons(nlp, next_x) - nlp.meta.ucon
        next_norm_c = norm(next_c)
        vpred = norm_cx - norm(A*v + cx)
        upred = 0.5 * (u'*ZWZ*u)[1] + dot(ZWv,u)
        μ_bar = 0.1 + upred / vpred # auxiliary variable for μ update

        μ_plus = max(μ, μ_bar) # another auxiliary variable
        if μ_plus > μ && μ_plus < 5 * μ && μ > last_accepted_μ && norm_cx > 0.2 * last_accepted_norm_c && (last_rejected || last_but_one_rejected)
            μ_plus = min(5 * μ, μ_plus + 25 * (μ_plus - last_accepted_μ))
        end
        if μ_plus == μ && norm(v) < relax_param * tr.radius/10 && norm_cx < 1e4 * atol
            μ_plus = max(μ0, μ_bar, norm(y))
        end

        #ϕ(x) = obj(nlp, x) + μ_plus * norm(cons(nlp, x))
     	#nlp_aux = ADNLPModel(ϕ, x)
        ϕx = fx + μ_plus * norm_cx
        ϕn = next_f + μ_plus * next_norm_c
        Δm =  μ * vpred - upred

        ared, pred = aredpred(nlp, ϕn, ϕx, Δm, next_x, d, dot(d, gx) - μ_plus * norm_cx)
        ρ = ared / pred
        set_property!(tr, :ratio, ρ)

	norm_v = norm(v)
	norm_Zu = norm(Zu)


        if acceptable(tr) 
            x = next_x
            fx = next_f
            cx = next_c
            norm_cx = next_norm_c
            A = jac(nlp, x)
            gx = grad(nlp, x)
            y = lsmr(A', gx)[1]
            W = Symmetric(hess(nlp, x, y), :L)
            Z = nullspace(Matrix(A))
            last_but_one_rejected = last_rejected
            last_rejected = false
            last_accepted_μ = μ
            last_accepted_norm_c = norm_cx
	elseif norm_v < 0.8 * relax_param * tr.radius && norm_v < 0.1 * norm_Zu
		aux_sist = cg(A*A', next_c)[1]
		d -= A'*aux_sist
		next_x = x + d
		next_f = obj(nlp, next_x)
		next_cx = cons(nlp, next_x) - nlp.meta.ucon
		next_norm_c = norm(next_cx)
   		ϕx = fx + μ_plus * norm_cx
        		ϕn = next_f + μ_plus * next_norm_c
        	Δm =  μ * vpred - upred
		ared, pred = aredpred(nlp, ϕn, ϕx, Δm, next_x, d, dot(d, gx) - μ_plus * norm_cx)
        	ρ = ared / pred
        	set_property!(tr, :ratio, ρ)
		x = next_x
            	fx = next_f
            	cx = next_c
            	norm_cx = next_norm_c
		A = jac(nlp, x)
            	gx = grad(nlp, x)
            	y = lsmr(A', gx)[1]
         	W = Symmetric(hess(nlp, x, y), :L)
       	     	Z = nullspace(Matrix(A))
            	last_but_one_rejected = last_rejected
            	last_rejected = false
            	last_accepted_μ = μ
            	last_accepted_norm_c = norm_cx
	else
		last_but_one_rejected = last_rejected
            	last_rejected = true
        end

        μ = μ_plus

        update!(tr, norm(d))

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
