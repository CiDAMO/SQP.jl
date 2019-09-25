module SQP


export sqp


using LinearAlgebra
using Krylov
using NLPModels
using SolverTools
using JSOSolvers


function sqp(nlp;
             precision = 1e-8,
             max_iter = 1000,
             time_lim = 30.0,
             relax_param = 0.5,
             trust_reg = 3.0,
             μ0 = 1.0)


    if bound_constrained(nlp)
        return tron(nlp, x = copy(nlp.meta.x0), max_time = time_lim, atol = precision)
    end

    if unconstrained(nlp)
        return trunk(nlp, x = copy(nlp.meta.x0), max_time = time_lim, nm_itmax = max_iter, atol = precision)
    end

    if equality_constrained(nlp) == false || has_bounds(nlp) == true
        error("This SQP implementation only works for equality constraints and unbounded variables")
    end

    iter = 1
    time_0 = time()
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    x = copy(nlp.meta.x0)
    y = copy(nlp.meta.y0)
    fx = obj(nlp, x)
    cx = cons(nlp, x) - nlp.meta.ucon
    gx = grad(nlp, x)
    A = jac(nlp, x)
    W = Symmetric(hess(nlp, x, y = y), :L)
    Z = nullspace(A)
    norm_cx = norm(cx)
    tr = TrustRegion(trust_reg)
    ρ = 0.0
    μ = μ0
    last_accepted_μ = μ0 # for the first iteration
    last_accepted_norm_c = norm_cx
    last_rejected = false # the information from the last two steps is needed for the μ update
    last_but_one_rejected = false

    exitflag = :unknow
    norm_first = norm(A'*y - gx)
    success = norm_first < precision && norm_cx < precision
    if success
        exitflag = :first_order
    end
    now = time() - time_0
    tired = now > time_lim || iter > max_iter
    if tired
         if now > time_lim
            exitflag = :max_time
        else
            exitflag = :max_iter
        end
    end

    @info log_header([:iter, :time, :dual, :normcx, :radius, :ratio], [Int, Float64, Float64, Float64, Float64, Float64])
    @info log_row(Any[iter, now, norm_first, norm_cx, tr.radius, ρ])

    while !(success || tired)
        v = lsmr(A, -cx, radius = relax_param * tr.radius)[1]
        ZWZ = Z' * W * Z
        ZWv = Z' * (W*v + gx)
        u = cg(ZWZ, -ZWv, radius = sqrt(tr.radius^2 - norm(v)^2))[1]
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
        if μ_plus == μ && norm(v) < relax_param * tr.radius/10 && norm_cx < 1e4 * precision
            μ_plus = max(μ0, μ_bar, norm(y))
        end

        ϕ(x) = obj(nlp, x) + μ_plus * norm(cons(nlp, x))
        nlp_aux = ADNLPModel(ϕ, x)
        ϕx = fx + μ_plus * norm_cx
        ϕn = next_f + μ_plus * next_norm_c
        Δm = μ * vpred - upred

        ared, pred = aredpred(nlp_aux, ϕn, ϕx, Δm, next_x, d, dot(d, grad(nlp_aux,x)))
        ρ = ared / pred
        set_property!(tr, :ratio, ρ)

        if acceptable(tr)
            x = next_x
            fx = next_f
            cx = next_c
            norm_cx = next_norm_c
            A = jac(nlp, x)
            gx = grad(nlp, x)
            y = lsmr(A', gx)[1]
            W = Symmetric(hess(nlp, x, y = y), :L)
            Z = nullspace(A)
            last_but_one_rejected = last_rejected
            last_rejected = true
            last_accepted_μ = μ
            last_accepted_norm_c = norm_cx
        end

        μ = μ_plus

        update!(tr, norm(d))

        norm_first = norm(A'*y - gx)
        success = norm_first < precision && norm_cx < precision
        if success
            exitflag = :first_order
        end
        now = time() - time_0
        iter += 1
        tired = now > time_lim || iter > max_iter
        if tired
            if now > time_lim
                exitflag = :max_time
            else
                exitflag = :max_iter
            end
        end

        @info log_row(Any[iter, now, norm_first, norm_cx, tr.radius, ρ])

    end

    cx = cx + nlp.meta.ucon

    return GenericExecutionStats(exitflag, nlp, solution = x, objective = fx, dual_feas = norm_first, primal_feas = norm_cx, iter = iter, elapsed_time = now)

end


end # module
