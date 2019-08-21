module SQP


export sqp


using LinearAlgebra
using Krylov
using NLPModels
using SolverTools


function sqp(nlp;
             precision = 1e-8,
             max_iter = 1000,
             time_lim = 30,
             relax_param = 0.5,
             trust_reg = 1.0)

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
    tr = TrustRegion(eval(trust_reg))
    ρ = 0.0

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

    @info log_header([:iter, :time, :firt_order, :normcx, :radius, :ratio], [Int, Float64, Float64, Float64, Float64, Float64])
    @info log_row(Any[iter, now, norm_first, norm_cx, tr.radius, ρ])

    while !(success || tired)
        v = lsmr(A, -cx, radius = relax_param * trust_reg)[1]
        ZWZ = Z' * W * Z
        ZWv = Z' * (W*v + gx)
        u = cg(ZWZ, -ZWv, radius = sqrt(trust_reg^2 - norm(v)^2))[1]
        d = v + Z * u
        next_x = x + d
        next_f = obj(nlp, next_x)
        next_c = cons(nlp, next_x) - nlp.meta.ucon
        next_norm_c = norm(next_c)
        vpred = norm(A*v + cx) - norm_cx
        upred = (-0.5*u'*ZWZ*u)[1] - dot(ZWv,u)
        μ = norm(y, Inf)
        ϕ(x) = obj(nlp, x) + μ * norm(cons(nlp, x))
        nlp_aux = ADNLPModel(ϕ, x)
        ϕx = fx + μ * norm_cx
        ϕn = next_f + μ * next_norm_c
        Δm = μ * vpred - upred

        ared, pred = aredpred(nlp_aux, ϕx, ϕn, Δm, next_x, d, dot(d, grad(nlp_aux,x)))
        ρ = ared / pred
        set_property!(tr, :ratio, ρ)
        tr = update!(tr, norm(d))

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
        end

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
