export merit_function

using LinearAlgebra

abstract type merit_function end

mutable struct L2 <: merit_function
    function_eval ::Float64
    directional_derivative ::Float64

    function L2(fx, cx, gx, μ, d)
        norm2_cx = μ * norm_cx
        gᵗd = dot(gx, d)
        new(fx + norm2_cx, gᵗd - norm2_cx)
    end
end
