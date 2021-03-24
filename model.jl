using DifferentialEquations, DynamicalSystems, ChaosTools, LinearAlgebra

h(x, γ) = map(val ->
    if val < 0
        -abs(val^γ)
    else
        val^γ
    end,
    x
)

function Jac(P, H, g)
    Z = P * 0
    z = Array(H[:,1]) * 0
    return [
        P   Z   Z   -H[:,2]  H[:,3]  z;
        Z   P   Z   -H[:,1]  z       H[:,3];
        Z   Z   P    z      -H[:,1] -H[:,2]
    ]
end

function construct_3d_hopfield_model(attractor::Dataset, r::Float64 = 0.0, Δt::Float64 = 1.0, s::Int64 = 1)
    # attractor dimensions
    N, M = size(attractor)
    if M != 3
        error("The method is implemented only for 3d systems for now.")
    end
    if N < M
        error("Attractor can't have fewer points than its dimensions")
    end
    # number of training elements
    m = N-1
    # initial values
    Y = zeros(15)
    γ = 1
    # optimization algorithm
    best_error = Inf
    best_k = 0
    best_C₀ = nothing
    best_C = nothing
    best_S = nothing
    #     (1 g₁₁ g₂₁ g₃₁)
    # P = ( . . . . . . )
    #     (1 g₁ₘ a₂ₘ g₃ₘ)
    g = Array(reduce(hcat, attractor[1:end-1])')
    P = Array([ones(m) g])
    #      (  gᵢ₁ - gᵢ₀  )
    # Dᵢ = (     ...     ) / Δt
    #      ( gᵢₘ - gᵢₘ₋₁ )
    D = Array(reduce(hcat, attractor[2:end] - attractor[1:end-1])' / Δt)
    k = 0
    R = zeros(0,m)
    while true
        γ = 1 + k * s
        #         (  gᵢ₁ - gᵢ₀  )
        # h(aᵢ) = (     ...     )
        #         ( gᵢₘ - gᵢₘ₋₁ )
        H = h(g, γ)
        Wₖ = Jac(P, H, g)
        μ = if rank(Wₖ) == 15 0 else rand() end
        # c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, d12, d13, d23
        Y = (Wₖ' * Wₖ + μ * I)^(-1) * Wₖ' * reduce(vcat, D)
        C₀ = Y[1:3]
        C = [Y[2:4] Y[5:7] Y[8:10]]
        S = [
            -r        Y[13]    Y[14];
            -Y[13]   -r        Y[15];
            -Y[14]   -Y[15]   -r
        ]
        R = zeros(m,M)
        for j ∈ 1:m
            R[j,:] = C₀ + C*g[j,:] + S*H[j,:]
        end
        MSE = sum(reduce(vcat, R) - reduce(vcat, D))^2
        if k < M
            k += 1
            old_best_error = best_error
            best_error = min(MSE, best_error)
            if best_error != old_best_error
                best_k = k
                best_C₀ = C₀
                best_C = C
                best_S = S
            end
        else
            break
        end
    end
    println("k = $best_k; mse = $best_error")
    println("C₀ = $best_C₀")
    println("C = $best_C")
    println("S = $best_S")
    println("γ = $γ")
    return best_C₀, best_C, best_S, γ
end

function test(C₀::Array{Float64,1}, C::Array{Float64,2}, S::Array{Float64,2}, γ::Int64)
    function f(dx,x,p,t)
        H = h(x, γ)
        dx[:] = C₀ + C*x + S*H
    end
    x0 = [10.,10.,-10.]
    tspan = (0.,20000.)
    prob = ODEProblem(f, x0, tspan)
    return solve(prob)
end