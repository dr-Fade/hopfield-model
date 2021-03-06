using DifferentialEquations, DynamicalSystems, ChaosTools, LinearAlgebra

f(x, λ) = map(val ->
    if val < 0
        -abs(val^λ)
    else
        val^λ
    end,
    x
)

h(x, γ) = map(val ->
    if val < 0
        -(-val)^γ
    else
        val^γ
    end,
    x
)

u(x, α, β) = map(val ->
    if val < 0
        (-val)^β
    else
        val^α
    end,
    x
)

function Jac(P, Q, V, H, W, g)
    m, _ = size(P)
    Zₚ = P * 0
    Zₒ = Q * 0
    z = Array(H[:,1]) * 0
    return [
        P   Zₚ  Zₚ  Q   Zₒ  Zₒ  V[1:m,:]        -H[:,1]  H[:,2]  H[:,3]  z      V[1:m,:];
        Zₚ  P   Zₚ  Zₒ  Q   Zₒ  V[m+1:2*m,:]    -H[:,2] -H[:,1]  z       H[:,3] V[m+1:2*m,:];
        Zₚ  Zₚ  P   Zₒ  Zₒ  Q   V[2*m+1:3*m,:]  -H[:,3]  z      -H[:,1] -H[:,2] V[2*m+1:3*m,:]
    ]
end

C0_of_Y(Y) = [Y[1]; Y[5]; Y[9]]
C_of_Y(Y) = [Y[2:4] Y[6:8] Y[10:12]]
B_of_Y(Y) = [Y[13:15] Y[16:18] Y[19:21]]
α_of_Y(Y) = Y[22]
β_of_Y(Y) = Y[23]
S_of_Y(Y) = [
    -Y[24]    Y[25]   Y[26];
    -Y[25]   -Y[24]   Y[27];
    -Y[26]   -Y[27]  -Y[24]
]
γ_of_Y(Y) = Y[28]
δ_of_Y(Y) = Y[29]

function get_V(g, B, Q)
    m, _ = size(H)
    Va = map(val ->
        if val < 0 0
        else log(val) end,
        g
    )
    Vb = map(val ->
        if val < 0 (log(-val))
        else 0 end,
        g
    )
    v_column(V) = reduce(vcat,
        [[sum(B[:,j] .* Q[i,:] .* V[i,:]) for i = 1:m] for j = 1:3]
    )
    return [
        v_column(Va) v_column(Vb)
    ]
end

function get_W(g, S, H)
    m, _ = size(H)
    Wg = map(val ->
        if val < 0 0
        else log(val) end,
        g
    )
    Wd = map(val ->
        if val < 0 (-log(-val))
        else 0 end,
        g
    )
    w_column(W) = reduce(vcat,
        [[sum(S[:,j] .* H[i,:] .* W[i,:]) for i = 1:m] for j = 1:3]
    )
    return [
        w_column(Wg) w_column(Wd)
    ]
end

function construct_3d_hopfield_model(attractor::Dataset, Δt::Float64 = 1.0)
    # attractor dimensions
    N, M = size(attractor)
    if M != 3
        error("The method is implemented only for 3d systems for now.")
    end
    if N < M
        error("Attractor can't have fewer points than its dimensions")
    end
    m = N-1
    λ = 1
    g = Array(reduce(hcat, attractor[1:end-1])')
    P = Array([ones(m) f(g, λ)])
    D = Array(reduce(hcat, attractor[2:end] - attractor[1:end-1])' / Δt)

    Y = [ones(21); 1; 1; 0; 1; 1; 1; 2; 2]

    function model_error(Y)
        # α = α_of_Y(Y)
        # γ = γ_of_Y(Y)
        # β = β_of_Y(Y)
        # δ = δ_of_Y(Y)

        # Q = u(g, α, β)
        # V = get_V(g, B_of_Y(Y), Q)
        # H = h(g, γ)
        # W = get_W(g, S_of_Y(Y), H)

        # Wₖ = Jac(P, Q, V, H, W, g)
        # μ = if rank(Wₖ) == 29 0 else rand() end

        # Y = (Wₖ' * Wₖ + μ * I)^(-1) * Wₖ' * reduce(vcat, D)

        α = α_of_Y(Y)
        γ = γ_of_Y(Y)
        β = β_of_Y(Y)
        δ = δ_of_Y(Y)
        C0 = C0_of_Y(Y)
        C = C_of_Y(Y)
        B = B_of_Y(Y)
        S = S_of_Y(Y)
        H = h(g, γ)

        R = reduce(hcat,
            [C0 + C*g[i,:] + B*g[i,:] + S*H[i,:] for i = 1:m]
        )
        MSE = sum(reduce(vcat, R) - reduce(vcat, D))^2
        return MSE, Y
    end

    println("k = $best_k; mse = $best_error")
    println("C₀ = $best_C₀")
    println("C = $best_C")
    println("S = $best_S")
    println("γ = $γ")
    return best_C₀, best_C, best_S, γ
end

function test(C0::Array{Float64,1}, C::Array{Float64,2}, S::Array{Float64,2}, γ::Int64)
    function f(dx,x,p,t)
        H = h(x, γ)
        dx[:] = C0 + C*x + B*x + S*H
    end
    x0 = [10.,10.,-10.]
    tspan = (0.,200000.)
    prob = ODEProblem(f, x0, tspan)
    return solve(prob)
end