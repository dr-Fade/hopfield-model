import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def h(x, γ):
    result = x.copy()
    N, M = x.shape
    for i in range(0, N):
        for j in range(0, M):
            val = result[i,j]
            result[i,j] = -abs(val**γ) if val < 0 else val**γ
    return result

def Jac(P, H, g):
    Z = P * 0
    z = np.array([H[:,0]]).T * 0
    row1 = P
    row1 = np.append(row1, Z, 1)
    row1 = np.append(row1, Z, 1)
    row1 = np.append(row1, np.array([H[:,1]]).T, 1)
    row1 = np.append(row1, np.array([H[:,2]]).T, 1)
    row1 = np.append(row1, z, 1)
    row2 = Z
    row2 = np.append(row2, P, 1)
    row2 = np.append(row2, Z, 1)
    row2 = np.append(row2, np.array([H[:,0]]).T, 1)
    row2 = np.append(row2, z, 1)
    row2 = np.append(row2, np.array([H[:,2]]).T, 1)
    row3 = Z
    row3 = np.append(row3, Z, 1)
    row3 = np.append(row3, P, 1)
    row3 = np.append(row3, z, 1)
    row3 = np.append(row3, np.array([H[:,0]]).T, 1)
    row3 = np.append(row3, np.array([H[:,1]]).T, 1)
    return np.append(np.append(row1, row2, 0), row3, 0)

def construct_3d_hopfield_model(attractor: np.ndarray, r = 0, Δt = 1.0, step = 1):
    # attractor dimensions
    N, M = attractor.shape
    if M != 3:
        raise Exception("The method is implemented only for 3d systems for now.")
    # number of training elements
    m = 15
    # initial values
    Y = np.zeros(15)
    γ = 1
    # optimization algorithm
    best_error = np.inf
    best_k = 0
    best_C0 = np.zeros(M)
    best_C = np.zeros((M,M))
    best_S = np.zeros((M,M))
    for i in range(1, N-m-1):
        #     (1 a₁₁ a₂₁ a₃₁)
        # P = ( . . . . . . )
        #     (1 a₁ₘ a₂ₘ a₃ₘ)
        g = attractor[i:i+m]
        P = np.append(np.ones((m,1)), g, axis = 1)
        #      (  aᵢ₁ - aᵢ₀  )
        # Dᵢ = (     ...     ) / Δt
        #      ( aᵢₘ - aᵢₘ₋₁ )
        D = (attractor[i:i+m] - attractor[i-1:i+m-1]) / Δt
        k = 1
        R = np.zeros((0,m))
        while (True):
            γ = 1 + k * step
            #         (  aᵢ₁ - aᵢ₀  )
            # h(aᵢ) = (     ...     ) / Δt
            #         ( aᵢₘ - aᵢₘ₋₁ )
            H = h(g, γ) / Δt
            Wₖ = Jac(P, H, g)
            μ = 0
            if np.linalg.matrix_rank(Wₖ) == 15:
                μ = np.random.random_sample()
            Wₖᵀ = np.transpose(Wₖ)
            Wₖˢ = np.matmul(Wₖᵀ, Wₖ)
            inverse_square = np.linalg.inv(Wₖˢ + μ * np.identity(max(Wₖˢ.shape)))
            c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, d12, d13, d23 = np.matmul(np.matmul(inverse_square, Wₖᵀ), D.flatten())
            C0 = np.array([[c10], [c20], [c30]])
            C = np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])
            S = np.array(np.append([-r], [d12, d13]))
            S = np.append([S], np.array([[-d12, -r, d23]]), 0)
            S = np.append(S, np.array([[-d13, -d23, -r]]), 0)
            R = np.zeros((M,m))
            for i in range(0,M):
                for j in range(0,m):
                    R[i,j] = C0[i] + C[i,0]*g[j,0] + C[i,1]*g[j,1] + C[i,2]*g[j,2] + S[i,0]*H[j,0] + S[i,1]*H[j,1] + S[i,2]*H[j,2]
            MSE = sum((D.flatten() - R.flatten()) ** 2)
            if k < M:
                k += 1
                old_best_error = best_error
                best_error = min(MSE, best_error)
                if best_error != old_best_error:
                    best_k = k
                    best_C0 = C0
                    best_C = C
                    best_S = S
            else:
                break
    print("k = " + str(best_k) + "; mse = " + str(best_error))
    print("C0 = " + str(best_C0))
    print("C = " + str(best_C))
    print("S = " + str(best_S))
    print("γ = " + str(γ))
    return best_k, γ, best_C0, best_C, best_S