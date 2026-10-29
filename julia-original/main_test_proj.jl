using Pkg
Pkg.activate(".")
include("src/PDSDP.jl")
problem = load_problem_from_spot_npz_to_cpu("kappa_2_n_94_moment_info.npz", false)
# n, k = Int64(problem.m), 10
# A_pred = build_sparse_matrix(n, k)
# problem.A = A_pred * problem.A
# problem.A_t = sparse(transpose(problem.A))
# problem.b = A_pred * problem.b
