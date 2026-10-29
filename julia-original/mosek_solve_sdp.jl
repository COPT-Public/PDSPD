using NPZ
using MosekTools  
using LinearAlgebra
using JuMP     
using Pkg
Pkg.activate(".")
include("src/PDSDP.jl")
function solve_sdp_mosek(problem::SemidefiniteProgram)
    model = Model(optimizer_with_attributes(Mosek.Optimizer,
                                            "LOG" => 10,
                                            "INTPNT_MAX_ITERATIONS" => 100))
    X_blocks = map(problem.tI_size) do d
        @variable(model, [1:d, 1:d], PSD)
    end
    if problem.var_ptr[end] - problem.var_ptr[end - 1] > 0
        x_lin = @variable(model, [1:(problem.var_ptr[end] - problem.var_ptr[end - 1])], )
    end

    C_rows = problem.C
    obj = @expression(model, 0.0)
    for block_idx in eachindex(problem.tI_size)
        d = problem.tI_size[block_idx]
        X = X_blocks[block_idx]
        start_idx = problem.var_ptr[block_idx] 
        end_idx = problem.var_ptr[block_idx + 1] - 1
        for row_idx in 1:d
            for col_idx in 1:row_idx
                tri_idx = start_idx + (row_idx - 1) * row_idx ÷ 2 + col_idx - 1
                val = C_rows[tri_idx]
               
                    if row_idx == col_idx
                        obj += val * X[row_idx, col_idx]
                    else
                        obj += (1/sqrt(2.0)) * val * (X[row_idx, col_idx] + X[col_idx, row_idx])
                    end
            end
        end
    end
    if problem.var_ptr[end] - problem.var_ptr[end - 1] > 0
        lin_start_idx = problem.var_ptr[end - 1]
        lin_end_idx = problem.var_ptr[end] - 1
        for lin_idx in lin_start_idx:lin_end_idx
            val = C_rows[lin_idx]
            obj += val * x_lin[lin_idx - lin_start_idx + 1]
        end
    end

    @objective(model, Min, obj)

    for i in 1:problem.m
        expr = @expression(model, 0.0)
        for block_idx in eachindex(problem.tI_size)
            X = X_blocks[block_idx]
            d = problem.tI_size[block_idx]
            start_idx = problem.var_ptr[block_idx] 
            end_idx = problem.var_ptr[block_idx + 1] - 1
            for row_idx in 1:d 
                for col_idx in 1:row_idx
                    tri_idx = start_idx + (row_idx - 1) * row_idx ÷ 2 + col_idx - 1
                    val = problem.A[i, tri_idx]
                    if val != 0.0
                        if row_idx == col_idx
                            expr += val * X[row_idx, col_idx]
                        else
                            expr += (1/sqrt(2.0)) * val * (X[row_idx, col_idx] + X[col_idx, row_idx])
                        end
                    else
                        continue
                    end
                end
            end
        end
        if problem.var_ptr[end] - problem.var_ptr[end - 1] > 0
            lin_start_idx = problem.var_ptr[end - 1]
            lin_end_idx = problem.var_ptr[end] - 1
            for lin_idx in lin_start_idx:lin_end_idx
                val = problem.A[i, lin_idx]
                if val != 0.0
                    expr += val * x_lin[lin_idx - lin_start_idx + 1]
                end
            end
        end
        @constraint(model, expr == problem.b[i])
    end 

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        println("Optimal objective: ", objective_value(model))
        X_sol = [value.(X) for X in X_blocks]
        return X_sol, objective_value(model)
    else
        println("Mosek did not converge: ", termination_status(model))
        return nothing, nothing
    end
end


problem = load_problem_from_spot_npz_to_cpu("/data/SOS_group/sos/new_problem/PushT_MF_SOS_k_3_N_10.npz")
x_sol, obj_val = solve_sdp_mosek(problem)
