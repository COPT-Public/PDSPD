using NPZ
using MosekTools  # 需要安装：] add MosekTools
using LinearAlgebra
using JuMP     
"""
    tri_idx(r, c, n)
给定 1-based 下三角索引 (r, c)，返回 MOSEK 线性化索引（0-based）。
"""
function tri_idx(r::Int, c::Int)
    k, l = max(r,c)-1, min(r,c)-1
    return k * (k+1) ÷ 2 + l
end
function read_spot_npz_numeric_data(filename::String)
    # Some NPZ files contain string metadata arrays (for example dtype U3), which
    # NPZ.jl cannot decode. Only read the numeric fields used by the solver.
    required_keys = ["tI_size", "A", "b", "C", "a", "c"]
    return npzread(filename, required_keys)
end
"""
    solve_with_mosek(npz_path::String)
读取 .npz 并调用 Mosek 求解。
"""
function solve_with_mosek(npz_path::String,problem::SemidefiniteProgram)
    # data = npzread(npz_path)
    data = read_spot_npz_numeric_data(npz_path)
    tI_size = Vector{Int}(data["tI_size"])
    n_blk   = length(tI_size)


    # ---------- 构造 Mosek 模型 ----------
        model = Model(optimizer_with_attributes(Mosek.Optimizer,
            "LOG" => 10,
            "INTPNT_MAX_ITERATIONS" => 100,
            
            # --- 新增精度控制代码 ---
            "INTPNT_CO_TOL_PFEAS"   => 1.0e-8,  # 默认通常是 1e-8
            "INTPNT_CO_TOL_DFEAS"   => 1.0e-8,  # 默认通常是 1e-8
            "INTPNT_CO_TOL_REL_GAP" => 1.0e-8,  # 默认通常是 1e-8
            "INTPNT_CO_TOL_MU_RED"  => 1.0e-8   # 互补性收敛速度
            # -----------------------
        ))

    # 为每个块创建 PSD 变量

    X_blocks = map(tI_size) do d
        @variable(model, [1:d, 1:d], PSD)
    end

    println(problem.var_ptr[end] - problem.var_ptr[end - 1])
    x_lin = @variable(model, [1:(problem.var_ptr[end] - problem.var_ptr[end - 1])], )
    # println(data)
    # ---------- 目标函数 ----------
    data_c = reshape(data["c"], :)
    # println(data_c)
    n_lin = length(data_c)
    obj = @expression(model, 0.0)
    for i in 1:n_lin   
            obj += data_c[i] * x_lin[i]
    end

    @objective(model, Max, obj)

    # ---------- 约束 ----------
    b = Vector{Float64}(data["b"])
    A_rows = data["A"]
    m = length(b)

    # 先收集每个约束的稀疏系数
    constr_coeff = Dict{Int, Dict{Tuple{Int,Int,Int}, Float64}}()
    for row in eachrow(A_rows)
        con_idx = Int(row[1] + 1)    # 1-based
        blk     = Int(row[2] + 1)
        r       = Int(row[3] + 1)
        c       = Int(row[4] + 1)
        val     = row[5]

        coeff = get!(constr_coeff, con_idx) do
            Dict{Tuple{Int,Int,Int}, Float64}()
        end
        key = (blk, r, c)
        if r == c
            coeff[key] = get(coeff, key, 0.0) + val
        else
            coeff[key] = get(coeff, key, 0.0) + val
            # coeff[(blk, c, r)] = get(coeff, (blk, c, r), 0.0) + val
        end
    end
    a_rows = data["a"]
    constr_lin_coeff = Dict{Int, Dict{Int, Float64}}()
    for row in eachrow(a_rows)
        row_idx = Int(row[1] + 1)    # 1-based
        col_idx = Int(row[2] + 1)
        val     = row[3]

        coeff = get!(constr_lin_coeff, row_idx) do
            Dict{Int, Float64}()
        end
        key = (col_idx)

        coeff[key] = get(coeff, key, 0.0) + val

    end
    # 构造约束表达式
    for i in 1:m
        expr = @expression(model, 0.0)
        if haskey(constr_coeff, i)
            for ((blk, r, c), val) in constr_coeff[i]
                X = X_blocks[blk]
                if r == c
                    expr += val * X[r, c]
                else
                    expr += val * (X[r, c] + X[c, r])
                end
            end
        end
    if haskey(constr_lin_coeff, i)
        for (col_idx, val) in constr_lin_coeff[i]
            expr += val * x_lin[col_idx]
        end
    end

        @constraint(model, expr == b[i])
    end

    # ---------- 求解并返回 ----------
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



file_path = "/data/SOS_group/sos/test/PushT_kappa_3_N_10_sos_info.npz"
problem = load_problem_from_spot_npz_to_cpu(file_path)

x_sol, _ = solve_with_mosek(file_path, problem)

# println(x_sol)
