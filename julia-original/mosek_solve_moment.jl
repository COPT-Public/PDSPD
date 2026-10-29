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

"""
    solve_with_mosek(npz_path::String)
读取 .npz 并调用 Mosek 求解。
"""
function solve_with_mosek(npz_path::String)
    data = npzread(npz_path)

    tI_size = Vector{Int}(data["tI_size"])
    n_blk   = length(tI_size)

    # ---------- 变量偏移 ----------
    var_ptr = Vector{Int}(undef, n_blk + 1)
    var_ptr[1] = 0
    for i in 1:n_blk
        var_ptr[i+1] = var_ptr[i] + (tI_size[i] * (tI_size[i] + 1)) ÷ 2
    end
    n_vars = var_ptr[end]

    # ---------- 构造 Mosek 模型 ----------
    model = Model(optimizer_with_attributes(Mosek.Optimizer,
                                            "LOG" => 10,
                                            "INTPNT_MAX_ITERATIONS" => 100))

    # 为每个块创建 PSD 变量
    X_blocks = map(tI_size) do d
        @variable(model, [1:d, 1:d], PSD)
    end

    # ---------- 目标函数 ----------
    C_rows = data["C"]
    obj = @expression(model, 0.0)
    for row in eachrow(C_rows)
        blk = Int(row[1] + 1)        # 0-based → 1-based
        r   = Int(row[2] + 1)
        c   = Int(row[3] + 1)
        val = row[4]
        X = X_blocks[blk]
        if r == c
            obj += val * X[r, c]
        else
            obj += val * (X[r, c] + X[c, r])   # 对称
        end
    end
    @objective(model, Min, obj)

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

    # 构造约束表达式
    for i in 1:m
        expr = @expression(model, 0.0)
        for ((blk, r, c), val) in constr_coeff[i]
            X = X_blocks[blk]
            if r == c
                expr += val * X[r, c]
            else
                expr += val * (X[r, c] + X[c, r])  # 对称平均，避免重复
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

x_sol, _ = solve_with_mosek("/data/SOS_group/moment/PushBox/PushBox_kappa_2_N_5_moment_info.npz")
# println(x_sol)