function dense_block_bb_kernel!(
    next_primal::CuDeviceArray{Float64,1},
    next_primal_obj_product::CuDeviceArray{Float64,1},
    dense_block::CuDeviceArray{Float64,2},  # 当前块的密集矩阵
    obj_vec::CuDeviceArray{Float64,1},
    current_primal::CuDeviceArray{Float64,1},
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64,
    max_CG_iter::Int,
    block_size::Int,
    block_idx::Int
)
    # 计算全局索引
    thread_id = threadIdx().x
    global_idx = (block_idx - 1) * block_size + thread_id
    
    # 检查是否在有效范围内
    if thread_id > block_size || global_idx > length(current_primal)
        return nothing
    end
    
    # 共享内存分配
    sh_current = @cuDynamicSharedMem(Float64, (block_size,))
    sh_next = @cuDynamicSharedMem(Float64, (block_size,), sizeof(Float64)*block_size)
    sh_grad = @cuDynamicSharedMem(Float64, (block_size,), sizeof(Float64)*2*block_size)
    sh_last_grad = @cuDynamicSharedMem(Float64, (block_size,), sizeof(Float64)*3*block_size)
    sh_reduce = @cuDynamicSharedMem(Float64, (block_size,), sizeof(Float64)*4*block_size)
    
    # 加载当前解到共享内存
    sh_current[thread_id] = current_primal[global_idx]
    sync_threads()
    
    # 初始化局部变量
    rho = primal_weight / step_size
    k = 1
    
    # 计算初始梯度（仅使用当前块）
    grad_val = 0.0
    for col in 1:block_size
        col_idx = (block_idx - 1) * block_size + col
        grad_val += dense_block[thread_id, col] * current_primal[col_idx]
    end
    grad_val += rho * sh_current[thread_id] + obj_vec[global_idx]
    sh_grad[thread_id] = grad_val
    sh_last_grad[thread_id] = grad_val
    sync_threads()
    
    # 计算初始步长 - 手动归约
    norm_val = 0.0
    for i in 1:block_size
        norm_val += sh_grad[i]^2
    end
    
    # 使用共享内存归约
    sh_reduce[thread_id] = norm_val
    sync_threads()
    
    if thread_id == 1
        total_norm = 0.0
        for i in 1:block_size
            total_norm += sh_reduce[i]
        end
        sh_reduce[1] = total_norm
    end
    sync_threads()
    
    norm_val = sh_reduce[1]
    alpha = 1.0 / (norm_val + rho + 1e-16)
    
    # 更新解
    sh_next[thread_id] = max(sh_current[thread_id] - alpha * sh_grad[thread_id], 0.0)
    sync_threads()
    
    # BB迭代
    while k <= max_CG_iter
        # 计算变化量
        delta = sh_next[thread_id] - sh_current[thread_id]
        
        # 计算新梯度（仅使用当前块）
        new_grad = 0.0
        for col in 1:block_size
            col_idx = (block_idx - 1) * block_size + col
            new_grad += dense_block[thread_id, col] * sh_next[col]
        end
        new_grad += rho * sh_next[thread_id] + obj_vec[global_idx]
        
        # 计算步长分母
        grad_diff = new_grad - sh_last_grad[thread_id]
        
        # 更新解
        if abs(delta) > 1e-12 && abs(grad_diff) > 1e-12
            alpha = delta / grad_diff
            sh_current[thread_id] = sh_next[thread_id]
            sh_next[thread_id] = max(sh_next[thread_id] - alpha * new_grad, 0.0)
        end
        
        sh_grad[thread_id] = new_grad
        sh_last_grad[thread_id] = new_grad
        sync_threads()
        
        # 检查收敛 - 手动归约
        delta_val = (sh_next[thread_id] - sh_current[thread_id])^2
        sh_reduce[thread_id] = delta_val
        sync_threads()
        
        if thread_id == 1
            total_delta = 0.0
            for i in 1:block_size
                total_delta += sh_reduce[i]
            end
            sh_reduce[1] = total_delta
        end
        sync_threads()
        
        delta_val = sh_reduce[1]
        if sqrt(delta_val) < CG_bound
            break
        end
        
        k += 1
    end
    
    # 保存最终解
    next_primal[global_idx] = sh_next[thread_id]
    
    # # 计算目标函数值（仅使用当前块）
    # obj_val = 0.0
    # for col in 1:block_size
    #     col_idx = (block_idx - 1) * block_size + col
    #     obj_val += dense_block[thread_id, col] * sh_next[col]
    # end
    # next_primal_obj_product[global_idx] = obj_val
    
    return 
end

# 修改后的主函数 - 循环启动Kernel
function primal_parallel_in_gpu!(
    problem::cuMLC_problem,
    current_primal_solution::CuArray{Float64,1},
    current_primal_obj_product::CuArray{Float64,1},
    current_dual_product::CuArray{Float64,1},
    current_gradient::CuArray{Float64,1},
    last_gradient::CuArray{Float64,1},
    inner_delta_primal::CuArray{Float64,1},
    next_primal::CuArray{Float64,1},
    last_primal::CuArray{Float64,1},
    next_primal_obj_product::CuArray{Float64,1},
    next_primal_product::CuArray{Float64,1},
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64 = 1e-5
)
    nblocks = problem.K_scenario^(problem.T_stage-1)
    block_size = problem.n_dim * problem.T_stage
    total_size = nblocks * block_size
    
    # 创建CUDA流
    max_streams = min(32, nblocks)
    streams = [CUDA.CuStream() for _ in 1:max_streams]
    
    # 共享内存大小
    shmem_size = sizeof(Float64) * 5 * block_size
    
    # 为每个块启动Kernel
    for bi in 1:nblocks
        stream_idx = mod1(bi, max_streams)
        stream = streams[stream_idx]
        
        # 获取当前块的密集矩阵
        dense_block = problem.obj_matrix_blocks[bi]
        
        # 启动Kernel
        @cuda stream=stream threads=block_size blocks=1 shmem=shmem_size dense_block_bb_kernel!(
            next_primal,
            next_primal_obj_product,
            dense_block,  # 当前块的密集矩阵
            problem.obj_vec,
            current_primal_solution,
            step_size,
            primal_weight,
            CG_bound,
            100,  # max_CG_iter
            block_size,
            bi
        )
    end
    
    # 等待所有流完成
    for s in streams
        CUDA.synchronize(s)
    end
    
    # 在外部计算约束矩阵乘积
    CUDA.CUSPARSE.mv!(
        'N', 1.0, problem.constraint_matrix, next_primal,
        0.0, next_primal_product, 'O',
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
    
    # 在外部计算目标矩阵乘积（全局）
    CUDA.CUSPARSE.mv!(
        'N', 1.0, problem.obj_matrix, next_primal,
        0.0, next_primal_obj_product, 'O',
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
    
    return fill(0, nblocks)
end



####下面这个是原本CG的代码

    function primal_total_in_gpu!(
        problem::cuMLC_problem,
        current_primal_solution::CuArray{Float64,1},
        current_primal_obj_product::CuArray{Float64,1},
        current_dual_product::CuArray{Float64,1},
        current_gradient::CuArray{Float64,1},
        last_gradient::CuArray{Float64,1},
        inner_delta_primal::CuArray{Float64,1},
        next_primal::CuArray{Float64,1},
        last_primal::CuArray{Float64,1},
        next_primal_obj_product::CuArray{Float64,1},
        next_primal_product::CuArray{Float64,1},
        step_size::Float64,
        primal_weight::Float64,        
        CG_bound::Float64=1e-5,
        )

        norm_Q = CUDA.norm(problem.obj_matrix, 2)
        max_CG_iter = 100
        k = 1
        alpha = 1.0 / (norm_Q + primal_weight / step_size)
        CUDA.copyto!(last_primal, current_primal_solution)
        current_gradient .= current_primal_obj_product .+ problem.obj_vec .- current_dual_product
        CUDA.copyto!(last_gradient, current_gradient)
        next_primal .= current_primal_solution .- alpha .* current_gradient
        next_primal .=max.(next_primal,0)#可以优化
        while k <= max_CG_iter
            inner_delta_primal .= next_primal .- last_primal
            gg = CUDA.dot(inner_delta_primal, inner_delta_primal)
            if sqrt(gg) <= min(0.05*CG_bound, 1e-2)*alpha
                break
            end
            CUDA.CUSPARSE.mv!('N', 1.0, problem.obj_matrix, next_primal, 0.0, current_gradient,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            current_gradient .= current_gradient .+ (primal_weight / step_size) .* (next_primal .- current_primal_solution) .+ problem.obj_vec .- current_dual_product
            alpha = gg / CUDA.dot(inner_delta_primal, current_gradient .- last_gradient)

            CUDA.copyto!(last_primal, next_primal)
            CUDA.copyto!(last_gradient, current_gradient)
        
            next_primal .= next_primal .- alpha .* current_gradient
            next_primal .=max.(next_primal,0)
            k += 1
        end
        CUDA.CUSPARSE.mv!('N', 1.0, problem.obj_matrix, next_primal, 0.0, next_primal_obj_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

        CG_iter = min(k, max_CG_iter)
        return CG_iter
    end