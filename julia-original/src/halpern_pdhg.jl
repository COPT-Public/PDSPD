

function build_var_ptr_from_tI_size(tI_size::Vector{<:Integer})
    var_ptr = Vector{Int32}(undef, length(tI_size) + 1)
    var_ptr[1] = 1
    for i in 2:length(var_ptr)
        sz = Int(tI_size[i - 1])
        var_ptr[i] = var_ptr[i - 1] + Int32(sz * (sz + 1) ÷ 2)
    end
    return var_ptr
end

function pdsdp_env_flag(name::String, default::Bool=false)
    value = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return value in ("1", "true", "yes", "on")
end

function unpack_primal_blocks_from_tri_solution(
    tri_solution::AbstractVector,
    tI_size::Vector{<:Integer},
    var_ptr::Vector{<:Integer},
)
    block_mats = Vector{Matrix{PDSDP_FLOAT}}(undef, length(tI_size))
    for block_id in eachindex(tI_size)
        n = Int(tI_size[block_id])
        start_idx = Int(var_ptr[block_id])
        end_idx = Int(var_ptr[block_id + 1] - 1)
        tri_len = n * (n + 1) ÷ 2
        if end_idx - start_idx + 1 != tri_len
            error("Block $(block_id) length mismatch: expected $(tri_len), got $(end_idx - start_idx + 1)")
        end

        tri_view = @view tri_solution[start_idx:end_idx]
        X = zeros(PDSDP_FLOAT, n, n)
        k = 1
        for i in 1:n
            for j in 1:i
                v = PDSDP_FLOAT(tri_view[k])
                if i == j
                    X[i, j] = v
                else
                    sym_v = v / sqrt(PDSDP_FLOAT(2.0))
                    X[i, j] = sym_v
                    X[j, i] = sym_v
                end
                k += 1
            end
        end
        block_mats[block_id] = X
    end
    return block_mats
end

function save_pdhg_solution_to_mat(
    output_mat_path::String,
    solver::PdsdpSolver,
    primal_solution::Vector,
    dual_solution::Vector,
    primal_solution_original_order::Vector,
)
    col_norm = Array(solver.scaling_info.col_norm)
    row_norm = Array(solver.scaling_info.row_norm)

    primal_solution_recovered = primal_solution ./ col_norm .* solver.scaling_info.b_scale
    dual_solution_recovered = dual_solution ./ row_norm .* solver.scaling_info.C_scale

    original_tI_size = isnothing(solver.problem.original_tI_size) ? solver.problem.tI_size : solver.problem.original_tI_size
    original_var_ptr = isnothing(solver.problem.original_tI_size) ? solver.problem.var_ptr : build_var_ptr_from_tI_size(original_tI_size)
    original_psd_len = Int(original_var_ptr[end] - 1)
    n_cones = length(original_tI_size)
    solver_psd_len = Int(solver.problem.var_ptr[n_cones + 1] - 1)

    if solver_psd_len != original_psd_len
        error("Cone dimension mismatch between solver and original order: solver cone len $(solver_psd_len), original cone len $(original_psd_len)")
    end

    function reorder_cone_part_only(name::String, v::AbstractVector)
        if length(v) < solver_psd_len
            error("$(name) length mismatch for cone restore: expected at least $(solver_psd_len), got $(length(v))")
        end

        # Reorder only the cone prefix; keep remaining 1-D variables unchanged.
        out = collect(v)
        old2new = isnothing(solver.problem.cone_old2new) ? collect(1:n_cones) : Int.(solver.problem.cone_old2new)
        for old_cone_idx in 1:n_cones
            new_cone_idx = old2new[old_cone_idx]
            src_lo = Int(solver.problem.var_ptr[new_cone_idx])
            src_hi = Int(solver.problem.var_ptr[new_cone_idx + 1]) - 1
            dst_lo = Int(original_var_ptr[old_cone_idx])
            dst_hi = Int(original_var_ptr[old_cone_idx + 1]) - 1
            out[dst_lo:dst_hi] .= v[src_lo:src_hi]
        end
        return out
    end

    function reorder_cone_vector_only(name::String, v::AbstractVector)
        if length(v) != solver_psd_len
            error("$(name) cone vector length mismatch: expected $(solver_psd_len), got $(length(v))")
        end

        out = similar(collect(v), original_psd_len)
        old2new = isnothing(solver.problem.cone_old2new) ? collect(1:n_cones) : Int.(solver.problem.cone_old2new)
        for old_cone_idx in 1:n_cones
            new_cone_idx = old2new[old_cone_idx]
            src_lo = Int(solver.problem.var_ptr[new_cone_idx])
            src_hi = Int(solver.problem.var_ptr[new_cone_idx + 1]) - 1
            dst_lo = Int(original_var_ptr[old_cone_idx])
            dst_hi = Int(original_var_ptr[old_cone_idx + 1]) - 1
            out[dst_lo:dst_hi] .= v[src_lo:src_hi]
        end
        return out
    end

    primal_solution_original_order = reorder_cone_part_only("primal_solution", primal_solution)
    primal_solution_recovered_original_order = reorder_cone_part_only("primal_solution_recovered", primal_solution_recovered)
    dual_solution_original_order = collect(dual_solution)
    dual_solution_recovered_original_order = collect(dual_solution_recovered)

    # The full historical MAT output duplicates every dense block several
    # times.  A campaign checkpoint only needs the complete recovered primal
    # and dual vectors plus the order/scaling metadata used to audit them.
    if pdsdp_env_flag("PDSDP_MINIMAL_CHECKPOINT", false)
        cone_old2new = isnothing(solver.problem.cone_old2new) ? Int32[] : solver.problem.cone_old2new
        cone_new2old = isempty(cone_old2new) ? Int32[] : Int32.(invperm(Int.(cone_old2new)))
        matwrite(output_mat_path, Dict{String, Any}(
            "primal_solution_recovered_original_order" => primal_solution_recovered_original_order,
            "dual_solution_recovered_original_order" => dual_solution_recovered_original_order,
            "cone_tI_size_solver_sorted" => solver.problem.tI_size,
            "cone_tI_size_original" => original_tI_size,
            "cone_old2new" => cone_old2new,
            "cone_new2old" => cone_new2old,
            "scaling_b_scale" => solver.scaling_info.b_scale,
            "scaling_C_scale" => solver.scaling_info.C_scale,
            "scaling_row_norm" => row_norm,
            "scaling_col_norm" => col_norm,
            "checkpoint_format" => "PDSDP_MINIMAL_ORIGINAL_COORDINATES_V1",
        ))
        print_info_banner("Saved minimal primal/dual checkpoint to MAT", output_mat_path; status_char="✅")
        return nothing
    end

    primal_blocks_original = unpack_primal_blocks_from_tri_solution(
        @view(primal_solution_recovered_original_order[1:original_psd_len]),
        original_tI_size,
        original_var_ptr,
    )

    dual_blocks_original = Any[]

    c_plus_Aty_psd_solver_order = PDSDP_FLOAT[]
    c_plus_Aty_psd_original_order = PDSDP_FLOAT[]
    c_plus_Aty_psd_blocks_original = Any[]
    c_plus_Aty_psd_projected_original_order = PDSDP_FLOAT[]
    c_plus_Aty_psd_projected_blocks_original = Any[]

    if original_psd_len > 0
        dual_solution_gpu = CuArray(dual_solution)
        c_plus_Aty_scaled_gpu = CUDA.zeros(PDSDP_FLOAT, solver.problem.n)
        CUDA.CUSPARSE.mv!(
            'N',
            1.0,
            solver.problem.A_t,
            dual_solution_gpu,
            0.0,
            c_plus_Aty_scaled_gpu,
            'O',
            CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2,
        )
        c_plus_Aty_scaled = Array(c_plus_Aty_scaled_gpu)
        c_plus_Aty_scaled .+= Array(solver.problem.C)

        c_plus_Aty_psd_solver_order = c_plus_Aty_scaled[1:solver_psd_len]
        c_plus_Aty_psd_solver_order .*= (solver.scaling_info.C_scale .* col_norm[1:solver_psd_len])

        c_plus_Aty_psd_original_order = reorder_cone_vector_only(
            "c_plus_Aty_psd_solver_order",
            c_plus_Aty_psd_solver_order,
        )

        c_plus_Aty_psd_blocks = unpack_primal_blocks_from_tri_solution(
            c_plus_Aty_psd_original_order,
            original_tI_size,
            original_var_ptr,
        )
        c_plus_Aty_psd_blocks_original = Any[
            c_plus_Aty_psd_blocks[i] for i in eachindex(c_plus_Aty_psd_blocks)
        ]

        c_plus_Aty_psd_projected_gpu = CuArray(c_plus_Aty_psd_original_order)
        batch_projection!(c_plus_Aty_psd_projected_gpu, original_tI_size, original_var_ptr)
        c_plus_Aty_psd_projected_original_order = Array(c_plus_Aty_psd_projected_gpu)
        c_plus_Aty_psd_projected_blocks = unpack_primal_blocks_from_tri_solution(
            c_plus_Aty_psd_projected_original_order,
            original_tI_size,
            original_var_ptr,
        )
        c_plus_Aty_psd_projected_blocks_original = Any[
            c_plus_Aty_psd_projected_blocks[i] for i in eachindex(c_plus_Aty_psd_projected_blocks)
        ]
    end

    cone_old2new = isnothing(solver.problem.cone_old2new) ? Int32[] : solver.problem.cone_old2new
    cone_new2old = isempty(cone_old2new) ? Int32[] : Int32.(invperm(Int.(cone_old2new)))

    mat_data = Dict{String, Any}(
        "primal_solution" => primal_solution,
        "dual_solution" => dual_solution,
        "primal_solution_original_order" => primal_solution_original_order,
        "dual_solution_original_order" => dual_solution_original_order,
        "primal_solution_recovered" => primal_solution_recovered,
        "dual_solution_recovered" => dual_solution_recovered,
        "primal_solution_recovered_original_order" => primal_solution_recovered_original_order,
        "dual_solution_recovered_original_order" => dual_solution_recovered_original_order,
        "cone_tI_size_solver_sorted" => solver.problem.tI_size,
        "cone_tI_size_original" => original_tI_size,
        "cone_old2new" => cone_old2new,
        "cone_new2old" => cone_new2old,
        "primal_solution_X_blocks_original_order" => Any[
            primal_blocks_original[i] for i in eachindex(primal_blocks_original)
        ],
        "dual_solution_X_blocks_original_order" => dual_blocks_original,
        "C_plus_Aty_psd_solver_order" => c_plus_Aty_psd_solver_order,
        "C_plus_Aty_psd_original_order" => c_plus_Aty_psd_original_order,
        "C_plus_Aty_psd_blocks_original_order" => c_plus_Aty_psd_blocks_original,
        "C_plus_Aty_psd_projected_original_order" => c_plus_Aty_psd_projected_original_order,
        "C_plus_Aty_psd_projected_blocks_original_order" => c_plus_Aty_psd_projected_blocks_original,
        "dual_primal_variable_X_blocks_original_order" => c_plus_Aty_psd_projected_blocks_original,
        "scaling_b_scale" => solver.scaling_info.b_scale,
        "scaling_C_scale" => solver.scaling_info.C_scale,
        "scaling_row_norm" => row_norm,
        "scaling_col_norm" => col_norm,
    )

    matwrite(output_mat_path, mat_data)
    print_info_banner("Saved primal/dual solutions to MAT", output_mat_path; status_char="✅")
    return nothing
end

function pdhg_sdp(
        sdp_problem::Union{CuSemidefiniteProgram, SemidefiniteProgram};
        option::PdhgOption = PdhgOption(),
        save_solution_mat::Bool = false,
        output_mat_path::Union{Nothing, String} = nothing,
    )
    # Tune GPU stream fan-out via env var to avoid excessive async overhead.
    streams_per_device = Int32(parse(Int, get(ENV, "PDSDP_STREAMS_PER_DEVICE", "8")))
    # if option.ruiz_rescaling || option.pock_chambolle_rescaling || option.bc_rescaling
        printstyled("Applying Rescaling to the SDP problem...", color=:yellow)
        println()
        start_scaling = time()
        
            # scaling_cpu!(sdp_problem, use_Ruiz_scaling=false, use_Pock_Chambolle_scaling=false, use_bc_scaling = true)
        cpu_scaling_info = scaling_cpu!(sdp_problem, use_Ruiz_scaling=option.ruiz_rescaling, use_Pock_Chambolle_scaling=option.pock_chambolle_rescaling, use_bc_scaling = option.bc_rescaling)
       
            print_timing_banner(
            "Rescaling Completed",
            time() - start_scaling,
            status_char="✅",
        )
    # else
    #     cpu_scaling_info = ScalingInfoCpu()
    # end
    # println("Starting PDHG for SDP...")
    if typeof(sdp_problem) == SemidefiniteProgram
        printstyled("Converting SemidefiniteProgram to CuSemidefiniteProgram...", color=:yellow)
        println()
        start_convert = time()
        sdp_problem = cpu_to_gpu(sdp_problem)
        print_timing_banner(
            "Conversion Completed",
            time() - start_convert,
            status_char="✅",
        )
    end
    println(CYAN, "🚀 Starting PDHG for SDP...", RESET)
    println(CYAN, "Problem dimensions: m = ", sdp_problem.m, ", n = ", sdp_problem.n, RESET)
    println(CYAN, "Number of PSD cones: ", length(sdp_problem.tI_size), RESET)
    count_bin_psd_cones = countmap(sdp_problem.tI_size)
    for (size, count) in count_bin_psd_cones
        println(CYAN, "Size of PSD cone: ", size, ", Count: ", count, RESET)
    end
    print_solver_option(option)
    residual_record = Float64[]
    buffer = BufferState(
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
    )
    state = PdhgState(
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        0.001,
        0.001,
        false,
        0,
        0
    )

    restart_info = RestartInfo(
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.n),
        CUDA.zeros(PDSDP_FLOAT, sdp_problem.m),
        100.0,
        100.0,
        0,
        0.0,
        0.0,
        0,
        0.0
    )

    residual_info = ResidualInfo()
    solver = PdsdpSolver(
        sdp_problem,
        state,
        restart_info,
        buffer,
        residual_info,
        option,
        SolverInfo(),
        ScalingInfoGpu(cpu_scaling_info),
        TableLogger(),
        HyperParameters(),
        BatchConeInfo(sdp_problem.tI_size),
        DeviceInfo(streams_per_device),
        PIDController(Kp=option.Kp, Ki=option.Ki, Kd=option.Kd, i_smooth=option.i_smooth)
    )
    # print_device_info(solver.device_info)
    
    alloc_proj_method!(solver)
    # Balance projection workload across GPUs and split large batches to improve throughput.
    prepare_task_balance!(solver; further_split_batch=true, target_batch_size=128)
    warmup_projection!(solver; runs=1, verbose=true)
    
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.state.current_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    time_dict = Dict{String, PDSDP_FLOAT}(
        "total_time" => 0.0,
        "termination_time" => 0.0,
        "projection_time" => 0.0,
        "primal_update_time" => 0.0,
        "dual_update_time" => 0.0,
        "convergence_check_time" => 0.0,
        "step_size_time" => 0.0
    )

    restart = false
    restart_condition = 0
    
    start_time = time()
    solver.solver_info.estimate_eignvalue = power_iteration_gpu(solver.problem.A, solver.problem.A_t)
    
    solver.solver_info.time_dict = time_dict
    if isnothing(option.init_step_size)
        # solver.state.step_size = 1 / sqrt(solver.problem.m)
        solver.state.step_size = 0.5 / solver.solver_info.estimate_eignvalue
    else
        solver.state.step_size = option.init_step_size
    end
    if isnothing(option.init_primal_weight)
        solver.state.primal_weight = default_primal_weight(solver.problem)
    else
        solver.state.primal_weight = option.init_primal_weight
    end
    println("stepsize: ",solver.state.step_size, ", primal weight: ", solver.state.primal_weight)
    solver.state.avg_dual_solution .= solver.state.current_dual_solution
    solver.state.avg_primal_solution .= solver.state.current_primal_solution


    # println("Estimated largest eigenvalue of A^T*A: ", solver.solver_info.estimate_eignvalue)
    print_info_banner("Estimated largest eigenvalue of A^T*A: ", solver.solver_info.estimate_eignvalue)
    print_timing_banner(
        "Power Iteration Completed",
        time() - start_time,
        status_char="✅",
    )
    start_core_algo = time()
    iter_window = max(1, Int(parse(Int, get(ENV, "PDSDP_ITER_TIME_WINDOW", "100"))))
    window_start_time = start_core_algo
    window_start_iter = 1
    while(true)
        # Keep the live timer current.  The uploaded code only assigned
        # total_time after leaving this loop, which made option.time_limit
        # ineffective and prevented bounded benchmark runs.
        solver.solver_info.time_dict["total_time"] = time() - start_time
        if solver.state.num_outer_iterations >= option.max_iterations
            println("Maximum iterations reached without convergence.")
            solver.solver_info.status = PDSDP_OUT_OF_ITER
            solver.solver_info.time_dict["termination_time"] = time() - start_time
            break
        end
        if solver.solver_info.time_dict["total_time"] > option.time_limit
            println("Time limit reached without convergence.")
            solver.solver_info.status = PDSDP_OUT_OF_TIME
            solver.solver_info.time_dict["termination_time"] = time() - start_time
            break
        end
        solver.state.num_outer_iterations += 1
        solver.restart_info.interval_iterations += 1
        if solver.option.stepsize_scheme == PDSDP_CUSTOM_STEPSIZE
            pdsdp_custom_stepsize_iter!(solver)
        elseif solver.option.stepsize_scheme == PDSDP_rAPDHG_STEPSIZE
            pdsdp_dynamic_stepsize_iter!(solver)
        elseif solver.option.stepsize_scheme == PDSDP_PID_STEPSIZE
            pdsdp_dynamic_stepsize_iter!(solver)
        else
            error("Unknown stepsize scheme")
        end
        halpern_restart_update!(solver)
  
        restart = false
        restart_condition = 0
        restart, restart_condition = fixed_point_restart_check(solver) 
        if solver.state.num_outer_iterations % option.evaluate_frequency == 0
            # || solver.state.num_outer_iterations < 40
        # ===========Check Residuals and Gaps===========
            
            halpern_scheme_evaluate_kkt!(solver)
            residual = solver.residual_info.residual
            # println("residual: ", residual)
            push!(residual_record, residual)
            if residual <= solver.option.tol
                Info = "✅ Converged at iteration $(solver.state.num_outer_iterations) with last solution, residual = $(solver.residual_info.residual)"
                print_info(solver.logger, Info)
                solver.solver_info.status = PDSDP_CONVERGED
                solver.solver_info.time_dict["termination_time"] = time() - start_time
                break
            end
        end
        # if restart && solver.state.num_outer_iterations > solver.option.restart_skip_iterations
        #     solver.restart_info.restart_count += 1
        #     #Add logo of restart
        #     info_string = "🔄 Restarting with Condition $(restart_condition) at iteration $(solver.state.num_outer_iterations) with fix point residual $(solver.residual_info.last_residual)"
        #       print_info(solver.logger, info_string)
        #     apply_halpern_restart!(solver)
        #     solver.restart_info.interval_iterations = 0
        #     if solver.option.stepsize_scheme != PDSDP_CUSTOM_STEPSIZE
        #         update_primal_weight!(solver)
        #     end
        #     solver.restart_info.last_restart_primal_solution .= solver.state.current_primal_solution
        #     solver.restart_info.last_restart_dual_solution .= solver.state.current_dual_solution
        # end

        if (restart && solver.state.num_outer_iterations > solver.option.restart_skip_iterations)
        # println("solver.residual_info.residual: ",solver.residual_info.residual)
        # println("solver.restart_info.last_residual: ", solver.residual_info.last_residual)
            # if solver.residual_info.residual <= 10*solver.residual_info.last_residual
                solver.restart_info.restart_count += 1
                #Add logo of restart
                info_string = "🔄 Restarting with Condition $(restart_condition) at iteration $(solver.state.num_outer_iterations) with fix point residual $(solver.restart_info.last_residual)"
                print_info(solver.logger, info_string)
                apply_halpern_restart!(solver)
                solver.restart_info.interval_iterations = 0
                if solver.option.stepsize_scheme != PDSDP_CUSTOM_STEPSIZE
                    update_primal_weight!(solver)
                end
                solver.restart_info.last_restart_primal_solution .= solver.state.current_primal_solution
                solver.restart_info.last_restart_dual_solution .= solver.state.current_dual_solution
                solver.residual_info.last_residual = solver.residual_info.residual
            # else
            #     solver.restart_info.restart_count += 1
            #     info_string = "🔄 Restarting with Condition $(restart_condition) at iteration $(solver.state.num_outer_iterations) old_res $(solver.residual_info.last_residual) new_res $(solver.residual_info.residual) "
            #     print_info(solver.logger, info_string)
            #     # solver.restart_info.interval_iterations = 0
            #     if isnothing(solver.option.init_primal_weight)
            #         solver.state.primal_weight = norm(solver.problem.C, 2) / norm(solver.problem.b, 2)
            #     else
            #         solver.state.primal_weight = solver.option.init_primal_weight 
            #     end
            #     # solver.restart_info.last_restart_primal_solution .= solver.state.current_primal_solution
            #     # solver.restart_info.last_restart_dual_solution .= solver.state.current_dual_solution
            # end
        end
            # println(solver.solver_info.time_dict)
        if (solver.state.num_outer_iterations - 1) % solver.option.print_frequency == 0 
            if solver.option.stepsize_scheme == PDSDP_rAPDHG_STEPSIZE || solver.option.stepsize_scheme == PDSDP_PID_STEPSIZE
                print_row(solver.logger,
                    ["Iter", "PrimalRes", "DualRes", "Gap", "PrObj", "DuObj", "StepSize", "PrimalWeight", "Time"],
                    [solver.state.num_outer_iterations,
                    solver.solver_info.primal_residual,
                    solver.solver_info.dual_residual,
                    solver.solver_info.primal_dual_gap,
                    solver.solver_info.primal_obj,
                    solver.solver_info.dual_obj,
                    solver.state.step_size,
                    solver.state.primal_weight,
                    time() - start_core_algo])
            elseif solver.option.stepsize_scheme == PDSDP_CUSTOM_STEPSIZE
                print_row(solver.logger,
                    ["Iter", "PrimalRes", "DualRes", "Gap", "PrObj", "DuObj", "PrStepSize", "DualStepSize", "Level"],
                    [solver.state.num_outer_iterations,
                    solver.solver_info.primal_residual,
                    solver.solver_info.dual_residual,
                    solver.solver_info.primal_dual_gap,
                    solver.solver_info.primal_obj,
                    solver.solver_info.dual_obj,
                    solver.hyper_parameters.primal_step_size,
                    solver.hyper_parameters.dual_step_size,
                    solver.hyper_parameters.adaptive_level])
            end
        end

        if solver.state.num_outer_iterations % iter_window == 0
            elapsed = time() - window_start_time
            iters = solver.state.num_outer_iterations - window_start_iter + 1
            avg_iter_ms = elapsed / max(iters, 1) * 1000
            print_info_banner(
                "Rolling Iter Time (ms)",
                "iter $(window_start_iter)-$(solver.state.num_outer_iterations): $(round(avg_iter_ms, digits=3))"
            )
            window_start_time = time()
            window_start_iter = solver.state.num_outer_iterations + 1
        end

    end

    if pdsdp_env_flag("PDSDP_SAVE_RESIDUAL_PLOT", true)
        plot_residual_trajectory(residual_record,
                                filename="residual_log.png",
                                yscale=:log10)
    end

    close_table(solver.logger)
    current_time = time()
    solver.solver_info.time_dict["total_time"] = current_time - start_time
    for (k, v) in solver.solver_info.time_dict
        print_timing_banner(
            k * " time",
            v,
            status_char="⏱️",
        )
    end
    primal_solution = Array(solver.state.current_primal_solution)
    dual_solution = Array(solver.state.current_dual_solution)
    primal_solution_original_order = reorder_primal_solution_to_original_cone_order(
        solver.problem,
        primal_solution,
    )
    dual_solution_original_order = copy(dual_solution)

    saved_solution_mat_path = nothing
    finite_solution = all(isfinite, primal_solution) && all(isfinite, dual_solution)
    if !finite_solution
        solver.solver_info.status = PDSDP_NUMERICAL_ERROR
        @warn "PDSDP produced NaN/Inf; refusing to export an invalid checkpoint"
    end
    if save_solution_mat && finite_solution
        saved_solution_mat_path = isnothing(output_mat_path) ? "pdhg_solution.mat" : output_mat_path
        save_pdhg_solution_to_mat(
            saved_solution_mat_path,
            solver,
            primal_solution,
            dual_solution,
            primal_solution_original_order,
        )
    end

    # if !isnothing(solver.problem.original_tI_size)
    #     println(CYAN, "Cone order (original): ", solver.problem.original_tI_size, RESET)
    # end
    # println(CYAN, "Cone order (solver sorted): ", solver.problem.tI_size, RESET)
    # if !isnothing(solver.problem.cone_old2new)
    #     println(CYAN, "Cone index map old->new: ", solver.problem.cone_old2new, RESET)
    # end

    return Dict(
    "iterations" => solver.state.num_outer_iterations,
    "status" => solver.solver_info.status,
    "total_time" => solver.solver_info.time_dict["total_time"],
    "termination_time" => solver.solver_info.time_dict["termination_time"],
    "residual" => solver.residual_info.residual,
    "primal_residual" => solver.solver_info.primal_residual,
    "dual_residual" => solver.solver_info.dual_residual,
    "primal_dual_gap" => solver.solver_info.primal_dual_gap,
    "primal_obj" => solver.solver_info.primal_obj,
    "dual_obj" => solver.solver_info.dual_obj,
    "trajectory" => solver.solver_info.traj_buffer,
    "primal_solution" => primal_solution,
    "dual_solution" => dual_solution,
    "primal_solution_original_order" => primal_solution_original_order,
    "dual_solution_original_order" => dual_solution_original_order,
    "solution_mat_path" => saved_solution_mat_path,
)
end
