

function pdhg_sdp(
        sdp_problem::Union{CuSemidefiniteProgram, SemidefiniteProgram};
        option::PdhgOption = PdhgOption(),
    )
    if option.rescaling
        printstyled("Applying Rescaling to the SDP problem...", color=:yellow)
        println()
        start_scaling = time()
        scaling_cpu!(sdp_problem, use_Ruiz_scaling=true, use_Pock_Chambolle_scaling=false, use_bc_scaling = true)
        print_timing_banner(
            "Rescaling Completed",
            time() - start_scaling,
            status_char="✅",
        )
    end
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
        0.0,
        0.0,
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
        ScalingInfoGpu(),
        TableLogger(),
        HyperParameters(),
        BlockInfo(sdp_problem.tI_size),
        PIDController(Kp=option.Kp, Ki=option.Ki, Kd=option.Kd)
    )
    
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.state.current_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    time_dict = Dict{String, PDSDP_FLOAT}(
        "total_time" => 0.0,
        "projection_time" => 0.0,
        "primal_update_time" => 0.0,
        "dual_update_time" => 0.0,
        "convergence_check_time" => 0.0,
        "step_size_time" => 0.0
    )
    solver.solver_info.time_dict = time_dict
    if isnothing(option.init_step_size)
        solver.state.step_size = 1 / sqrt(solver.problem.m)
    else
        solver.state.step_size = option.init_step_size
    end
    if isnothing(option.init_primal_weight)
        solver.state.primal_weight = norm(solver.problem.C, 2) / norm(solver.problem.b, 2)
    else
        solver.state.primal_weight = option.init_primal_weight
    end
    println("stepsize: ",solver.state.step_size, ", primal weight: ", solver.state.primal_weight)
    solver.state.avg_dual_solution .= solver.state.current_dual_solution
    solver.state.avg_primal_solution .= solver.state.current_primal_solution

    restart = false
    restart_condition = 0
    start_time = time()
    solver.solver_info.estimate_eignvalue = power_iteration_gpu(solver.problem.A, solver.problem.A_t)
    println("Estimated largest eigenvalue of A^T*A: ", solver.solver_info.estimate_eignvalue)
    print_timing_banner(
        "Power Iteration Completed",
        time() - start_time,
        status_char="✅",
    )
    print_info_banner("Starting PDHG with the following options:", nothing)
    print_info_banner("- Tolerance:", option.tol)
    print_info_banner("- Restart Scheme:", 
                      option.restart_scheme == PDSDP_AVERAGE_RESTART ? "Average Restart" : "Halpern Restart")
    print_info_banner("- Maximum Iterations:", option.max_iterations)
    print_info_banner("- Evaluate Frequency:", option.evaluate_frequency)
    print_info_banner("- Primal Weight Update Smoothing:", option.primal_weight_update_smoothing)
    print_info_banner("- Sufficient Reduction Threshold:", option.sufficient_reduction_threshold)
    while(true)
        solver.state.num_outer_iterations += 1
        solver.restart_info.interval_iterations += 1
        if solver.state.num_outer_iterations > option.max_iterations
            println("Maximum iterations reached without convergence.")
            solver.solver_info.status = PDSDP_OUT_OF_ITER
            break
        end
        if solver.solver_info.time_dict["total_time"] > option.time_limit
            println("Time limit reached without convergence.")
            solver.solver_info.status = PDSDP_OUT_OF_TIME
            break
        end
        if solver.option.stepsize_scheme == PDSDP_CUSTOM_STEPSIZE
            pdsdp_custom_stepsize_iter!(solver)
        elseif solver.option.stepsize_scheme == PDSDP_rAPDHG_STEPSIZE
            pdsdp_dynamic_stepsize_iter!(solver)
        elseif solver.option.stepsize_scheme == PDSDP_PID_STEPSIZE
            pdsdp_dynamic_stepsize_iter!(solver)
        else
            error("Unknown stepsize scheme")
        end
        if option.restart_scheme == PDSDP_AVERAGE_RESTART
            avg_restart_update!(solver)
        elseif option.restart_scheme == PDSDP_HALPERN_RESTART
            halpern_restart_update!(solver)
            # halpern_eig_restart_update!(solver)
        end
        
        restart = false
        restart_condition = 0
        if solver.option.restart_scheme == PDSDP_HALPERN_RESTART
            restart, restart_condition = fixed_point_restart_check(solver)
        end
        if solver.state.num_outer_iterations % option.evaluate_frequency == 0
        # ===========Check Residuals and Gaps===========
            avg_scheme_evaluate_kkt!(solver)
            better_residual = min(solver.residual_info.residual, solver.residual_info.avg_residual)
            if better_residual < solver.option.tol
                if solver.residual_info.avg_residual < solver.residual_info.residual
                    # println("Converged at iteration ", solver.state.num_outer_iterations, " with average solution, residual = ", solver.residual_info.avg_residual)
                    Info = "✅ Converged at iteration $(solver.state.num_outer_iterations) with average solution, residual = $(solver.residual_info.avg_residual)"
                    print_info(solver.logger, Info)
                    solver.state.current_primal_solution .= solver.state.avg_primal_solution
                    solver.state.current_dual_solution .= solver.state.avg_dual_solution
                else
                    # println("Converged at iteration ", solver.state.num_outer_iterations, " with last solution, residual = ", solver.residual_info.residual)
                    Info = "✅ Converged at iteration $(solver.state.num_outer_iterations) with last solution, residual = $(solver.residual_info.residual)"
                    print_info(solver.logger, Info)
                end
                solver.solver_info.status = PDSDP_CONVERGED
                break
            end
            if option.restart_scheme == PDSDP_AVERAGE_RESTART
                restart, restart_condition = kkt_restart_check(solver)
            end
        end
        if restart && solver.state.num_outer_iterations > solver.option.restart_skip_iterations
            solver.restart_info.last_residual = min(solver.residual_info.residual, solver.residual_info.avg_residual)
            solver.restart_info.restart_count += 1
            #Add logo of restart
            # info_string = "🔄 Restarting with Condition $(restart_condition) at iteration $(solver.state.num_outer_iterations), residual = $(solver.restart_info.last_residual)"
            #   print_info(solver.logger, info_string)
            if solver.option.restart_scheme == PDSDP_AVERAGE_RESTART
                apply_avg_restart!(solver)
            elseif solver.option.restart_scheme == PDSDP_HALPERN_RESTART
                apply_halpern_restart!(solver)
            end
            solver.restart_info.interval_iterations = 0
            if solver.option.stepsize_scheme != PDSDP_CUSTOM_STEPSIZE
                update_primal_weight!(solver)
            end
            solver.restart_info.last_restart_primal_solution .= solver.state.current_primal_solution
            solver.restart_info.last_restart_dual_solution .= solver.state.current_dual_solution
        end
            # println(solver.solver_info.time_dict)
        if (solver.state.num_outer_iterations - 1) % solver.option.print_frequency == 0 
            if solver.option.stepsize_scheme == PDSDP_rAPDHG_STEPSIZE || solver.option.stepsize_scheme == PDSDP_PID_STEPSIZE
                print_row(solver.logger,
                    ["Iter", "PrimalRes", "DualRes", "Gap", "PrObj", "DuObj", "StepSize", "PrimalWeight"],
                    [solver.state.num_outer_iterations,
                    solver.solver_info.primal_residual,
                    solver.solver_info.dual_residual,
                    solver.solver_info.primal_dual_gap,
                    solver.solver_info.primal_obj,
                    solver.solver_info.dual_obj,
                    solver.state.step_size,
                    solver.state.primal_weight])
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
    return Dict(
    "iterations" => solver.state.num_outer_iterations,
    "status" => solver.solver_info.status,
    "residual" => solver.residual_info.residual,
    "trajectory" => solver.solver_info.traj_buffer,
)
end