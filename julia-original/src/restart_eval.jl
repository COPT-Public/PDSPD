function kkt_restart_check(solver::PdsdpSolver)
    ratio = solver.residual_info.residual / solver.restart_info.last_residual
    restart = false
    restart_condition = 0
    better_residual = min(solver.residual_info.residual, solver.residual_info.avg_residual)
    if better_residual <= solver.option.sufficient_reduction_threshold * solver.restart_info.last_residual
        restart = true
        restart_condition = 1
    else
        if better_residual <= solver.option.necessary_reduction_threshold * solver.restart_info.last_residual && ratio > solver.restart_info.last_ratio
            restart = true
            restart_condition = 2
        else
            if solver.restart_info.interval_iterations >= solver.option.artificial_restart_threshold * solver.state.num_outer_iterations
                restart = true
                restart_condition = 3
            else
                restart = false
            end
        end
    end
    solver.restart_info.last_ratio = ratio
    if solver.option.closed_all_trick
        restart = false
    end
    if solver.restart_info.interval_iterations > 2000
        restart = true
        restart_condition = 4
    end
    return restart, restart_condition
end

function calculate_fix_point_error(
    delta_primal::PDSDP_GPU_VECTOR_FLOAT,
    delta_dual::PDSDP_GPU_VECTOR_FLOAT,
    delta_primal_product::PDSDP_GPU_VECTOR_FLOAT,
    primal_weight::PDSDP_FLOAT,
    step_size::PDSDP_FLOAT,
)
    result_1 = dot(delta_primal, delta_primal) * primal_weight / step_size
    result_2 = dot(delta_dual, delta_dual) / (primal_weight * step_size)
    result_3 = dot(delta_primal_product, delta_dual) * 2.0
    return result_1 + result_2 + result_3
end

function fixed_point_restart_check(solver::PdsdpSolver)
    restart = false
    restart_condition = 0
    
    CUDA.CUSPARSE.mv!(
        'N',
        1.0,
        solver.problem.A,
        solver.buffer.delta_primal,
        0.0,
        solver.buffer.delta_primal_product,
        'O',
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
    residual = calculate_fix_point_error(
        solver.buffer.delta_primal,
        solver.buffer.delta_dual,
        solver.buffer.delta_primal_product,
        solver.state.primal_weight,
        solver.state.step_size
    )
    # println("Fixed Point Residual: $residual")
    if solver.restart_info.interval_iterations <= 1
        solver.restart_info.last_residual = residual
        # println("Initial Fixed Point Residual: $residual")
        return false, 0
    end

    if residual <= solver.option.sufficient_reduction_threshold * solver.restart_info.last_residual
        restart = true
        restart_condition = 1
    else
        if residual <= solver.option.necessary_reduction_threshold * solver.restart_info.last_residual && residual > solver.residual_info.last_residual
            restart = true
            restart_condition = 2
        else
            if solver.restart_info.interval_iterations >= solver.option.artificial_restart_threshold * solver.state.num_outer_iterations
                restart = true
                restart_condition = 3
            else
                restart = false
            end
        end
    end
    # solver.residual_info.last_residual = residual
    if solver.option.closed_all_trick
        restart = false
    end
    if solver.restart_info.interval_iterations > 10000
        restart = true
        restart_condition = 4
    end
    return restart, restart_condition
end

function apply_avg_restart!(solver::PdsdpSolver)
    if solver.residual_info.residual < solver.residual_info.avg_residual
            solver.state.avg_primal_solution .= solver.state.current_primal_solution
            solver.state.avg_dual_solution .= solver.state.current_dual_solution
            solver.buffer.avg_primal_product .= solver.buffer.primal_product
            solver.restart_info.last_restart_residual = solver.residual_info.residual
            solver.residual_info.avg_residual = solver.residual_info.residual
    else
        solver.state.current_primal_solution .= solver.state.avg_primal_solution
        solver.state.current_dual_solution .= solver.state.avg_dual_solution
        solver.buffer.primal_product .= solver.buffer.avg_primal_product
        solver.restart_info.last_restart_residual = solver.residual_info.avg_residual
        solver.residual_info.residual = solver.residual_info.avg_residual
        primal_residual_value = avg_primal_residual_value
        dual_residual_value = avg_dual_residual_value
        primal_dual_gap_value = avg_primal_dual_gap_value
        update_residual_obj_info!(
            solver.solver_info,
            primal_residual_value,
            dual_residual_value,
            primal_dual_gap_value,
            avg_primal_obj,
            avg_dual_obj
        )
    end
end
function require_scaling_back(solver::PdsdpSolver) 
    return (solver.option.ruiz_rescaling || solver.option.pock_chambolle_rescaling || solver.option.bc_rescaling) && solver.option.stop_criterion == ORIGIN_PROBLEM
end
function evaluate_sol(solver::PdsdpSolver, primal_solution, dual_solution,
	primal_product, C_dual_product, C_dual_product_projected,scaling_info::ScalingInfoGpu)
    problem = solver.problem
    CUDA.@sync begin
            primal_residual_val = primal_residual(solver, primal_product, problem.b)
            dual_residual_val = dual_residual(solver, C_dual_product, C_dual_product_projected)
            primal_dual_gap_val, primal_obj, dual_obj = primal_dual_gap_with_obj(solver, primal_solution, problem.C, problem.b, dual_solution)
            # primal_dual_gap_val, primal_obj, dual_obj = primal_dual_gap_def2_with_obj(primal_solution, problem.C, problem.b, dual_solution, primal_product)
    end
	return primal_residual_val, dual_residual_val, primal_dual_gap_val, primal_obj, dual_obj
end


function apply_halpern_restart!(solver::PdsdpSolver)
    solver.state.avg_primal_solution .= solver.state.current_primal_solution
    solver.state.avg_dual_solution .= solver.state.current_dual_solution
    solver.buffer.avg_primal_product .= solver.buffer.primal_product
    solver.restart_info.last_restart_residual = solver.residual_info.residual
    solver.residual_info.avg_residual = solver.residual_info.residual
end

function avg_scheme_evaluate_kkt!(solver::PdsdpSolver)
        start_convergence_check = time()
        
        solver.buffer.C_dual_product_projected .= solver.buffer.C_dual_product
        solver.buffer.avg_C_dual_product_projected .= solver.buffer.avg_C_dual_product

        start_projection = time()
        dispatch_projection!(solver.buffer.C_dual_product_projected, solver)
        dispatch_projection!(solver.buffer.avg_C_dual_product_projected, solver)
        solver.solver_info.time_dict["projection_time"] += time() - start_projection


        primal_residual_value, dual_residual_value, primal_dual_gap_value, primal_obj, dual_obj = evaluate_sol(
            solver,
            solver.state.current_primal_solution,
            solver.state.current_dual_solution,
            solver.buffer.primal_product,
            solver.buffer.C_dual_product,
            solver.buffer.C_dual_product_projected
        )

        avg_primal_residual_value, avg_dual_residual_value, avg_primal_dual_gap_value, avg_primal_obj, avg_dual_obj = evaluate_sol(
            solver,
            solver.state.avg_primal_solution,
            solver.state.avg_dual_solution,
            solver.buffer.avg_primal_product,
            solver.buffer.avg_C_dual_product,
            solver.buffer.avg_C_dual_product_projected
        )
        
        solver.residual_info.residual = maximum([primal_residual_value, dual_residual_value, primal_dual_gap_value])
        solver.residual_info.avg_residual = maximum([avg_primal_residual_value, avg_dual_residual_value, avg_primal_dual_gap_value])

        update_residual_obj_info!(
            solver.solver_info,
            primal_residual_value,
            dual_residual_value,
            primal_dual_gap_value,
            primal_obj,
            dual_obj
        )

        solver.solver_info.time_dict["convergence_check_time"] += time() - start_convergence_check
end

function halpern_scheme_evaluate_kkt!(solver::PdsdpSolver)
        start_convergence_check = time()
        if require_scaling_back(solver)
            solver.buffer.C_dual_product_projected .= solver.scaling_info.C_scale.*solver.scaling_info.col_norm .*solver.buffer.C_dual_product
        else
            solver.buffer.C_dual_product_projected .= solver.buffer.C_dual_product
        end
        start_projection = time()
        dispatch_projection!(solver.buffer.C_dual_product_projected, solver)
        solver.solver_info.time_dict["projection_time"] += time() - start_projection


        primal_residual_value, dual_residual_value, primal_dual_gap_value, primal_obj, dual_obj = evaluate_sol(
            solver,
            solver.state.current_primal_solution,
            solver.state.current_dual_solution,
            solver.buffer.primal_product,
            solver.buffer.C_dual_product,
            solver.buffer.C_dual_product_projected,
            solver.scaling_info,
        )

        solver.residual_info.residual = maximum([primal_residual_value, dual_residual_value, primal_dual_gap_value])

        update_residual_obj_info!(
            solver.solver_info,
            primal_residual_value,
            dual_residual_value,
            primal_dual_gap_value,
            primal_obj,
            dual_obj
        )

        solver.solver_info.time_dict["convergence_check_time"] += time() - start_convergence_check
end