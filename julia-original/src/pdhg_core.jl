
function pdhg_norm(x)
	return norm(x, Inf)
end

function default_primal_weight(problem)
	c_norm = norm(problem.C, 2)
	b_norm = norm(problem.b, 2)
	if !isfinite(c_norm) || !isfinite(b_norm) || c_norm <= eps(PDSDP_FLOAT) || b_norm <= eps(PDSDP_FLOAT)
		return 1.0
	end
	return c_norm / b_norm
end
# function primal_residual(primal_product, b)
# 	return norm(primal_product - b, 2) / (1 + norm(b, 2))
# end

# function dual_residual(C_dual_product, C_dual_product_projected)
# 	return norm(C_dual_product - C_dual_product_projected, 2) / (1 + max(norm(C_dual_product, 2), norm(C_dual_product_projected, 2)))
# end

# function primal_dual_gap_with_obj(primal_solution, C, b, dual_solution)
# 	dot_1 = dot(primal_solution, C)
# 	dot_2 = dot(dual_solution, b)
# 	return abs(dot_1 + dot_2) / (1 + abs(dot(C, C) + abs(dot(b, b)))), dot_1, dot_2
# end
function primal_residual(
	solver::PdsdpSolver, 
	primal_product::PDSDP_GPU_VECTOR_FLOAT, 
	b::PDSDP_GPU_VECTOR_FLOAT
	)
	if !require_scaling_back(solver)
		return pdhg_norm(primal_product - b) / (1 + max(pdhg_norm(primal_product), pdhg_norm(b)))
	else
		# Mittelmann/CARDAL contract: ||A(X)-b||_2 / (1+||b||_inf),
		# evaluated after restoring the original row and b scaling.
		original_residual = solver.scaling_info.b_scale .* solver.scaling_info.row_norm .* (primal_product .- b)
		return norm(original_residual, 2) / (1 + solver.scaling_info.norm_b_ori)
	end
end
function dual_residual(
	solver::PdsdpSolver, 
	C_dual_product::PDSDP_GPU_VECTOR_FLOAT, 
	C_dual_product_projected::PDSDP_GPU_VECTOR_FLOAT
	)
	if !require_scaling_back(solver)
		return pdhg_norm(C_dual_product - C_dual_product_projected) / (1 + max(pdhg_norm(C_dual_product), pdhg_norm(C_dual_product_projected)))
	else
		# The Euclidean svec distance to the product PSD cone is the
		# Frobenius norm of the negative spectral parts.  It upper-bounds the
		# benchmark's worst negative eigenvalue (and is exact for 1x1/LP
		# cones), so convergence here is a conservative sufficient condition
		# for the independent dual threshold with the same 1+||C||_inf
		# denominator.
		original_slack = solver.scaling_info.C_scale .* solver.scaling_info.col_norm .* C_dual_product
		return norm(original_slack - C_dual_product_projected, 2) / (1 + solver.scaling_info.norm_C_ori)
	end
end

function primal_dual_gap_with_obj(
	solver::PdsdpSolver, 
	primal_solution::PDSDP_GPU_VECTOR_FLOAT, 
	C::PDSDP_GPU_VECTOR_FLOAT, 
	b::PDSDP_GPU_VECTOR_FLOAT, 
	dual_solution::PDSDP_GPU_VECTOR_FLOAT
	)
		
	dot_1 = dot(primal_solution, C)
	dot_2 = dot(dual_solution, b)
	
	if require_scaling_back(solver)
		scaling_constant = solver.scaling_info.C_scale * solver.scaling_info.b_scale
		dot_1 *= scaling_constant
		dot_2 *= scaling_constant
	end
	return pdhg_norm(dot_1 + dot_2) / (1 + pdhg_norm(dot_1) + pdhg_norm(dot_2)), dot_1, dot_2
end

function primal_dual_gap_def2_with_obj(primal_solution, C, b, dual_solution, primal_product)
	dot_1 = dot(primal_solution, C)
	dot_2 = dot(dual_solution, b)
	dot_3 = dot(dual_solution, primal_product)
	return pdhg_norm(dot_1 + dot_3) / (1 + max(pdhg_norm(dot_1), pdhg_norm(dot_3))), dot_1, dot_2
end




function primal_update!(
	solver::PdsdpSolver, 
	primal_solution::PDSDP_GPU_VECTOR_FLOAT,
	step_size::PDSDP_FLOAT;
	augmented=false
	)
	if solver.option.save_trajectory push!(solver.solver_info.traj_buffer,  vcat([0.0], Array(primal_solution))) end
	if augmented
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		solver.buffer.primal_product .-= solver.problem.b
		buffer = similar(primal_solution)
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.primal_product, 0.0, buffer, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		buffer .*= 0.001
	end
	primal_solution .= primal_solution - step_size * solver.buffer.C_dual_product
	if augmented
		primal_solution .-= buffer
	end
	
	if solver.option.save_trajectory push!(solver.solver_info.traj_buffer, vcat([1.0], Array(primal_solution))) end
	start_projection = time()
	# batch_projection!(primal_solution, solver.problem.tI_size, solver.problem.var_ptr)
	dispatch_projection!(primal_solution, solver)
	# dispatch_projection!(primal_solution, solver.problem.var_ptr, solver.block_info.batch_ptr, solver.block_info.per_block_dim, solver.block_info.block_num)
	solver.solver_info.time_dict["projection_time"] += time() - start_projection
	if solver.option.save_trajectory push!(solver.solver_info.traj_buffer, vcat([2.0], Array(primal_solution))) end
end

function dual_update!(
	solver::PdsdpSolver, 
	dual_solution::PDSDP_GPU_VECTOR_FLOAT,
	primal_product::PDSDP_GPU_VECTOR_FLOAT,
	step_size::PDSDP_FLOAT
	)
	if solver.option.save_trajectory push!(solver.solver_info.traj_buffer,  vcat([3.0], Array(dual_solution))) end
	dual_solution .= dual_solution + step_size * (primal_product - solver.problem.b)
end

function pdsdp_dynamic_stepsize_iter!(solver::PdsdpSolver)
	solver.state.last_primal_solution .= solver.state.current_primal_solution
	solver.state.last_dual_solution .= solver.state.current_dual_solution
	solver.buffer.last_primal_product .= solver.buffer.primal_product
	step_size = solver.state.step_size
	for _ in 1:solver.option.step_size_trace_back_limit
		solver.buffer.tmp_primal_solution .= solver.state.last_primal_solution
		solver.buffer.tmp_dual_solution .= solver.state.last_dual_solution
		primal_step_size = step_size / solver.state.primal_weight
		dual_step_size = step_size * solver.state.primal_weight

		start_primal_update = time()
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.tmp_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		solver.buffer.C_dual_product .+= solver.problem.C
		primal_update!(solver, solver.buffer.tmp_primal_solution, primal_step_size)
		solver.solver_info.time_dict["primal_update_time"] += time() - start_primal_update
		start_dual_update = time()
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.buffer.tmp_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		last_primal_product = similar(solver.buffer.primal_product)
		theta = 1.0
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.buffer.tmp_primal_solution + theta .* (solver.buffer.tmp_primal_solution - solver.state.last_primal_solution), 0.0, last_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		dual_update!(solver, solver.buffer.tmp_dual_solution, last_primal_product, dual_step_size)

		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.tmp_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		solver.buffer.C_dual_product .+= solver.problem.C

		solver.solver_info.time_dict["dual_update_time"] += time() - start_dual_update
		start_interaction_movement = time()
		interaction, movement = compute_interaction_and_movement(
			solver.state.primal_weight,
			solver.buffer.tmp_primal_solution,
			solver.state.last_primal_solution,
			solver.buffer.tmp_dual_solution,
			solver.state.last_dual_solution,
			solver.buffer.primal_product,
			solver.buffer.last_primal_product,
			solver.buffer.delta_primal,
			solver.buffer.delta_dual,
			solver.buffer.delta_primal_product,
		)
		step_size_limit = Inf
		if interaction > 0
			step_size_limit = movement / interaction
			if movement == 0
				solver.state.numerical_error = true
				break
			end
		end

		first_term = (1.0 - 1.0 / (solver.restart_info.interval_iterations + 1)^0.3) * step_size_limit
		second_term = (1.0 + 1.0 / (solver.restart_info.interval_iterations + 1)^0.9) * step_size
		solver.solver_info.time_dict["step_size_time"] += time() - start_interaction_movement
		if ~solver.option.closed_all_trick
			# println("stepsize_limit: ",step_size_limit, ", first_term: ", first_term, ", second_term: ", second_term)
			step_size = min(first_term, second_term)
			# step_size = 0.998
			# step_size = max(5/sqrt(solver.problem.n + solver.problem.m),step_size)
			# solver.state.step_size = min(max(solver.state.step_size, 0.01 / sqrt(solver.problem.n + solver.problem.m)), 0.2 / sqrt(solver.problem.n + solver.problem.m))
		end

		if step_size <= step_size_limit
			break
		end
	end

	solver.state.step_size = step_size
	solver.state.current_primal_solution .= solver.buffer.tmp_primal_solution
	solver.state.current_dual_solution .= solver.buffer.tmp_dual_solution
end

function pdsdp_constant_stepsize_iter!(solver::PdsdpSolver)

	"Constant stepsize scheme 
	The stepsize is not changed during the iterations and is set to be 0.99/||A^T A||_2"

	solver.state.last_primal_solution .= solver.state.current_primal_solution
	solver.state.last_dual_solution .= solver.state.current_dual_solution
	solver.buffer.last_primal_product .= solver.buffer.primal_product
	step_size = 0.99 / solver.solver_info.estimate_eignvalue
	for _ in 1:1
		solver.buffer.tmp_primal_solution .= solver.state.last_primal_solution
		solver.buffer.tmp_dual_solution .= solver.state.last_dual_solution
		primal_step_size = step_size / solver.state.primal_weight
		dual_step_size = step_size * solver.state.primal_weight

		start_primal_update = time()
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.tmp_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		solver.buffer.C_dual_product .+= solver.problem.C
		primal_update!(solver, solver.buffer.tmp_primal_solution, primal_step_size)
		solver.solver_info.time_dict["primal_update_time"] += time() - start_primal_update
		start_dual_update = time()
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.buffer.tmp_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		last_primal_product = similar(solver.buffer.primal_product)
		theta = 1.0
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.buffer.tmp_primal_solution + theta .* (solver.buffer.tmp_primal_solution - solver.state.last_primal_solution), 0.0, last_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		dual_update!(solver, solver.buffer.tmp_dual_solution, last_primal_product, dual_step_size)

		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.tmp_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		solver.buffer.C_dual_product .+= solver.problem.C

		solver.solver_info.time_dict["dual_update_time"] += time() - start_dual_update
	end

	solver.state.step_size = step_size
	solver.state.current_primal_solution .= solver.buffer.tmp_primal_solution
	solver.state.current_dual_solution .= solver.buffer.tmp_dual_solution
end

# This scheme is reproduced for the paper "Tuning-Free Primal-Dual Splitting Algorithm for Large-Scale Semidefinite Programming"
# https://arxiv.org/pdf/2402.00311
function tuning_free_scheme(solver::PdsdpSolver)
    solver.state.last_primal_solution .= solver.state.current_primal_solution
	solver.state.last_dual_solution .= solver.state.current_dual_solution
	solver.buffer.last_primal_product .= solver.buffer.primal_product
	solver.buffer.last_C_dual_product .= solver.buffer.C_dual_product
    start_time = time()
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.state.last_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    solver.buffer.C_dual_product .+= solver.problem.C
    primal_update!(solver, solver.state.current_primal_solution, solver.hyper_parameters.primal_step_size)
    solver.solver_info.time_dict["primal_update_time"] += time() - start_time
    start_time = time()
    norm1 = CUDA.norm(solver.state.current_primal_solution, 2)
    norm2 =  CUDA.norm(solver.state.current_primal_solution - solver.state.last_primal_solution +  solver.hyper_parameters.primal_step_size * (
        solver.buffer.C_dual_product - solver.problem.C
    ), 2)
    solver.hyper_parameters.θ = clamp(norm1 / norm2, solver.hyper_parameters.θ_min, solver.hyper_parameters.θ_max)
    ω = 2^(-solver.state.num_outer_iterations / solver.hyper_parameters.ω_norm)
    println("primal step size: ", solver.hyper_parameters.primal_step_size, ", θ: ", solver.hyper_parameters.θ, ", ω: ", ω)
    solver.hyper_parameters.primal_step_size *= (1.0 - (1.0 - solver.hyper_parameters.θ) * ω)
    solver.hyper_parameters.dual_step_size = 1.0 / (solver.hyper_parameters.primal_step_size * solver.solver_info.estimate_eignvalue)
    solver.solver_info.time_dict["step_size_time"] += time() - start_time
    start_time = time()
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, (1.0 + solver.hyper_parameters.θ) .* solver.state.current_primal_solution .- solver.hyper_parameters.θ .* solver.state.last_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    dual_update!(solver, solver.state.current_dual_solution, solver.buffer.primal_product, solver.hyper_parameters.dual_step_size)
    solver.solver_info.time_dict["dual_update_time"] += time() - start_time
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.state.current_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    solver.buffer.C_dual_product .+= solver.problem.C
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.state.current_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function pdsdp_custom_stepsize_iter!(solver::PdsdpSolver)
	if solver.hyper_parameters.update_times == 0
		solver.hyper_parameters.primal_step_size = 1.0 / sqrt(solver.solver_info.estimate_eignvalue) * 5
		solver.hyper_parameters.dual_step_size = 1.0 / (solver.hyper_parameters.primal_step_size * solver.solver_info.estimate_eignvalue)
	end
    solver.state.last_primal_solution .= solver.state.current_primal_solution
	solver.state.last_dual_solution .= solver.state.current_dual_solution
	solver.buffer.last_primal_product .= solver.buffer.primal_product
	solver.buffer.last_C_dual_product .= solver.buffer.C_dual_product
    start_time = time()
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.state.last_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    solver.buffer.C_dual_product .+= solver.problem.C
    primal_update!(solver, solver.state.current_primal_solution, solver.hyper_parameters.primal_step_size)
    solver.solver_info.time_dict["primal_update_time"] += time() - start_time
    start_time = time()
    start_time = time()
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, 2 .* solver.state.current_primal_solution .- solver.state.last_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    dual_update!(solver, solver.state.current_dual_solution, solver.buffer.primal_product, solver.hyper_parameters.dual_step_size)
    solver.solver_info.time_dict["dual_update_time"] += time() - start_time
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.state.current_dual_solution, 0.0, solver.buffer.C_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    solver.buffer.C_dual_product .+= solver.problem.C
    CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.state.current_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

	if solver.state.num_outer_iterations % 200 == 0
		start_step_size = time()
		solver.buffer.delta_primal .= solver.state.current_primal_solution - solver.state.last_primal_solution
		solver.buffer.delta_dual .= solver.state.current_dual_solution - solver.state.last_dual_solution
		solver.buffer.delta_primal_product .= solver.buffer.primal_product - solver.buffer.last_primal_product
		delta_dual_product = similar(solver.buffer.C_dual_product)
		CUDA.CUSPARSE.mv!('N', 1.0, solver.problem.A_t, solver.buffer.delta_dual, 0.0, delta_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		norm_primal = CUDA.norm(solver.buffer.delta_primal / solver.hyper_parameters.primal_step_size .- delta_dual_product, 2)
		norm_dual = CUDA.norm(solver.buffer.delta_dual / solver.hyper_parameters.dual_step_size .- solver.buffer.delta_primal_product, 2)
		s = solver.hyper_parameters.primal_dual_scale
		Δ = solver.hyper_parameters.residual_size
		η = solver.hyper_parameters.adaptive_diminish
		
		if norm_primal > s * norm_dual * Δ
			solver.hyper_parameters.primal_step_size /= (1.0 - solver.hyper_parameters.adaptive_level)
			solver.hyper_parameters.dual_step_size *= (1.0 - solver.hyper_parameters.adaptive_level)
			solver.hyper_parameters.adaptive_level *= η
		end

		if norm_primal < norm_dual * s / Δ
			solver.hyper_parameters.primal_step_size *= (1.0 - solver.hyper_parameters.adaptive_level)
			solver.hyper_parameters.dual_step_size /= (1.0 - solver.hyper_parameters.adaptive_level)
			solver.hyper_parameters.adaptive_level *= η
		end
		solver.hyper_parameters.update_times += 1
		solver.solver_info.time_dict["step_size_time"] += time() - start_step_size
	end
	# if s * norm_dual / Δ <= norm_primal <= norm_dual * s * Δ
end

function update_primal_weight!(solver::PdsdpSolver)
	start_weight_update = time()

	# solver.restart_info.restart_primal_distance = pdhg_norm(solver.state.current_primal_solution - solver.restart_info.last_restart_primal_solution)
	# solver.restart_info.restart_dual_distance = pdhg_norm(solver.state.current_dual_solution - solver.restart_info.last_restart_dual_solution)
	# if solver.solver_info.dual_residual < 5e-5
	# 	return 
	# end
	if isinf(solver.residual_info.primal_residual)
		solver.residual_info.primal_residual = 0
		solver.residual_info.dual_residual = 0
	end
	# solver.restart_info.restart_primal_distance = abs(solver.solver_info.primal_residual - solver.residual_info.primal_residual) 
	# solver.restart_info.restart_dual_distance = abs(solver.solver_info.dual_residual - solver.residual_info.dual_residual) 
	# solver.restart_info.restart_primal_distance = solver.solver_info.dual_residual 
	# solver.restart_info.restart_dual_distance = solver.solver_info.primal_residual
	solver.restart_info.restart_primal_distance = norm(solver.state.current_primal_solution - solver.restart_info.last_restart_primal_solution, 2)
	solver.restart_info.restart_dual_distance = norm(solver.state.current_dual_solution - solver.restart_info.last_restart_dual_solution, 2)
	solver.residual_info.primal_residual = solver.solver_info.primal_residual
	solver.residual_info.dual_residual = solver.solver_info.dual_residual

	if ~solver.option.closed_all_trick
		if solver.option.stepsize_scheme == PDSDP_rAPDHG_STEPSIZE
			solver.state.primal_weight = compute_new_primal_weight_rAPDHG(
				solver.restart_info.restart_primal_distance,
				solver.restart_info.restart_dual_distance,
				solver.state.primal_weight,
				solver.option.primal_weight_update_smoothing,
			)
		else
			solver.state.primal_weight = compute_new_primal_weight_PID(
				solver.restart_info.restart_primal_distance,
				solver.restart_info.restart_dual_distance,
				solver.state.primal_weight,
				solver.pid_controller
			)

			# solver.state.primal_weight = compute_new_primal_weight_PID_abs(
			# solver.residual_info.primal_residual,
			# solver.residual_info.dual_residual,
			# solver.state.primal_weight,
			# solver.pid_controller
			# )

			# solver.state.primal_weight = compute_new_primal_weight_rAPDHG(
			# solver.restart_info.restart_primal_distance,
			# solver.restart_info.restart_dual_distance,
			# solver.state.primal_weight,
			# solver.option.primal_weight_update_smoothing,
			# )
		end
		if solver.state.primal_weight < 0.0001 || solver.state.primal_weight > 10000
			if isnothing(solver.option.init_primal_weight)
				solver.state.primal_weight = default_primal_weight(solver.problem)
			else
		    	solver.state.primal_weight = solver.option.init_primal_weight 
			end
		    print_info_banner(
		        "Primal weight reset to initial value due to extreme value, current value: " ,
		        solver.state.primal_weight,
		        status_char="⚠️")

		end
	end
	# solver.state.primal_weight = norm(solver.problem.C, 2) / norm(solver.problem.b, 2)
	solver.solver_info.time_dict["step_size_time"] += time() - start_weight_update
end

function avg_restart_update!(solver::PdsdpSolver)
        restart_weight = 1.0 / (1.0 + solver.restart_info.interval_iterations)
        solver.state.avg_primal_solution .+= restart_weight * (solver.state.current_primal_solution .- solver.state.avg_primal_solution)
        solver.state.avg_dual_solution .+= restart_weight * (solver.state.current_dual_solution .- solver.state.avg_dual_solution)
        solver.buffer.avg_primal_product .+= restart_weight * (solver.buffer.primal_product .- solver.buffer.avg_primal_product)
        solver.buffer.avg_C_dual_product .+= restart_weight * (solver.buffer.C_dual_product .- solver.buffer.avg_C_dual_product)
end

function halpern_restart_update!(solver::PdsdpSolver)
		gamma = 0.9
		alpha_2 = 1.0 / (solver.restart_info.interval_iterations + 2)
        alpha_1 = (solver.restart_info.interval_iterations + 1) / (solver.restart_info.interval_iterations + 2)
        solver.state.current_primal_solution .= alpha_1 * ((1.0 + gamma) * solver.state.current_primal_solution - gamma * solver.state.last_primal_solution) + alpha_2 * solver.state.avg_primal_solution
        solver.state.current_dual_solution .= alpha_1 * ((1.0 + gamma) * solver.state.current_dual_solution - gamma * solver.state.last_dual_solution) + alpha_2 * solver.state.avg_dual_solution
        solver.buffer.primal_product .= alpha_1 * ((1.0 + gamma) * solver.buffer.primal_product - gamma * solver.buffer.last_primal_product) + alpha_2 * solver.buffer.avg_primal_product
        solver.buffer.C_dual_product .= alpha_1 * ((1.0 + gamma) * solver.buffer.C_dual_product - gamma * solver.buffer.last_C_dual_product) + alpha_2 * solver.buffer.avg_C_dual_product
end    

function halpern_eig_restart_update!(solver::PdsdpSolver)
		gamma = 0.9
		alpha_2 = 1.0 / (solver.restart_info.interval_iterations + 2)
        alpha_1 = (solver.restart_info.interval_iterations + 1) / (solver.restart_info.interval_iterations + 2)
		solver.state.current_primal_solution .= batch_eigenvalue_add(
			solver.state.current_primal_solution, solver.state.last_primal_solution, solver.problem.tI_size, solver.problem.var_ptr, 1.0 + gamma, -gamma)
		solver.state.current_primal_solution .= batch_eigenvalue_add(
			solver.state.current_primal_solution, solver.state.avg_primal_solution, solver.problem.tI_size, solver.problem.var_ptr, alpha_1, alpha_2)
		solver.state.current_dual_solution .= alpha_1 * ((1.0 + gamma) * solver.state.current_dual_solution - gamma * solver.state.last_dual_solution) + alpha_2 * solver.state.avg_dual_solution
		CUSPARSE.mv!('N', 1.0, solver.problem.A, solver.state.current_primal_solution, 0.0, solver.buffer.primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
		
		solver.buffer.C_dual_product .= alpha_1 * ((1.0 + gamma) * solver.buffer.C_dual_product - gamma * solver.buffer.last_C_dual_product) + alpha_2 * solver.buffer.avg_C_dual_product

end    

function plot_residual_trajectory(residual_record::Vector{Float64}; 
                                 filename::String="residual_trajectory.png",
                                 title::String="PDHG Residual Convergence",
                                 yscale::Symbol=:log10,
                                 size_tuple::Tuple=(800, 600),
                                 dpi::Int=300)

    iterations = 1:length(residual_record)
    
    plt = plot(iterations, residual_record,
               seriestype=:line,
               linewidth=2,
               color=:blue,
               markershape=:circle,
               markercolor=:red,
               markersize=3,
               markerstrokewidth=0,
               label="Residual",
               xlabel="Iteration",
               ylabel="Residual Value",
               title=title,
               yscale=yscale,
               grid=true,
               gridstyle=:dot,
               gridalpha=0.5,
               legend=:topright,
               size=size_tuple,
               dpi=dpi)
    
    if length(residual_record) > 1
        min_residual = minimum(residual_record)
        final_residual = residual_record[end]
        
        annotate!(length(residual_record), final_residual, 
                 text("Final: $(round(final_residual, sigdigits=4))", 8, :right))
        
        if min_residual != final_residual
            min_idx = findfirst(x -> x == min_residual, residual_record)
            annotate!(min_idx, min_residual, 
                     text("Min: $(round(min_residual, sigdigits=4))", 8, :left))
        end
    end
    
    savefig(plt, filename)
    println("✅ Residual trajectory plot saved as: ", filename)
    

    return plt
end
