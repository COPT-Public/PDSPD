function compute_interaction_and_movement(
    primal_weight::PDSDP_FLOAT,
    current_primal_solution::PDSDP_GPU_VECTOR_FLOAT,
    last_primal_solution::PDSDP_GPU_VECTOR_FLOAT,
    current_dual_solution::PDSDP_GPU_VECTOR_FLOAT,
    last_dual_solution::PDSDP_GPU_VECTOR_FLOAT,
    primal_product::PDSDP_GPU_VECTOR_FLOAT,
    last_primal_product::PDSDP_GPU_VECTOR_FLOAT,
    delta_primal::PDSDP_GPU_VECTOR_FLOAT,
    delta_dual::PDSDP_GPU_VECTOR_FLOAT,
    delta_primal_product::PDSDP_GPU_VECTOR_FLOAT,
)
    delta_primal .= current_primal_solution .- last_primal_solution
    delta_dual .= current_dual_solution .- last_dual_solution
    delta_primal_product .= primal_product .- last_primal_product
    interaction = dot(delta_dual, delta_primal_product) 
    interaction = abs(interaction)

    norm_primal_squared = dot(delta_primal, delta_primal)
    norm_dual_squared = dot(delta_dual, delta_dual)
    movement = 0.5 * (primal_weight * norm_primal_squared + (1.0 / primal_weight) * norm_dual_squared)
    return interaction, movement
end

function compute_new_primal_weight_rAPDHG(
    restart_primal_distance::PDSDP_FLOAT,
    restart_dual_distance::PDSDP_FLOAT,
    primal_weight::PDSDP_FLOAT,
    primal_weight_update_smoothing::PDSDP_FLOAT;
    primal_dual_scale_weight::PDSDP_FLOAT = 1.0
)
    if restart_primal_distance > 0.0 && restart_dual_distance > 0.0
        new_estimate = restart_dual_distance / restart_primal_distance / primal_dual_scale_weight

        log_estimate = primal_weight_update_smoothing * log(new_estimate) + 
                       (1.0 - primal_weight_update_smoothing) * log(primal_weight)
        primal_weight = exp(log_estimate)
    end
    return primal_weight
end

function compute_new_primal_weight_PID(
    restart_primal_distance::PDSDP_FLOAT,
    restart_dual_distance::PDSDP_FLOAT,
    primal_weight::PDSDP_FLOAT,
    pid_controller::PIDController
)
    if restart_primal_distance < 0.0 || restart_dual_distance < 0.0
        return primal_weight
    end
    error = log((restart_primal_distance * sqrt(primal_weight)) / (restart_dual_distance  / sqrt(primal_weight)+ eps()) )
    if isnan(error) || isinf(error)
        println("Distance: primal = $restart_primal_distance, dual = $restart_dual_distance, primal_weight = $primal_weight")
        @assert false "Error in PID controller is NaN or Inf"
    end
    # error = -error
    pid_controller.integral_error = pid_controller.integral_error * pid_controller.i_smooth + error
    log_primal_weight = log(primal_weight) - pid_controller.Kp * error - pid_controller.Ki * pid_controller.integral_error - pid_controller.Kd * (error - pid_controller.last_error)
    pid_controller.last_error = error
    return exp(log_primal_weight)
end

function compute_new_primal_weight_PID_abs(
    primal_residual::PDSDP_FLOAT,
    dual_residual::PDSDP_FLOAT,
    primal_weight::PDSDP_FLOAT,
    pid_controller::PIDController
)
    if restart_primal_distance < 0.0 || restart_dual_distance < 0.0
        return primal_weight
    end
    error = log((primal_residual * sqrt(primal_weight)) / (dual_residual  / sqrt(primal_weight)+ eps()) )
    if isnan(error) || isinf(error)
        println("Distance: primal = $restart_primal_distance, dual = $restart_dual_distance, primal_weight = $primal_weight")
        @assert false "Error in PID controller is NaN or Inf"
    end
    # error = -error
    pid_controller.integral_error = pid_controller.integral_error * pid_controller.i_smooth + error
    log_primal_weight = log(primal_weight) - pid_controller.Kp * error - pid_controller.Ki * pid_controller.integral_error - pid_controller.Kd * (error - pid_controller.last_error)
    pid_controller.last_error = error
    return exp(log_primal_weight)
end