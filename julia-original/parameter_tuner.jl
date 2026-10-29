include("src/PDSDP.jl")
using Pkg
Pkg.activate(".")
using DataFrames
using CSV
using Statistics
function tune_pid_hierarchical(problem_input; 
    coarse_Kp = 0.0:0.1:0.5,
    coarse_Ki = 0.0:0.01:0.1,
    coarse_Kd = 0.0:0.05:0.5,
    fine_radius = 0.05,   
    fine_steps = 3,      
    max_iterations = 9000,
    tol = 1e-6,
    further = false
)

    results = []

    println("========== [Coarse Search] ==========")
    best_score = (Inf, Inf)
    best_params = (0.0, 0.0, 0.0)

    for Kp in coarse_Kp, Ki in coarse_Ki, Kd in coarse_Kd
        problem = copy(problem_input) # Reset problem for each run
        option = PdhgOption(
        tol=1e-4,
        max_iterations=40000,
        evaluate_frequency=40,
        restart_skip_iterations=500,
        rescaling = true,
        sufficient_reduction_threshold = 0.2,
        necessary_reduction_threshold = 0.8,
        primal_weight_update_smoothing = 0.2,
        artificial_restart_threshold = 0.5,
        print_frequency =400,
        restart_scheme = PDSDP_HALPERN_RESTART,
        stepsize_scheme = PDSDP_PID_STEPSIZE, 
            Kp = Kp, Ki = Ki, Kd = Kd
        )

        println("Testing coarse: Kp=$Kp, Ki=$Ki, Kd=$Kd ... ")
        iter = pdhg_sdp(problem, option=option)

        score = (iter["iterations"], iter["residual"]) 
        println(" -> score=$score")

        push!(results, ("coarse", Kp, Ki, Kd, score))
        #Save in CSV
        df = DataFrame(results, [:stage, :Kp, :Ki, :Kd, :score])
        CSV.write("pid_tuning_results.csv", df)
        if score[1] < best_score[1] || (score[1] == best_score[1] && score[2] < best_score[2])
            best_score = score
            best_params = (Kp, Ki, Kd)
        end
    end

    println("Best coarse params: Kp=$(best_params[1]), Ki=$(best_params[2]), Kd=$(best_params[3]), score=$best_score")
    if further == false
        return results, best_params
    end
    println("========== [Fine Search around best coarse] ==========")

    fine_Kp = range(max(0.0, best_params[1]-fine_radius), best_params[1]+fine_radius, length=fine_steps)
    fine_Ki = range(max(0.0, best_params[2]-fine_radius), best_params[2]+fine_radius, length=fine_steps)
    fine_Kd = range(max(0.0, best_params[3]-fine_radius), best_params[3]+fine_radius, length=fine_steps)

    best_fine_score = best_score
    best_fine_params = best_params

    for Kp in fine_Kp, Ki in fine_Ki, Kd in fine_Kd
        option = PdhgOption(
            tol = tol,
            max_iterations = max_iterations,
            evaluate_frequency = 40,
            restart_skip_iterations = 500,
            rescaling = false,
            sufficient_reduction_threshold = 0.2,
            primal_weight_update_smoothing = 0.2,
            print_frequency = 400,
            restart_scheme = PDSDP_HALPERN_RESTART,
            stepsize_scheme = PDSDP_PID_STEPSIZE,
            Kp = Kp, Ki = Ki, Kd = Kd
        )

        print("Testing fine: Kp=$Kp, Ki=$Ki, Kd=$Kd ... ")
        iter = pdhg_sdp(problem, option=option)

        score = (iter["iterations"], iter["residual"])
        println(" -> score=$score")

        push!(results, ("fine", Kp, Ki, Kd, score))

        if score[1] < best_fine_score[1] || (score[1] == best_fine_score[1] && score[2] < best_fine_score[2])
            best_fine_score = score
            best_fine_params = (Kp, Ki, Kd)
        end
    end

    println("========== [Final Result] ==========")
    println("Best params: Kp=$(best_fine_params[1]), Ki=$(best_fine_params[2]), Kd=$(best_fine_params[3])")
    println("Best score: $best_fine_score")

    return results, best_fine_params
end

problem = load_problem_from_spot_npz_to_cpu("PlanarHand/PushBoxTunnel_kappa_2_N_5_moment_info.npz")
results, best = tune_pid_hierarchical(problem)
