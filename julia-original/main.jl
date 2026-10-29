ENV["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
using LinearAlgebra
using Pkg
Pkg.activate(".")

# Avoid nested parallelism slowdown: Julia @threads handles outer parallelism,
# while BLAS stays single-threaded inside each worker task.
const PDSDP_BLAS_THREADS = parse(Int, get(ENV, "PDSDP_BLAS_THREADS", "32"))
BLAS.set_num_threads(PDSDP_BLAS_THREADS)

include("src/PDSDP.jl")

function cone_count_summary(tI_size)
    counts = Dict{Int, Int}()
    for sz in tI_size
        key = Int(sz)
        counts[key] = get(counts, key, 0) + 1
    end
    return join(["$(dim):$(cnt)" for (dim, cnt) in sort(collect(counts), by=first)], ";")
end

function csv_escape(value)
    s = isnothing(value) ? "" : string(value)
    s = replace(s, '"' => "\"\"")
    return occursin(r"[,\"\n]", s) ? "\"$(s)\"" : s
end

function write_summary_csv(csv_path, rows)
    headers = [
        "file_name",
        "problem_path",
        "status",
        "iterations",
        "total_time_sec",
        "residual",
        "primal_residual",
        "dual_residual",
        "primal_dual_gap",
        "primal_obj",
        "dual_obj",
        "m",
        "n",
        "num_cones",
        "cone_summary",
        "solution_mat_path",
        "error",
    ]

    open(csv_path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join([csv_escape(get(row, key, "")) for key in headers], ","))
        end
    end
end

npz_dir = "/data/SOS_group/sos/test"
output_path = "/home/huangyicheng/SOS/PDSDP_new/PDSDP/output_path"
mkpath(output_path)
summary_csv_path = joinpath(output_path, "batch_summary.csv")

if !isdir(npz_dir)
    error("NPZ directory does not exist: $(npz_dir)")
end

npz_files = sort(filter(path -> endswith(lowercase(path), ".npz"), readdir(npz_dir; join=true)))

if isempty(npz_files)
    error("No .npz files found in directory: $(npz_dir)")
end

# n, k = Int64(problem.m), 10
# A_pred = build_sparse_matrix(n, k)
# problem.A = A_pred * problem.A
# problem.A_t = sparse(transpose(problem.A))
# problem.b = A_pred * problem.b
option = PdhgOption(
    tol=1e-4,
    max_iterations=20000,
    evaluate_frequency=40,
    restart_skip_iterations=40,
    ruiz_rescaling =true,
    bc_rescaling = true,
    pock_chambolle_rescaling = true,
    sufficient_reduction_threshold = 0.2,
    necessary_reduction_threshold = 0.9,
    primal_weight_update_smoothing = 0.1,
    artificial_restart_threshold = 0.2,
    print_frequency =40,
    restart_scheme = PDSDP_HALPERN_RESTART,
    stepsize_scheme = PDSDP_rAPDHG_STEPSIZE, #PDSDP_rAPDHG_STEPSIZE, PDSDP_CUSTOM_STEPSIZE
    Kp = 0.90,
    Ki = 0.01,
    Kd = 0.0,
    # init_primal_weight =500,
    stop_criterion = ORIGIN_PROBLEM, #PRECONDITIONED_PROBLEM, ORIGIN_PROBLEM 
    # init_step_size = 0.2,
)

failed_files = String[]
summary_rows = Dict{String, Any}[]

for (file_idx, problem_path) in enumerate(npz_files)
    println("==================================================")
    println("Processing file $(file_idx)/$(length(npz_files)): $(problem_path)")

    row = Dict{String, Any}(
        "file_name" => basename(problem_path),
        "problem_path" => problem_path,
        "status" => "FAILED",
        "iterations" => "",
        "total_time_sec" => "",
        "residual" => "",
        "primal_residual" => "",
        "dual_residual" => "",
        "primal_dual_gap" => "",
        "primal_obj" => "",
        "dual_obj" => "",
        "m" => "",
        "n" => "",
        "num_cones" => "",
        "cone_summary" => "",
        "solution_mat_path" => "",
        "error" => "",
    )

    try
        problem = load_problem_from_spot_npz_to_cpu(problem_path)
        output_mat_path = joinpath(
            output_path,
            replace(basename(problem_path), r"\.npz$" => "_pdhg_solution.mat"),
        )

        row["m"] = Int(problem.m)
        row["n"] = Int(problem.n)
        row["num_cones"] = length(problem.tI_size)
        row["cone_summary"] = cone_count_summary(problem.tI_size)

        result = pdhg_sdp(
            problem,
            option=option,
            save_solution_mat=true,
            output_mat_path=output_mat_path,
        )

        row["status"] = get(result, "status", "UNKNOWN")
        row["iterations"] = get(result, "iterations", "")
        row["total_time_sec"] = get(result, "total_time", "")
        row["residual"] = get(result, "residual", "")
        row["primal_residual"] = get(result, "primal_residual", "")
        row["dual_residual"] = get(result, "dual_residual", "")
        row["primal_dual_gap"] = get(result, "primal_dual_gap", "")
        row["primal_obj"] = get(result, "primal_obj", "")
        row["dual_obj"] = get(result, "dual_obj", "")
        row["solution_mat_path"] = get(result, "solution_mat_path", output_mat_path)

        println("Finished: $(problem_path)")
    catch err
        push!(failed_files, problem_path)
        row["error"] = sprint(showerror, err)
        println("Failed: $(problem_path)")
        showerror(stdout, err, catch_backtrace())
        println()
    end

    push!(summary_rows, row)
    write_summary_csv(summary_csv_path, summary_rows)
end

println("==================================================")
println("Batch run completed. Total files: $(length(npz_files)), failed: $(length(failed_files))")
println("Summary CSV: $(summary_csv_path)")

if !isempty(failed_files)
    println("Failed files:")
    for path in failed_files
        println(path)
    end
end
