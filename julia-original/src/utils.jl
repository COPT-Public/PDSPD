function self_add_weighted_diff_kernel!(
    src::CuDeviceVector{Float64},
    a ::CuDeviceVector{Float64},
    weight ::Float64,
    N::Int64,
)
    tid = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if tid <= N
        src[tid] = src[tid] + weight * (a[tid] - src[tid])
    end
    return
end

function launch_self_add_weighted_diff!(
    src::CuVector{Float64},
    a ::CuVector{Float64},
    weight ::Float64,
)
    N = length(src)
    blocks = div(N + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@cuda threads=ThreadPerBlock blocks=blocks self_add_weighted_diff_kernel!(src, a, weight, N)
    return
end

function print_log_entry(description::String, time_in_s::Number; 
                         status_char::String="✅", color::String=GREEN, width::Int=40)
    
    # Use string interpolation for the format string itself to make the width dynamic
    @printf("%s%s %-*s - %8.4fs%s\n", color, status_char, width, description, time_in_s, RESET)
end

function print_timing_banner(
    name::AbstractString,
    value::Real; 
    width::Int = 60,
    digits::Int = 4,
    status_char::String = "",
    text_color::Symbol = :white,
    frame_color::Symbol = :light_black,
    background_color::Symbol = :cyan,
    value_color::Symbol = :yellow
)
    ts = Dates.format(now(), "HH:MM:SS")

    # The entire line is now one single printstyled call for efficiency
    printstyled("──"; color=frame_color)
    printstyled("[$ts] "; color=background_color)
    
    formatted_name = name
    if status_char != ""
        formatted_name = status_char * " " * name
    end

    printstyled(rpad(formatted_name, width); bold=true, color=text_color)
    printstyled("── "; color=frame_color)
    printstyled(string(round(value; digits=digits)); color=value_color)
    println(" s")
    
    return nothing
end

function print_info_banner(
    info::AbstractString,
    value,
    width::Int = 60;
    status_char::String = " ",
    text_color::Symbol = :white,
    frame_color::Symbol = :light_black,
    background_color::Symbol = :blue,
    value_color::Symbol = :yellow
)
    ts = Dates.format(now(), "HH:MM:SS")
    printstyled("──"; color=frame_color)
    fomat_info = info
    if status_char != ""
        fomat_info = status_char * " " * info
    end
    printstyled("[$ts] "; color=background_color)
    printstyled(rpad(fomat_info, width); bold=true, color=text_color)
    if value !== nothing
        printstyled("── "; color=frame_color)
        printstyled(string(value); color=value_color)
    end
    println(" ")
    return nothing
end

function check_symmetry(problem::Union{SemidefiniteProgram, CuSemidefiniteProgram})
    println("Checking symmetry of the problem...")
    printstyled("Notice: This function is for debugging purposes only and may not be efficient for large problems.", color=:red)
    println()
    densen_A = Matrix(problem.A)
    C = Vector(problem.C)

    for row in eachrow(densen_A)
        for sz_i in 1:length(problem.tI_size)
            sz = problem.tI_size[sz_i]
            #test each one are Symmetric
            @assert reshape(row[problem.var_ptr[sz_i]:(problem.var_ptr[sz_i] + sz * sz - 1)], (sz, sz)) ≈
                reshape(row[problem.var_ptr[sz_i]:(problem.var_ptr[sz_i] + sz * sz - 1)], (sz, sz))'
        end
    end
    for sz_i in 1:length(problem.tI_size)
        sz = problem.tI_size[sz_i]
        @assert reshape(C[problem.var_ptr[sz_i]:(problem.var_ptr[sz_i] + sz * sz - 1)], (sz, sz))  ≈
            reshape(C[problem.var_ptr[sz_i]:(problem.var_ptr[sz_i] + sz * sz - 1)], (sz, sz))'
    end
end

function power_iteration_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q
        if CUDA.norm(q) < tolerance
            return sqrt(lambda_max)
        end
    end
    # println("Power iteration did not converge within the specified tolerance.")
    # println("The maximum iteration is ", max_iterations, " and the error is ", CUDA.norm(q))
    print_info_banner("Warning: Power iteration did not converge within the specified tolerance. Current error: ", 
                      CUDA.norm(q); text_color=:yellow, frame_color=:yellow, value_color=:yellow)
    return sqrt(lambda_max)
end


function print_solver_option(option::PdhgOption)
    println("========================================")
    print_info_banner("Solver Options:", nothing; background_color=:magenta)
    println("----------------------------------------")
    for field in fieldnames(typeof(option))
        value = getfield(option, field)
        printstyled(" - $(field): ", color=:blue)
        println(value)
    end
    println("========================================")
end


function print_batch_cone_info(bci::BatchConeInfo)
    println("========================================")
    print_info_banner("Batch Cone Information:", nothing; background_color=:magenta)
    println("----------------------------------------")
    println("Number of distinct cone sizes: $(length(bci.batch_dim))")

    Dict_name = Dict(
        PDSDP_GPU_DIRECT_SVD => "GPU Direct SVD",
        PDSDP_GPU_JACOBI_SVD => "GPU Jacobi SVD",
        PDSDP_CPU_SVD        => "CPU SVD"
    )

    size_str = ["$(d)×$(d)" for d in bci.cone_dim]        
    num_str  = string.(bci.cone_num)                         
    meth_str = [m === nothing ? "Not Assigned" : Dict_name[m] for m in bci.svd_method]

    w1 = maximum(length.(size_str)) 
    w2 = maximum(length.(num_str))  
    w3 = maximum(length.(meth_str))   

    for i in eachindex(bci.cone_dim)
        if Dict_name[bci.svd_method[i]] == PDSDP_GPU_JACOBI_SVD
            @printf("Cone Size: %*s, Number of Cones: %*s, SVD Method: %-*s\n",
                w1, size_str[i],
                w2, num_str[i],
                w3, meth_str[i])
        else
            @printf("Cone Size: %*s, Number of Cones: %*s, SVD Method: %-*s\n",
                    w1, size_str[i],
                    w2, num_str[i],
                    w3, meth_str[i])
        end
    end

    println("========================================")
    return nothing
end

function print_proj_task_info(proj_task, count_info)
    println("========================================")
    # print(Crayon(foreground=:yellow, bold=true)("Projection Task Distribution:"))
    print_info_banner("Projection Task Distribution:", nothing; background_color=:magenta)
    println()
    println("----------------------------------------")

    for k in sort(collect(keys(proj_task)))
        if k == -1
            # CPU
            # println(Crayon(foreground=:cyan, bold=true)("CPU SVD tasks:"))
            printstyled("CPU SVD tasks:", color=:CYAN, bold=true)
            println()
            for (kk, vv) in count_info[k]
                println("   - Cone Size: $(kk)×$(kk), number: $(vv)")
            end
            println("-----------------------------------------")
        else
            # GPU
            # gpu_header = Crayon(foreground=:green, bold=true)("GPU $(k) tasks:")
            
            # println(gpu_header)
            printstyled("GPU $(k) tasks:", color=:CYAN, bold=true)
            println()
            for (kk, vv) in count_info[k]
                if length(kk) == 2
                    println("   - Batched Cone Size: ($(kk[1]), $(kk[2])×$(kk[2])), number: $(vv)")
                else
                    println("   - Cone Size: $(kk)×$(kk), number: $(vv)")
                end
            end
            println("-----------------------------------------")
        end
    end

    println("========================================")
end

function print_device_info(device_info::DeviceInfo)
    println("========================================")
    print_info_banner("Detected CUDA Devices:", nothing; background_color=:magenta)
    println("----------------------------------------")
    for i in 1:device_info.device_num
        println("Device ID: $(device_info.device_id[i]), Memory: $(round(device_info.mem_info[i], digits=2)) MB, Name: $(CUDA.name(device_info.devices[i]))")
    end
    println("Using $(device_info.stream_per_device) streams per device.")
    println("========================================")
end

function build_sparse_matrix(n::Int, k::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    0 <= k < n || throw(ArgumentError("k 必须满足 0 ≤ k < n"))
    
    I = Vector{Int}()
    J = Vector{Int}()    
    V = Vector{Float64}() 

    for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
    end

    for i in 1:n
        cols = [1:i-1; i+1:n]
        selected = shuffle(rng, cols)[1:k]
        val = 1.0 / (k + 1)
        for j in selected
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end

    sparse(I, J, V, n, n)
end
