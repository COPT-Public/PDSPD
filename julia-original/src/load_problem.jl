

function sdpnpzread(filename::String)
    return NPZ.load(filename)
end

function read_spot_npz_numeric_data(filename::String)
    # Some NPZ files contain string metadata arrays (for example dtype U3), which
    # NPZ.jl cannot decode. Only read the numeric fields used by the solver.
    required_keys = ["tI_size", "A", "b", "C", "a", "c"]
    return npzread(filename, required_keys)
end


function read_sdpa_file_info(fname::String)
    file = open(fname, "r")
    try
        # Read the lines until the necessary information is gathered
        while true
            line = readline(file)
            # Skip comments
            if startswith(line, '*') || startswith(line, '"')
                continue
            end

            # First non-comment line gives n_constrs
            n_constrs = parse(LORADS_INT, line)
            # Next line gives n_cones
            n_cones = parse(LORADS_INT, readline(file))
            # Next line gives blk_dims
            blk_dim_line = readline(file)
            blk_dims = parse.(LORADS_INT, split(replace(blk_dim_line, r"[\(\)\{\},;]" => " ")))
            blk_dims, blk_index_map = map_blk_idx(blk_dims)
            
            pos_blk_dims = findall(x -> x >= 0, blk_dims)
            neg_blk_dims = findall(x -> x < 0, blk_dims)

            lp_dim = 0

            if sum(neg_blk_dims) > 0
                lp_dim = abs(blk_dims[neg_blk_dims[1]])
                blk_dims = blk_dims[1:end-1]
            else
                lp_dim = 0
            end
        
            blk_dims_sq_cum = zeros(LORADS_INT, length(blk_dims) + 1)
            blk_dims_sq_cum[1] = 1
        
            for blk in 2:length(blk_dims) + 1
                blk_dims_sq_cum[blk] = blk_dims_sq_cum[blk - 1] + blk_dims[blk - 1]^2 
            end        

            n_cols = sum(d^2 for d in blk_dims) + lp_dim

            return n_constrs, n_cols, blk_dims, length(blk_dims), blk_dims_sq_cum, lp_dim
        end
    finally
        close(file)
    end
end

function load_problem_from_npz(filename::String)
    data = read_spot_npz_numeric_data(filename)

    # ---------- 把数据搬到 GPU ----------
    A_dense_cpu = data["A"]                       # CPU dense
    A_gpu       = CuSparseMatrixCSR(CuArray(A_dense_cpu))  # 关键改动！ # CSR 格式（CUSPARSE 要求行主）
    b_gpu       = CuVector{Float64}(data["b"])
    C_gpu       = CuVector{Float64}(data["C"])
    tI_size_gpu = Vector{Int32}(data["tI_size"])  # 32 位索引足够

    # ---------- 构造 var_ptr ----------
    tI_size_cpu = Array(tI_size_gpu)      # 拉回 CPU
    var_ptr_cpu = Vector{Int32}(undef, length(tI_size_cpu)+1)
    var_ptr_cpu[1] = one(Int32)
    for i in 2:length(var_ptr_cpu)
        var_ptr_cpu[i] = var_ptr_cpu[i-1] + tI_size_cpu[i-1]^2
    end
    var_ptr_gpu = Vector{Int32}(var_ptr_cpu)  # 再搬回 GPU

    m, n = size(A_dense_cpu)

    return CuSemidefiniteProgram(
        A_gpu,                         # 稀疏 CSR
        CuSparseMatrixCSR(transpose(A_gpu)),                        # 转置仍为 CuSparseMatrixCSR
        b_gpu,
        C_gpu,
        tI_size_gpu,
        var_ptr_gpu,
        n,
        m,
        true,
        false,
    )
end


# Constants for ANSI colors and emojis for a clean log


function load_problem_from_spot_npz(filename::String)
    error("NOT IMPLEMENTED")
    println(CYAN, "🚀 Starting problem load for: ", filename, RESET)
    start_time = time()

    # --- Cache Check ---
    cache_start = time()
    cache_dir = "problem_cache"
    if !isdir(cache_dir)
        mkdir(cache_dir)
    end
    cache_file = joinpath(cache_dir, replace(basename(filename), ".npz" => ".jls"))
    if isfile(cache_file)
        println(GREEN, "💾 Loading from cache: ", cache_file, RESET)
        return deserialize(cache_file)
    end
    println(YELLOW, "⏳ Cache not found. Proceeding with NPZ read...", RESET)
    cache_time = time() - cache_start

    # --- Read NPZ File ---
    npz_read_start = time()
    data = read_spot_npz_numeric_data(filename)
    tI_size = data["tI_size"]
    npz_read_time = time() - npz_read_start
    print_timing_banner(
        "NPZ file read",
        npz_read_time,
        status_char="✅",
    )

    # --- Data Permutation ---
    perm_start = time()
    perm = sortperm(tI_size)
    old2new = similar(perm)
    old2new[perm] = 1:length(perm)
    permuted_tI_size = tI_size[perm]
    perm_time = time() - perm_start
    print_timing_banner(
        "Data permutation",
        perm_time,
        status_char="✅",
    )

    # --- Construct C Vector ---
    C_start = time()
    var_ptr = Vector{PDSDP_INT}(undef, length(permuted_tI_size) + 1)
    var_ptr[1] = one(PDSDP_INT)
    for i in 2:(length(permuted_tI_size) + 1)
        sz = permuted_tI_size[i - 1]
        var_ptr[i] = var_ptr[i-1] + sz * sz
    end
    C = zeros(var_ptr[end] - 1)
    data_C = data["C"]
    for i in eachindex(data_C[:, 1])
        block_idx = old2new[Int(data_C[i, 1] + 1)]
        col_idx = Int(data_C[i, 2] + 1)
        row_idx = Int(data_C[i, 3] + 1)
        value = data_C[i, 4]
        block_dim = permuted_tI_size[block_idx]
        if col_idx == row_idx
            C[var_ptr[block_idx] + (row_idx - 1) * block_dim + col_idx - 1] = value
        else
            C[var_ptr[block_idx] + (row_idx - 1) * block_dim + col_idx - 1] = value
            C[var_ptr[block_idx] + (col_idx - 1) * block_dim + row_idx - 1] = value
        end
    end
    C_gpu = convert(PDSDP_GPU_VECTOR_FLOAT, C)
    C_time = time() - C_start
    print_timing_banner(
        "C vector construction",
        C_time,
        status_char="✅",
    )

    # --- Construct A Matrix ---
    A_start = time()
    A_row_idx = []
    A_col_idx = []
    A_values = []
    data_A = data["A"]
    for i in eachindex(data_A[:, 1])
        cons_idx = Int(data_A[i, 1] + 1)
        block_idx = old2new[Int(data_A[i, 2] + 1)]
        col_idx = Int(data_A[i, 3] + 1)
        row_idx = Int(data_A[i, 4] + 1)
        value = data_A[i, 5]
        block_dim = permuted_tI_size[block_idx]
        if col_idx == row_idx
            A_row_idx = push!(A_row_idx, cons_idx)
            A_col_idx = push!(A_col_idx, var_ptr[block_idx] + (row_idx - 1) * block_dim + col_idx - 1)
            A_values = push!(A_values, value)
        else
            A_row_idx = push!(A_row_idx, cons_idx)
            A_col_idx = push!(A_col_idx, var_ptr[block_idx] + (row_idx - 1) * block_dim + col_idx - 1)
            A_values = push!(A_values, value)
            A_row_idx = push!(A_row_idx, cons_idx)
            A_col_idx = push!(A_col_idx, var_ptr[block_idx] + (col_idx - 1) * block_dim + row_idx - 1)
            A_values = push!(A_values, value)
        end
    end
    A_time = time() - A_start
    print_timing_banner(
        "A matrix construction",
        A_time,
        status_char="✅",
    )
    
    # --- CUDA CSR Sparse Matrix A ---
    A_CSR_start = time()
    order = sortperm(1:length(A_row_idx), by = i -> (A_row_idx[i], A_col_idx[i]))
    A_row_sorted = A_row_idx[order]
    A_col_sorted = A_col_idx[order]
    A_values_sorted = A_values[order]
    m = length(data["b"])
    counts = zeros(PDSDP_INT, m)
    for r in A_row_sorted
        counts[r] += 1
    end
    row_ptr = vcat(1, cumsum(counts) .+ 1)
    gpu_row_ptr = convert(PDSDP_GPU_VECTOR_INT, row_ptr)
    gpu_col_idx = convert(PDSDP_GPU_VECTOR_INT, A_col_sorted)
    gpu_values = convert(PDSDP_GPU_VECTOR_FLOAT, A_values_sorted)
    n = var_ptr[end] - 1
    sparse_A = CUSPARSE.CuSparseMatrixCSR(
        CuArray(gpu_row_ptr),
        CuArray(gpu_col_idx),
        CuArray(gpu_values),
        (m, n),
    )
    b_gpu = convert(PDSDP_GPU_VECTOR_FLOAT, data["b"])
    convert_time = time() - A_CSR_start
    print_timing_banner(
        "CUDA matrix construction",
        convert_time,
        status_char="✅",
    )

    # --- CUDA CSR Sparse Matrix A Transpose ---
    A_transpose_start = time()
    sparse_A_T = CUSPARSE.CuSparseMatrixCSR(transpose(sparse_A))
    print_timing_banner(
        "CUDA transpose matrix construction",
        time() - A_transpose_start,
        status_char="✅",
    )

    final_start = time()
    problem = CuSemidefiniteProgram(
        sparse_A,
        sparse_A_T,
        b_gpu,
        C_gpu,
        permuted_tI_size,
        var_ptr,
        n,
        m
    )
    final_time = time() - final_start
    print_timing_banner(
        "Problem construction",
        final_time,
        status_char="✅",
    )

    # --- Caching and Total Time ---
    serialize(cache_file, problem)
    println(MAGENTA, "💾 Problem cached to: ", cache_file, RESET)
    total_time = time() - start_time
    @printf("%s - Total Problem Load Time: %.4f seconds%s\n", CYAN, total_time, RESET)
    return problem
end

function load_problem_from_spot_npz_to_cpu(filename::String)
    println(CYAN, "🚀 Starting problem load for: ", filename, RESET)
    println(CYAN, "NOTE: Current Version Change to Save only Lower Triangular Part of Each PSD Cone", RESET)
    start_time = time()
    # --- Cache Check ---
    cache_start = time()
    cache_dir = "problem_cache_cpu"
    if !isdir(cache_dir)
        mkdir(cache_dir)
    end
    cache_file = joinpath(cache_dir, replace(basename(filename), ".npz" => ".jls"))
    # if isfile(cache_file)
    #     println(GREEN, "💾 Loading from cache: ", cache_file, RESET)
    #     return deserialize(cache_file)
    # end
    println(YELLOW, "⏳ Cache not found. Proceeding with NPZ read...", RESET)
    cache_time = time() - cache_start

    # --- Read NPZ File ---
    npz_read_start = time()
    data = read_spot_npz_numeric_data(filename)
    tI_size = data["tI_size"]
    npz_read_time = time() - npz_read_start
    print_timing_banner(
        "NPZ file read",
        npz_read_time,
        status_char="✅",
    )
    # --- Data Permutation ---
    perm_start = time()
    perm = sortperm(tI_size)
    # println(perm)
    old2new = similar(perm)
    old2new[perm] = 1:length(perm)
    original_tI_size = convert(Vector{PDSDP_INT}, tI_size)
    cone_old2new = convert(Vector{PDSDP_INT}, old2new)
    permuted_tI_size = tI_size[perm]
    println(permuted_tI_size)
    perm_time = time() - perm_start
    print_timing_banner(
        "Data permutation",
        perm_time,
        status_char="✅",
    )

    # --- Construct C Vector ---
    C_start = time()
    var_ptr = Vector{PDSDP_INT}(undef, length(permuted_tI_size) + 1)
    var_ptr[1] = one(PDSDP_INT)
    for i in 2:(length(permuted_tI_size) + 1)
        sz = permuted_tI_size[i - 1]
        var_ptr[i] = var_ptr[i-1] + sz * (sz + 1) ÷ 2
    end
    if !haskey(data, "C") 
        psd_var_coeff = false
    else
        psd_var_coeff = true
    end
    if !haskey(data, "c") 
        lin_var_coeff = false
    else
        lin_var_coeff = true
    end
    if !lin_var_coeff && !psd_var_coeff
        error("At least one of linear or PSD variable coefficients must be present.")
    end

    if psd_var_coeff
        data_C = data["C"]
        C = zeros(var_ptr[end] - 1)
        dict_obj = Dict{Tuple{Int}, Float64}()
        C_non_diag_flag = fill(false, var_ptr[end] - 1)
        for i in eachindex(data_C[:, 1])
            block_idx = old2new[Int(data_C[i, 1] + 1)]
            col_idx = Int(data_C[i, 2] + 1)
            row_idx = Int(data_C[i, 3] + 1)
            value = data_C[i, 4]
            block_dim = permuted_tI_size[block_idx]
            
            if col_idx == row_idx
                idx_c = var_ptr[block_idx] + (row_idx - 1) * row_idx ÷ 2 + col_idx - 1
                C[idx_c] = value
                C_non_diag_flag[idx_c] = false
            else
                if col_idx > row_idx
                    idx_c = var_ptr[block_idx] + col_idx * (col_idx - 1) ÷ 2 + row_idx - 1
                    C[var_ptr[block_idx] + col_idx * (col_idx - 1) ÷ 2 + row_idx - 1] = 2.0 * value
                    C_non_diag_flag[idx_c] = true
                else
                    idx_c = var_ptr[block_idx] + col_idx * (col_idx - 1) ÷ 2 + row_idx - 1
                    C[var_ptr[block_idx] + row_idx * (row_idx - 1) ÷ 2 + col_idx - 1] = 2.0 * value
                    C_non_diag_flag[idx_c] = true
                end
            end
            if !isnothing(get(dict_obj, (idx_c,), nothing))
                error("Duplicate entry found for C at index $idx_c")
            end
            dict_obj[(idx_c,)] = value
        end
    end
    if lin_var_coeff
        data_c = reshape(data["c"], :)
        n_lin = length(data_c)
        c = zeros(n_lin)
        for i in 1:n_lin
            c[i] = data_c[i]
        end
        if !psd_var_coeff
            C = vcat(zeros(var_ptr[end] - 1), -c)
            C_non_diag_flag = fill(false, length(C))
        else
            C = vcat(C, c)
            C_non_diag_flag = vcat(C_non_diag_flag, fill(false, length(c)))
        end
    end
    



    C_time = time() - C_start
    print_timing_banner(
        "Objective construction",
        C_time,
        status_char="✅",
    )

    # --- Construct A Matrix ---
    A_start = time()
    A_row_idx = []
    A_col_idx = []
    A_values = []
    A_non_diag_flag = Bool[]
    data_A = data["A"]
    has_a = haskey(data, "a")

    if has_a || lin_var_coeff
        max_idx_a = 0
        if has_a
            data_a = data["a"]
            max_idx_a = Int64(maximum(data_a[:, 2] .+ 1))
        end
        push!(var_ptr, var_ptr[end] + max(Int64(n_lin), Int64(max_idx_a)))
    else
        push!(var_ptr, var_ptr[end])
    end

    processed_entries = Dict{Tuple{Int, Int}, Float64}()

    for row_data in eachrow(data_A)
        cons_idx = Int(row_data[1] + 1)
        block_idx = old2new[Int(row_data[2] + 1)]
        col_idx = Int(row_data[3] + 1)
        row_idx = Int(row_data[4] + 1)
        value = row_data[5]
    
        if col_idx > row_idx
            row_idx, col_idx = col_idx, row_idx 
        end
    
        var_idx = var_ptr[block_idx] + row_idx * (row_idx - 1) ÷ 2 + col_idx - 1

        if !isnothing(get(processed_entries, (cons_idx, var_idx), nothing))
            error("Duplicate entry found for constraint $cons_idx, variable index $var_idx")
        end
        processed_entries[(cons_idx, var_idx)] = value

        push!(A_row_idx, cons_idx)
        push!(A_col_idx, var_idx)
    
        if row_idx == col_idx
            push!(A_values, value)
            push!(A_non_diag_flag, false)
        else
            push!(A_values, 2.0 * value)
            push!(A_non_diag_flag, true)
        end
    end
    if has_a
        for row_data in eachrow(data_a)
            cons_idx = Int(row_data[1] + 1)
            col_idx = Int(row_data[2] + 1)
            value = row_data[3]
            var_idx = var_ptr[end - 1] + col_idx - 1  # linear variable part starts from var_ptr[end - 1]
            if !isnothing(get(processed_entries, (cons_idx, var_idx), nothing))
                error("Duplicate entry found for constraint $cons_idx, variable index $var_idx")
            end
            processed_entries[(cons_idx, var_idx)] = value
            push!(A_row_idx, cons_idx)
            push!(A_col_idx, var_idx)
            push!(A_values, value)
            push!(A_non_diag_flag, false)
        end
    end

    A_time = time() - A_start
    print_timing_banner(
        "Constraint matrix construction",
        A_time,
        status_char="✅",
    )
    
    # --- CUDA CSR Sparse Matrix A ---
    A_CSR_start = time()
    order = sortperm(1:length(A_row_idx), by = i -> (A_col_idx[i], A_row_idx[i]))
    A_row_sorted = A_row_idx[order]
    A_col_sorted = A_col_idx[order]
    A_values_sorted = A_values[order]
    A_non_diag_flag_sorted = A_non_diag_flag[order]
    m = length(data["b"])
    n = var_ptr[end] - 1

    counts = zeros(PDSDP_INT, n)
    for r in A_col_sorted
        counts[r] += 1
    end
    col_ptr = vcat(1, cumsum(counts) .+ 1)
    # gpu_row_ptr = convert(PDSDP_CPU_VECTOR_INT, row_ptr)
    # gpu_col_idx = convert(PDSDP_CPU_VECTOR_INT, A_col_sorted)
    # gpu_values = convert(PDSDP_CPU_VECTOR_FLOAT, A_values_sorted)
    colptr = convert(PDSDP_CPU_VECTOR_INT, col_ptr)
    rowval = convert(PDSDP_CPU_VECTOR_INT, A_row_sorted)
    nzval = convert(PDSDP_CPU_VECTOR_FLOAT, A_values_sorted)
    # sparse_A = CUSPARSE.CuSparseMatrixCSR(
    #     CuArray(gpu_row_ptr),
    #     CuArray(gpu_col_idx),
    #     CuArray(gpu_values),
    #     (m, n),
    # )
    sparse_A = SparseMatrixCSC(m, n, colptr, rowval, nzval)
        
    b_gpu = convert(PDSDP_CPU_VECTOR_FLOAT, data["b"])
    convert_time = time() - A_CSR_start
    print_timing_banner(
        "CPU matrix construction",
        convert_time,
        status_char="✅",
    )

    # --- CUDA CSR Sparse Matrix A Transpose ---
    
    final_start = time()
    m,n = size(sparse_A)
    # println(Matrix(sparse_A))
    # dense_matrix = Matrix(sparse_A)
    #  writedlm("lascasbeN2.csv", dense_matrix, ',')
    # println("save sccucess")
    # sleep(5)
    problem = SemidefiniteProgram(
        sparse_A,
        sparse(transpose(sparse_A)),
        b_gpu,
        C,
        permuted_tI_size,
        var_ptr,
        n,
        m,
        psd_var_coeff,
        lin_var_coeff,
        cone_old2new,
        original_tI_size,
    )
    final_time = time() - final_start
    print_timing_banner(
        "Problem construction",
        final_time,
        status_char="✅",
    )

    # --- Caching and Total Time ---
    # serialize(cache_file, problem)
    println(MAGENTA, "💾 Problem cached to: ", cache_file, RESET)
    total_time = time() - start_time
    @printf("%s - Total Problem Load Time: %.4f seconds%s\n", CYAN, total_time, RESET)
    sc_time = time() 
    println(MAGENTA, "Scaling before convert to primal-dual compatible format", RESET)
    # scaling_abs_one_T!(problem)
    # scaling_abs_one_parallel!(problem)
    # scaling_chain!(problem)
    print_timing_banner(
        "Scaling",
        time() - sc_time,
        status_char="✅",
    )
    # problem.C[C_non_diag_flag] ./= sqrt(2.0)
    # problem.A.nzval[A_non_diag_flag_sorted] ./= sqrt(2.0)
    # problem.C[C_non_diag_flag] ./= sqrt(2.0)
    # problem.A.nzval[A_non_diag_flag_sorted] ./= sqrt(2.0)
    # A_transpose_start = time()
    # A_T  = SparseArrays.SparseMatrixCSC(transpose(problem.A))
    
    # Atb = sparse_A_T * b_gpu
    # AAt = problem.A * A_T
    # print_timing_banner(
    #     "CPU transpose matrix construction",
    #     time() - A_transpose_start,
    #     status_char="✅",
    # )
    # problem.A_t = A_T
    # problem.AAt = AAt
    sc2_time = time()
    convert_problem_to_primal_dual_compat_new!(problem)
    row_norms = sqrt.(sum(abs2,problem.A,dims=2))
    row_norms[row_norms .== 0.0] .= 1.0
    problem.A .= spdiagm(vec(1.0 ./ row_norms)) * problem.A
    problem.b .= problem.b ./ row_norms

    print_timing_banner(
    "Scaling2",
    time() - sc2_time,
    status_char="✅",
)
    return problem
end

function reorder_primal_solution_to_original_cone_order(
    problem::Union{SemidefiniteProgram, CuSemidefiniteProgram},
    primal_solution::AbstractVector,
)
    if isnothing(problem.cone_old2new) || isnothing(problem.original_tI_size)
        return collect(primal_solution)
    end

    old2new = Int.(problem.cone_old2new)
    original_tI_size = Int.(problem.original_tI_size)
    n_cones = length(original_tI_size)

    original_var_ptr = Vector{Int}(undef, n_cones + 1)
    original_var_ptr[1] = 1
    for i in 2:(n_cones + 1)
        sz = original_tI_size[i - 1]
        original_var_ptr[i] = original_var_ptr[i - 1] + sz * (sz + 1) ÷ 2
    end

    primal_perm = collect(primal_solution)
    primal_original = similar(primal_perm)

    for old_cone_idx in 1:n_cones
        new_cone_idx = old2new[old_cone_idx]
        src_lo = Int(problem.var_ptr[new_cone_idx])
        src_hi = Int(problem.var_ptr[new_cone_idx + 1]) - 1
        dst_lo = original_var_ptr[old_cone_idx]
        dst_hi = original_var_ptr[old_cone_idx + 1] - 1
        primal_original[dst_lo:dst_hi] .= primal_perm[src_lo:src_hi]
    end

    original_psd_dim = original_var_ptr[end] - 1
    if length(primal_perm) > original_psd_dim
        primal_original[(original_psd_dim + 1):end] .= primal_perm[(original_psd_dim + 1):end]
    end

    return primal_original
end

function find_diag_indices(tI_size::Vector{Int32}, var_ptr::Vector{Int32})
    diag_indices = Int32[]
    for i in 1:length(tI_size)
        sz = tI_size[i]
        start_idx = var_ptr[i]
        for d in 1:sz
            diag_idx = start_idx + (d * (d - 1)) ÷ 2 + (d - 1)
            push!(diag_indices, diag_idx)
        end
    end
    return diag_indices
end

function find_non_diag_indices(tI_size::Vector{Int32}, var_ptr::Vector{Int32})
    non_diag_indices = fill(true, var_ptr[end] - 1)
    for i in 1:length(tI_size)
        sz = tI_size[i]
        start_idx = var_ptr[i]
        for d in 1:sz
            diag_idx = start_idx + (d * (d - 1)) ÷ 2 + (d - 1)
            non_diag_indices[diag_idx] = false
        end
    end
    return non_diag_indices
end

function convert_problem_to_primal_dual_compat(problem::SemidefiniteProgram)
    A = problem.A
    tI_size = problem.tI_size
    var_ptr = problem.var_ptr
    diag_indices = find_diag_indices(tI_size, var_ptr)
    A[:, diag_indices] .= A[:, diag_indices] *= sqrt(2.0)
    A .= A ./= sqrt(2.0)
    C = problem.C
    C[diag_indices] .= C[diag_indices] *= sqrt(2.0)
    C .= C ./= sqrt(2.0)
    return SemidefiniteProgram(
        A,
        sparse(transpose(problem.A)),
        problem.b,
        C,
        tI_size,
        var_ptr,
        problem.n,
        problem.m,
        problem.psd_var_coeff,
        problem.lin_var_coeff
    )
end

function convert_problem_to_primal_dual_compat_new!(problem::SemidefiniteProgram)
    A = problem.A
    tI_size = problem.tI_size
    var_ptr = problem.var_ptr
    non_diag_indices = find_non_diag_indices(tI_size, var_ptr)
    # println("length_non_diag: ",length(non_diag_indices))
    # println("size_A_m: ",size(problem.A,2)," siza_A_n: ",size(problem.A,1))
    @assert length(non_diag_indices) == size(problem.A, 2)
    problem.C[non_diag_indices] ./= sqrt(2.0)
    # A = A[:, non_diag_indices] ./ sqrt(2.0)
    # Design for CSC 
    row_idx = problem.A.rowval
    col_ptr = problem.A.colptr
    nzval = problem.A.nzval
    nz_needs_scaling = fill(false, length(nzval))
    for col in 1:(length(col_ptr)-1)
        if non_diag_indices[col]
            nz_needs_scaling[col_ptr[col]:(col_ptr[col+1]-1)] .= true
        end
    end
    nzval[nz_needs_scaling] ./= sqrt(2.0)
    # problem.A.nzval .= nzval
    problem.A_t = transpose(problem.A)
end




# function load_problem_core(
#     cone_size::Vector{Int},
#     b::Vector{Float64},           # 新增参数：右端项 b
#     psd_obj_coneId::Vector{Int},
#     psd_obj_rowVal::Vector{Int},
#     psd_obj_colVal::Vector{Int},
#     psd_obj_nnzVal::Vector{Float64},
#     lin_obj_idx::Vector{Int},     # 修改：线性目标通常只需要 index (或者 row)
#     lin_obj_val::Vector{Float64},
#     psd_con_consId::Vector{Int},
#     psd_con_coneId::Vector{Int},
#     psd_con_rowVal::Vector{Int},
#     psd_con_colVal::Vector{Int},
#     psd_con_nnzVal::Vector{Float64},
#     lin_con_consId::Vector{Int},
#     lin_con_idx::Vector{Int}, 
#     lin_con_nnzVal::Vector{Float64}
# )
#     n_cones = length(cone_size)
#     var_ptr = Vector{Int}(undef, n_cones + 1)
#     var_ptr[1] = 1
#     for i in 1:n_cones
#         sz = cone_size[i]
#         var_ptr[i+1] = var_ptr[i] + sz * (sz + 1) ÷ 2
#     end
    
#     n_psd_vars = var_ptr[end] - 1
#     max_lin_idx = isempty(lin_obj_idx) && isempty(lin_con_idx) ? 0 : 
#                   max(
#                       isempty(lin_obj_idx) ? 0 : maximum(lin_obj_idx),
#                       isempty(lin_con_idx) ? 0 : maximum(lin_con_idx)
#                   )
    
    
#     n_vars_total = n_psd_vars + max_lin_idx
    
#     if max_lin_idx > 0
#         push!(var_ptr, var_ptr[end] + max_lin_idx)
#     else
#         push!(var_ptr, var_ptr[end])
#     end

#     psd_var_coeff = !isempty(psd_obj_nnzVal)
#     lin_var_coeff = !isempty(lin_obj_val)

#     C = zeros(Float64, n_vars_total)
    
#     for k in eachindex(psd_obj_nnzVal)
#         bid = psd_obj_coneId[k]
#         r   = psd_obj_rowVal[k]
#         c   = psd_obj_colVal[k]
#         val = psd_obj_nnzVal[k]
#         if c > r
#             r, c = c, r
#         end
#         idx = var_ptr[bid] + (r - 1) * r ÷ 2 + c - 1
#         if r != c
#             C[idx] = 2.0 * val
#         else
#             C[idx] = val
#         end
#     end
#     lin_offset = var_ptr[n_cones + 1] - 1
#     for k in eachindex(lin_obj_val)
#         idx_rel = lin_obj_idx[k]
#         val     = lin_obj_val[k]
#         C[lin_offset + idx_rel] = val
#     end

#     A_I = Int[]
#     A_J = Int[]
#     A_V = Float64[]
    
#     for k in eachindex(psd_con_nnzVal)
#         cons_id = psd_con_consId[k]
#         bid     = psd_con_coneId[k]
#         r       = psd_con_rowVal[k]
#         c       = psd_con_colVal[k]
#         val     = psd_con_nnzVal[k]
        
#         if c > r
#             r, c = c, r
#         end
        
#         col_idx = var_ptr[bid] + (r - 1) * r ÷ 2 + c - 1
        
#         push!(A_I, cons_id)
#         push!(A_J, col_idx)
        
#         if r != c
#             push!(A_V, 2.0 * val)
#         else
#             push!(A_V, val)
#         end
#     end

#     for k in eachindex(lin_con_nnzVal)
#         cons_id = lin_con_consId[k]
#         idx_rel = lin_con_idx[k]
#         val     = lin_con_nnzVal[k]
        
#         col_idx = lin_offset + idx_rel
        
#         push!(A_I, cons_id)
#         push!(A_J, col_idx)
#         push!(A_V, val)
#     end

#     m = length(b)
#     n = n_vars_total
    
#     sparse_A = sparse(A_I, A_J, A_V, m, n)

#     problem = SemidefiniteProgram(
#         sparse_A,
#         sparse(transpose(sparse_A)),
#         b,
#         C,
#         cone_size,
#         var_ptr,
#         n,
#         m,
#         psd_var_coeff,
#         lin_var_coeff
#     )
    
#     return problem
# end

# function load_problem_from_spot_npz_to_cpu(filename::String)
#     println(CYAN, "🚀 Starting problem load for: ", filename, RESET)
#     println(CYAN, "NOTE: Current Version Change to Save only Lower Triangular Part of Each PSD Cone", RESET)
#     start_time = time()
    
#     # --- Cache Check ---
#     cache_dir = "problem_cache_cpu"
#     if !isdir(cache_dir) mkdir(cache_dir) end
#     cache_file = joinpath(cache_dir, replace(basename(filename), ".npz" => ".jls"))
#     println(YELLOW, "⏳ Cache not found. Proceeding with NPZ read...", RESET)

#     # --- Read NPZ File ---
#     npz_read_start = time()
#     data = npzread(filename)
#     tI_size = data["tI_size"]
#     print_timing_banner("NPZ file read", time() - npz_read_start, status_char="✅")

#     # --- Data Permutation ---
#     perm_start = time()
#     perm = sortperm(tI_size)
#     old2new = similar(perm)
#     old2new[perm] = 1:length(perm)
    
#     # 确保 cone_size 是 Int 类型，防止传入 Int32 导致 core 函数签名不匹配
#     permuted_tI_size = Vector{Int}(tI_size[perm]) 
#     print_timing_banner("Data permutation", time() - perm_start, status_char="✅")

#     # --- Prepare Data for Core ---
#     prep_start = time()

#     # 1. Prepare Objective (C)
#     psd_obj_coneId = Int[]
#     psd_obj_rowVal = Int[]
#     psd_obj_colVal = Int[]
#     psd_obj_nnzVal = Float64[]

#     # 严格检查 C 是否存在
#     has_psd_C = haskey(data, "C")

#     if has_psd_C
#         data_C = data["C"]
#         for i in 1:size(data_C, 1)
#             orig_block = Int(data_C[i, 1] + 1)
#             new_block  = old2new[orig_block]
#             col        = Int(data_C[i, 2] + 1)
#             row        = Int(data_C[i, 3] + 1)
#             val        = data_C[i, 4]

#             push!(psd_obj_coneId, new_block)
#             push!(psd_obj_rowVal, row)
#             push!(psd_obj_colVal, col)
#             push!(psd_obj_nnzVal, val)
#         end
#     end

#     lin_obj_idx = Int[]
#     lin_obj_val = Float64[]
    
#     if haskey(data, "c")
#         data_c = reshape(data["c"], :)
#         for i in eachindex(data_c)
#             push!(lin_obj_idx, i)
#             # 关键修复：原代码逻辑
#             # if !psd_var_coeff: C = vcat(..., -c)  -> 只有线性时取负
#             # else: C = vcat(C, c) -> 混合时取正
#             if !has_psd_C
#                 push!(lin_obj_val, data_c[i])
#             else
#                 push!(lin_obj_val, data_c[i])
#             end
#         end
#     end
    
#     if isempty(lin_obj_val) && isempty(psd_obj_nnzVal)
#          error("At least one of linear or PSD variable coefficients must be present.")
#     end

#     # 2. Prepare Constraints (A)
#     psd_con_consId = Int[]
#     psd_con_coneId = Int[]
#     psd_con_rowVal = Int[]
#     psd_con_colVal = Int[]
#     psd_con_nnzVal = Float64[]

#     data_A = data["A"]
#     for i in 1:size(data_A, 1)
#         cons_idx   = Int(data_A[i, 1] + 1)
#         orig_block = Int(data_A[i, 2] + 1)
#         new_block  = old2new[orig_block]
#         col        = Int(data_A[i, 3] + 1)
#         row        = Int(data_A[i, 4] + 1)
#         val        = data_A[i, 5]

#         push!(psd_con_consId, cons_idx)
#         push!(psd_con_coneId, new_block)
#         push!(psd_con_rowVal, row)
#         push!(psd_con_colVal, col)
#         push!(psd_con_nnzVal, val)
#     end

#     lin_con_consId = Int[]
#     lin_con_idx    = Int[]
#     lin_con_nnzVal = Float64[]

#     if haskey(data, "a")
#         data_a = data["a"]
#         for i in 1:size(data_a, 1)
#             cons_idx = Int(data_a[i, 1] + 1)
#             col_idx  = Int(data_a[i, 2] + 1)
#             val      = data_a[i, 3]

#             push!(lin_con_consId, cons_idx)
#             push!(lin_con_idx, col_idx)
#             push!(lin_con_nnzVal, val)
#         end
#     end

#     # 3. Prepare b
#     b = Vector{Float64}(vec(data["b"]))
    
#     print_timing_banner("Data preparation", time() - prep_start, status_char="✅")

#     # --- Call Core Construction ---
#     core_start = time()
#     problem = load_problem_core(
#         permuted_tI_size,
#         b,
#         psd_obj_coneId, psd_obj_rowVal, psd_obj_colVal, psd_obj_nnzVal,
#         lin_obj_idx, lin_obj_val,
#         psd_con_consId, psd_con_coneId, psd_con_rowVal, psd_con_colVal, psd_con_nnzVal,
#         lin_con_consId, lin_con_idx, lin_con_nnzVal
#     )
#     print_timing_banner("Problem construction (Core)", time() - core_start, status_char="✅")

#     # --- Post-Processing / Scaling ---
#     # serialize(cache_file, problem)
#     println(MAGENTA, "💾 Problem cached to: ", cache_file, RESET)
    
#     total_time = time() - start_time
#     @printf("%s - Total Problem Load Time: %.4f seconds%s\n", CYAN, total_time, RESET)

#     sc2_time = time()
#     println(MAGENTA, "Scaling and converting to primal-dual compatible format", RESET)
    
#     convert_problem_to_primal_dual_compat_new!(problem)
    
#     # Row Normalization
#     row_norms = sqrt.(sum(abs2, problem.A, dims=2))
#     row_norms[row_norms .== 0.0] .= 1.0
    
#     # Efficient scaling: A = D * A, b = D * b
#     d = vec(1.0 ./ row_norms)
#     # 必须注意: spdiagm 生成的类型是否兼容，通常是 Float64
#     problem.A = spdiagm(d) * problem.A
#     problem.b .= problem.b .* d 

#     print_timing_banner("Scaling", time() - sc2_time, status_char="✅")

#     return problem
# end