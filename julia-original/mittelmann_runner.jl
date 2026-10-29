using LinearAlgebra
using SparseArrays
using Pkg

Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "PDSDP.jl"))

const SQRT2 = sqrt(2.0)

function numeric_tokens(line::AbstractString)
    stripped = lstrip(line)
    if isempty(stripped) || first(stripped) in ('*', '"', '#')
        return SubString{String}[]
    end
    normalized = replace(line, r"[,{}()=:]" => " ")
    return split(normalized)
end

"""Load an SDPA-sparse problem into PDSDP's packed-symmetric representation.

Positive SDPA blocks remain PSD cones.  Each coordinate of a negative
(diagonal nonnegative) SDPA block becomes a 1x1 PSD cone, because PDSDP's
trailing linear variables are free rather than nonnegative.
"""
function load_sdpa_sparse(filename::String)
    io = open(filename, "r")
    try
        header = String[]
        target = typemax(Int)
        while length(header) < target
            eof(io) && error("Unexpected EOF in SDPA header")
            tokens = numeric_tokens(readline(io))
            isempty(tokens) && continue
            append!(header, String.(tokens))
            if length(header) >= 2 && target == typemax(Int)
                nblocks = parse(Int, header[2])
                target = 2 + nblocks
            elseif target != typemax(Int) && length(header) >= target
                m = parse(Int, header[1])
                target = 2 + parse(Int, header[2]) + m
            end
        end

        m = parse(Int, header[1])
        nblocks = parse(Int, header[2])
        expected = 2 + nblocks + m
        length(header) == expected || error(
            "RHS and matrix entries must be separated at a line boundary",
        )
        raw_dims = parse.(Int, header[3:(2 + nblocks)])
        any(iszero, raw_dims) && error("Zero-dimensional SDPA blocks are unsupported")
        b = parse.(Float64, header[(3 + nblocks):expected])

        # Original PDSDP cone order is all genuine PSD blocks followed by the
        # scalar coordinates of SDPA diagonal/LP blocks.  PDSDP internally
        # sorts by dimension to make same-size projections contiguous.
        original_dims = Int[]
        raw_psd_to_original = zeros(Int, nblocks)
        for raw_block in eachindex(raw_dims)
            dim = raw_dims[raw_block]
            if dim > 0
                push!(original_dims, dim)
                raw_psd_to_original[raw_block] = length(original_dims)
            end
        end
        raw_lp_offset = zeros(Int, nblocks)
        for raw_block in eachindex(raw_dims)
            dim = raw_dims[raw_block]
            if dim < 0
                raw_lp_offset[raw_block] = length(original_dims)
                append!(original_dims, ones(Int, -dim))
            end
        end

        perm = sortperm(original_dims; alg=Base.Sort.MergeSort)
        old2new = invperm(perm)
        sorted_dims = original_dims[perm]
        var_ptr = Vector{Int32}(undef, length(sorted_dims) + 2)
        var_ptr[1] = 1
        for cone in eachindex(sorted_dims)
            dim = sorted_dims[cone]
            next_ptr = Int64(var_ptr[cone]) + dim * (dim + 1) ÷ 2
            next_ptr <= typemax(Int32) || error("Packed problem exceeds PDSDP Int32 capacity")
            var_ptr[cone + 1] = Int32(next_ptr)
        end
        # PDSDP reserves the last var_ptr interval for unconstrained linear
        # variables.  There are none here, so append an empty interval.
        var_ptr[end] = var_ptr[end - 1]
        n = Int(var_ptr[end]) - 1

        function packed_col(original_cone::Int, row::Int, col::Int)
            row < col && ((row, col) = (col, row))
            sorted_cone = old2new[original_cone]
            return Int(var_ptr[sorted_cone]) + row * (row - 1) ÷ 2 + col - 1
        end

        A_i = Int32[]
        A_j = Int32[]
        A_v = Float64[]
        C = zeros(Float64, n)
        entry_count = 0

        for line in eachline(io)
            tokens = numeric_tokens(line)
            isempty(tokens) && continue
            length(tokens) == 5 || error("Invalid SDPA matrix line: $line")
            matrix_id = parse(Int, tokens[1])
            raw_block = parse(Int, tokens[2])
            row = parse(Int, tokens[3])
            col = parse(Int, tokens[4])
            value = parse(Float64, tokens[5])
            0 <= matrix_id <= m || error("Matrix id out of range: $matrix_id")
            1 <= raw_block <= nblocks || error("Block id out of range: $raw_block")
            dim = raw_dims[raw_block]

            original_cone = 0
            coefficient = value
            if dim > 0
                1 <= row <= dim && 1 <= col <= dim || error("PSD index out of range")
                original_cone = raw_psd_to_original[raw_block]
                row != col && (coefficient *= SQRT2)
            else
                row == col || error("Non-diagonal entry in SDPA diagonal block $raw_block")
                1 <= row <= -dim || error("Diagonal-block index out of range")
                original_cone = raw_lp_offset[raw_block] + row
                row = 1
                col = 1
            end
            variable = packed_col(original_cone, row, col)
            if matrix_id == 0
                # Sparse SDPA stores F0, while the primal objective matrix is
                # C=-F0 and PDSDP uses the slack C+A'*lambda.
                C[variable] -= coefficient
            else
                push!(A_i, Int32(matrix_id))
                push!(A_j, Int32(variable))
                push!(A_v, coefficient)
            end
            entry_count += 1
        end

        A64 = sparse(A_i, A_j, A_v, m, n)
        A = convert(SparseMatrixCSC{Float64, Int32}, A64)
        problem = SemidefiniteProgram(
            A,
            convert(SparseMatrixCSC{Float64, Int32}, sparse(transpose(A))),
            b,
            C,
            Int32.(sorted_dims),
            var_ptr,
            Int32(n),
            Int32(m),
            true,
            false,
            Int32.(old2new),
            Int32.(original_dims),
        )
        println(
            "PDSDP_SDPA_LOAD m=$m n=$n raw_blocks=$nblocks cones=$(length(sorted_dims)) " *
            "lp_scalars=$(count(<(0), raw_dims) == 0 ? 0 : sum(-d for d in raw_dims if d < 0)) " *
            "entries=$entry_count nnzA=$(nnz(A))",
        )
        return problem
    finally
        close(io)
    end
end

function usage()
    println(stderr, "usage: julia mittelmann_runner.jl PROBLEM.dat-s OUTPUT.mat [TIME_SEC] [MAX_ITER] [TOL]")
end

if length(ARGS) < 2
    usage()
    exit(2)
end

problem_path = abspath(ARGS[1])
output_path = abspath(ARGS[2])
time_limit = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3600.0
max_iterations = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20000
tol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-4

BLAS.set_num_threads(parse(Int, get(ENV, "PDSDP_BLAS_THREADS", "32")))
mkpath(dirname(output_path))
benchmark_start = time()
problem = load_sdpa_sparse(problem_path)
option = PdhgOption(
    tol=tol,
    time_limit=time_limit,
    max_iterations=max_iterations,
    evaluate_frequency=parse(Int, get(ENV, "PDSDP_EVALUATE_FREQUENCY", "40")),
    restart_skip_iterations=40,
    ruiz_rescaling=true,
    bc_rescaling=true,
    pock_chambolle_rescaling=true,
    sufficient_reduction_threshold=0.2,
    necessary_reduction_threshold=0.9,
    primal_weight_update_smoothing=0.1,
    artificial_restart_threshold=0.2,
    print_frequency=parse(Int, get(ENV, "PDSDP_PRINT_FREQUENCY", "200")),
    restart_scheme=PDSDP_HALPERN_RESTART,
    stepsize_scheme=PDSDP_rAPDHG_STEPSIZE,
    Kp=0.90,
    Ki=0.01,
    Kd=0.0,
    stop_criterion=ORIGIN_PROBLEM,
)

result = pdhg_sdp(
    problem;
    option=option,
    save_solution_mat=true,
    output_mat_path=output_path,
)
benchmark_time = time() - benchmark_start
status = result["status"]
iterations = result["iterations"]
solver_time_sec = result["total_time"]
termination_time_sec = result["termination_time"]
reported_primal_residual = result["primal_residual"]
reported_dual_residual = result["dual_residual"]
reported_primal_dual_gap = result["primal_dual_gap"]
reported_primal_obj = result["primal_obj"]
reported_dual_obj = result["dual_obj"]
println(
    "PDSDP_RESULT status=$status iterations=$iterations " *
    "solver_time_sec=$solver_time_sec termination_time_sec=$termination_time_sec " *
    "benchmark_time_sec=$benchmark_time " *
    "primal=$reported_primal_residual dual=$reported_dual_residual " *
    "gap=$reported_primal_dual_gap primal_obj=$reported_primal_obj " *
    "dual_obj=$reported_dual_obj output=$output_path",
)
