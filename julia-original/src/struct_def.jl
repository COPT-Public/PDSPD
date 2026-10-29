mutable struct CuSemidefiniteProgram
    A::PDSDP_GPU_MATRIX_FLOAT
    A_t::PDSDP_GPU_MATRIX_FLOAT
    b::PDSDP_GPU_VECTOR_FLOAT
    C::PDSDP_GPU_VECTOR_FLOAT
    tI_size::Vector{PDSDP_INT}
    var_ptr::Vector{PDSDP_INT}
    n::PDSDP_INT
    m::PDSDP_INT
    psd_var_coeff::Bool
    lin_var_coeff::Bool
    cone_old2new::Union{Vector{PDSDP_INT}, Nothing}
    original_tI_size::Union{Vector{PDSDP_INT}, Nothing}
end

function CuSemidefiniteProgram(
    A::PDSDP_GPU_MATRIX_FLOAT,
    A_t::PDSDP_GPU_MATRIX_FLOAT,
    b::PDSDP_GPU_VECTOR_FLOAT,
    C::PDSDP_GPU_VECTOR_FLOAT,
    tI_size::Vector{PDSDP_INT},
    var_ptr::Vector{PDSDP_INT},
    n::PDSDP_INT,
    m::PDSDP_INT,
    psd_var_coeff::Bool,
    lin_var_coeff::Bool,
)
    return CuSemidefiniteProgram(
        A,
        A_t,
        b,
        C,
        tI_size,
        var_ptr,
        n,
        m,
        psd_var_coeff,
        lin_var_coeff,
        nothing,
        nothing,
    )
end

mutable struct SemidefiniteProgram
    A::PDSDP_CPU_MATRIX_FLOAT
    A_t::PDSDP_CPU_MATRIX_FLOAT
    b::PDSDP_CPU_VECTOR_FLOAT
    C::PDSDP_CPU_VECTOR_FLOAT
    tI_size::Vector{PDSDP_INT}
    var_ptr::Vector{PDSDP_INT}
    n::PDSDP_INT
    m::PDSDP_INT
    psd_var_coeff::Bool
    lin_var_coeff::Bool
    cone_old2new::Union{Vector{PDSDP_INT}, Nothing}
    original_tI_size::Union{Vector{PDSDP_INT}, Nothing}
end

function SemidefiniteProgram(
    A::PDSDP_CPU_MATRIX_FLOAT,
    A_t::PDSDP_CPU_MATRIX_FLOAT,
    b::PDSDP_CPU_VECTOR_FLOAT,
    C::PDSDP_CPU_VECTOR_FLOAT,
    tI_size::Vector{PDSDP_INT},
    var_ptr::Vector{PDSDP_INT},
    n::PDSDP_INT,
    m::PDSDP_INT,
    psd_var_coeff::Bool,
    lin_var_coeff::Bool,
)
    return SemidefiniteProgram(
        A,
        A_t,
        b,
        C,
        tI_size,
        var_ptr,
        n,
        m,
        psd_var_coeff,
        lin_var_coeff,
        nothing,
        nothing,
    )
end

function copy(sdp::SemidefiniteProgram)
    SemidefiniteProgram(
        copy(sdp.A),         
        copy(sdp.A_t),
        copy(sdp.b),        
        copy(sdp.C),
        copy(sdp.tI_size),  
        copy(sdp.var_ptr),   
        sdp.n,              
        sdp.m,
        sdp.psd_var_coeff,
        sdp.lin_var_coeff,
        isnothing(sdp.cone_old2new) ? nothing : copy(sdp.cone_old2new),
        isnothing(sdp.original_tI_size) ? nothing : copy(sdp.original_tI_size),
    )
end

function cpu_to_gpu(sdp::SemidefiniteProgram)

    return CuSemidefiniteProgram(
        CuSparseMatrixCSR(sdp.A),
        CuSparseMatrixCSR(sdp.A_t),
        CuArray(sdp.b),
        CuArray(sdp.C),
        sdp.tI_size,
        sdp.var_ptr,
        sdp.n,
        sdp.m,
        sdp.psd_var_coeff,
        sdp.lin_var_coeff,
        isnothing(sdp.cone_old2new) ? nothing : copy(sdp.cone_old2new),
        isnothing(sdp.original_tI_size) ? nothing : copy(sdp.original_tI_size),
    )
end

struct ProjTask
    start_idx::PDSDP_INT   
    end_idx::PDSDP_INT
    dim::PDSDP_INT
    batch_size::PDSDP_INT
    method::PDSDP_INT      
    gpu_id::PDSDP_INT         
    stream::Union{CUDA.CuStream, Nothing}
    buffer_cpu::Union{Vector{PDSDP_FLOAT}, Nothing}
    packed_buffer::Union{PDSDP_GPU_VECTOR_FLOAT, Nothing}
    buffer_gpu::Union{PDSDP_GPU_VECTOR_FLOAT, Nothing}
    extra_buffer::Union{PDSDP_GPU_VECTOR_FLOAT, Nothing}
    event::Union{CUDA.CuEvent, Nothing}
end


mutable struct PdhgState
    current_primal_solution::PDSDP_GPU_VECTOR_FLOAT
    current_dual_solution::PDSDP_GPU_VECTOR_FLOAT
    last_primal_solution::PDSDP_GPU_VECTOR_FLOAT
    last_dual_solution::PDSDP_GPU_VECTOR_FLOAT
    avg_primal_solution::PDSDP_GPU_VECTOR_FLOAT
    avg_dual_solution::PDSDP_GPU_VECTOR_FLOAT
    primal_gradient::PDSDP_GPU_VECTOR_FLOAT
    step_size::PDSDP_FLOAT
    primal_weight::PDSDP_FLOAT
    numerical_error::Bool
    num_outer_iterations::PDSDP_INT
    num_inner_iterations::PDSDP_INT
end

mutable struct RestartInfo
    last_restart_primal_solution::PDSDP_GPU_VECTOR_FLOAT
    last_restart_dual_solution::PDSDP_GPU_VECTOR_FLOAT
    last_residual::PDSDP_FLOAT
    last_restart_residual::PDSDP_FLOAT
    restart_count::PDSDP_INT
    restart_primal_distance::PDSDP_FLOAT
    restart_dual_distance::PDSDP_FLOAT
    interval_iterations::PDSDP_INT
    last_ratio::PDSDP_FLOAT
end

mutable struct BufferState
    primal_product::PDSDP_GPU_VECTOR_FLOAT
    avg_C_dual_product_projected::PDSDP_GPU_VECTOR_FLOAT
    C_dual_product::PDSDP_GPU_VECTOR_FLOAT
    C_dual_product_projected::PDSDP_GPU_VECTOR_FLOAT
    last_primal_product::PDSDP_GPU_VECTOR_FLOAT
    last_C_dual_product::PDSDP_GPU_VECTOR_FLOAT
    avg_primal_product::PDSDP_GPU_VECTOR_FLOAT
    avg_C_dual_product::PDSDP_GPU_VECTOR_FLOAT
    delta_primal::PDSDP_GPU_VECTOR_FLOAT
    delta_dual::PDSDP_GPU_VECTOR_FLOAT
    delta_primal_product::PDSDP_GPU_VECTOR_FLOAT
    tmp_primal_solution::PDSDP_GPU_VECTOR_FLOAT
    tmp_dual_solution::PDSDP_GPU_VECTOR_FLOAT
end

mutable struct ResidualInfo
    primal_residual::PDSDP_FLOAT
    dual_residual::PDSDP_FLOAT
    primal_dual_gap::PDSDP_FLOAT
    best_primal_residual::PDSDP_FLOAT
    best_dual_residual::PDSDP_FLOAT
    best_primal_dual_gap::PDSDP_FLOAT
    avg_primal_residual::PDSDP_FLOAT
    avg_dual_residual::PDSDP_FLOAT
    avg_primal_dual_gap::PDSDP_FLOAT
    residual::PDSDP_FLOAT
    avg_residual::PDSDP_FLOAT
    last_residual::PDSDP_FLOAT
    # Construct
    function ResidualInfo()
        return new(
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf,
            Inf
        )
    end
end

mutable struct ScalingInfoCpu
    b_scale::PDSDP_FLOAT
    C_scale::PDSDP_FLOAT
    row_norm::PDSDP_CPU_VECTOR_FLOAT
    col_norm::PDSDP_CPU_VECTOR_FLOAT
    norm_b::PDSDP_FLOAT
    norm_C::PDSDP_FLOAT
    norm_b_ori::PDSDP_FLOAT
    norm_C_ori::PDSDP_FLOAT
end
function get_batch_cone_info(tI_size)
	tI_size
	nbatch = length(tI_size)
	batch_dim = Int32[]
	batch_ptr = Int32[1]
	block_dim = Int32[]
	last_size = tI_size[1]
	current_size = tI_size[1] * tI_size[1]
	cul_ptr = 1
	for i in 2:nbatch
		if tI_size[i] == last_size
			cul_ptr += 1
			current_size += tI_size[i] * tI_size[i]
		else
			push!(batch_dim, Int32(current_size))
			push!(batch_ptr, Int32(batch_ptr[end] + cul_ptr))
			push!(block_dim, Int32(tI_size[i-1]))
			last_size = tI_size[i]
			current_size = tI_size[i] * tI_size[i]
			cul_ptr = 1
		end
	end
	push!(batch_dim, Int32(current_size))
	push!(batch_ptr, Int32(batch_ptr[end] + cul_ptr))
	push!(block_dim, Int32(tI_size[end]))
    cone_num = Int32[]
    for (i, sz) in enumerate(batch_ptr[1:end-1])
        count = batch_ptr[i+1] - batch_ptr[i]
        push!(cone_num, count)
    end
    return (batch_dim, batch_ptr, block_dim, cone_num)
end
mutable struct BatchConeInfo
    batch_dim::Vector{PDSDP_INT}
    batch_ptr::Vector{PDSDP_INT}
    cone_dim::Vector{PDSDP_INT}
    cone_num::Vector{PDSDP_INT}
    svd_method::Union{Vector{PDSDP_INT}, Nothing} 
    proj_task::Union{Dict{PDSDP_INT, Vector{ProjTask}}, Nothing}
    function BatchConeInfo(tI_size::Vector{PDSDP_INT})
        batch_dim, batch_ptr, cone_dim, cone_num  = get_batch_cone_info(tI_size)
        return new(
            batch_dim,
            batch_ptr,
            cone_dim,
            cone_num,
            nothing,
            nothing
        )
    end
end



mutable struct ScalingInfoGpu
    b_scale::PDSDP_FLOAT
    C_scale::PDSDP_FLOAT
    row_norm::PDSDP_GPU_VECTOR_FLOAT
    col_norm::PDSDP_GPU_VECTOR_FLOAT
    norm_b::PDSDP_FLOAT
    norm_C::PDSDP_FLOAT
    norm_b_ori::PDSDP_FLOAT
    norm_C_ori::PDSDP_FLOAT
    function ScalingInfoGpu(cpu_scaling_info::ScalingInfoCpu)
        return new(
            cpu_scaling_info.b_scale,
            cpu_scaling_info.C_scale,
            CuArray(cpu_scaling_info.row_norm),
            CuArray(cpu_scaling_info.col_norm),
            cpu_scaling_info.norm_b,
            cpu_scaling_info.norm_C,
            cpu_scaling_info.norm_b_ori,
            cpu_scaling_info.norm_C_ori
        )
    end
    function ScalingInfoGpu()
        return new(
            1.0,
            1.0,
            CuArray(ones(PDSDP_FLOAT, 1)),
            CuArray(ones(PDSDP_FLOAT, 1)),
            1.0,
            1.0,
            1.0,
            1.0
        )
    end
end

struct PdhgOption
    max_iterations::PDSDP_INT
    tol::PDSDP_FLOAT
    time_limit::PDSDP_FLOAT
    evaluate_frequency::PDSDP_INT
    restart_skip_iterations::PDSDP_INT
    artificial_restart_threshold::PDSDP_FLOAT
    sufficient_reduction_threshold::PDSDP_FLOAT
    necessary_reduction_threshold::PDSDP_FLOAT
    primal_weight_update_smoothing ::PDSDP_FLOAT
    step_size_trace_back_limit::PDSDP_INT
    closed_all_trick::Bool
    print_frequency::PDSDP_INT
    restart_scheme::String
    stepsize_scheme::String
    Kp::PDSDP_FLOAT
    Ki::PDSDP_FLOAT
    Kd::PDSDP_FLOAT
    i_smooth::PDSDP_FLOAT
    save_trajectory::Bool
    init_step_size::Union{PDSDP_FLOAT, Nothing}
    init_primal_weight::Union{PDSDP_FLOAT, Nothing}
    ruiz_rescaling::Bool
    pock_chambolle_rescaling::Bool
    bc_rescaling::Bool
    stop_criterion::String
    function PdhgOption(
        ; max_iterations = 10000,
        tol = 1e-6,
        time_limit = 3600.0,
        evaluate_frequency = 100,
        restart_skip_iterations = 200,
        artificial_restart_threshold = 0.2,
        sufficient_reduction_threshold = 0.2,
        necessary_reduction_threshold = 0.8,
        primal_weight_update_smoothing = 0.2,
        step_size_trace_back_limit = 10,
        closed_all_trick = false,
        print_frequency = 1000,
        restart_scheme = PDSDP_AVERAGE_RESTART,
        stepsize_scheme = PDSDP_rAPDHG_STEPSIZE,
        Kp = 0.99,
        Ki = 0.01,
        Kd = 0.0,
        i_smooth = 0.3,
        save_trajectory = false,
        init_step_size = nothing,
        init_primal_weight = nothing,
        ruiz_rescaling = true,
        pock_chambolle_rescaling = false,
        bc_rescaling = true,
        stop_criterion = ORIGIN_PROBLEM
    )
        return new(
            max_iterations,
            tol,
            time_limit,
            evaluate_frequency,
            restart_skip_iterations,
            artificial_restart_threshold,
            sufficient_reduction_threshold,
            necessary_reduction_threshold,
            primal_weight_update_smoothing,
            step_size_trace_back_limit,
            closed_all_trick,
            print_frequency,
            restart_scheme,
            stepsize_scheme,
            Kp,
            Ki, Kd, i_smooth,
            save_trajectory,
            init_step_size,
            init_primal_weight,
            ruiz_rescaling,
            pock_chambolle_rescaling,
            bc_rescaling,
            stop_criterion
        )
        
    end
end
mutable struct PIDController
    integral_error::PDSDP_FLOAT
    last_error::PDSDP_FLOAT
    Kp::PDSDP_FLOAT
    Ki::PDSDP_FLOAT
    Kd::PDSDP_FLOAT
    i_smooth::PDSDP_FLOAT
    function PIDController(
        ;Kp::PDSDP_FLOAT = 0.99,
        Ki::PDSDP_FLOAT = 0.01,
        Kd::PDSDP_FLOAT = 0.0,
        i_smooth::PDSDP_FLOAT = 0.3
    )
        return new(0.0, 0.0, Kp, Ki, Kd, i_smooth)
    end
end
mutable struct SolverInfo
    time_dict::Dict{String, PDSDP_FLOAT}
    iteration_count::PDSDP_INT
    status::String
    primal_residual::PDSDP_FLOAT
    dual_residual::PDSDP_FLOAT
    primal_dual_gap::PDSDP_FLOAT
    primal_obj::PDSDP_FLOAT
    dual_obj::PDSDP_FLOAT
    estimate_eignvalue::Union{PDSDP_FLOAT, Nothing}  
    traj_buffer 
    function SolverInfo()
        return new(
            Dict(),
            0,
            PDSDP_NOT_STARTED,
            Inf,
            Inf,
            Inf,
            Inf,
            -Inf,
            nothing,
            []
        )     
    end
end

function update_residual_obj_info!(solver_info::SolverInfo, primal_residual::PDSDP_FLOAT, dual_residual::PDSDP_FLOAT, primal_dual_gap::PDSDP_FLOAT, primal_obj::PDSDP_FLOAT, dual_obj::PDSDP_FLOAT)
    solver_info.primal_residual = primal_residual
    solver_info.dual_residual = dual_residual
    solver_info.primal_dual_gap = primal_dual_gap
    solver_info.primal_obj = primal_obj
    solver_info.dual_obj = dual_obj
end



mutable struct DeviceInfo
    device_num::Int32
    device_id::Vector{Int32}
    mem_info::Vector{Float64}
    devices::Vector{CUDA.CuDevice}
    stream_pool::Dict{Int32, CircularBuffer{CUDA.CuStream}}
    stream_per_device::Int32
    threads::Int32

    function DeviceInfo(stream_per_device::Int32)
        device_id = Int32[]
        mem_info = Float64[]
        devices = CUDA.devices()
        stream_pool = Dict{Int32, CircularBuffer{CUDA.CuStream}}()
        threads = nthreads()
        if threads < 4
            @warn "Current number of threads ($threads) may be insufficient for optimal performance. Consider increasing the number of threads by setting the JULIA_NUM_THREADS environment variable or use julia -t N to start Julia with N threads."
        end
            for (i, device) in enumerate(devices)
            CUDA.device!(device)
            mem = CUDA.memory_info()[1]/ 1024.0 / 1024.0  
            push!(mem_info, mem)
            push!(device_id, i - 1)  
            pool = CircularBuffer{CUDA.CuStream}(Int64(stream_per_device))
            for _ in 1:stream_per_device
                push!(pool, CUDA.CuStream())
            end
            stream_pool[i - 1] = pool
        end
        
        new(
            length(devices),
            device_id,
            mem_info,
            collect(devices),
            stream_pool,
            stream_per_device,
            threads
        )
    end
end



mutable struct TableLogger
    header::Vector{String}
    widths::Vector{Int}
    header_done::Bool
    TableLogger() = new(String[], Int[], false)
end

# Função auxiliar para centralizar o texto
function center_text(text::String, width::Int)
    pad_total = width - length(text)
    pad_left = div(pad_total, 2)
    pad_right = pad_total - pad_left
    return " "^pad_left * text * " "^pad_right
end

# Inicialização interna
function _init!(t::TableLogger, header::Vector{String})
    t.header_done && return
    t.header = header
    # Adiciona 2 espaços de preenchimento (1 de cada lado)
    t.widths = max.(20, length.(header) .+ 2)
end
function print_info(t::TableLogger, info::String)
    # This only works if the header has been printed (so we know the widths)
    if !t.header_done
        @warn "Table not initialized. Cannot print info line."
        return
    end

    bold = "\e[1;37m"; reset = "\e[0m"

    # Calculate the total internal width of the table
    total_width = sum(t.widths) + length(t.widths) - 1

    # Prepare the content with padding
    padded_info = "-" * info * " "

    # Calculate how many dashes are needed to fill the rest of the line
    num_dashes = max(0, total_width - length(padded_info))
    dashes = "─"^num_dashes

    # Build the full line, ensuring it doesn't exceed the total width
    # Use first() to safely handle multi-byte characters and avoid StringIndexError
    line_content = first(padded_info * dashes, total_width-1)

    println(bold * "│" * line_content * "│" * reset)
end
# Cabeçalho preguiçoso
function print_header(t::TableLogger)
    if t.header_done
        return
    end
    
    # As larguras são para o conteúdo, as linhas de separação se ajustam a elas
    sep = "┌" * join(["─"^w for w in t.widths], "┬") * "┐"
    mid = "├" * join(["─"^w for w in t.widths], "┼") * "┤"
    bold = "\e[1;37m"; reset = "\e[0m"
    
    println(bold * sep * reset)
    
    # Centraliza cada cabeçalho dentro da sua largura de coluna
    cells = [center_text(h, w) for (h, w) in zip(t.header, t.widths)]
    
    println(bold * "│" * join(cells, "│") * "│" * reset)
    println(bold * mid * reset)
    t.header_done = true
end

# Cada chamada: o cabeçalho só tem efeito na primeira vez
function print_row(t::TableLogger, header::Vector{String}, values::Vector{<:Real})
    _init!(t, header)  # Registra o cabeçalho na primeira vez
    print_header(t)
    
    bold = "\e[1;37m"; reset = "\e[0m"
    
    # Formata os valores para se alinharem corretamente dentro da largura definida
    cells = [Printf.@sprintf("%*.3e", w, v) for (v, w) in zip(values, t.widths)]
    
    println(bold * "│" * join(cells, "│") * "│" * reset)
end

# Função para fechar a tabela com uma linha inferior
function close_table(t::TableLogger)
    if t.header_done
        bot = "└" * join(["─"^w for w in t.widths], "┴") * "┘"
        bold = "\e[1;37m"; reset = "\e[0m"
        println(bold * bot * reset)
    end
end

mutable struct HyperParameters
    primal_step_size::PDSDP_FLOAT
    dual_step_size::PDSDP_FLOAT
    adaptive_level::PDSDP_FLOAT
    adaptive_diminish::PDSDP_FLOAT
    primal_dual_scale::PDSDP_FLOAT
    residual_size::PDSDP_FLOAT
    update_times::PDSDP_INT

    function HyperParameters(
        ; primal_step_size = 0.1,
        dual_step_size = 0.1,
        adaptive_level = 0.95,
        adaptive_diminish = 0.95,
        primal_dual_scale = 1.0,
        residual_size = 1.5
    )
        return new(
            primal_step_size,
            dual_step_size,
            adaptive_level,
            adaptive_diminish,
            primal_dual_scale,
            residual_size,
            0
        )
    end
end

mutable struct PdsdpSolver
    problem::CuSemidefiniteProgram
    state::PdhgState
    restart_info::RestartInfo
    buffer::BufferState
    residual_info::ResidualInfo
    option::PdhgOption
    solver_info::SolverInfo
    scaling_info::ScalingInfoGpu
    logger::TableLogger
    hyper_parameters::HyperParameters
    batch_cone_info::BatchConeInfo
    device_info::DeviceInfo
    pid_controller::PIDController
end