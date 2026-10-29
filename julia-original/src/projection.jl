const MAX_STREAMS = 64
const PROJECTION_PROFILE_SWITCH = Ref(false)

function set_projection_profile!(enabled::Bool)
	PROJECTION_PROFILE_SWITCH[] = enabled
	return PROJECTION_PROFILE_SWITCH[]
end

function is_projection_profile_enabled()
	return PROJECTION_PROFILE_SWITCH[] || get(ENV, "PDSDP_PROFILE_PROJECTION", "0") == "1"
end


function batch_projection!(x::PDSDP_GPU_VECTOR_FLOAT,
	batch_dim::Vector{PDSDP_INT},
	batch_ptr::Vector{PDSDP_INT};
	min_value::Float64 = 0.0)
	"""
	Will be Deprecated. 
	"""

	x_cpu = batch_projection_cpu!(Array(x), batch_dim, batch_ptr; min_value=min_value)
	copyto!(x, x_cpu)
	# parallel_projection!(x, batch_dim, batch_ptr; min_value=min_value)
	# x = batch_projection_gpu!(x,batch_dim, batch_ptr)
	return x
end
function pack_lower_triangular_rowmajor!(
    tri_x::AbstractVector{Float64},
    full_x::AbstractMatrix{Float64},
    n::Integer,
)
    k = 1
    @inbounds for i in 1:Int(n)
        for j in 1:i
			if i == j
				tri_x[k] = full_x[i, j]
			else
            	tri_x[k] = sqrt(2.0) * full_x[i, j]
			end
            k += 1
        end
    end
    nothing
end

function unpack_lower_triangular_rowmajor!(
    tri_x::AbstractVector{Float64},
    full_x::AbstractMatrix{Float64},
    n::Integer,
)
    k = 1
    @inbounds for i in 1:Int(n)
        for j in 1:i
            v = tri_x[k]
			if i == j
           	 	full_x[i, j] = v
			end
            if i != j
                full_x[j, i] = 1/sqrt(2.0) * v
				full_x[i, j] = 1/sqrt(2.0) * v
            end
			k += 1
        end
    end
    nothing
end

function batch_projection_cpu!(
    x::Vector{Float64},
    batch_dim::Vector{Int32},
    batch_ptr::Vector{Int32};
    min_value::Float64 = 0.0)
    
    x_cpu = Array(x)
    nbatch = length(batch_dim)
    
    @threads for k in 1:nbatch
        n = batch_dim[k]
        idx = batch_ptr[k]:(batch_ptr[k+1]-1)
        # assert(length(idx) == n * (n + 1) ÷ 2)
		if length(idx) != n * (n + 1) ÷ 2
			error("Inconsistent dimension in batch projection! Expected length $(n * (n + 1) ÷ 2), got $(length(idx))")
		end
        full_mat = Matrix{Float64}(undef, n, n)
        
        unpack_lower_triangular_rowmajor!(view(x_cpu, idx), full_mat, n)
        
        F = eigen(Symmetric(full_mat))
        F.values .= max.(F.values, min_value)
        full_mat .= F.vectors * Diagonal(F.values) * F.vectors'
        
        pack_lower_triangular_rowmajor!(view(x_cpu, idx), full_mat, n)
    end
    return x_cpu
end


function alloc_proj_method!(solver::PdsdpSolver)

	if !isnothing(solver.batch_cone_info.svd_method)
		Warning("SVD method has been allocated! This may cause extra unecessary computation.")
	end
	svd_method = []
	for idx in 1:length(solver.batch_cone_info.cone_dim)
		size = solver.batch_cone_info.cone_dim[idx]
		number = solver.batch_cone_info.cone_num[idx]
		if false && number <= 20  && size > 30
			push!(svd_method, PDSDP_CPU_SVD)
		elseif size == 1
			push!(svd_method, PDSDP_SCALAR_PROJECTION)
		elseif size > 100
		# elseif size > 100			
			push!(svd_method, PDSDP_GPU_DIRECT_SVD)
		else
			push!(svd_method, PDSDP_GPU_JACOBI_SVD)
		end
	end
	solver.batch_cone_info.svd_method = svd_method
	return nothing
end

function prepare_task!(
    solver::PdsdpSolver
)
    svd_method = solver.batch_cone_info.svd_method
    if isnothing(svd_method)
        error("SVD method has not been allocated! Please call `alloc_proj_method!` before projection.")
    end

    # Get solver parameters
    batch_ptr = solver.batch_cone_info.batch_ptr
    var_ptr = solver.problem.var_ptr[1:end - 1] # exclude the last part, unbounded variable
    cone_dim = solver.batch_cone_info.cone_dim
    cone_num = solver.batch_cone_info.cone_num

    num_gpus = length(CUDA.devices())
	proj_task = Dict{PDSDP_INT, Vector{ProjTask}}()
	batch_gpu_num = 0
	direct_gpu_num = 0
	scalar_gpu_num = 0
	count_info = Dict()
	for idx in 1:length(cone_dim)
		n = cone_dim[idx]
		len = n * (n + 1) ÷ 2
		start_idx = var_ptr[batch_ptr[idx]]
		end_idx = var_ptr[batch_ptr[idx+1]] - 1
		if svd_method[idx] == PDSDP_CPU_SVD
			proj_task[-1] = get!(proj_task, -1, Vector{ProjTask}())
			count_info[-1] = get(count_info, -1, Dict())
			for i in 1:cone_num[idx]
				push!(proj_task[-1], ProjTask(
					PDSDP_INT(start_idx + (i - 1) * len),
					PDSDP_INT(start_idx + i * len - 1),
					PDSDP_INT(n),	
					PDSDP_INT(1),
					PDSDP_CPU_SVD,
					Int32(-1),
					nothing,
					fill(0, len),
					nothing,
					nothing,
					nothing,
					nothing
				))
			end
			num_count = get(count_info[-1], n, 0)
			count_info[-1][n] = num_count + cone_num[idx]
		elseif svd_method[idx] == PDSDP_GPU_JACOBI_SVD
			gpu_id = batch_gpu_num % num_gpus
			stream_id = (batch_gpu_num + direct_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
			count_info[gpu_id] = get(count_info, gpu_id, Dict())
			proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
			CUDA.device!(gpu_id)
			batch_size = (Int(end_idx)-Int(start_idx)+1)÷(n * (n + 1) ÷ 2)
			push!(proj_task[gpu_id], ProjTask(
				PDSDP_INT(start_idx),
				PDSDP_INT(end_idx),
				PDSDP_INT(n),
				PDSDP_INT(batch_size),
				PDSDP_GPU_JACOBI_SVD,
				gpu_id,
				solver.device_info.stream_pool[gpu_id][stream_id],
				nothing,
				CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * len)),
				CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
				CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
				CUDA.CuEvent()
			))
			size_count = get(count_info[gpu_id], (batch_size, Int32(n)), 0)
			count_info[gpu_id][(batch_size, Int32(n))] = size_count + 1
			batch_gpu_num += 1
		elseif svd_method[idx] == PDSDP_GPU_DIRECT_SVD
			for i in 1:cone_num[idx]
				gpu_id = direct_gpu_num % num_gpus
				stream_id = (batch_gpu_num + direct_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
				count_info[gpu_id] = get(count_info, gpu_id, Dict())
				proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
				CUDA.device!(gpu_id)
				push!(proj_task[gpu_id], ProjTask(
					PDSDP_INT(start_idx + (i - 1) * len),
					PDSDP_INT(start_idx + i * len - 1),
					PDSDP_INT(n),
					PDSDP_INT(1),
					PDSDP_GPU_DIRECT_SVD,
					gpu_id,
					solver.device_info.stream_pool[gpu_id][stream_id],
					nothing,
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, len)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, n * n)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, n * n)),
					CUDA.CuEvent()
				))
				direct_gpu_num += 1
				num_count = get(count_info[gpu_id], n, 0)
				count_info[gpu_id][n] = num_count + 1
			end
		elseif svd_method[idx] == PDSDP_SCALAR_PROJECTION
			gpu_id = 0
			proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
			count_info[gpu_id] = get(count_info, gpu_id, Dict())
			
			stream_id = (batch_gpu_num + direct_gpu_num + scalar_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
			batch_size = (Int(end_idx)-Int(start_idx)+1)
			push!(proj_task[gpu_id], ProjTask(
				PDSDP_INT(start_idx),
				PDSDP_INT(end_idx),
				PDSDP_INT(1),	
				PDSDP_INT(batch_size),
				PDSDP_SCALAR_PROJECTION,
				Int32(gpu_id),
				solver.device_info.stream_pool[gpu_id][stream_id],
				nothing,
				nothing,
				nothing,
				nothing,
				nothing
			))
			num_count = get(count_info[gpu_id], 1, 0)
			count_info[gpu_id][(batch_size, 1)] = num_count + 1
			scalar_gpu_num += 1
		else
			error("Unknown SVD method: $(svd_method[idx])")
		end		
	end
	CUDA.device!(0)
	solver.batch_cone_info.proj_task = proj_task
	print_proj_task_info(proj_task, count_info)
	if !check_independent_projection(solver)
		error("Projection tasks are not independent! Please check the problem data.")
	else
		# println("Projection tasks are independent.")
		print_info_banner("Examined projection tasks are independent.", nothing; status_char="✅")
	end
	return nothing
end

function prepare_task_balance!(
    solver::PdsdpSolver; further_split_batch::Bool=false, target_batch_size::Int = 128
)
    svd_method = solver.batch_cone_info.svd_method
    if isnothing(svd_method)
        error("SVD method has not been allocated! Please call `alloc_proj_method!` before projection.")
    end

    # Get solver parameters
    batch_ptr = solver.batch_cone_info.batch_ptr
    var_ptr = solver.problem.var_ptr[1:end - 1] # exclude the last part, unbounded variable
    cone_dim = solver.batch_cone_info.cone_dim
    cone_num = solver.batch_cone_info.cone_num

    num_gpus = length(CUDA.devices())
	proj_task = Dict{PDSDP_INT, Vector{ProjTask}}()
	batch_gpu_num = 0
	direct_gpu_num = 0
	scalar_gpu_num = 0
	count_info = Dict()
	workload = zeros(Int, num_gpus)
	for idx in 1:length(cone_dim)
		n = cone_dim[idx]
		len = n * (n + 1) ÷ 2
		start_idx = var_ptr[batch_ptr[idx]]
		end_idx = var_ptr[batch_ptr[idx+1]] - 1
		if svd_method[idx] == PDSDP_CPU_SVD
			proj_task[-1] = get!(proj_task, -1, Vector{ProjTask}())
			count_info[-1] = get(count_info, -1, Dict())
			for i in 1:cone_num[idx]
				push!(proj_task[-1], ProjTask(
					PDSDP_INT(start_idx + (i - 1) * len),
					PDSDP_INT(start_idx + i * len - 1),
					PDSDP_INT(n),	
					PDSDP_INT(1),
					PDSDP_CPU_SVD,
					Int32(-1),
					nothing,
					fill(0, len),
					nothing,
					nothing,
					nothing,
					nothing
				))
			end
			num_count = get(count_info[-1], n, 0)
			count_info[-1][n] = num_count + cone_num[idx]
		elseif svd_method[idx] == PDSDP_GPU_JACOBI_SVD
			if !further_split_batch || cone_num[idx] <= target_batch_size
				gpu_id = argmin(workload) - 1
				stream_id = (batch_gpu_num + direct_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
				count_info[gpu_id] = get(count_info, gpu_id, Dict())
				proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
				CUDA.device!(gpu_id)
				batch_size = (Int(end_idx)-Int(start_idx)+1)÷(n * (n + 1) ÷ 2)
				push!(proj_task[gpu_id], ProjTask(
					PDSDP_INT(start_idx),
					PDSDP_INT(end_idx),
					PDSDP_INT(n),
					PDSDP_INT(batch_size),
					PDSDP_GPU_JACOBI_SVD,
					gpu_id,
					solver.device_info.stream_pool[gpu_id][stream_id],
					nothing,
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * len)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
					CUDA.CuEvent()
				))
				size_count = get(count_info[gpu_id], (batch_size, Int32(n)), 0)
				count_info[gpu_id][(batch_size, Int32(n))] = size_count + 1
				batch_gpu_num += 1
				workload[gpu_id + 1] += cone_num[idx] * n^3
			else 
				#find a number for futher split cone number
				# We want let each batch contain 100-200 matrices
				
				num_cones = cld(cone_num[idx], target_batch_size)
				for i in 1:num_cones
					gpu_id = argmin(workload) - 1
					stream_id = (batch_gpu_num + direct_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
					count_info[gpu_id] = get(count_info, gpu_id, Dict())
					proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
					CUDA.device!(gpu_id)
					batch_size = min(target_batch_size, cone_num[idx] - (i - 1) * target_batch_size)
					push!(proj_task[gpu_id], ProjTask(
						PDSDP_INT(start_idx + (i - 1) * target_batch_size * len),
						PDSDP_INT(min(start_idx + i * target_batch_size * len - 1, end_idx)),
						PDSDP_INT(n),
						PDSDP_INT(batch_size),
						PDSDP_GPU_JACOBI_SVD,
						gpu_id,
						solver.device_info.stream_pool[gpu_id][stream_id],
						nothing,
						CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * len)),
						CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
						CuArray{PDSDP_FLOAT, 1}(fill(0.0, batch_size * n * n)),
						CUDA.CuEvent()
					))
					size_count = get(count_info[gpu_id], (batch_size, Int32(n)), 0)
					count_info[gpu_id][(batch_size, Int32(n))] = size_count + 1
					batch_gpu_num += 1
					workload[gpu_id + 1] += batch_size * n^3
				end
			end



		elseif svd_method[idx] == PDSDP_GPU_DIRECT_SVD
			for i in 1:cone_num[idx]
				gpu_id = argmin(workload) - 1
				stream_id = (batch_gpu_num + direct_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
				count_info[gpu_id] = get(count_info, gpu_id, Dict())
				proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
				CUDA.device!(gpu_id)
				push!(proj_task[gpu_id], ProjTask(
					PDSDP_INT(start_idx + (i - 1) * len),
					PDSDP_INT(start_idx + i * len - 1),
					PDSDP_INT(n),
					PDSDP_INT(1),
					PDSDP_GPU_DIRECT_SVD,
					gpu_id,
					solver.device_info.stream_pool[gpu_id][stream_id],
					nothing,
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, len)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, n * n)),
					CuArray{PDSDP_FLOAT, 1}(fill(0.0, n * n)),
					CUDA.CuEvent()
				))
				direct_gpu_num += 1
				num_count = get(count_info[gpu_id], n, 0)
				count_info[gpu_id][n] = num_count + 1
				workload[gpu_id + 1] += n^3
			end
		elseif svd_method[idx] == PDSDP_SCALAR_PROJECTION
			gpu_id = 0
			proj_task[gpu_id] = get!(proj_task, gpu_id, Vector{ProjTask}())
			count_info[gpu_id] = get(count_info, gpu_id, Dict())
			
			stream_id = (batch_gpu_num + direct_gpu_num + scalar_gpu_num) % length(solver.device_info.stream_pool[gpu_id]) + 1
			batch_size = (Int(end_idx)-Int(start_idx)+1)
			push!(proj_task[gpu_id], ProjTask(
				PDSDP_INT(start_idx),
				PDSDP_INT(end_idx),
				PDSDP_INT(1),	
				PDSDP_INT(batch_size),
				PDSDP_SCALAR_PROJECTION,
				Int32(gpu_id),
				solver.device_info.stream_pool[gpu_id][stream_id],
				nothing,
				nothing,
				nothing,
				nothing,
				nothing
			))
			num_count = get(count_info[gpu_id], 1, 0)
			count_info[gpu_id][(batch_size, 1)] = num_count + 1
			scalar_gpu_num += 1
		else
			error("Unknown SVD method: $(svd_method[idx])")
		end		
	end
	CUDA.device!(0)
	solver.batch_cone_info.proj_task = proj_task
	print_proj_task_info(proj_task, count_info)
	if !check_independent_projection(solver)
		error("Projection tasks are not independent! Please check the problem data.")
	else
		println("Projection tasks are independent.")
	end
	return nothing
end


function check_independent_projection(solver::PdsdpSolver)
	flag_vector = fill(false, solver.problem.var_ptr[end - 1] - 1)
	for task in values(solver.batch_cone_info.proj_task)
		for t in task
			if any(flag_vector[t.start_idx:t.end_idx])
				return false
			else
				flag_vector[t.start_idx:t.end_idx] .= true
			end
		end
	end
	if !all(flag_vector)
		return false
	end
	return true
end


function unpack_low_tri_row_major_kernel!(
	tri_x::CuDeviceVector{PDSDP_FLOAT},
	full_x::CuDeviceVector{PDSDP_FLOAT},
	n::PDSDP_INT,
	batch_size::PDSDP_INT
)
	tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	tid > n * (n + 1) ÷ 2 * batch_size && return
	batch_id, k= divrem(tid - 1, n * (n + 1) ÷ 2)
	k += 1
	i = PDSDP_INT(ceil((sqrt(8.0 * k + 1) - 1) / 2))
	j = k - (i * (i - 1)) ÷ 2
	full_x_idx = (i - 1) * n + j
	full_x_offset = batch_id * n * n
	tri_x_offset = batch_id * n * (n + 1) ÷ 2
	val = tri_x[k + tri_x_offset]
	if i == j
		full_x[full_x_idx + full_x_offset] = val
	else
		full_x[full_x_idx + full_x_offset] = val / sqrt(2.0)
		full_x[(j - 1) * n + i + full_x_offset] = val / sqrt(2.0)
	end
	return
end

function pack_low_tri_row_major_kernel!(
	tri_x::CuDeviceVector{PDSDP_FLOAT},
	full_x::CuDeviceVector{PDSDP_FLOAT},
	n::PDSDP_INT,
	batch_size::PDSDP_INT
)
	tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	tid > n * (n + 1) ÷ 2 * batch_size && return
	batch_id, k= divrem(tid - 1, n * (n + 1) ÷ 2)
	k += 1
	i = PDSDP_INT(ceil((sqrt(8.0 * k + 1) - 1) / 2))
	j = k - (i * (i - 1)) ÷ 2
	full_x_idx = (i - 1) * n + j
	full_x_offset = batch_id * n * n
	tri_x_offset = batch_id * n * (n + 1) ÷ 2
	if i == j
		tri_x[k + tri_x_offset] = full_x[full_x_idx + full_x_offset]
	else
		tri_x[k + tri_x_offset]  = sqrt(2.0) * full_x[full_x_idx + full_x_offset]
	end
	return
end

function dispatch_projection!(
	x::PDSDP_GPU_VECTOR_FLOAT,
	solver::PdsdpSolver
)
	if isnothing(solver.batch_cone_info.proj_task)
		error("Projection tasks have not been prepared! Please call `prepare_task!` before projection.")
	end
	profile_projection = is_projection_profile_enabled()
	total_start = profile_projection ? time() : 0.0
	stage_unpack_start = profile_projection ? time() : 0.0
	stage_compute_start = 0.0
	stage_pack_start = 0.0
	cpu_d2h_time = 0.0
	cpu_proj_time = 0.0
	cpu_h2d_time = 0.0

	function sync_all_projection_devices!(proj_task)
		for dev_id in keys(proj_task)
			if dev_id == -1
				continue
			end
			CUDA.device!(dev_id)
			CUDA.synchronize()
		end
		CUDA.device!(0)
		return nothing
	end

	proj_task = solver.batch_cone_info.proj_task
	# all_proj_task = reduce(vcat, values(proj_task))
	# #sort it by method 
	# sort!(all_proj_task, by = t -> t.method)
	@sync begin
		@async begin
			if haskey(proj_task, -1)
				t0 = profile_projection ? time() : 0.0
				@threads for task in proj_task[-1]
					copyto!(task.buffer_cpu, view(x, task.start_idx:task.end_idx))
				end
				if profile_projection
					cpu_d2h_time += time() - t0
				end
			end
		end
		for dev in keys(proj_task)
			if dev == -1
				continue
			end
			@async begin
				CUDA.device!(dev) do
					for task in proj_task[dev]
						if task.method == PDSDP_SCALAR_PROJECTION
							continue
						end
						CUDA.stream!(task.stream) do
							length = task.end_idx - task.start_idx + 1
							copyto!(task.packed_buffer, view(x, task.start_idx:task.end_idx))
							blocks = cld(length, ThreadsPerBlock)
							@cuda threads=ThreadsPerBlock blocks=blocks unpack_low_tri_row_major_kernel!(task.packed_buffer, task.buffer_gpu, task.dim, task.batch_size)
						end
					end
					if !profile_projection
						CUDA.synchronize()
					end
				end
			end
		end
	end
	if profile_projection
		sync_all_projection_devices!(proj_task)
	end
	stage_unpack_time = profile_projection ? (time() - stage_unpack_start) : 0.0
	CUDA.device!(0)

	if profile_projection
		stage_compute_start = time()
	end
	@sync begin
		@async begin
			if haskey(proj_task, -1)
				t0 = profile_projection ? time() : 0.0
				@threads for task in proj_task[-1]
					projection_cpu_direct_entire!(task.buffer_cpu, task.dim)
				end
				if profile_projection
					cpu_proj_time += time() - t0
				end
			end
		end
		
		for dev_id in keys(proj_task)
			if dev_id == -1
				continue
			end
			@async begin
				CUDA.device!(dev_id) do
					for task in proj_task[dev_id]
						if task.method == PDSDP_SCALAR_PROJECTION
							continue
						end
						CUDA.stream!(task.stream) do
							if task.method == PDSDP_GPU_JACOBI_SVD
								projection_gpu_batched_entire!(task.buffer_gpu, task.extra_buffer, task.batch_size, task.dim)
							elseif task.method == PDSDP_GPU_DIRECT_SVD
								projection_gpu_direct_entire!(task.buffer_gpu, task.extra_buffer, task.dim)
							else
								error("Unknown projection method: $(task.method)")
							end
						end
					end
				end
			end
		end
	end
	if profile_projection
		sync_all_projection_devices!(proj_task)
	end
	stage_compute_time = profile_projection ? (time() - stage_compute_start) : 0.0

	if profile_projection
		stage_pack_start = time()
	end
	@sync begin
		@async begin
			if haskey(proj_task, -1)
				t0 = profile_projection ? time() : 0.0
				@threads for task in proj_task[-1]
					copyto!(view(x, task.start_idx:task.end_idx), task.buffer_cpu)
				end
				if profile_projection
					cpu_h2d_time += time() - t0
				end
			end
		end
		for dev in keys(proj_task)
			if dev == -1
				continue
			end
			@async begin
				CUDA.device!(dev) do
					for task in proj_task[dev]
						if task.method == PDSDP_SCALAR_PROJECTION
							CUDA.stream!(task.stream) do
								view(x, task.start_idx:task.end_idx) .= max.(view(x, task.start_idx:task.end_idx), 0.0)
							end
						elseif task.method != PDSDP_GPU_JACOBI_SVD
							CUDA.stream!(task.stream) do
								length = task.end_idx - task.start_idx + 1
								blocks = cld(length, ThreadsPerBlock)
								@cuda threads=ThreadsPerBlock blocks=blocks pack_low_tri_row_major_kernel!(task.packed_buffer, task.buffer_gpu, task.dim, Int32(1))
								copyto!(view(x, task.start_idx:task.end_idx), task.packed_buffer)
							end
						else
							CUDA.stream!(task.stream) do
								length = task.end_idx - task.start_idx + 1
								blocks = cld(length, ThreadsPerBlock)
								@cuda threads=ThreadsPerBlock blocks=blocks pack_low_tri_row_major_kernel!(task.packed_buffer, task.extra_buffer, task.dim, task.batch_size)
								copyto!(view(x, task.start_idx:task.end_idx), task.packed_buffer)
							end
						end
					end
					if !profile_projection
						CUDA.synchronize()
					end
				end
			end
		end
	end
	if profile_projection
		sync_all_projection_devices!(proj_task)
	end
	stage_pack_time = profile_projection ? (time() - stage_pack_start) : 0.0
	CUDA.device!(0)

	if profile_projection
		total_time = time() - total_start
		time_dict = solver.solver_info.time_dict
		time_dict["projection_profile_total_time"] = get(time_dict, "projection_profile_total_time", 0.0) + total_time
		time_dict["projection_profile_unpack_time"] = get(time_dict, "projection_profile_unpack_time", 0.0) + stage_unpack_time
		time_dict["projection_profile_compute_time"] = get(time_dict, "projection_profile_compute_time", 0.0) + stage_compute_time
		time_dict["projection_profile_pack_time"] = get(time_dict, "projection_profile_pack_time", 0.0) + stage_pack_time
		time_dict["projection_profile_cpu_d2h_time"] = get(time_dict, "projection_profile_cpu_d2h_time", 0.0) + cpu_d2h_time
		time_dict["projection_profile_cpu_compute_time"] = get(time_dict, "projection_profile_cpu_compute_time", 0.0) + cpu_proj_time
		time_dict["projection_profile_cpu_h2d_time"] = get(time_dict, "projection_profile_cpu_h2d_time", 0.0) + cpu_h2d_time
	end
	# CUDA.synchronize()
end

function warmup_projection!(
	solver::PdsdpSolver;
	runs::Int = 1,
	verbose::Bool = true
)
	if runs < 1
		return nothing
	end
	if isnothing(solver.batch_cone_info.proj_task)
		error("Projection tasks have not been prepared! Please call `prepare_task!` before warmup.")
	end
	warm_vec = similar(solver.state.current_primal_solution)
	fill!(warm_vec, 0.0)
	t0 = time()
	for _ in 1:runs
		dispatch_projection!(warm_vec, solver)
	end
	CUDA.synchronize()
	if verbose
		print_timing_banner(
			"Projection Warmup Completed",
			time() - t0,
			status_char="🔥",
		)
	end
	return nothing
end

function projection_gpu_batched!(
	x::CuArray{Float64, 1},
	start_idx::Int32,
	end_idx::Int32,
	dim::Int32;
	min_value::Float64 = 0.0)
	batch_size = PDSDP_INT((end_idx - start_idx + 1) ÷ (dim * dim))
	x_view = view(x, start_idx:end_idx)
	projection_gpu_batched_entire!(x_view, batch_size, dim; min_value=min_value)
	return nothing
end


function projection_cpu_direct!(
	x::CuArray{Float64, 1},
	start_idx::Int32,
	end_idx::Int32,
	dim::Int32;
	min_value::Float64 = 0.0)
	x_cpu = collect(view(x, start_idx:end_idx))

	projection_cpu_direct_entire!(x_cpu, dim)

	copyto!(view(x, start_idx:end_idx), vec(x_cpu))
	return
end


function projection_gpu_direct!(
	x_gpu::CuVector{PDSDP_FLOAT},
	start_idx::PDSDP_INT,
	end_idx::PDSDP_INT,
	dim::PDSDP_INT;
	min_value::PDSDP_FLOAT = 0.0)
	x_view = view(x_gpu, start_idx:end_idx)
	projection_gpu_direct_entire!(x_view, dim)
	return nothing
end

function projection_cpu_direct_entire!(
	x::Vector{Float64},
	dim::Int32;
	min_value::Float64 = 0.0)
	n = dim
	x = reshape(x, (n, n))
	F = eigen!(Symmetric(x))	
	@. F.values = max(F.values, min_value)
	mul!(x, F.vectors, Diagonal(F.values) * F.vectors')
	return
end


# function projection_gpu_direct_entire!(
# 	x::CuVector{PDSDP_FLOAT},
# 	dim::PDSDP_INT;
# 	min_value::PDSDP_FLOAT = 0.0)
# 	x = reshape(x, (dim, dim))
# 	Λ, Q = CUSOLVER.syevd!('V', 'U', x)
# 	Λ .= max.(Λ, min_value)
# 	CUDA.CUBLAS.gemm!('N', 'T', 1.0, Q * Diagonal(Λ), Q, 0.0, x)
# 	return nothing
# end
function projection_gpu_direct_entire!(
	x::CuVector{PDSDP_FLOAT},
	buffer::CuVector{PDSDP_FLOAT},
	dim::PDSDP_INT;
	min_value::PDSDP_FLOAT = 0.0)
	x = reshape(x, (dim, dim))
	buffer = reshape(buffer, (dim, dim))
	Λ, Q = CUSOLVER.syevd!('V', 'U', x)
	Λ .= max.(Λ, min_value)
	function row_scale!(out::CuDeviceMatrix{Float64}, Q::CuDeviceMatrix{Float64}, Λ::CuDeviceVector{Float64}, N::PDSDP_INT)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i > N && return
        for j in 1:size(Q,2)
            out[i, j] = Q[i, j] * Λ[j]
        end
        return
    end
    threads = ThreadsPerBlock
    blocks  = cld(dim, threads)
    @cuda threads=threads blocks=blocks row_scale!(buffer, Q, Λ, dim)
	CUDA.CUBLAS.gemm!('N', 'T', 1.0, buffer, Q, 0.0, x)
	return nothing
end
function projection_gpu_batched_entire!(
	x::CuArray{Float64, 1},
	buffer::CuArray{Float64, 1},
	batch_size::Int32,
	dim::Int32;
	min_value::Float64 = 0.0)
	X3d = reshape(x, (dim, dim, batch_size))
	Λ, U = CUSOLVER.syevjBatched!('V', 'U', X3d)
	batched_proj!(reshape(buffer, (dim, dim, batch_size)), Λ, U)
	return nothing
end

function batched_proj!(X3d::CuArray{Float64, 3}, Λ::CuArray{Float64, 2}, U::CuArray{Float64, 3};
                       min_value::Float64 = 0.0)
    n, _, cnt = size(X3d)

    function trunc_Λ(Λ)
        k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        k > size(Λ, 2) && return
        @inbounds for i in 1:size(Λ, 1)
            Λ[i, k] = max(Λ[i, k], min_value)
        end
        return
    end
    threads_Λ = ThreadsPerBlock
    blocks_Λ  = cld(cnt, threads_Λ)
    @cuda threads = threads_Λ blocks = blocks_Λ trunc_Λ(Λ)

    function kernel_ijk(X, Λ, U)
        k = blockIdx().z
        i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        (k > size(X, 3) || i > size(X, 1) || j > size(X, 2)) && return

        tmp = 0.0
        @inbounds for l in 1:size(U, 2)
            tmp += U[i, l, k] * Λ[l, k] * U[j, l, k]
        end
        @inbounds X[i, j, k] = tmp
        return
    end
    threads = (16, 16, 1)
    blocks  = (cld(n, 16), cld(n, 16), cnt)
    @cuda threads = threads blocks = blocks kernel_ijk(X3d, Λ, U)

    return X3d
end


function batched_proj_inplace!(Λ::CuArray{Float64, 2}, U::CuArray{Float64, 3}; min_value::Float64 = 0.0)
	n, _, cnt = size(U)
	function kernel_shm(Λ, U)
		k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		k > cnt && return
		shm = @cuDynamicSharedMem(Float64, n * blockDim().x)
		for i in 1:n
			Λ[i, k] = max(Λ[i, k], 0.0)
		end
		for i in 1:n
			for j in i:n
				# tmp = 0.0
				shm[(threadIdx().x-1)*n+j] = 0.0
				for l in 1:n
					# tmp += U[i, l, k] * Λ[l, k] * U[j, l, k]
					shm[(threadIdx().x-1)*n+j] += U[i, l, k] * Λ[l, k] * U[j, l, k]
				end
			end
			for j in i:n
				U[i, j, k] = shm[(threadIdx().x-1)*n+j]
			end
		end

		for i in 2:n
			for j in 1:i-1
				U[i, j, k] = U[j, i, k]
			end
		end

		return
	end

	threads = min(cnt, ThreadsPerBlock)
	blocks = cld(cnt, threads)
	shm_size = sizeof(Float64) * n * threads
	# if n >= 32
	# 	@cuda threads = threads blocks = blocks shmem = shm_size kernel_shm(Λ, U)
	# else
	# 	@cuda threads = threads blocks = blocks kernel_tmp(Λ, U, n, cnt)
	# end
	@cuda threads = threads blocks = blocks shmem = shm_size kernel_shm(Λ, U)

	return
end




function batched_proj_shared!(X3d::CuArray{Float64, 3},
	Λ::CuArray{Float64, 2},
	U::CuArray{Float64, 3})
	n, _, cnt = size(X3d)

	function kernel(X, Λ, U)
		k = blockIdx().x
		k > size(X, 3) && return

		shU = @cuDynamicSharedMem(Float64, (n, n))
		shλ = @cuDynamicSharedMem(Float64, n)

		tid = threadIdx().x
		for idx in tid:n*n:n*n
			i = (idx - 1) ÷ n + 1
			j = (idx - 1) % n + 1
			@inbounds shU[i, j] = U[i, j, k]
		end

		for i in tid:n
			@inbounds shλ[i] = max(Λ[i, k], 0.0)
		end
		sync_threads()

		i = (threadIdx().x - 1) ÷ n + 1
		j = (threadIdx().x - 1) % n + 1
		if i ≤ n && j ≤ n
			tmp = 0.0
			@inbounds for l in 1:n
				tmp += shU[i, l] * shλ[l] * shU[j, l]
			end
			@inbounds X[i, j, k] = tmp
		end
		return
	end

	threads = min(n * n, ThreadsPerBlock)
	blocks  = cnt
	shmem   = sizeof(Float64) * (n * n + n)

	@cuda threads = threads blocks = blocks shmem = shmem kernel(X3d, Λ, U)
	return X3d
end