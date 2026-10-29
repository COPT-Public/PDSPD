function scaling_cpu!(
    sdp::SemidefiniteProgram; 
    use_Ruiz_scaling::Bool = true, 
    use_Pock_Chambolle_scaling::Bool= true, 
    use_bc_scaling::Bool = true
    )
    m, n = sdp.m, sdp.n
    row_norm = ones(m)
    col_norm = ones(n)

    # Preserve the benchmark normalizers in original coordinates before any
    # Ruiz/Pock-Chambolle/b-c scaling mutates the problem.  PDSDP stores each
    # PSD block as an orthonormal svec (sqrt(2) on off-diagonals), whereas the
    # benchmark's ||C||_inf is the largest matrix entry.
    b_norm_inf_original = norm(sdp.b, Inf)
    C_norm_inf_original = 0.0
    for block in eachindex(sdp.tI_size)
        dim = Int(sdp.tI_size[block])
        cursor = Int(sdp.var_ptr[block])
        for row in 1:dim
            for col in 1:row
                value = abs(sdp.C[cursor])
                if row != col
                    value /= sqrt(2.0)
                end
                C_norm_inf_original = max(C_norm_inf_original, value)
                cursor += 1
            end
        end
    end
    cone_end = Int(sdp.var_ptr[length(sdp.tI_size) + 1])
    if cone_end <= length(sdp.C)
        C_norm_inf_original = max(
            C_norm_inf_original,
            maximum(abs, @view(sdp.C[cone_end:end]); init=0.0),
        )
    end

    # Preallocate temporary arrays
    temp_norm1 = zeros(m)
    temp_norm2 = zeros(n)
    DA = spdiagm(temp_norm1)
    EA = spdiagm(temp_norm2)
    # scaling_info = Scaling_info_cpu(copy(sdp.l), copy(sdp.u), row_norm, col_norm, 1, 1, 1, 1, 1 + norm(sdp.b), 1 + norm(sdp.C))
    scaling_info = ScalingInfoCpu(
        1,
        1,
        row_norm,
        col_norm,
        1.0,
        1.0,
        b_norm_inf_original,
        C_norm_inf_original,
    )

    # Ruiz scaling
    if use_Ruiz_scaling
        for _ in 1:20
            temp_norm1 .= sqrt.(maximum(abs, sdp.A, dims=2)[:, 1])

            temp_norm1[iszero.(temp_norm1)] .= 1.0
            row_norm .*= temp_norm1
            DA .= spdiagm(1.0 ./ temp_norm1)
            temp_norm2 .= sqrt.(maximum(abs, sdp.A, dims=1)[1, :])
            temp_norm2[iszero.(temp_norm2)] .= 1.0

            for i in 1:length(sdp.var_ptr)-2
                block_indices = sdp.var_ptr[i]:(sdp.var_ptr[i+1]-1)
                max_block_norm = maximum(temp_norm2[block_indices])
                if max_block_norm != 0.0
                    temp_norm2[block_indices] .= max_block_norm
                else
                    temp_norm2[block_indices] .= 1.0
                end
            end

            col_norm .*= temp_norm2
            EA .= spdiagm(1.0 ./ temp_norm2)
            sdp.b ./= temp_norm1
            sdp.A .= DA * sdp.A * EA
            sdp.C ./= temp_norm2
        end
    end

    # Pock-Chambolle scaling
    if use_Pock_Chambolle_scaling
        temp_norm1 .= sqrt.(sum(abs, sdp.A, dims=2)[:, 1])
        temp_norm1[iszero.(temp_norm1)] .= 1.0
        row_norm .*= temp_norm1
        DA .= spdiagm(1.0 ./ temp_norm1)
        temp_norm2 .= sqrt.(sum(abs, sdp.A, dims=1)[1, :])
        temp_norm2[iszero.(temp_norm2)] .= 1.0

        for i in 1:length(sdp.var_ptr)-2
            block_indices = sdp.var_ptr[i]:(sdp.var_ptr[i+1]-1)
            max_block_norm = maximum(temp_norm2[block_indices])
            if max_block_norm != 0.0
                temp_norm2[block_indices] .= max_block_norm
            else
                temp_norm2[block_indices] .= 1.0
            end
        end

        col_norm .*= temp_norm2
        EA .= spdiagm(1.0 ./ temp_norm2)
        sdp.b ./= temp_norm1
        sdp.A .= DA * sdp.A * EA
        sdp.C ./= temp_norm2
    end

    # scaling for b and c
    if use_bc_scaling
        b_scale = 1 + norm(sdp.b)
        C_scale = 1 + norm(sdp.C)
        sdp.b ./= b_scale
        sdp.C ./= C_scale
        scaling_info.b_scale = b_scale
        scaling_info.C_scale = C_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.C_scale = 1.0
    end

    scaling_info.norm_b = norm(sdp.b)
    scaling_info.norm_C = norm(sdp.C)
    sdp.A_t = sparse(transpose(sdp.A))
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    
    return scaling_info
end

function scaling_abs_one_T!(sdp::SemidefiniteProgram)
    println("Nonzeros in A before scaling: $(nnz(sdp.A))")
    A = sdp.A  
    b = sdp.b     
    Ab_T = sparse(transpose(hcat(A,b)))
    A = Ab_T
    m, n = size(A)
    rows = rowvals(A) 
    vals = nonzeros(A) 
    # for i in 1:m
    #     rng = A[i, :]     
    #     nnz_col_idx = findall(x -> x != 0.0, rng)
    #     length(nnz_col_idx) == -1 || continue
    #     cols = collect(nnz_col_idx)
    #     v    = collect(rng[nnz_col_idx])
    #     a = abs.(v)
    #     if (1.0 ∈ a) && count(==(0.5), a) == 2
    #         # 找出 j(±1) 和 k,h(±0.5)
            
    #         j_pos = findfirst(x -> abs(x) == 1.0, v)
    #         kh_pos = findall(x -> abs(x) == 0.5, v)
    #         j_col = cols[j_pos]
    #         k_col = cols[kh_pos[1]]
    #         h_col = cols[kh_pos[2]]
    #         j_val = v[j_pos]
    #         k_val = v[kh_pos[1]]
    #         h_val = v[kh_pos[2]]

    #         orig_i_j = A[i, j_col]
    #         orig_i_k = A[i, k_col]
    #         orig_i_h = A[i, h_col]

    #         A[:, j_col] .= 0.5 .* sign.(-j_val) .* A[:, j_col] .+
    #                        0.25 .* sign.(k_val) .* A[:, k_col] .+
    #                        0.25 .* sign.(h_val) .* A[:, h_col]

    #         A[i, j_col] = orig_i_j
    #         A[i, k_col] = orig_i_k
    #         A[i, h_col] = orig_i_h
    #     end
    # end
    for i in 1:m
        rng = A[i, :]   
        nnz_col_idx = findall(x -> x != 0.0, rng)
        length(nnz_col_idx) == 2 || continue
        cols = collect(nnz_col_idx)
        v    = collect(rng[nnz_col_idx])

        if Set(v) == Set([1.0, -1.0]) && Set(v) == Set([2.0, -2.0])
            j_col, k_col = cols
            orig_i_j = A[i, j_col]
            orig_i_k = A[i, k_col]

            mean_col = (A[:, j_col] .+ A[:, k_col]) ./ 2.0
            A[:, j_col] .= mean_col
            A[:, k_col] .= mean_col

            A[i, j_col] = orig_i_j
            A[i, k_col] = orig_i_k
        end

    end

    Ab = sparse(transpose(A))
    sdp.A = Ab[:,1:end-1]
    sdp.b = Ab[:,end]
    println("Nonzeros in A after scaling: $(nnz(sdp.A))")
end

# function scaling_abs_one2!(sdp::SemidefiniteProgram)
#     println("Nonzeros in A before scaling: $(nnz(sdp.A))")
#     A = sdp.A    
#     A_t = sdp.A_t   
#     A_t_colptr = A_t.colptr
#     m, n = sdp.m, sdp.n
#     row_nz_number = A_t_colptr[2:end] .- A_t_colptr[1:end-1]
#     two_ele_cols = findall(x -> x == 2, row_nz_number)
#     abs_two_ele_cols_one_A = []
#     for col in two_ele_cols
#         nnz_col = A_t.nzval[A_t_colptr[col]:(A_t_colptr[col+1]-1)]
#         if Set(nnz_col) == Set([1.0, -1.0])
#             push!(abs_two_ele_cols_one_A, (col, A_t.rowval[A_t_colptr[col]], A_t.rowval[A_t_colptr[col]+1], nnz_col[1], nnz_col[2]))
#         end
#     end
#     ds = IntDisjointSets(n)
#     for item in abs_two_ele_cols_one_A
#         row, col1, col2, val1, val2 = item
#         union!(ds, col1, col2)
#     end
#     groups = Dict{Int, Vector{Int}}()
#     for (i, item) in enumerate(abs_two_ele_cols_one_A)
#         row, col1, col2, val1, val2 = item
#         root = find_root!(ds, col1)
#         push!(get!(groups, root, Int[]), i)
#     end
#     # println(num_groups(ds))
#     groups_values = collect(values(groups))
#     multi_threads_result = Vector{Matrix{Float64}}(undef, length(groups_values))
#     groups_cols = [unique(vcat(map(idx -> [abs_two_ele_cols_one_A[idx][2], abs_two_ele_cols_one_A[idx][3]], g)...))
#                for g in groups_values]
    
#     groups_buffer = [zeros(m, length(cols)) for cols in groups_cols]
#     @threads for group_idx in 1:length(groups_values)
#         A_sub = copy(A[:, groups_cols[group_idx]])
#         groups_cols_idx = Dict(col => i for (i, col) in enumerate(groups_cols[group_idx]))
#         for idx in groups_values[group_idx]
            
#     groups_values = collect(values(groups))
#     multi_threads_result = Vector{Matrix{Float64}}(undef, length(groups_values))
#     groups_cols = [unique(vcat(map(idx -> [abs_two_ele_cols_one_A[idx][2], abs_two_ele_cols_one_A[idx][3]], g)...))
#                for g in groups_values]
    
#     groups_buffer = [zeros(m, length(cols)) for cols in groups_cols]
#     @threads for group_idx in 1:length(groups_values)
#         A_sub = copy(A[:, groups_cols[group_idx]])
#         groups_cols_idx = Dict(col => i for (i, col) in enumerate(groups_cols[group_idx]))
#         for idx in groups_values[group_idx]
            
#             row, col1, col2, val1, val2 = abs_two_ele_cols_one_A[idx]
#             mean_col = (A[:, col1] .+ A[:, col2]) ./ 2.0
#             # mean_col_change[:, idx] = mean_col
#             # A[:, col1] .= mean_col
#             # A[:, col2] .= mean_col
#             A_sub[:, groups_cols_idx[col1]] .= mean_col
#             A_sub[:, groups_cols_idx[col2]] .= mean_col
#             A_sub[row, groups_cols_idx[col1]] = val1
#             A_sub[row, groups_cols_idx[col2]] = val2
#         end
#         multi_threads_result[group_idx] = copy(A_sub)
#     end
#     for group_idx in 1:length(groups_values)
#         A[:, groups_cols[group_idx]] .= multi_threads_result[group_idx]
#     end

#     # for item in abs_two_ele_cols_one_A
#     #     row, col1, col2, val1, val2 = item
#     #     row_idx_1 = A.rowval[A.colptr[col1]:(A.colptr[col1+1]-1)]
#     #     row_idx_2 = A.rowval[A.colptr[col2]:(A.colptr[col2+1]-1)]
#     #     val_idx_1 = A.nzval[A.colptr[col1]:(A.colptr[col1+1]-1)]
#     #     val_idx_2 = A.nzval[A.colptr[col2]:(A.colptr[col2+1]-1)]
#     #     idx_1_union_2 = union(row_idx_1, row_idx_2)
#     #     var_union = Vector{Float64}(undef, length(idx_1_union_2))
#     #     view(var_union, row_idx_1) .= val_idx_1 ./ 2.0
#     #     view(var_union, row_idx_2) .= val_idx_2 ./ 2.
#     #     row_pos = findfirst(==row, idx_1_union_2)
#     #     new_val_1 = 
#     # rows = rowvals(A) 
#     # vals = nonzeros(A) 
#     # for i in 1:m
#     #     rng = A[i, :]   
#     #     nnz_col_idx = findall(x -> x != 0.0, rng)
#     #     length(nnz_col_idx) == 2 || continue
#     #     cols = collect(nnz_col_idx)
#     #     v    = collect(rng[nnz_col_idx])

#     #     if Set(v) == Set([1.0, -1.0])
#     #         j_col, k_col = cols
#     #         orig_i_j = A[i, j_col]
#     #         orig_i_k = A[i, k_col]

#     #         mean_col = (A[:, j_col] .+ A[:, k_col]) ./ 2.0
#     #         A[:, j_col] .= mean_col
#     #         A[:, k_col] .= mean_col

#     #         A[i, j_col] = orig_i_j
#     #         A[i, k_col] = orig_i_k
#     #     end
#     # end
#     println("Nonzeros in A after scaling: $(nnz(sdp.A))")
# end

# function scaling_abs_one_parallel!(sdp::SemidefiniteProgram)
#     println("Nonzeros in A before scaling: $(nnz(sdp.A))")
#     A = sdp.A    
#     A_t = sdp.A_t   
#     A_t_colptr = A_t.colptr
#     m, n = sdp.m, sdp.n
#     row_nz_number = A_t_colptr[2:end] .- A_t_colptr[1:end-1]
#     two_ele_cols = findall(x -> x == 2, row_nz_number)
#     abs_two_ele_cols_one_A = []
#     for col in two_ele_cols
#         nnz_col = A_t.nzval[A_t_colptr[col]:(A_t_colptr[col+1]-1)]
#         if Set(nnz_col) == Set([1.0, -1.0])
#             push!(abs_two_ele_cols_one_A, (col, A_t.rowval[A_t_colptr[col]], A_t.rowval[A_t_colptr[col]+1], nnz_col[1], nnz_col[2]))
#         end
#     end
#     ds = IntDisjointSets(n)
#     for item in abs_two_ele_cols_one_A
#         row, col1, col2, val1, val2 = item
#         union!(ds, col1, col2)
#     end
#     groups = Dict{Int, Vector{Int}}()
#     for (i, item) in enumerate(abs_two_ele_cols_one_A)
#         row, col1, col2, val1, val2 = item
#         root = find_root!(ds, col1)
#         push!(get!(groups, root, Int[]), i)
#     end
#     # println(num_groups(ds))
#     groups_values = collect(values(groups))
#     multi_threads_result = Vector{Matrix{Float64}}(undef, length(groups_values))
#     groups_cols = [unique(vcat(map(idx -> [abs_two_ele_cols_one_A[idx][2], abs_two_ele_cols_one_A[idx][3]], g)...))
#                for g in groups_values]
    
#     groups_buffer = [zeros(m, length(cols)) for cols in groups_cols]
#     @threads for group_idx in 1:length(groups_values)
#         A_sub = copy(A[:, groups_cols[group_idx]])
#         groups_cols_idx = Dict(col => i for (i, col) in enumerate(groups_cols[group_idx]))
#         for idx in groups_values[group_idx]
            
#             row, col1, col2, val1, val2 = abs_two_ele_cols_one_A[idx]
#             mean_col = (A[:, col1] .+ A[:, col2]) ./ 2.0
#             # mean_col_change[:, idx] = mean_col
#             # A[:, col1] .= mean_col
#             # A[:, col2] .= mean_col
#             A_sub[:, groups_cols_idx[col1]] .= mean_col
#             A_sub[:, groups_cols_idx[col2]] .= mean_col
#             A_sub[row, groups_cols_idx[col1]] = val1
#             A_sub[row, groups_cols_idx[col2]] = val2
#         end
#         multi_threads_result[group_idx] = copy(A_sub)
#     end
#     for group_idx in 1:length(groups_values)
#         A[:, groups_cols[group_idx]] .= multi_threads_result[group_idx]
#     end
#     println("Nonzeros in A after scaling: $(nnz(sdp.A))")
# end

function scaling_chain!(sdp::SemidefiniteProgram)
    println("Nonzeros in A before scaling: $(nnz(sdp.A))")
    A = sdp.A
    m, n = sdp.m, sdp.n
    edges = []
    for i in 1:m
        rng = A[i, :]
        nnz_col_idx = findall(x -> x != 0.0, rng)
        length(nnz_col_idx) == 2 || continue
        cols = collect(nnz_col_idx)
        v    = collect(rng[nnz_col_idx])
        if Set(v) == Set([1.0, -1.0])
            col_neg = findfirst(==( -1.0), v)
            col_pos = findfirst(==(  1.0), v)
            push!(edges, (i, cols[col_neg], cols[col_pos]))
        end
    end

    graph = Dict{Int, Vector{Int}}()
    for (row, col_neg, col_pos) in edges
        if haskey(graph, col_neg)
            push!(graph[col_neg], col_pos)
        else
            graph[col_neg] = [col_pos]
        end
        if haskey(graph, col_pos)
            push!(graph[col_pos], col_neg)
        else
            graph[col_pos] = [col_neg]
        end
    end
    # println(graph)
    visited = falses(n)
    components = []
    for node in keys(graph)
        
        if !visited[node]
            stack = [node]
            component = []
            while !isempty(stack)
                current = pop!(stack)
                if !visited[current]
                    push!(component, current)
                    visited[current] = true
                    for neighbor in graph[current]
                        if !visited[neighbor]
                            push!(stack, neighbor)
                        end
                    end
                end
            end
            push!(components, component)
        end
    end

    for cols in components
        sort!(cols)
        len_cols = length(cols)
        if len_cols < 2
            continue
        end

        block = copy(sdp.A[:, cols])
        sum_cols = sum(block, dims=2)

        for idx in eachindex(cols)
            col = cols[idx]
            sdp.A[:, col] .= block[:, idx] / 2 + sum_cols[:, 1] / (2 * len_cols)
        end
    end
    println("Nonzeros in A after scaling: $(nnz(sdp.A))")
end
