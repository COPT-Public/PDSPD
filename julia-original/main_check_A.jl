using Pkg
using Plots
Pkg.activate(".")
include("src/PDSDP.jl")
problem = load_problem_from_spot_npz_to_cpu("kappa_2_n_3_moment_info.npz")
println(problem.tI_size)
println("m = ", problem.m)
println("n = ", problem.n)
#get non zero elements in A
last_var_start = problem.var_ptr[1]
last_var_end = problem.var_ptr[end] - 1
nnz = problem.A.nzval
row_idx = problem.A.rowval
col_ptr = problem.A.colptr
diff_col = diff(col_ptr)
col_idx = Int[]
for col in 1:length(col_ptr)-1
    for _ in 1:(col_ptr[col+1] - col_ptr[col]) 
        push!(col_idx, col)
    end
end
println("Total non-zero elements in A: ", length(nnz))
# vals  = nnz[last_var_start:last_var_end]
# rows  = row_idx[last_var_start:last_var_end]  # convert to 1-based indexing
# cols  = col_idx[last_var_start:last_var_end]  # convert to 1-based indexing 
mask = (col_idx .>= last_var_start) .& (col_idx .<= last_var_end)
vals = nnz[mask]
rows = row_idx[mask] 
cols = col_idx[mask]


m = maximum(rows)
n = maximum(cols) 


min_row = minimum(rows)
max_row = maximum(rows)
min_col = minimum(cols) 
max_col = maximum(cols) 
A_dense = fill(NaN, max_row - min_row + 1, max_col - min_col + 1)
# 4. 填充数值
for (k, (i, j, v)) in enumerate(zip(rows, cols, vals))
    A_dense[i - min_row + 1, j - min_col + 1] = v
end
println("FILLING DONE")
println("Row range: $min_row to $max_row")
println("Col range: $min_col to $max_col")
cmap = cgrad(:viridis, rev=false)  
zmin, zmax = -1.0, 1.0

heatmap(A_dense,
        clims=(zmin, zmax), 
        c=cmap,
        colorbar_title="value",
        title="Sparse slice heatmap")

plot!(seriestype=:heatmap, background_color=:white, foreground_color_border=:white)

savefig("A_slice_heatmap.svg")
# savefig("A_slice_heatmap.html")
