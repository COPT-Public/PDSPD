include("src/PDSDP.jl")
problem = load_problem_from_spot_npz_to_cpu("kappa_2_n_22_moment_info.npz")

option = PdhgOption(
    tol=1e-6,
    max_iterations=500000,
    evaluate_frequency=40,
    restart_skip_iterations=500,
    rescaling = true,
    sufficient_reduction_threshold = 0.2,
    primal_weight_update_smoothing = 0.2,
    print_frequency =40,
    restart_scheme = PDSDP_HALPERN_RESTART,
    stepsize_scheme = PDSDP_PID_STEPSIZE, #PDSDP_rAPDHG_STEPSIZE, PDSDP_CUSTOM_STEPSIZE
    Kp = 0.0,
    Ki = 0.0,
    Kd = 0.0,
    save_trajectory = true
)

result_dict = pdhg_sdp(problem, option=option)
raw = result_dict["trajectory"]
using Plots
plotlyjs()  
primal_raw_idx = findall(x -> x[1] == 2.0, raw)
dual_raw_idx = findall(x -> x[1] == 3.0, raw)
eigs_raw_idx = findall(x -> x[1] == 4.0, raw)
primal_raw = raw[primal_raw_idx]
dual_raw = raw[dual_raw_idx]
eigs_raw = raw[eigs_raw_idx]
Primal = reduce(hcat, primal_raw)'   # (2160 × 4) 的矩阵，第 i 行就是第 i 步的 4 维点
Dual = reduce(hcat, dual_raw)'   # (2160 × 1) 的矩阵，第 i 行就是第 i 步的 2 维点
Eigs = reduce(hcat, eigs_raw)'   # (2160 × 2) 的矩阵，第 i 行就是第 i 步的 2 维点
# marker_color = X[:, 1]  # 用第一维来给点上色
x1 = Primal[:, 2]              # 第一维
x2 = Primal[:, 3]              # 第二维
x3 = Primal[:, 4]              # 第三维
x4 = Primal[:, 5]              # 第四维
y1 = Dual[:, 2]              # 第一维
eig1 = Eigs[:, 2]              # 第一维
eig2 = Eigs[:, 3]              # 第二维

# #----------------------------------------------------------

# cmap = cgrad([:red, :green, :blue])

# plt = plot(x, y, z,
#      linewidth  = 1,
#      linecolor  = :gray,        # 轨迹线
#      markersize = 2,
#      markeralpha = 0.5,
#      markercolor = cmap[Int.(marker_color) .+ 1],  # 关键：逐点颜色
#      marker     = :circle,
#      label      = "trajectory",
#      xlabel     = "x₁", ylabel = "x₂", zlabel = "x₃",
#      title      = "3-D trajectory")

# savefig(plt, "trajectory_3d.html")
# savefig(plt, "trajectory_3d.png")

#draw 2D graph (consider only marker_color = 0, 2)
# selected_indices = findall(x -> x in (0.0, 2.0), marker_color)
x_2d = x1
y_2d = y1
plt2d = plot(x_2d, y_2d,
     linewidth  = 1,
     label      = "trajectory",
     xlabel     = "x₁", ylabel = "x₂",
     title      = "2-D trajectory (only marker_color = 0, 2)")
savefig(plt2d, "trajectory_2d.png")
savefig(plt2d, "trajectory_2d.html")

