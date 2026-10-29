include("src/PDSDP.jl")
problem = generate_draw_graph_problem()

option = PdhgOption(
    tol=1e-6,
    max_iterations=500000,
    evaluate_frequency=40,
    restart_skip_iterations=500,
    rescaling = true,
    sufficient_reduction_threshold = 0.2,
    primal_weight_update_smoothing = 0.2,
    print_frequency =400,
    restart_scheme = PDSDP_HALPERN_RESTART,
    stepsize_scheme = PDSDP_rAPDHG_STEPSIZE, #PDSDP_rAPDHG_STEPSIZE, PDSDP_CUSTOM_STEPSIZE
    Kp = 0.00,
    Ki = 0.0,
    Kd = 0.2,
    save_trajectory = true
)

result_dict = pdhg_sdp(problem, option=option)
raw = result_dict["trajectory"]
using Plots
plotlyjs()  
X = reduce(hcat, raw)'   # (2160 × 4) 的矩阵，第 i 行就是第 i 步的 4 维点
marker_color = X[:, 1]  # 用第一维来给点上色
x = X[:, 2]              # 第一维
y = X[:, 3]              # 第二维
z = X[:, 5]              # 第三维（如果你想换维度，改这里即可）

#----------------------------------------------------------

cmap = cgrad([:red, :green, :blue])

plt = plot(x, y, z,
     linewidth  = 1,
     linecolor  = :gray,        # 轨迹线
     markersize = 2,
     markeralpha = 0.5,
     markercolor = cmap[Int.(marker_color) .+ 1],  # 关键：逐点颜色
     marker     = :circle,
     label      = "trajectory",
     xlabel     = "x₁", ylabel = "x₂", zlabel = "x₃",
     title      = "3-D trajectory")

savefig(plt, "trajectory_3d.html")
savefig(plt, "trajectory_3d.png")

#draw 2D graph (consider only marker_color = 0, 2)
selected_indices = findall(x -> x in (0.0, 2.0), marker_color)
x_2d = x[selected_indices]
y_2d = y[selected_indices]
marker_color_2d = marker_color[selected_indices]
plt2d = plot(x_2d, y_2d,
     linewidth  = 1,
     markersize = 1,
     markeralpha = 0.5,
     markercolor = cmap[Int.(marker_color_2d) .+ 1],  # 关键：逐点颜色
     marker     = :circle,
     label      = "trajectory",
     xlabel     = "x₁", ylabel = "x₂",
     title      = "2-D trajectory (only marker_color = 0, 2)")
savefig(plt2d, "trajectory_2d.png")
savefig(plt2d, "trajectory_2d.html")

