using NPZ
using LinearAlgebra
using Base.Threads
using CUDA
using SparseArrays
using StatsBase
using CUDA.CUSOLVER
using CUDA.CUSPARSE
using Serialization
using Printf
using Dates
using Random
using StaticArrays
import Base: copy
using Plots
using DataStructures
using CSV
using DelimitedFiles
using MAT
# using DisjointSets
include("unrolled_kernel.jl")
include("const.jl")
include("struct_def.jl")
include("utils.jl")
include("projection.jl")
include("saddle_point.jl")
include("preprocess.jl")
include("load_problem.jl")
include("halpern_pdhg.jl")
include("scaling.jl")
include("pdhg_core.jl")
include("restart_eval.jl")
