using Pkg
Pkg.activate(".")
using CUDA
using Base.Threads
function compute(x)
    Λ, Q = CUSOLVER.syevd!('V', 'U', x)
    return Q
end
function run(x_list, device_id)
    results = [similar(x_list[i]) for i in 1:length(x_list)]

    # computation
    @sync begin
        # Threads.@spawn begin
        #     results[1] = Array(compute(a,b))
        #     nothing # JuliaLang/julia#40626
        # end
        # Threads.@spawn begin
        #     results[2] = Array(compute(a,b))
        #     nothing # JuliaLang/julia#40626
        # end
        for i in 1:length(x_list)
            Threads.@spawn begin
                CUDA.device!(device_id[i])
                    results[i] = compute(x_list[i])
                    nothing # JuliaLang/julia#40626
                end
        end
     
    end
    # println(results)

end


function main(N=4096)
    a = CUDA.rand(N, N)
    x = a * a'
    x_list = [x for i in 1:128]
    device_id = [1 for i in 1:Int(length(x_list)/length(CUDA.devices()))]
    for i in 1:length(CUDA.devices())
        append!(device_id, [i for j in 1:Int(length(x_list)/length(CUDA.devices()))])
    end
    # make sure this data can be used by other tasks!
    synchronize()

    run(x_list, device_id)
end

main()