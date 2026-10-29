function preprocess_sdp!(CuSemidefiniteProgram; tol=1e-10)
    zero_rows = vec(all(CuSemidefiniteProgram.A .== 0, dims=2))

    qrA = qr(CuSemidefiniteProgram.A', ColumnNorm())   
    r   = count(i -> abs(qrA.R[i, i]) > tol, 1:size(qrA.R, 1))
    m   = size(CuSemidefiniteProgram.A, 1)
    redundant_rows = qrA.p[(r + 1):m]  

    drop_rows = unique!([findall(zero_rows); redundant_rows])
    keep_rows = setdiff(1:m, drop_rows)

    if !isempty(drop_rows)
        println("Removing $(length(drop_rows)) rows from A...")
        CuSemidefiniteProgram.A   = CuSemidefiniteProgram.A[keep_rows, :]
        CuSemidefiniteProgram.A_t = CuSemidefiniteProgram.A'
        CuSemidefiniteProgram.b   = CuSemidefiniteProgram.b[keep_rows]
        CuSemidefiniteProgram.m   = length(keep_rows)
        println("Final m = ", CuSemidefiniteProgram.m,
                ", rank = ", r,
                ", cond = ", cond(CuSemidefiniteProgram.A))
    end
end