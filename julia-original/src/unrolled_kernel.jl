function kernel_tmp(Λ, U, n, cnt)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k > cnt && return
    shm1, shm2, shm3, shm4, shm5, shm6, shm7, shm8, shm9, shm10, shm11, shm12, shm13, shm14, shm15, shm16,
    shm17, shm18, shm19, shm20, shm21, shm22, shm23, shm24, shm25, shm26, shm27, shm28, shm29, shm30, shm31, shm32 = @SVector zeros(Float64, 32)
    for i in 1:n
        Λ[i, k] = max(Λ[i, k], 0.0)
    end
    for i in 1:n

       if 1 <= n - i + 1; shm1 = 0.0; end
       if 2 <= n - i + 1; shm2 = 0.0; end
       if 3 <= n - i + 1; shm3 = 0.0; end
       if 4 <= n - i + 1; shm4 = 0.0; end
       if 5 <= n - i + 1; shm5 = 0.0; end
       if 6 <= n - i + 1; shm6 = 0.0; end
       if 7 <= n - i + 1; shm7 = 0.0; end
       if 8 <= n - i + 1; shm8 = 0.0; end
       if 9 <= n - i + 1; shm9 = 0.0; end
       if 10 <= n - i + 1; shm10 = 0.0; end
       if 11 <= n - i + 1; shm11 = 0.0; end
       if 12 <= n - i + 1; shm12 = 0.0; end
       if 13 <= n - i + 1; shm13 = 0.0; end
       if 14 <= n - i + 1; shm14 = 0.0; end
       if 15 <= n - i + 1; shm15 = 0.0; end
       if 16 <= n - i + 1; shm16 = 0.0; end
       if 17 <= n - i + 1; shm17 = 0.0; end
       if 18 <= n - i + 1; shm18 = 0.0; end
       if 19 <= n - i + 1; shm19 = 0.0; end
       if 20 <= n - i + 1; shm20 = 0.0; end
       if 21 <= n - i + 1; shm21 = 0.0; end
       if 22 <= n - i + 1; shm22 = 0.0; end
       if 23 <= n - i + 1; shm23 = 0.0; end
       if 24 <= n - i + 1; shm24 = 0.0; end
       if 25 <= n - i + 1; shm25 = 0.0; end
       if 26 <= n - i + 1; shm26 = 0.0; end
       if 27 <= n - i + 1; shm27 = 0.0; end
       if 28 <= n - i + 1; shm28 = 0.0; end
       if 29 <= n - i + 1; shm29 = 0.0; end
       if 30 <= n - i + 1; shm30 = 0.0; end
       if 31 <= n - i + 1; shm31 = 0.0; end
       if 32 <= n - i + 1; shm32 = 0.0; end
       for l in 1:n
           if 1 <= n - i + 1; shm1 += U[i, l, k] * Λ[l, k] * U[1 + i - 1, l, k]; end
           if 2 <= n - i + 1; shm2 += U[i, l, k] * Λ[l, k] * U[2 + i - 1, l, k]; end
           if 3 <= n - i + 1; shm3 += U[i, l, k] * Λ[l, k] * U[3 + i - 1, l, k]; end
           if 4 <= n - i + 1; shm4 += U[i, l, k] * Λ[l, k] * U[4 + i - 1, l, k]; end
           if 5 <= n - i + 1; shm5 += U[i, l, k] * Λ[l, k] * U[5 + i - 1, l, k]; end
           if 6 <= n - i + 1; shm6 += U[i, l, k] * Λ[l, k] * U[6 + i - 1, l, k]; end
           if 7 <= n - i + 1; shm7 += U[i, l, k] * Λ[l, k] * U[7 + i - 1, l, k]; end
           if 8 <= n - i + 1; shm8 += U[i, l, k] * Λ[l, k] * U[8 + i - 1, l, k]; end
           if 9 <= n - i + 1; shm9 += U[i, l, k] * Λ[l, k] * U[9 + i - 1, l, k]; end
           if 10 <= n - i + 1; shm10 += U[i, l, k] * Λ[l, k] * U[10 + i - 1, l, k]; end
           if 11 <= n - i + 1; shm11 += U[i, l, k] * Λ[l, k] * U[11 + i - 1, l, k]; end
           if 12 <= n - i + 1; shm12 += U[i, l, k] * Λ[l, k] * U[12 + i - 1, l, k]; end
           if 13 <= n - i + 1; shm13 += U[i, l, k] * Λ[l, k] * U[13 + i - 1, l, k]; end
           if 14 <= n - i + 1; shm14 += U[i, l, k] * Λ[l, k] * U[14 + i - 1, l, k]; end
           if 15 <= n - i + 1; shm15 += U[i, l, k] * Λ[l, k] * U[15 + i - 1, l, k]; end
           if 16 <= n - i + 1; shm16 += U[i, l, k] * Λ[l, k] * U[16 + i - 1, l, k]; end
           if 17 <= n - i + 1; shm17 += U[i, l, k] * Λ[l, k] * U[17 + i - 1, l, k]; end
           if 18 <= n - i + 1; shm18 += U[i, l, k] * Λ[l, k] * U[18 + i - 1, l, k]; end
           if 19 <= n - i + 1; shm19 += U[i, l, k] * Λ[l, k] * U[19 + i - 1, l, k]; end
           if 20 <= n - i + 1; shm20 += U[i, l, k] * Λ[l, k] * U[20 + i - 1, l, k]; end
           if 21 <= n - i + 1; shm21 += U[i, l, k] * Λ[l, k] * U[21 + i - 1, l, k]; end
           if 22 <= n - i + 1; shm22 += U[i, l, k] * Λ[l, k] * U[22 + i - 1, l, k]; end
           if 23 <= n - i + 1; shm23 += U[i, l, k] * Λ[l, k] * U[23 + i - 1, l, k]; end
           if 24 <= n - i + 1; shm24 += U[i, l, k] * Λ[l, k] * U[24 + i - 1, l, k]; end
           if 25 <= n - i + 1; shm25 += U[i, l, k] * Λ[l, k] * U[25 + i - 1, l, k]; end
           if 26 <= n - i + 1; shm26 += U[i, l, k] * Λ[l, k] * U[26 + i - 1, l, k]; end
           if 27 <= n - i + 1; shm27 += U[i, l, k] * Λ[l, k] * U[27 + i - 1, l, k]; end
           if 28 <= n - i + 1; shm28 += U[i, l, k] * Λ[l, k] * U[28 + i - 1, l, k]; end
           if 29 <= n - i + 1; shm29 += U[i, l, k] * Λ[l, k] * U[29 + i - 1, l, k]; end
           if 30 <= n - i + 1; shm30 += U[i, l, k] * Λ[l, k] * U[30 + i - 1, l, k]; end
           if 31 <= n - i + 1; shm31 += U[i, l, k] * Λ[l, k] * U[31 + i - 1, l, k]; end
           if 32 <= n - i + 1; shm32 += U[i, l, k] * Λ[l, k] * U[32 + i - 1, l, k]; end
       end

        if 1 <= n - i + 1; U[i, 1 + i - 1, k] = shm1; end
        if 2 <= n - i + 1; U[i, 2 + i - 1, k] = shm2; end
        if 3 <= n - i + 1; U[i, 3 + i - 1, k] = shm3; end
        if 4 <= n - i + 1; U[i, 4 + i - 1, k] = shm4; end
        if 5 <= n - i + 1; U[i, 5 + i - 1, k] = shm5; end
        if 6 <= n - i + 1; U[i, 6 + i - 1, k] = shm6; end
        if 7 <= n - i + 1; U[i, 7 + i - 1, k] = shm7; end
        if 8 <= n - i + 1; U[i, 8 + i - 1, k] = shm8; end
        if 9 <= n - i + 1; U[i, 9 + i - 1, k] = shm9; end
        if 10 <= n - i + 1; U[i, 10 + i - 1, k] = shm10; end
        if 11 <= n - i + 1; U[i, 11 + i - 1, k] = shm11; end
        if 12 <= n - i + 1; U[i, 12 + i - 1, k] = shm12; end
        if 13 <= n - i + 1; U[i, 13 + i - 1, k] = shm13; end
        if 14 <= n - i + 1; U[i, 14 + i - 1, k] = shm14; end
        if 15 <= n - i + 1; U[i, 15 + i - 1, k] = shm15; end
        if 16 <= n - i + 1; U[i, 16 + i - 1, k] = shm16; end
        if 17 <= n - i + 1; U[i, 17 + i - 1, k] = shm17; end
        if 18 <= n - i + 1; U[i, 18 + i - 1, k] = shm18; end
        if 19 <= n - i + 1; U[i, 19 + i - 1, k] = shm19; end
        if 20 <= n - i + 1; U[i, 20 + i - 1, k] = shm20; end
        if 21 <= n - i + 1; U[i, 21 + i - 1, k] = shm21; end
        if 22 <= n - i + 1; U[i, 22 + i - 1, k] = shm22; end
        if 23 <= n - i + 1; U[i, 23 + i - 1, k] = shm23; end
        if 24 <= n - i + 1; U[i, 24 + i - 1, k] = shm24; end
        if 25 <= n - i + 1; U[i, 25 + i - 1, k] = shm25; end
        if 26 <= n - i + 1; U[i, 26 + i - 1, k] = shm26; end
        if 27 <= n - i + 1; U[i, 27 + i - 1, k] = shm27; end
        if 28 <= n - i + 1; U[i, 28 + i - 1, k] = shm28; end
        if 29 <= n - i + 1; U[i, 29 + i - 1, k] = shm29; end
        if 30 <= n - i + 1; U[i, 30 + i - 1, k] = shm30; end
        if 31 <= n - i + 1; U[i, 31 + i - 1, k] = shm31; end
        if 32 <= n - i + 1; U[i, 32 + i - 1, k] = shm32; end
    end

    for i in 2:n
        for j in 1:i-1
            U[i, j, k] = U[j, i, k]
        end
    end

    return
end
