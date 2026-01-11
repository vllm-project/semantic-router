// func dotProductAVX2(a, b []float32) float32
TEXT ·dotProductAVX2(SB), $0-52
    MOVQ    a_base+0(FP), AX      // AX = &a[0]
    MOVQ    b_base+24(FP), BX     // BX = &b[0]
    MOVQ    a_len+8(FP), CX       // CX = len(a)

    // Initialize accumulator to zero
    VXORPS  Y0, Y0, Y0            // Y0 = accumulator (8x float32)

    // Calculate number of full 8-element chunks
    MOVQ    CX, DX
    SHRQ    $3, DX                // DX = len / 8
    JZ      remainder             // Jump if less than 8 elements

loop_avx2:
    // Load 8 float32s from a and b
    VMOVUPS (AX), Y1              // Y1 = a[i:i+8]
    VMOVUPS (BX), Y2              // Y2 = b[i:i+8]

    // Multiply and accumulate: Y0 += Y1 * Y2
    VFMADD231PS Y1, Y2, Y0        // Y0 = Y0 + (Y1 * Y2) [FMA instruction]

    // Advance pointers
    ADDQ    $32, AX               // AX += 32 bytes (8 * 4 bytes)
    ADDQ    $32, BX               // BX += 32 bytes

    DECQ    DX
    JNZ     loop_avx2

remainder:
    // Horizontal sum of Y0 (8 float32s -> 1 float32)
    VEXTRACTF128 $1, Y0, X1       // X1 = upper 4 elements of Y0
    VADDPS  X0, X1, X0            // X0 = sum of lower and upper halves
    VHADDPS X0, X0, X0            // Horizontal add (4->2)
    VHADDPS X0, X0, X0            // Horizontal add (2->1)

    // Handle remaining elements (scalar)
    MOVQ    CX, DX
    ANDQ    $7, DX                // DX = len % 8
    JZ      done

remainder_loop:
    VMOVSS  (AX), X1
    VMOVSS  (BX), X2
    VMULSS  X1, X2, X1
    VADDSS  X0, X1, X0

    ADDQ    $4, AX
    ADDQ    $4, BX
    DECQ    DX
    JNZ     remainder_loop

done:
    VMOVSS  X0, ret+48(FP)
    RET

// func dotProductAVX512(a, b []float32) float32
TEXT ·dotProductAVX512(SB), $0-52
    MOVQ    a_base+0(FP), AX      // AX = &a[0]
    MOVQ    b_base+24(FP), BX     // BX = &b[0]
    MOVQ    a_len+8(FP), CX       // CX = len(a)

    // Initialize accumulator to zero
    VXORPS  Z0, Z0, Z0            // Z0 = accumulator (16x float32)

    // Calculate number of full 16-element chunks
    MOVQ    CX, DX
    SHRQ    $4, DX                // DX = len / 16
    JZ      remainder512          // Jump if less than 16 elements

loop_avx512:
    // Load 16 float32s from a and b
    VMOVUPS (AX), Z1              // Z1 = a[i:i+16]
    VMOVUPS (BX), Z2              // Z2 = b[i:i+16]

    // Multiply and accumulate: Z0 += Z1 * Z2
    VFMADD231PS Z1, Z2, Z0        // Z0 = Z0 + (Z1 * Z2)

    // Advance pointers
    ADDQ    $64, AX               // AX += 64 bytes (16 * 4 bytes)
    ADDQ    $64, BX               // BX += 64 bytes

    DECQ    DX
    JNZ     loop_avx512

remainder512:
    // Horizontal sum of Z0 (16 float32s -> 1 float32)
    VEXTRACTF32X8 $1, Z0, Y1      // Y1 = upper 8 elements
    VADDPS  Y0, Y1, Y0            // Y0 = sum of lower and upper halves (8 elements)
    VEXTRACTF128 $1, Y0, X1       // X1 = upper 4 elements
    VADDPS  X0, X1, X0            // X0 = 4 elements
    VHADDPS X0, X0, X0            // 4->2
    VHADDPS X0, X0, X0            // 2->1

    // Handle remaining elements (scalar)
    MOVQ    CX, DX
    ANDQ    $15, DX               // DX = len % 16
    JZ      done512

remainder512_loop:
    VMOVSS  (AX), X1
    VMOVSS  (BX), X2
    VMULSS  X1, X2, X1
    VADDSS  X0, X1, X0

    ADDQ    $4, AX
    ADDQ    $4, BX
    DECQ    DX
    JNZ     remainder512_loop

done512:
    VMOVSS  X0, ret+48(FP)
    RET
