//go:build amd64 && !purego

#include "textflag.h"

// Each kernel keeps four vector accumulators so consecutive FMAs do not wait
// on each other, reduces them in a fixed tree, then folds the scalar tail
// into lane 0.

// func dotAVX2(a, b []float32) float32
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), SI
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), DI
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	CMPQ CX, $32
	JL   dot_by8

dot_by32:
	VMOVUPS     (SI), Y4
	VMOVUPS     32(SI), Y5
	VMOVUPS     64(SI), Y6
	VMOVUPS     96(SI), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ        $128, SI
	ADDQ        $128, DI
	SUBQ        $32, CX
	CMPQ        CX, $32
	JGE         dot_by32

dot_by8:
	CMPQ        CX, $8
	JL          dot_reduce
	VMOVUPS     (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         dot_by8

dot_reduce:
	VADDPS       Y1, Y0, Y0
	VADDPS       Y3, Y2, Y2
	VADDPS       Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

dot_by1:
	TESTQ       CX, CX
	JE          dot_done
	VMOVSS      (SI), X4
	VFMADD231SS (DI), X4, X0
	ADDQ        $4, SI
	ADDQ        $4, DI
	DECQ        CX
	JMP         dot_by1

dot_done:
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET

// func dot64AVX2(a, b []float64) float64
TEXT ·dot64AVX2(SB), NOSPLIT, $0-56
	MOVQ   a_base+0(FP), SI
	MOVQ   a_len+8(FP), CX
	MOVQ   b_base+24(FP), DI
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	CMPQ   CX, $16
	JL     dot64_by4

dot64_by16:
	VMOVUPD     (SI), Y4
	VMOVUPD     32(SI), Y5
	VMOVUPD     64(SI), Y6
	VMOVUPD     96(SI), Y7
	VFMADD231PD (DI), Y4, Y0
	VFMADD231PD 32(DI), Y5, Y1
	VFMADD231PD 64(DI), Y6, Y2
	VFMADD231PD 96(DI), Y7, Y3
	ADDQ        $128, SI
	ADDQ        $128, DI
	SUBQ        $16, CX
	CMPQ        CX, $16
	JGE         dot64_by16

dot64_by4:
	CMPQ        CX, $4
	JL          dot64_reduce
	VMOVUPD     (SI), Y4
	VFMADD231PD (DI), Y4, Y0
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $4, CX
	JMP         dot64_by4

dot64_reduce:
	VADDPD       Y1, Y0, Y0
	VADDPD       Y3, Y2, Y2
	VADDPD       Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD       X1, X0, X0
	VHADDPD      X0, X0, X0

dot64_by1:
	TESTQ       CX, CX
	JE          dot64_done
	VMOVSD      (SI), X4
	VFMADD231SD (DI), X4, X0
	ADDQ        $8, SI
	ADDQ        $8, DI
	DECQ        CX
	JMP         dot64_by1

dot64_done:
	VZEROUPPER
	MOVSD X0, ret+48(FP)
	RET

// func squaredDistance64AVX2(a, b []float64) float64
TEXT ·squaredDistance64AVX2(SB), NOSPLIT, $0-56
	MOVQ   a_base+0(FP), SI
	MOVQ   a_len+8(FP), CX
	MOVQ   b_base+24(FP), DI
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	CMPQ   CX, $16
	JL     dist_by4

dist_by16:
	VMOVUPD     (SI), Y4
	VMOVUPD     32(SI), Y5
	VMOVUPD     64(SI), Y6
	VMOVUPD     96(SI), Y7
	VSUBPD      (DI), Y4, Y4
	VSUBPD      32(DI), Y5, Y5
	VSUBPD      64(DI), Y6, Y6
	VSUBPD      96(DI), Y7, Y7
	VFMADD231PD Y4, Y4, Y0
	VFMADD231PD Y5, Y5, Y1
	VFMADD231PD Y6, Y6, Y2
	VFMADD231PD Y7, Y7, Y3
	ADDQ        $128, SI
	ADDQ        $128, DI
	SUBQ        $16, CX
	CMPQ        CX, $16
	JGE         dist_by16

dist_by4:
	CMPQ        CX, $4
	JL          dist_reduce
	VMOVUPD     (SI), Y4
	VSUBPD      (DI), Y4, Y4
	VFMADD231PD Y4, Y4, Y0
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $4, CX
	JMP         dist_by4

dist_reduce:
	VADDPD       Y1, Y0, Y0
	VADDPD       Y3, Y2, Y2
	VADDPD       Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD       X1, X0, X0
	VHADDPD      X0, X0, X0

dist_by1:
	TESTQ       CX, CX
	JE          dist_done
	VMOVSD      (SI), X4
	VSUBSD      (DI), X4, X4
	VFMADD231SD X4, X4, X0
	ADDQ        $8, SI
	ADDQ        $8, DI
	DECQ        CX
	JMP         dist_by1

dist_done:
	VZEROUPPER
	MOVSD X0, ret+48(FP)
	RET
