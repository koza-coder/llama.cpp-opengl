// Shared helpers for the ggml OpenGL backend kernels, ported from the D3D11 backend's common.hlsli.
//
// The host prepends "#version 430", WG_SIZE and the variant defines. Every tensor is a uint[] SSBO bound
// at an aligned offset at or before the tensor; element offsets/strides in the Params block are in
// elements of the buffer's type. The Params block is std140 with only scalar members, so it has the
// same layout as the D3D11 cbuffer: one 32-bit word per member, in order.
// Runtime values are plain locals, not const: GLSL 4.30 compilers may reject const with a
// non-constant initializer.

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

// flat thread index over a 2D dispatch grid: nwg_x groups along x
uint flat_index(uvec3 gid, uint nwg_x) {
    return gid.x + nwg_x * uint(WG_SIZE) * gid.y;
}

// typed load/store on uint[] buffers; indices are element indices
#define LOAD_F32(buf, i)     uintBitsToFloat(buf[(i)])
#define STORE_F32(buf, i, v) buf[(i)] = floatBitsToUint(v)
#define LOAD_I32(buf, i)     int(buf[(i)])
#define STORE_I32(buf, i, v) buf[(i)] = uint(v)

// f32 -> f16 with round-to-nearest-even in software: packHalf2x16 does not define its rounding
uint f32_to_f16_rne(float f) {
    uint x    = floatBitsToUint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint ax   = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {                       // inf, nan
        return sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u);
    }
    if (ax >= 0x477ff000u) {                       // >= 65520 rounds to inf
        return sign | 0x7c00u;
    }
    if (ax < 0x38800000u) {                        // below 2^-14: f16 subnormal or zero
        uint sh = 126u - (ax >> 23);               // shift of the 24-bit mantissa, >= 14
        if (sh > 24u) {
            return sign;
        }
        uint m    = (ax & 0x7fffffu) | 0x800000u;
        uint mant = m >> sh;
        uint rem  = m & ((1u << sh) - 1u);
        uint hlf  = 1u << (sh - 1u);
        if (rem > hlf || (rem == hlf && (mant & 1u) != 0u)) {
            mant++;
        }
        return sign | mant;
    }
    uint mant = (ax & 0x7fffffu) >> 13;
    uint rem  = ax & 0x1fffu;
    uint h    = (((ax >> 23) - 112u) << 10) | mant;   // a mantissa carry rolls into the exponent
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u) != 0u)) {
        h++;
    }
    return sign | h;
}

float f16_to_f32(uint bits) {
    return unpackHalf2x16(bits & 0xFFFFu).x;
}

// no 16-bit storage: read the word, and write a half with two atomics so the other half is kept.
// v is converted before the atomicAnd: in place, v can read the very half that the atomicAnd clears.
#define LOAD_F16(buf, i) f16_to_f32(buf[(i) >> 1] >> (((i) & 1u) * 16u))
#define STORE_F16(buf, i, v) { \
    uint _h  = f32_to_f16_rne(v); \
    uint _sh = ((i) & 1u) * 16u; \
    atomicAnd(buf[(i) >> 1], ~(0xFFFFu << _sh)); \
    atomicOr(buf[(i) >> 1], _h << _sh); \
}

// unaligned loads by byte address. Every shift stays below 32 (a shift by 32 is undefined), the high
// half is dropped with a mask instead - see the note in the D3D11 common.hlsli.
#define LOAD_U32_UNALIGNED(buf, addr, res) { \
    uint _a0 = (addr) >> 2; \
    uint _s  = ((addr) & 3u) * 8u; \
    uint _lo = buf[_a0]; \
    uint _hi = buf[_a0 + 1u]; \
    res = (_lo >> _s) | ((_hi << ((32u - _s) & 31u)) & (_s == 0u ? 0u : 0xFFFFFFFFu)); \
}
#define LOAD_U16_UNALIGNED(buf, addr, res) { \
    res = (buf[(addr) >> 2] >> (((addr) & 2u) * 8u)) & 0xFFFFu; \
}
