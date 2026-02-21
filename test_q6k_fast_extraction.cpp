#include <cstdio>
#include <cstdint>
#include <cassert>

// Reference extraction
void extract_ref(const uint8_t* ql, const uint8_t* qh, int base, float* qs) {
    for (int k=0; k<4; k++) {
        int i = base + k;
        int half = i / 128;
        int rem = i % 128;
        int col = rem / 32;
        int l = rem % 32;
        
        int ql_idx = 64 * half;
        int qh_idx = 32 * half + l;
        int ql_shift = 0;
        int qh_shift = 0;

        if (col == 0) {
            ql_idx += l;
        } else if (col == 1) {
            ql_idx += l + 32;
            qh_shift = 2;
        } else if (col == 2) {
            ql_idx += l;
            ql_shift = 4;
            qh_shift = 4;
        } else if (col == 3) {
            ql_idx += l + 32;
            ql_shift = 4;
            qh_shift = 6;
        }

        int lo = (ql[ql_idx] >> ql_shift) & 0xF;
        int hi = (qh[qh_idx] >> qh_shift) & 0x3;
        qs[k] = static_cast<float>((lo | (hi << 4)) - 32);
    }
}

// Fast extraction
void extract_fast(const uint8_t* blk_ql, const uint8_t* blk_qh, int base, float* qs) {
    const int half = base / 128;
    const int col  = (base % 128) / 32; // 0..3
    const int l    = (base % 32);       // 0..28

    const uint8_t* ql_target = blk_ql + 64 * half + l + ((col == 1 || col == 3) ? 32 : 0);
    const uint8_t* qh_target = blk_qh + 32 * half + l;
    
    // We can't always load 32 bits if l > 28!
    // But since l % 4 == 0, base % 4 == 0. l can be 0, 4, 8, ..., 28.
    // If l == 28, l + 3 is 31. Loading 4 bytes reads 28, 29, 30, 31. That's within bounds!
    
    uint32_t ql4 = *reinterpret_cast<const uint32_t*>(ql_target);
    uint32_t qh4 = *reinterpret_cast<const uint32_t*>(qh_target);

    const int ql_shift = (col == 2 || col == 3) ? 4 : 0;
    const int qh_shift = col * 2;

    ql4 >>= ql_shift;
    qh4 >>= qh_shift;

    int lo0 = (ql4 >>  0) & 0xF;
    int hi0 = (qh4 >>  0) & 0x3;
    
    int lo1 = (ql4 >>  8) & 0xF;
    int hi1 = (qh4 >>  8) & 0x3;
    
    int lo2 = (ql4 >> 16) & 0xF;
    int hi2 = (qh4 >> 16) & 0x3;
    
    int lo3 = (ql4 >> 24) & 0xF;
    int hi3 = (qh4 >> 24) & 0x3;

    qs[0] = static_cast<float>((lo0 | (hi0 << 4)) - 32);
    qs[1] = static_cast<float>((lo1 | (hi1 << 4)) - 32);
    qs[2] = static_cast<float>((lo2 | (hi2 << 4)) - 32);
    qs[3] = static_cast<float>((lo3 | (hi3 << 4)) - 32);
}

int main() {
    uint8_t ql[128], qh[64];
    for (int i=0; i<128; i++) ql[i] = (i * 3 + 7) & 0xFF;
    for (int i=0; i<64; i++) qh[i] = (i * 5 + 11) & 0xFF;
    
    for (int base = 0; base < 256; base += 4) {
        float q_ref[4], q_fast[4];
        extract_ref(ql, qh, base, q_ref);
        extract_fast(ql, qh, base, q_fast);
        for (int k=0; k<4; k++) {
            if (q_ref[k] != q_fast[k]) {
                printf("Mismatch at base=%d, k=%d: ref=%f fast=%f\n", base, k, q_ref[k], q_fast[k]);
                return 1;
            }
        }
    }
    printf("Fast extraction matches reference perfectly.\n");
    return 0;
}
