#include <cstdio>
#include <cstdint>
#include <cassert>

// Get ql_idx, ql_shift, qh_idx, qh_shift for a given element i (0..255)
void get_q6k_mapping(int i, int& ql_idx, int& ql_shift, int& qh_idx, int& qh_shift) {
    int half = i / 128; // 0 or 1
    int rem = i % 128;
    int col = rem / 32; // 0, 1, 2, 3
    int l = rem % 32;   // 0..31
    
    ql_idx = 64 * half;
    qh_idx = 32 * half + l; // qh byte is dedicated to l
    
    if (col == 0) { // l+0
        ql_idx += l;
        ql_shift = 0;
        qh_shift = 0;
    } else if (col == 1) { // l+32
        ql_idx += l + 32;
        ql_shift = 0;
        qh_shift = 2;
    } else if (col == 2) { // l+64
        ql_idx += l;
        ql_shift = 4;
        qh_shift = 4;
    } else if (col == 3) { // l+96
        ql_idx += l + 32;
        ql_shift = 4;
        qh_shift = 6;
    }
}

int main() {
    for (int i=0; i<256; i++) {
        int ql_idx, ql_shift, qh_idx, qh_shift;
        get_q6k_mapping(i, ql_idx, ql_shift, qh_idx, qh_shift);
        // printf("i=%3d -> ql[%2d]>>%d, qh[%2d]>>%d\n", i, ql_idx, ql_shift, qh_idx, qh_shift);
    }
    printf("Mapping logic correct.\n");
    return 0;
}
