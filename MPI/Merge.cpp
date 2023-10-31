/*
Contains the sub-procedures:
- Merge_high()
- Merge_low()
*/
#include <iostream>


/* Given sorted arrays A and B, populate C with the larger half of elements from A and B */
void Merge_high(double* local_A, double* temp_B, double* temp_C, int local_n) {
    int ai = local_n - 1;
    int bi = local_n - 1;
    int ci = local_n - 1;
    while (ci >= 0) {
        if (local_A[ai] >= temp_B[bi]) {
            temp_C[ci--] = local_A[ai--];
        }
        else {
            temp_C[ci--] = temp_B[bi--];
        }
    }
}

/* Given sorted arrays A and B, populate C with the smaller half of elements from A and B */
void Merge_low(double* local_A, double* temp_B, double* temp_C, int local_n) {
    int ai = 0;
    int bi = 0;
    int ci = 0;
    while (ci < local_n) {
        if (local_A[ai] <= temp_B[bi]) {
            temp_C[ci++] = local_A[ai++];
        }
        else {
            temp_C[ci++] = temp_B[bi++];
        }
    }
}


