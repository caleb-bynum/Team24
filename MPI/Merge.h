/*
Contains two sub-procedures:
- Merge_Low()
- Merge_High()
*/

/* Given sorted arrays A and B, populate C with the larger half of elements from A and B.
   Then, copy the elements from C back into A. */
void Merge_High(double* local_A, double* temp_B, double* temp_C, int local_n) {
    int ai = local_n - 1;
    int bi = local_n - 1;
    int ci = local_n - 1;

    // perform merge
    while (ci >= 0) {
        if (local_A[ai] >= temp_B[bi]) {
            temp_C[ci--] = local_A[ai--];
        }
        else {
            temp_C[ci--] = temp_B[bi--];
        }
    }

    // copy elements from C to A
    for (int i = 0; i < local_n; i++) {
        local_A[i] = temp_C[i];
    }
}

/* Given sorted arrays A and B, populate C with the smaller half of elements from A and B */
void Merge_Low(double* local_A, double* temp_B, double* temp_C, int local_n) {
    int ai = 0;
    int bi = 0;
    int ci = 0;

    // perform merge
    while (ci < local_n) {
        if (local_A[ai] <= temp_B[bi]) {
            temp_C[ci++] = local_A[ai++];
        }
        else {
            temp_C[ci++] = temp_B[bi++];
        }
    }

    // copy elements from C to A
    for (int i = 0; i < local_n; i++) {
        local_A[i] = temp_C[i];
    }
}