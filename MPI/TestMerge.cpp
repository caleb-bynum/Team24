/* Unit tests for Merge_high() and Merge_low() */
#include <iostream>
#include <cassert>
#include "Merge.h"

int main() {
    // create unit tests for Merge_high() and Merge_low()
    // -create sorted arrays A and B
    // -create corresponding arrays C_high and C_low
    // -call Merge_high() and Merge_low()
    // -check that C_high and C_low are sorted

    // allocate C

    int n = 4;
    double A[4] = { 1, 5, 6, 7 };
    double B[4] = { 2, 3, 4, 8 };
    double C[4] = { 0, 0, 0, 0 };

    double C_high[4] = { 5, 6, 7, 8 };
    double C_low[4] = { 1, 2, 3, 4 };

    Merge_High(A, B, C, n);

    // assert that C_high and C are equal
    std::cout << "Merge_high..." << std::endl;
    for (int i = 0; i < n; i++) {
        assert(C_high[i] == C[i]);
        std::cout << C[i] << std::endl;
    }

    A[0] = 1;
    A[1] = 5;
    A[2] = 6;
    A[3] = 7;

    Merge_Low(A, B, C, n);

    // assert C_low and C are equal
    std::cout << "Merge_low..." << std::endl;
    for (int i = 0; i < n; i++) {
        assert(C_low[i] == C[i]);
        std::cout << C[i] << std::endl;
    }

    std::cout << "Unit test passed." << std::endl;

}