# CSCE 435 Group project

## 1. Group members:
1. Caleb Bynum
2. Josef Munduchirakal
3. Aazmir Lakhani
4. Tyler Roosth

---

## 2. _due 10/25_ Project topic
Implementing parallel sorting algorithms (mergeSort, sampleSort, enumerationSort, oddEvenTranspositionSort) using MPI and CUDA, and then performing comparison analysis on them.

---

## 3. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
### 1. MergeSort
#### MPI
#### CUDA
### 2. SampleSort
#### MPI 
#### CUDA
### 3. EnumerationSort
#### MPI
#### CUDA
### 4. OddEvenTranspositionSort
#### MPI
#### CUDA

## 4. Pseudocode
### OddEvenTranspositionSort MPI
# MPI Implementation Pseudocode

# Variables in main:
#     p : number of processes
#     global_n : number of elements in global array
#     local_n : number of elements in processes' local array
#     local_A : each processes' local array

# Main Procedure:
#     1. Perform MPI initialization functions
#     2. Allocate local_A
#     3. Populate local_A
#     4. Call Sort on each process
#     5. MPI Finalize

# Interpreting the sorted array:
# The global array can be formed by concatenating each local array
# in ascending order by process rank. In MPI, this can be done with MPI_Gather()

# Variables in Sort:
#     my_rank : the current processes' rank
#     comm : MPI_COMM_WORLD
#     phase : counter to track current phase, ranges from 0 to p
#     temp_B : temporary local storage for MPI_Sendrecv
#     temp_C : temporary local storage for Merge_high or Merge_low
#     local_n : Same as in the Main Procedure
#     local_A : Same as in the Main Procedure
#     p : Same as in the Main Procedue
#     even_partner : the process that will be communicated with during the even phase
#     odd_partner : the process that will be communicated with during the odd phase

# Sort Procedure:
#     1. Allocate local storage temp_B
#     2. Allocate local storage temp_C
#     3. Determine even_phase_partner
#     4. Determine odd_phase_partner
#     5. Call quickSort on local_A
#     6. For each phase:
#     7.   Call Odd-Even-Iterate
#     8. Deallocate temp_B
#     9. Deallocate temp_C

# Variables in Odd-Even-Iterate:
#     my_rank : the current processes' rank
#     comm : MPI_COMM_WORLD
#     status : MPI_status
#     phase : counter to track current phase, ranges from 0 to p
#     temp_B : temporary local storage for MPI_Sendrecv
#     temp_C : temporary local storage for Merge_high or Merge_low
#     local_n : Same as in the Main Procedure
#     local_A : Same as in the Main Procedure
#     p : Same as in the Main Procedue
#     even_partner : the process that will be communicated with during the even phase
#     odd_partner : the process that will be communicated with during the odd phase

# Odd-Even-Iterate Procedue:
#     1. if phase is even:
#     2.   Call MPI_Sendrecv to exchange values with even_partner
#     3.   if my_rank is odd:
#     4.     Call Merge_high
#     5.   else:
#     6.     Call Merge_low
#     7. else:
#     8.   Call MPI_Sendrecv to exchange values with odd_partner
#     9.   if my_rank is odd:
#     10.    call Merge_low
#     11.  else:
#     12.    call Merge_high

# Merge_high Variables:
#     local_A : local array for a process
#     temp_B : contains array sent from partner process
#     temp_C : to be filled with the larger half of values from local_A and temp_B
#     A_ptr : iterator for local_A
#     B_ptr : iterator for temp_B
#     C_ptr : iterator for temp_C

# Merge_high Procedure:
#     1. Initialize A_ptr, B_ptr, C_ptr to point at the end of their respective arrays
#     2. While C_ptr >= 0:
#     3.   if local_A[A_ptr] >= temp_B[B_ptr]:
#     4.     temp_C[C_ptr] = local_A[A_ptr]
#     5.     decrement C_ptr, A_ptr
#     6.   else:
#     7.     temp_C[C_ptr] = temp_B[B_ptr]
#     8.     decrement C_ptr, B_ptr

# Merge_low Variables:
#     local_A : local array for a process
#     temp_B : contains array sent from partner process
#     temp_C : to be filled with the larger half of values from local_A and temp_B
#     A_ptr : iterator for local_A
#     B_ptr : iterator for temp_B
#     C_ptr : iterator for temp_C

# Merge_low Procedure:
#     1. Initialize A_ptr, B_ptr, C_ptr to 0
#     2. While C_ptr < local_n:
#     3.   if local_A[A_ptr] <= temp_B[B_ptr]:
#     4.     temp_C[C_ptr] = local_A[A_ptr]
#     5.     increment C_ptr, A_ptr
#     6.   else:
#     7.     temp_C[C_ptr] = temp_B[B_ptr]
#     8.     increment C_ptr, B_ptr

# ### OddEvenTranspositionSort CUDA
# CUDA Implementation Pseudocode

# Variables:
#     n : number of elements to sorted
#     t : number of threads
#     A : array to be sorted

# Idea: avoid coordination between threads by strategically launching kernels
#       and subsequently synchronzing.

# Main Procedure:
#     1. Copy A to GPU memory
#     2. for each phase in {0, 1, ..., n-1}:
#     3.   if phase is even:
#     4.     launch EvenPhase kernel
#     5.   else:
#     6.     launch OddPhase kernel
#     7.   synchronize device
#     8. Copy A to device memory

# EvenPhase Kernel:
#     1. Id = threadId + blockDim * blockId
#     2. index1 = 2 * Id
#     3. index2 = index1 + 1
#     4. compare and swap array elements at index1, index2

# OddPhase Kernel:
#     1. Id = threadId + blockDim * blockId
#     2. index1 = 2 * Id + 1
#     3. index2 = index1 + 1
#     4. if thread is not the last thread
#     5.   compare and swap array elements at index1, index2


## 5. Evaluation plan - what and how we will measure performance
### For each algorithm, we plan to measure the runtime of various components with our programs with respect to input size.
### In testing, we plan to vary the array input sizes with the following counts {2^16, 2^20, 2^24} 
### Input properties of float arrays to be tested: { randomized, reversed, sorted }
### We plan to evaluate how our algorithms behave with respect to Strong Scaling (same problem size {2^24}, increase number of processors/nodes)
#### MPI Strong Scaling: increase number of cores {2, 4, 8, 16, 32, 64}
#### CUDA Strong Scaling: number of threads per block {64, 128, 512, 1024}
### Weak scaling (increase problem size, increase number of processors)
#### MPI Weak Scaling: 
##### Increase number of cores {2, 4, 8, 16, 32, 64}
##### Increase input array size {2^4, 2^8, 2^12, 2^16, 2^20, 2^24}
#### CUDA Weak Scaling: 
##### Increase number of threads per block {64, 128, 512, 1024}
##### Increase input array size {2^12, 2^16, 2^20, 2^24}

## 6. Incomplete Items for Every Team Member
### Every team member is at the same step in the development of the project. We have all completed our respective algorithm implementations in both MPI and CUDA. However, due to the sustained GRACE outage, every team member lacks their respective .cali files for each algorithm. We plan to obtain all .cali files immediately after GRACE becomes available.



