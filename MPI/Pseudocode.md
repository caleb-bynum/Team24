# CSCE435 Project

### Variables in main:
 
  Pivots: Array of random double to be sorted.
  Bucket_Num: Array to track the number of values that are being sent to each processor
  Displacements: Storage for displacement data for scatter and gather functionality
  D_list: final doubles list to store the sorted value

### Main Function Logic:
  1. MPI_Init, Size, and Rank to set up MPI parallel functionality
  2. Create all the random values in the main thread only
  3. Set the displacements and sort each value inti their respective buckets in the main thread only
  4. Scatter, Sort, and Gather each individual bucket in their own thread
  
### Compare Function Logic:
  This is a customer comparator function that is designed to help the custom_qsort function sort through doubles.

  1. Cast and dereference the two input values
  2. Subtract the two values
  3. If the first is greater than the second return 1
  4. If the second is greater than the first return -1
  5. If both values are equal return 0
  
### Custom Quicksort Function Logic:
  This function simply is a wrapper for the stl qsort function. 
  The only modification to the stl function is the use of the custom comparator mentioned above in the compare function logic.

### Find Bucket Logic:
  In order to complete the sample sort operation it is important to sort values into buckets for each processor.

  1. Loop through every processor
  2. Check if the current input value is lower than the upper limit of each processor
  3. Once a processor is found return the index number of the processor
