# CSCE435 Project

### Main Function Logic:

  1. Create vector of random floats
  2. Call bucket sort function
  3. Print out the sorted vector

### Bucket Sort Logic:

  1. Create arrays for data and bucket value counters
  2. Call the bucket_sort_kernel function and synchronize
  3. Use thrust sort to sort each individual bucket
  4. Free all the previously created data structures for data and bucket value counters

### Bucket Sort Kernel Logic:
  1. Check if the current value being passed to the function is less than the max bucket values
  2. Modify the bucket that we add the value to...
     If the current bucket is within the bound of the number of buckets we can add it if not we must go back a bucket
  3. Add the value to the appropriate bucket
