#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <ostream>
#include "../helper.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int THREADS;
int NUM_VALS;


int main (int argc, char *argv[])
{
   
cali::ConfigManager mgr;
mgr.start();
CALI_MARK_BEGIN("main");
NUM_VALS = atoi(argv[1]);
std::string input_type = argv[2];



int	numtasks,             
	taskid, 
	numworkers_inc_master,         
	source,             
	dest,               
	mtype,                 
	i, j, k, rc;           
MPI_Status status;





MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers_inc_master = numtasks;

int calculations_per_worker = NUM_VALS / numworkers_inc_master;
int rank[calculations_per_worker];
int rank_idx[calculations_per_worker];
float received_data[NUM_VALS];
float sorted_array[NUM_VALS];



/**************************** master task ************************************/
   if (taskid == MASTER)
   {
   
            const int n = NUM_VALS;
            float *h_array;
            CALI_MARK_BEGIN("data_init");
            h_array = (float*)malloc(n * sizeof(float));
            array_fill(h_array, n, input_type);
            CALI_MARK_END("data_init");

            /* Send matrix data to the worker tasks */
            mtype = FROM_MASTER;
            // printf("Sending array to tasks");

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Bcast");
            MPI_Bcast(h_array, NUM_VALS, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
            CALI_MARK_END("MPI_Bcast");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");


        //SEND ARRAY DATA PART FOR THE MASTER PROCESS ENDS HERE

        
        /* Do master thread calculations */
            int count = 0;
            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_large");
            for(int i = taskid; i < NUM_VALS; i += numworkers_inc_master){
                
                if (i < NUM_VALS) {
                    rank[count] = 0;
                    rank_idx[count] = i;
                    for (int j = 0; j < NUM_VALS; j++) {
                        if (h_array[j] < h_array[i] || (h_array[j] == h_array[i] && j < i)) {
                            rank[count]++;
                        }
                    }
                }
                
                count++;
            }
            CALI_MARK_END("comp_large");

            CALI_MARK_BEGIN("comp_small");
            for (int i = 0; i < calculations_per_worker; i++){
                sorted_array[rank[i]] = h_array[rank_idx[i]];
            }
            CALI_MARK_END("comp_small");
            CALI_MARK_END("comp");


        /* Receive results from worker tasks */
            mtype = FROM_WORKER;
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Recv");
            for (source=1; source<numworkers_inc_master; source++)
            {
                MPI_Recv(&rank, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&rank_idx, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

                for (int i = 0; i < calculations_per_worker; i++){
                    sorted_array[rank[i]] = h_array[rank_idx[i]];
                }
            }
            CALI_MARK_END("MPI_Recv");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");
      
      
        CALI_MARK_BEGIN("correctness_check");
        if (correctness_check(sorted_array, n)) {
            printf("Array correctly sorted!\n");
        } else {
            printf("Array sorting failed\n");
        }
        CALI_MARK_END("correctness_check");
        free(h_array);
      
      
   }


   if (taskid > MASTER)
   {
      //RECEIVING PART FOR WORKER PROCESS STARTS HERE
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Bcast");
        MPI_Bcast(received_data, NUM_VALS, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Bcast");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        // printf("worker recieved array");
        
      //RECEIVING PART FOR WORKER PROCESS ENDS HERE
      

      //CALCULATION PART FOR WORKER PROCESS STARTS HERE

        int count = 0;
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_small");
        for(int i = taskid; i < NUM_VALS; i += numworkers_inc_master){
            
            if (i < NUM_VALS) {
                rank[count] = 0;
                rank_idx[count] = i;
                for (int j = 0; j < NUM_VALS; j++) {
                    if (received_data[j] < received_data[i] || (received_data[j] == received_data[i] && j < i)) {
                        rank[count]++;
                    }
                }
            }
            count++;
        }
        CALI_MARK_END("comp_small");
        CALI_MARK_END("comp");
         
      //CALCULATION PART FOR WORKER PROCESS ENDS HERE
      
      
      //SENDING PART FOR WORKER PROCESS STARTS HERE
        
        mtype = FROM_WORKER;

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_small");
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(&rank, calculations_per_worker, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rank_idx, calculations_per_worker, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
        CALI_MARK_END("comm_small");
        CALI_MARK_END("comm");

      //SENDING PART FOR WORKER PROCESS ENDS HERE

   }
    MPI_Barrier(MPI_COMM_WORLD);

   // WHOLE PROGRAM COMPUTATION PART ENDS HERE
    CALI_MARK_END("main");

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    
   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
}