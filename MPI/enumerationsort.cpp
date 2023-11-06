#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <assert.h>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>

#define REG_EMPTY         (-1)
#define REG_TAG_X         256
#define REG_TAG_Y         257
#define REG_TAG_Z         258
#define REG_TAG_C         259
#define REG_TAG_ZC        260

using namespace std;

const char * FILE_NAME = "numbers";


int main(int argc, char *argv[]) {
	// number of procecs
   int numprocs;     
   int myid;       
	//status
   MPI_Status stat;  

	//registers
   int reg_x;     
   int reg_y;     
   int reg_z;    
   int compare;   
   size_t z_count;
   
   int N = 10;	
	int B;
	int T;
	
	int *a, *b, *c, *d;
	int *dev_a, *dev_b, *dev_c;
	int size;	
	
	struct timeval start, end;

   long mtime, seconds, useconds;  
   

   

   /*--------------------ENTER INPUT PARAMETERS AND ALLOCATE DATA -----------------------*/
	// keyboard input
   
	printf("Enter the value for N: ");
	scanf("%d", &N);
	//takes in input
	printf("Enter the number of threads: ");
	scanf("%d", &T);
   
	
	// size length
	size = N *N* sizeof(int);	

	a = (int*) malloc(size);	
	b = (int*) malloc(size);
	c = (int*) malloc(size);	
	d = (int*) malloc(size);	
	// load random numbers
	ofstream myfile;
  	myfile.open ("numbers", std::ofstream::out | std::ofstream::trunc);

	//initialize ranodm seed
	srand(3);	
	int i;
	for (i = 0; i < N; i++)
	{
		//load array with numbers
		a[i] = (int) rand();
		myfile << a[i] << " ";
	}
    
   myfile.close();

   // init
   MPI_Init(&argc, &(argv));
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   compare = 0;
   z_count = 0;
   reg_x = REG_EMPTY;
   reg_y = REG_EMPTY;
   reg_z = REG_EMPTY;

   if(myid == 0) { // main proc
      int number;
      fstream fin;
      fin.open("numbers");

      for (int i = 0; fin.good() && i < numprocs; ++i) {
          // read and print values at the same time
          number = fin.get();
          cout << number << endl;

          if(! fin.good()) break;

          if (i != 0){
             cout << " " << number;
          }
          else{
             cout << number; 
          }
          // send value to corresponding proc's reg X
          if (i != 0){
          	cout << " aee";
            MPI_Send(&number, 1, MPI_INT, i, REG_TAG_X, MPI_COMM_WORLD);
          }
          else{
             reg_x = number;
          }

          reg_y = number;
          compare += (reg_x < reg_y);

          MPI_Send(&reg_y, 1, MPI_INT, myid + 1, REG_TAG_Y, MPI_COMM_WORLD);
      }

      cout << endl; fin.close();
   } else { 
      MPI_Recv(&reg_x, 1, MPI_INT, 0, REG_TAG_X, MPI_COMM_WORLD, &stat);

		//gettimeofday(&start, NULL);
      for (int i = 0; i < numprocs; ++i) {
         MPI_Recv(&reg_y, 1, MPI_INT, myid - 1, REG_TAG_Y, MPI_COMM_WORLD, &stat);
         if (i < numprocs) {
            if (reg_x != REG_EMPTY && reg_x != REG_EMPTY)
               compare += (reg_x < reg_y);
         }


         if (myid != numprocs - 1)
            MPI_Send(&reg_y, 1, MPI_INT, myid + 1, REG_TAG_Y, MPI_COMM_WORLD);
      }
		//gettimeofday(&end, NULL);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   int value;     

   for (int i = 0; i < numprocs; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (myid == i) {
         if (compare != myid) {
            value = compare;
         } else {
            value = REG_EMPTY;
            reg_z = reg_x;
            z_count++;
         }
      } else
         value = REG_EMPTY;

      MPI_Bcast(&value, 1, MPI_INT, i, MPI_COMM_WORLD);

      if (myid == i && compare != myid) {
         MPI_Send(&reg_x, 1, MPI_INT, compare, REG_TAG_Z, MPI_COMM_WORLD);
      }

      if (value == myid) {
         MPI_Recv(&reg_z, 1, MPI_INT, i, REG_TAG_Z, MPI_COMM_WORLD, &stat);
         z_count++;
      }
   }

   for (int i = 0; i < numprocs; ++i) {
      if (myid == 0) {
         if (reg_z != REG_EMPTY) {
            for (size_t i = 0; i < z_count; i++)
               cout << reg_z << endl;
         }

      }

      if (myid != 0) {
         MPI_Send(&reg_z, 1, MPI_INT, myid - 1, REG_TAG_Z, MPI_COMM_WORLD);
         MPI_Send(&z_count, 1, MPI_INT, myid - 1, REG_TAG_ZC, MPI_COMM_WORLD);
      }
      if (myid != numprocs - 1) {
         MPI_Recv(&reg_z, 1, MPI_INT, myid + 1, REG_TAG_Z, MPI_COMM_WORLD, &stat);
         MPI_Recv(&z_count, 1, MPI_INT, myid + 1, REG_TAG_ZC, MPI_COMM_WORLD, &stat);
      }
   }

   seconds  = end.tv_sec  - start.tv_sec;
   useconds = end.tv_usec - start.tv_usec;

   mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
   printf("Elapsed time: %ld milliseconds\n", mtime);

   MPI_Finalize();
   return 0;
}

