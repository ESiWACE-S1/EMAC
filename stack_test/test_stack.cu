#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


//2048(threads)*1024 *8(arrays) *8(sizeof(double)

#define SIZE 1024*4

__global__ void mykernel(double *out, int *in){

  double A0[SIZE];
  double A1[SIZE];

  double res = 0.0;
  int i;
  

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  for(i=0; i<SIZE; i++){

    A0[i] = (double) (index % 25);
    A1[i] = (double) (index % 49);

  }


  for(i=0; i<SIZE; i++){

    A0[in[i]] += A1[i];
    A1[in[i]] += A0[i];

  }

  for(i=0; i<SIZE; i++)
    res += A0[i] + A1[i];


  out[index] = res;

}


int main(){

  int i;
  int nb_threads = 2048;

  double *out = (double *) malloc(nb_threads * sizeof(double));
  int *in     = (int *)    malloc(SIZE * sizeof(int)); 

  
  for(i=0; i<nb_threads; i++)
    out[i] = 0.0;

  for(i=0; i<SIZE; i++)
    in[i] = (i+127) % SIZE;


  double *d_out;
  int    *d_in;

  
  cudaMalloc((void **) &d_out, nb_threads * sizeof(double));
  cudaMalloc((void **) &d_in , SIZE * sizeof(double));

  cudaMemcpy(d_out, out, nb_threads * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in , in , SIZE * sizeof(int), cudaMemcpyHostToDevice);

  dim3 blocksize = 64;
  dim3 numblock  = (nb_threads + blocksize.x -1) / blocksize.x;

  for(i=0; i<1024*10; i++)
    mykernel<<<numblock,blocksize>>>(d_out, d_in);

  cudaMemcpy(out, d_out, nb_threads * sizeof(double), cudaMemcpyDeviceToHost);

  printf("%lf\n", out[5]);

}
