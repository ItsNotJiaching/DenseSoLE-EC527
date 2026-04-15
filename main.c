
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "sole_multi.h"
#include "sole_serial.h"

void initializeArray1D(float *arr, int len, int seed);
void print_array(float* v, int arr_len);


/* Serial MMM, Blocked */
void mmm(float* a, float* b, float* c, int length, int bsize) {
  long int i, j, k, jj, kk;
  float sum;
  int en = bsize * (length/bsize);

  for (kk = 0; kk < en; kk+=bsize){
    for (jj = 0; jj < en; jj+=bsize) {
      for (i = 0; i < length; i++) {
        for (j = jj; j < jj + bsize; j++) {
          sum = 0;
          for (k = kk; k < kk + bsize; k++) {
            sum += a[i*length+k] * b[k*length+j];
            // printf("Doing %f * %f to get %f\n", a[i*length+k], b[k*length+j], sum);
          }
          c[i*length+j] += sum;
          // printf("      Calculated C value: %f\n", c[i*length+j]);
        }
      }
    }
  }
}

int main(int argc, char **argv){

  printf("Hello world!");
  
  return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
  int i;
  float randNum;
  srand(seed);

  for (i = 0; i < len*len; i++) {
    randNum = (float) rand() / RAND_MAX;
    arr[i] = randNum;
  }
}

void print_array(float* v, int arr_len) {
  for (int i=0; i < arr_len; i++) {
    for (int j=0; j < arr_len; j++) {
      printf("%.3f ", v[i*arr_len + j]);
    }
    printf("\n");
  }
  printf("\n");
}