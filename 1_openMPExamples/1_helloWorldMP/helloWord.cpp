#include<iostream>
#include<math.h>
#include<omp.h>
#include<stdio.h>

using namespace std;

int main(int argc, char *argv[])
{    
  #pragma omp parallel
  {
    int noThreads = omp_get_num_threads();
    int id = omp_get_thread_num();
    cout << "hello from thread: " << id << " of " << noThreads << "\n";
  }
  return 0;
}
