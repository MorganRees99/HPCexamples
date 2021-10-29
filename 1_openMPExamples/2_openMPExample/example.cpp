#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_THREADS 8
#define PAD 8 // assume 64 byte L1 cache line size

static long steps = 1000000000;
double step;

void method1 () {

    int j;
    double pi;
    double start, delta;

    step = 1.0/(double) steps;

    // Compute parallel compute times for 1-MAX_THREADS
    for (j=1; j<= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);

        // This is the beginning of a single PI computation 
        omp_set_num_threads(j);

        double start = omp_get_wtime();
	double sum[j];

        #pragma omp parallel
	{
	    int i, id;
	    double x;
	    id = omp_get_thread_num();
	    sum[id]=0.0;
            for (i=id; i<steps; i=i+j) {
             	x = (i+0.5)*step;
              	sum[id] += 4.0 / (1.0+x*x); 
             }
	}

        // Out of the parallel region, finialize computation
        for(int i=0, pi=0.0; i<j; i++){
	    pi += sum[i]*step;
	}
        delta = omp_get_wtime() - start;
        printf("    PI = %.16g computed in %.4g seconds\n", pi, delta);

    }
    

}

void method2 () {

    int j;
    double pi;
    double start, delta;

    step = 1.0/(double) steps;

    // Compute parallel compute times for 1-MAX_THREADS
    for (j=1; j<= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);

        // This is the beginning of a single PI computation 
        omp_set_num_threads(j);

        double start = omp_get_wtime();
	double sum[j][PAD];

        #pragma omp parallel
	{
	    int i, id;
	    double x;
	    id = omp_get_thread_num();
	    sum[id][0] = 0.0;
            for (i=id; i<steps; i=i+j) {
             	x = (i+0.5)*step;
              	sum[id][0] += 4.0 / (1.0+x*x); 
             }
	}

        // Out of the parallel region, finialize computation
        for(int i=0, pi=0.0; i<j; i++){
	    pi += sum[i][0]*step;
	}
        delta = omp_get_wtime() - start;
        printf("    PI = %.16g computed in %.4g seconds\n", pi, delta);

    }
    

}

void method3 () {

    int j;
    double pi;
    double start, delta;

    step = 1.0/(double) steps;

    // Compute parallel compute times for 1-MAX_THREADS
    for (j=1; j<= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);

        // This is the beginning of a single PI computation 
        omp_set_num_threads(j);

        double start = omp_get_wtime();

        #pragma omp parallel
	{
	    int i, id, nthrds;
	    double x, sum=0.0;
	    id = omp_get_thread_num();
            for (i=id; i<steps; i=i+j) {
             	x = (i+0.5)*step;
              	sum += 4.0 / (1.0+x*x); 
             }
       	     #pragma omp critical
		pi += sum*step;
	}

        delta = omp_get_wtime() - start;
        printf("    PI = %.16g computed in %.4g seconds\n", pi, delta);

    }
    

}

void method4 () {

    int j;
    double pi;
    double start, delta;

    step = 1.0/(double) steps;

    // Compute parallel compute times for 1-MAX_THREADS
    for (j=1; j<= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);

        // This is the beginning of a single PI computation 
        omp_set_num_threads(j);

        double start = omp_get_wtime();

        #pragma omp parallel
	{
	    int i, id;
	    double x, sum;
	    id = omp_get_thread_num();
            for (i=id, sum=0.0; i<steps; i=i+j) {
             	x = (i+0.5)*step;
              	sum += 4.0 / (1.0+x*x); 
             }
       	     #pragma omp atomic
		pi += sum*step;
	}

        delta = omp_get_wtime() - start;
        printf("    PI = %.16g computed in %.4g seconds\n", pi, delta);

    }
    

}

void method5 () {

    int i,j;
    double x;
    double pi, sum = 0.0;
    double start, delta;

    step = 1.0/(double) steps;

    // Compute parallel compute times for 1-MAX_THREADS
    for (j=1; j<= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);

        // This is the beginning of a single PI computation 
        omp_set_num_threads(j);

        sum = 0.0;
        double start = omp_get_wtime();


        #pragma omp parallel for reduction(+:sum) private(x)
        for (i=0; i < steps; i++) {
            x = (i+0.5)*step;
            sum += 4.0 / (1.0+x*x); 
        }

        // Out of the parallel region, finialize computation
        pi = step * sum;
        delta = omp_get_wtime() - start;
        printf("    PI = %.16g computed in %.4g seconds\n", pi, delta);

    }
    

}

int main (int argc, const char *argv[]) {

    printf("-----------------------------------------------\n");
    printf("                   Method 1\n");
    printf("-----------------------------------------------\n");
    method1();

    printf("-----------------------------------------------\n");
    printf("                   Method 2 (padded)\n");
    printf("-----------------------------------------------\n");
    method2();

    printf("-----------------------------------------------\n");
    printf("                   Method 3 (critical)\n");
    printf("-----------------------------------------------\n");
    method3();

    printf("-----------------------------------------------\n");
    printf("                   Method 4 (atomic)\n");
    printf("-----------------------------------------------\n");
    method4();

    printf("-----------------------------------------------\n");
    printf("                   Method 5 (reduction)\n");
    printf("-----------------------------------------------\n");
    method5();

}
