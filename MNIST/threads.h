/**
 * Include this file to have access to the thread functionality as well as all
 * of the global variables necessary to create the network. These variables will
 * be defined in driver.cpp
 */

#ifndef THREADS_H
#define THREADS_H

#include <pthread.h>

extern int num_threads;
extern int epochs;
extern int num_inputs;
extern int batch_size;
extern int depth;
extern int *architecture;
extern double random_limit;
extern double (**f_activations)(double);
extern double (**d_f_activations)(double);
extern double (*f_cost)(double, double);
extern double (*d_f_cost)(double, double);
extern double training_rate;
extern double **inputs;
extern double **outputs;

extern double *read_data;
extern double *write_data;

extern pthread_mutex_t *mutexes;
extern pthread_barrier_t barrier;

// The thread function to be used in pthread_create
void *thread_func(void *ID);

#endif
