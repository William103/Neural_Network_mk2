/**
 * Include this file to have access to the thread functionality as well as all
 * of the global variables necessary to create the network. These variables will
 * be defined in driver.cpp
 */

#ifndef THREADS_H
#define THREADS_H

#include <pthreads.h>

extern const int num_threads;
extern const int epochs;
extern const int num_inputs;
extern const int batch_size;
extern const int depth;
extern const int architecture[depth];
extern const double (*(f_activations[depth-1]))(double, double);
extern const double (*(d_f_activations[depth-1]))(double, double);
extern const double (*f_cost)(double, double);
extern const double (*d_f_cost)(double, double);
extern const double training_rate;
extern const double inputs[num_inputs][architecture[0]];
extern const double outputs[num_inputs][architecture[depth-1]];

extern double *read_data;
extern double *write_data;

// The thread function to be used in pthread_create
void *thread_func(void *ID);

#endif
