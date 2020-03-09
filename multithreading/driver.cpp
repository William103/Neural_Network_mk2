#include <stdio.h>
#include <iostream>
#include <ctime>
#include "utils.h"
#include "network.h"
#include "threads.h"
#include "barrier.h"

int num_threads = 2;
int epochs = 50000;
int num_inputs = 4;
int batch_size = 4;
int depth = 4;
int *architecture;
double random_limit = 5;
double (**f_activations)(double);
double (**d_f_activations)(double);
double (*f_cost)(double, double);
double (*d_f_cost)(double, double);
double training_rate = 0.1;
double **inputs;
double **outputs;

double *read_data;
double *write_data;

pthread_mutex_t *mutexes;
pthread_barrier_t barrier;

typedef double (*activation_pointer)(double);

int main() {
    architecture = new int[4] { 2, 5, 5, 1 };
    f_activations = new activation_pointer[3] { sigmoid, sigmoid, sigmoid };
    d_f_activations = new activation_pointer[3] { d_sigmoid, d_sigmoid, d_sigmoid };
    f_cost = squared_error;
    d_f_cost = d_squared_error;
    inputs = new double*[4];
    inputs[0] = new double[2] { 0, 0 };
    inputs[1] = new double[2] { 1, 0 };
    inputs[2] = new double[2] { 0, 1 };
    inputs[3] = new double[2] { 1, 1 };
    outputs = new double*[4];
    outputs[0] = new double[1] { 0 };
    outputs[1] = new double[1] { 1 };
    outputs[2] = new double[1] { 1 };
    outputs[3] = new double[1] { 0 };

    int num_weights = 0;
    int num_neurons = architecture[0];
    for (int i = 1; i < depth; i++) {
        num_neurons += architecture[i];
        num_weights += architecture[i-1] * architecture[i];
    }

    read_data = new double[num_neurons + num_weights];
    write_data = new double[num_neurons + num_weights];
    for (int i = 0; i < num_neurons + num_weights; i++) {
        read_data[i] = write_data[i] = (((double) std::rand() / RAND_MAX) - 0.5) * random_limit;
    }

    pthread_t *threads = new pthread_t[num_threads];
    int *ids = new int[num_threads];

    mutexes = new pthread_mutex_t[num_weights + num_neurons];

    /* ----------------------------------------------------------------- */

    for (int i = 0; i < num_weights + num_neurons; i++) {
        pthread_mutex_init(mutexes + i, NULL);
    }

    pthread_barrier_init(&barrier, NULL, num_threads);

    for (int i = 0; i < num_threads; i++) {
        ids[i] = i;
        pthread_create(threads + i, NULL, thread_func, (void*)(ids + i));
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    for (int i = 0; i < num_weights + num_neurons; i++) {
        pthread_mutex_destroy(mutexes + i);
    }

    Network net(architecture, depth, f_activations, d_f_activations, f_cost, d_f_cost, read_data, write_data);
    double *y_hat;
    for (int i = 0; i < 4; i++) {
        y_hat = net.prop(inputs[i]);
        std::cout << inputs[i][0] << ' ' << inputs[i][1] << ": " << y_hat[0] << std::endl;
    }

    /* ----------------------------------------------------------------- */

    delete[] ids;
    delete[] threads;
    delete[] read_data;
    delete[] write_data;
    delete[] f_activations;
    delete[] d_f_activations;
    for (int i = 0; i < 4; i++) {
        delete[] inputs[i];
        delete[] outputs[i];
    }
    delete[] inputs;
    delete[] outputs;
}
