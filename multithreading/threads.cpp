#include "threads.h"
#include "network.h"
#include <cstring>
#include <iostream>

void *thread_func(void *ID_arg) {
    int ID = *((int*)ID_arg);
    Network net(architecture, depth, f_activations, d_f_activations, f_cost, d_f_cost, read_data, write_data);

    double *y_hat;
    double error;
    int n_inputs = num_inputs / num_threads;
    int thread_batch_size = batch_size / num_threads;
    double **thread_inputs = inputs + ID * num_inputs / num_threads;
    double **thread_outputs = outputs + ID * num_inputs / num_threads;
    for (int i = 0; i < epochs; i++) {
        //error = 0;
        for (int j = 0; j < n_inputs; j++) {
            y_hat = net.prop(thread_inputs[j]);

            net.back_prop(thread_inputs[j], thread_outputs[j], training_rate);
            if ((j+1) % thread_batch_size == 0) {
                pthread_barrier_wait(&barrier);
                net.update();
                pthread_barrier_wait(&barrier);
            }
        }
        //error /= n_inputs;
        //if (i == epochs - 1)
            //std::cout << "Epoch #" << i << " Error: " << error << std::endl;
    }
}
