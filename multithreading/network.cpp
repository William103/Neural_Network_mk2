#include "network.h"
#include "threads.h"

#include <cstring>
#include <iostream>

Network::Network(int *architecture, int depth, double (**f_activations)(double),
                 double (**d_f_activations)(double), double (*f_cost)(double, double),
                 double (*d_f_cost)(double, double), double *read_data, double *write_data) :
                     architecture(architecture), depth(depth),
                     f_activations(f_activations),
                     d_f_activations(d_f_activations), f_cost(f_cost), d_f_cost(d_f_cost), 
                     read_data(read_data), write_data(write_data)
{
    num_weights = 0;
    num_neurons = architecture[0];
    for (int i = 1; i < depth; i++) {
        num_neurons += architecture[i];
        num_weights += architecture[i-1] * architecture[i];
    }
    neuron_inputs = new double[num_neurons * 3];
    activations = neuron_inputs + num_neurons;
    deltas = neuron_inputs + num_neurons * 2;
    read_weights = read_data;
    read_biases = read_data + num_weights;
    write_weights = write_data;
    write_biases = write_data + num_weights;
    weight_layers = new int[depth];
    neuron_layers = new int[depth];
    neuron_layers[0] = 0;
    weight_layers[0] = 0;
    for (int i = 1; i < depth; i++) {
        neuron_layers[i] = neuron_layers[i-1] + architecture[i-1];
        weight_layers[i] = weight_layers[i-1] + architecture[i-1] * architecture[i];
    }
}

Network::~Network()
{
    delete[] neuron_inputs;
    delete[] weight_layers;
    delete[] neuron_layers;
}

double *Network::prop(double *input)
{
    std::memset(neuron_inputs, 0, num_neurons * 3 * sizeof(double));
    std::memcpy(neuron_inputs, input, architecture[0] * sizeof(double));
    std::memcpy(activations, neuron_inputs, architecture[0] * sizeof(double));
    for (int i = 1; i < depth; i++) {
        for (int k = 0; k < architecture[i]; k++) {
            for (int j = 0; j < architecture[i-1]; j++) {
                neuron_inputs[k + neuron_layers[i]] +=
                    activations[j + neuron_layers[i-1]] *
                    read_weights[weight_layers[i-1] + j * architecture[i] + k];
            } 
            activations[k + neuron_layers[i]] = f_activations[i-1](neuron_inputs[k + neuron_layers[i]] +
                read_biases[k + neuron_layers[i]]);
        }
    }
    return activations + neuron_layers[depth - 1];
}

double Network::back_prop(double *input, double *output, double training_rate)
{
    double error = 0;
    for (int i = 0; i < architecture[depth-1]; i++) {
        deltas[neuron_layers[depth-1]+i] += d_f_cost(activations[neuron_layers[depth-1]+i], output[i]) *
            d_f_activations[depth-2](neuron_inputs[neuron_layers[depth-1]+i] + read_biases[neuron_layers[depth-1]+i]);

        write_biases[neuron_layers[depth-1]+i] -= training_rate * deltas[neuron_layers[depth-1]+i];
    
        // TODO: Mutex stuff
        for (int m = 0; m < architecture[depth-2]; m++) {
            pthread_mutex_lock(&mutexes[weight_layers[depth-2] + m * architecture[depth-1] + i]);
            write_weights[weight_layers[depth-2] + m * architecture[depth-1] + i] -=
                training_rate * activations[neuron_layers[depth-2] + m] * deltas[neuron_layers[depth-1]+i];
            pthread_mutex_unlock(&mutexes[weight_layers[depth-2] + m * architecture[depth-1] + i]);
        }
    }
    for (int i = depth - 2; i >= 0; i--) {
        for (int j = 0; j < architecture[i]; j++) {
            for (int k = 0; k < architecture[i+1]; k++) {
                deltas[neuron_layers[i]+j] += deltas[neuron_layers[i+1]+k] * read_weights[weight_layers[i] + 
                    architecture[i+1]*j+k];
            }
            if (i != 0) {
                deltas[neuron_layers[i]+j] *= d_f_activations[i-1](neuron_inputs[neuron_layers[i]+j] +
                        read_biases[neuron_layers[i]+j]);

                pthread_mutex_lock(&mutexes[num_weights + neuron_layers[i] + j]);
                write_biases[neuron_layers[i]+j] -= training_rate * deltas[neuron_layers[i]+j];
                pthread_mutex_unlock(&mutexes[num_weights + neuron_layers[i] + j]);
                
                // TODO: Mutex stuff
                for (int m = 0; m < architecture[i-1]; m++) {
                    pthread_mutex_lock(&mutexes[weight_layers[i-1] + m * architecture[i] + j]);
                    write_weights[weight_layers[i-1] + m * architecture[i] + j] -=
                        training_rate * activations[neuron_layers[i-1] + m] * deltas[neuron_layers[i]+j];
                    pthread_mutex_unlock(&mutexes[weight_layers[i-1] + m * architecture[i] + j]);
                }
            }
        }
    }
    for (int i = 0; i < architecture[depth-1]; i++) {
        error += f_cost(activations[neuron_layers[depth-1]+i], output[i]);
    }
    return error / architecture[depth-1];
}

void Network::train(double training_rate, int epochs, int batch_size, double **inputs, double **outputs, int n_inputs)
{
    double *y_hat;
    double error;
    for (int i = 0; i < epochs; i++) {
        error = 0;
        for (int j = 0; j < n_inputs; j++) {
            y_hat = prop(inputs[j]);
            for (int k = 0; k < architecture[depth - 1]; k++) {
                if (i == epochs - 1)
                    std::cout << y_hat[k] << ' ';
            }
            if (i == epochs - 1)
                std::cout << std::endl;

            error += back_prop(inputs[j], outputs[j], training_rate);
            if ((j+1) % batch_size == 0) {
                update();
            }
        }
        error /= n_inputs;
        if (i == epochs - 1)
            std::cout << "Epoch #" << i << " Error: " << error << std::endl;
    }
}

void Network::update()
{
    std::memcpy(read_data, write_data, sizeof(double) * (num_weights + num_neurons));
}
