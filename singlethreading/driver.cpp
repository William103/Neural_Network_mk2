#include <stdio.h>
#include <iostream>
#include <ctime>
#include "utils.h"
#include "network.h"

int main() {
    std::srand((unsigned) time(NULL));
    int architecture[] = {2, 5, 5, 1};
    int depth = 4;
    double   (*f_activations[])(double) = {&sigmoid,   &sigmoid,   &sigmoid};
    double (*d_f_activations[])(double) = {&d_sigmoid, &d_sigmoid, &d_sigmoid};
    double   (*f_cost)(double, double)  =   &squared_error;
    double (*d_f_cost)(double, double)  = &d_squared_error;
    double random_limit = 5;
    double training_rate = 0.1;
    int epochs = 3000;
    int batch_size = 4;
    int n_inputs = 4;
    int number_of_weights = 0;
    int num_neurons = architecture[0];
    for (int i = 1; i < depth; i++) {
        num_neurons += architecture[i];
        number_of_weights += architecture[i-1] * architecture[i];
    }
    double *read_data = new double[num_neurons + number_of_weights];
    double *write_data = new double[num_neurons + number_of_weights];

    double in_layer1[2] =  {1,1};
    double in_layer2[2] =  {0,1};
    double in_layer3[2] =  {1,0};
    double in_layer4[2] =  {0,0};
    double out_layer1[2] = {0};
    double out_layer2[2] = {1};
    double out_layer3[2] = {1};
    double out_layer4[2] = {0};
    double *inputs[4] = {in_layer1, in_layer2, in_layer3, in_layer4};
    double *outputs[4] = {out_layer1, out_layer2, out_layer3, out_layer4};

    for (int i = 0; i < num_neurons + number_of_weights; i++) {
        read_data[i] = write_data[i] = (((double) std::rand() / RAND_MAX) - 0.5) * random_limit;
    }

    Network net(architecture, depth, f_activations, d_f_activations, f_cost, d_f_cost, read_data, write_data);
    net.train(training_rate, epochs, batch_size, inputs, outputs, n_inputs);

    delete[] read_data;
    delete[] write_data;
}
