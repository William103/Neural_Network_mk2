#include <stdio.h>
#include <iostream>
#include <ctime>
#include <iomanip>
#include "utils.h"
#include "network.h"

void generate_inputs();

#define NUM_INPUTS 10000

double **inputs, **outputs;

int main() {
    std::srand((unsigned) time(NULL));
    int architecture[] = {2, 10, 10, 10, 2};
    int depth = 5;
    double   (*f_activations[])(double) = {sigmoid,   sigmoid,   sigmoid,   sigmoid};
    double (*d_f_activations[])(double) = {d_sigmoid, d_sigmoid, d_sigmoid, d_sigmoid};
    double   (*f_cost)(double, double)  =   squared_error;
    double (*d_f_cost)(double, double)  = d_squared_error;
    double random_limit = 5;
    double training_rate = 0.1;
    int epochs = 100;
    int batch_size = 50;
    int num_inputs = NUM_INPUTS;
    int number_of_weights = 0;
    int num_neurons = architecture[0];
    generate_inputs();
    for (int i = 1; i < depth; i++) {
        num_neurons += architecture[i];
        number_of_weights += architecture[i-1] * architecture[i];
    }
    double *read_data = new double[num_neurons + number_of_weights];
    double *write_data = new double[num_neurons + number_of_weights];

    for (int i = 0; i < num_neurons + number_of_weights; i++) {
        read_data[i] = write_data[i] = (((double) std::rand() / RAND_MAX) - 0.5) * random_limit;
    }

    Network net(architecture, depth, f_activations, d_f_activations, f_cost, d_f_cost, read_data, write_data);
    net.train(training_rate, epochs, batch_size, inputs, outputs, num_inputs);

    double *y_hat;
    unsigned long correct = 0;
    for (int i = 0; i < NUM_INPUTS; i++) {
        y_hat = net.prop(inputs[i]);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Inputs: ";
        for (int j = 0; j < architecture[0]; j++) {
            if (inputs[i][j] >= 0) std::cout << ' ';
            std::cout << inputs[i][j] << ' ';
        }
        std::cout << "\t\tPredicted Output: ";
        for (int j = 0; j < architecture[depth-1]; j++) {
            std::cout << y_hat[j] << ' ';
        }
        std::cout << "\t\tActual Output: ";
        for (int j = 0; j < architecture[depth-1]; j++) {
            std::cout << outputs[i][j] << ' ';
        }
        std::cout << "\t\t";
        for (int j = 0; j < architecture[depth-1]; j++) {
            if (outputs[i][j] > 0.5 && y_hat[j] > 0.5 || outputs[i][j] < 0.5 && y_hat[j] < 0.5) {
                std::cout << "  correct ";
                correct++;
            } else
                std::cout << "incorrect ";
        }
        std::cout << std::endl;
    }
    std::cout << (double)correct / num_inputs * 50 << "% Correct" << std::endl;

    delete[] read_data;
    delete[] write_data;
}

void generate_inputs() {
    inputs = new double*[NUM_INPUTS];
    outputs = new double*[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++) {
        inputs[i] = new double[2];
        outputs[i] = new double[2];
        inputs[i][0] = std::rand() / (double) RAND_MAX * 2 - 1;
        inputs[i][1] = std::rand() / (double) RAND_MAX * 2 - 1;
        if (inputs[i][0] * inputs[i][0] + inputs[i][1] * inputs[i][1] < 1) {
            outputs[i][0] = 1;
        } else outputs[i][0] = 0;
        if (inputs[i][0] < 0.5 && inputs[i][0] > -0.5 && inputs[i][1] < 0.5 && inputs[i][1] > -0.5) {
            outputs[i][1] = 1;
        } else outputs[i][1] = 0;
    }
}
