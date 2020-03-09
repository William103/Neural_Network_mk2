#include <stdio.h>
#include <iostream>
#include <ctime>
#include "utils.h"
#include "network.h"

const int num_threads;
const int epochs;
const int num_inputs;
const int batch_size;
const int depth;
const int architecture[depth];
const double (*(f_activations[depth-1]))(double, double);
const double (*(d_f_activations[depth-1]))(double, double);
const double (*f_cost)(double, double);
const double (*d_f_cost)(double, double);
const double training_rate;
const double inputs[num_inputs][architecture[0]];
const double outputs[num_inputs][architecture[depth-1]];

double *read_data;
double *write_data;

int main() {

}
