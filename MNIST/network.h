#ifndef NETWORK_H
#define NETWORK_H

#include "threads.h"

/*
 * A class modelling a simple feed forward neural network
 */
class Network
{
public:

    /*
     * The constructor
     *      @param architecture:
     *          An array of ints describing the number of neurons per layer
     *      @param depth:
     *          The depth of the network, also the length of architecture
     *      @param f_activations:
     *          An array of length depth-1 of function pointers to functions of
     *          doubles returning doubles (the activation functions)
     *      @param d_f_activations:
     *          Similar to f_activations but the derivative of the activation function of each layer
     *      @param f_cost:
     *          A pointer to the cost function to be used
     *      @param d_f_cost:
     *          A pointer to the derivative of the cost function to be used
     */
    Network(int *architecture, int depth, double (**f_activations)(double),
            double (**d_f_activations)(double), double (*f_cost)(double, double),
            double (*d_f_cost)(double, double), double *read_data, double *write_data);

    /*
     * The deconstructor: does typical deconstructor things, i.e. clears up dynamic data
     */
    ~Network();

    /*
     * prop: does forward propagation for a given input
     *      @param input:
     *          An array of doubles; the input
     *      @returns: 
     *          An array of doubles; the output
     */
    double *prop(double *input);

    /*
     * back_prop: does backpropagation based on error signals accumulated during prop
     *      @param input:
     *          An array of doubles; the input
     *      @param output:
     *          The expected output
     *      @param training_rate:
     *          The calculated gradient gets multiplied by this parameter
     *      @returns:
     *          The total average error of the network for this input-output pair
     */
    double back_prop(double *input, double *output, double training_rate);

    // returns final error
    void train(double training_rate, int epochs, int batch_size, double **inputs, double **outputs, int n_inputs);

    void update();

private:
    // array of ints representing the structure of the network
    int *architecture;

    // the depth of the network, i.e. length of architecture
    int depth;

    // the activation functions in an array
    double (**f_activations)(double);

    // the derivative of the activation functions
    double (**d_f_activations)(double);

    // the pre-allocated data to be read from
    double *read_data;

    // the pre-allocated data to be written to
    double *write_data;

    double *read_weights;

    double *read_biases;

    double *write_weights;

    double *write_biases;

    double **layers;

    double *activations;

    double *neuron_inputs;

    double *deltas;

    // the cost function
    double (*f_cost)(double, double);

    // the derivative of the cost function
    double (*d_f_cost)(double, double);

    int *weight_layers;
    int *neuron_layers;
    int num_neurons;
    int num_weights;

};

#endif
