#ifndef NETWORK_H
#define NETWORK_H

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
     *      @param random_limit:
     *          The weights and biases will be chosen randomly from -random_limit/2 to random_limit/2
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
    Network(int *architecture, int depth, double random_limit, double (**f_activations)(double),
            double (**d_f_activations)(double), double (*f_cost)(double, double),
            double (*d_f_cost)(double, double));

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

private:
    // array of ints representing the structure of the network
    int *architecture;

    // the depth of the network, i.e. length of architecture
    int depth;

    // weights and biases generated in [-random_limit/2, random_limit/2]
    double random_limit;

    // the activation functions in an array
    double (**f_activations)(double);

    // the derivatives of the activation functions in an array
    double (**d_f_activations)(double);

    // a pointer to all the networks data: i.e. weights, biases, activations, and inputs
    double *data;

    // a pointer to all the delta data NOTE: will NOT be allocated by Network to facilitate multithreading
    double *delta_data;

    // a pointer to the start of the weights within data
    double *weights;

    // a pointer to the start of the biases within data
    double *biases;

    // a pointer to the start of the delta_weights within delta_data
    double *delta_weights;

    // a pointer to the start of the delta_biases within delta_data
    double *delta_biases;

    // an array of pointers to the starts of the layers within data
    double **layers;

    // a pointer to the start of the activations within data
    double *activations;

    // a pointer to the start of the inputs within data
    double *neuron_inputs;

    // the cost function
    double (*f_cost)(double, double);

    // the derivative of the cost function
    double (*d_f_cost)(double, double);
}

#endif
