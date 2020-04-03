#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include "utils.h"
#include "network.h"
#include "threads.h"
#include "barrier.h"
#include "mnist/mnist_reader.hpp"

void generate_inputs();

int num_threads = 4;
int epochs = 20;
int num_inputs = 60000;
int num_tests = 10000;
int batch_size = 1000;
int depth = 5;
int *architecture;
double random_limit = 5;
double (**f_activations)(double);
double (**d_f_activations)(double);
double (*f_cost)(double, double);
double (*d_f_cost)(double, double);
double training_rate = 0.1;
double **inputs;
double **outputs;
double **test_inputs;
double **test_outputs;

double *read_data;
double *write_data;

bool shuffle = true;

pthread_mutex_t *mutexes;
pthread_barrier_t barrier;

typedef double (*activation_pointer)(double);

int main() {
    std::srand(time(NULL));
    architecture = new int[depth] { 784, 25, 20, 10, 10 };
    f_activations = new activation_pointer[depth] { sigmoid, sigmoid, sigmoid, sigmoid };
    d_f_activations = new activation_pointer[depth] { d_sigmoid, d_sigmoid, d_sigmoid, d_sigmoid };
    f_cost = squared_error;
    d_f_cost = d_squared_error;

    int num_weights = 0;
    int num_neurons = architecture[0];
    for (int i = 1; i < depth; i++) {
        num_neurons += architecture[i];
        num_weights += architecture[i-1] * architecture[i];
    }

    generate_inputs();

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
    unsigned long correct = 0;
    for (int i = 0; i < num_tests; i++) {
        y_hat = net.prop(test_inputs[i]);
        std::cout << std::fixed << std::setprecision(4);
        //std::cout << "Inputs: ";
        /*
        for (int j = 0; j < architecture[0]; j++) {
            if (inputs[i][j] >= 0) std::cout << ' ';
            std::cout << inputs[i][j] << ' ';
        }
        */
        //std::cout << "\t\tPredicted Output: ";
        for (int j = 0; j < architecture[depth-1]; j++) {
            //std::cout << y_hat[j] << ' ';
        }
        //std::cout << "\t\tActual Output: ";
        for (int j = 0; j < architecture[depth-1]; j++) {
            //std::cout << outputs[i][j] << ' ';
        }
        //std::cout << "\t\t";
        double max = -100;
        int maxdex = -1;
        for (int j = 0; j < architecture[depth-1]; j++) {
            if (y_hat[j] > max) {
                max = y_hat[j];
                maxdex = j;
            }
        }
        if (test_outputs[i][maxdex]) correct++;
        //std::cout << std::endl;
    }
    std::cout << (double)correct / num_tests << " Accuracy" << std::endl;


    /* ----------------------------------------------------------------- */

    delete[] mutexes;
    delete[] architecture;
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

void generate_inputs() {
    inputs = new double*[num_inputs];
    outputs = new double*[num_inputs];
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    for (int i = 0; i < num_inputs; i++) {
        inputs[i] = new double[784];
        for (int j = 0; j < 784; j++) {
            inputs[i][j] = dataset.training_images[i][j] / 128.0 - 1;
        }
        outputs[i] = new double[10];
        for (int j = 0; j < 10; j++) outputs[i][j] = (double)((int)dataset.training_labels[i] == j);
    }

    test_inputs = new double*[10000];
    test_outputs = new double*[10000];
    for (int i = 0; i < 10000; i++) {
        test_inputs[i] = new double[784];
        for (int j = 0; j < 784; j++) {
            test_inputs[i][j] = dataset.test_images[i][j] / 128.0 - 1;
        }
        test_outputs[i] = new double[10];
        for (int j = 0; j < 10; j++) test_outputs[i][j] = (double)((int)dataset.test_labels[i] == j);
    }
    std::cout << "Done generating inputs" << std::endl;
}
