#include <cmath>

double sigmoid(double x) {
    return 1 / (1 + std::exp((double) -x));
}

double d_sigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double squared_error(double y_hat, double y) {
    return (y_hat - y) * (y_hat - y);
}

double d_squared_error(double y_hat, double y) {
    return 2 * (y_hat - y);
}

double relu(double x) {
    return x < 0 ? 0 : x;
}

double d_relu(double x) {
    return x < 0 ? 0 : 1;
}
