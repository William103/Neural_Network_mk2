#include <cmath>

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
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
