//
// Created by piotr on 18/06/2023.
//

#ifndef SVM_CPP_SVM_H
#define SVM_CPP_SVM_H

#endif //SVM_CPP_SVM_H

#include <vector>
#include <random>

const double DELTA = 0.001;

class SVM {
private:
    std::vector<double> weights;
    double offset;
    double lambda;
public:
    SVM() {
        offset = 0;
        lambda = 0;
    }

    // T - number of iterations
    void fit(std::vector<std::vector<double>> X, std::vector<int> y, int T, double lamb) {
        // SGD
        lambda = lamb;

        for (int t = 1; t <= T; t++) {

        }

    }

    int predict(std::vector<double> x) {

    }

    double learningRate(int t) {
        return 1.0 / t;
    }

    std::vector<double> numericalGradient(std::vector<std::vector<double>> X, std::vector<int> y) {
        std::vector<double> gradient;
        for (int i = 0; i <= weights.size(); i++) {
            gradient.push_back(numericalPartialDerivative(X, y, i));
        }
        return gradient;
    }

    double numericalPartialDerivative(std::vector<std::vector<double>> X, std::vector<int> y, size_t i) {
        if (i < weights.size()) {
            std::vector<double> w1 = weights;
            std::vector<double> w2 = weights;

            w1[i] += DELTA;
            w2[i] -= DELTA;

            return (objective(X, y, w1, offset) - objective(X, y, w2, offset)) / (2 * DELTA);
        }

        double b = offset;

        return (objective(X, y, weights, b + DELTA) - objective(X, y, weights, b - DELTA)) / (2 * DELTA);
    }

    double objective(std::vector<std::vector<double>> X, std::vector<int> y, std::vector<double> w, double b) {
        int n = X.size();
        std::vector<double> losses;
        for (int i = 0; i < n; i++) {
            losses.push_back(hingeLoss(X[i], y[i], w, b));
        }

        std::vector<double> squared;

        for (double & i : w) {
            squared.push_back(i * i);
        }

        double regularizer = lambda * std::accumulate(squared.begin(), squared.end(), 0.0);

        return 1.0/n * std::accumulate(losses.begin(), losses.end(), 0.0) + regularizer;
    }

    double hingeLoss(std::vector<double> x, int y, std::vector<double> w, double b) {
        double loss = 1 - y*(std::inner_product(w.begin(), w.end(), x.begin(), 0.0) - b);
        if (loss > 0)
            return loss;
        return 0;
    }
};
