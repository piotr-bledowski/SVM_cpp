//
// Created by piotr on 18/06/2023.
//

#ifndef SVM_CPP_SVM_H
#define SVM_CPP_SVM_H

#endif //SVM_CPP_SVM_H

#include <cstdlib> // for rand()
#include <vector>
#include <numeric>
//#include <random>

const double DELTA = 0.001;

class SVM {
private:
    std::vector<double> weights;
    double offset;
    double lambda;
    std::vector<std::vector<double>> supportVectors;
    bool plot;
public:
    SVM(bool p) {
        offset = 0.0;
        lambda = 0.0;
        plot = p;
    }

    // T - number of iterations
    void fit(std::vector<std::vector<double>> X, std::vector<int> y, int T, double lamb) {
        lambda = lamb;
        for (int j = 0; j < X.size(); j++) {
            weights.push_back(0.0);
        }

        // SGD
        for (int t = 1; t <= T; t++) {
            int i = rand() % X[0].size();
            std::vector<std::vector<double>> x;
            x.push_back({X[0][i], X[1][i]});
            std::vector<int> yy;
            yy.push_back(y[i]);
            std::vector<double> gradient = numericalGradient(x, yy);

            // update weights
            for (int j = 0; j < weights.size(); j++) {
                weights[j] -= learningRate(t) * gradient[j];
            }
            offset -= learningRate(t) * gradient[gradient.size() - 1];
        }

        // determine the support vectors
        if (plot) { // if plot is true, we assume the data is 2D
            double max_dist = 0.0;
            double min_dist = 0.0;

            for (int i = 0; i < y.size(); i++) {
                std::vector<double> x = {X[0][i], X[1][i]};
                if (hingeLoss(x, y[i], weights, offset) <= 1) {
                    supportVectors.push_back(x);
                    double dist = std::inner_product(x.begin(), x.end(), weights.begin(), 0.0);
                    if (dist < min_dist)
                        min_dist = dist;
                    if (dist > max_dist)
                        max_dist = dist;
                }
            }
        }

    }

    int predict(std::vector<double> x) {
        double product = std::inner_product(x.begin(), x.end(), weights.begin(), 0.0);
        if (product - offset > 0)
            return 1;
        return -1;
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
