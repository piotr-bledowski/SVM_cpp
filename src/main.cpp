//
// Created by piotr on 18/06/2023.
//
#include "../include/matplotlibcpp.h"
#include <iostream>
#include "svm.h"
#include "../include/rapidcsv.h"

namespace plt = matplotlibcpp;

void plotLine(std::vector<double> x, std::vector<double> w, double intercept, std::string linestyle="-") {
    std::vector<double> x0 = {0, intercept / w[1]};
    std::vector<double> y = std::vector<double>(x.size());
    y[0] = -(w[0] / w[1]) * (x[0] - x0[0]) + x0[1];
    y[1] = -(w[0] / w[1]) * (x[1] - x0[0]) + x0[1];
    plt::plot(x, y, {{"color", "black"}, {"linestyle", linestyle}});
}

void plotLine(std::vector<double> x, std::vector<double> w, std::vector<double> x0, std::string linestyle="-") {
    std::vector<double> y = std::vector<double>(x.size());
    y[0] = -(w[0] / w[1]) * (x[0] - x0[0]) + x0[1];
    y[1] = -(w[0] / w[1]) * (x[1] - x0[0]) + x0[1];
    plt::plot(x, y, {{"color", "black"}, {"linestyle", linestyle}});
}

void plotSVM(SVM model, std::vector<std::vector<double>> features, std::vector<int> target) {
    //std::cout << features[0][0] << " " << features[1][0] << " " << target[0];

    // Gotta separate the positives and negatives for plotting
    std::vector<double> x1_positives;
    std::vector<double> x2_positives;
    std::vector<double> x1_negatives;
    std::vector<double> x2_negatives;

    for (size_t i = 0; i < target.size(); i++) {
        if (target[i] == 1) {
            x1_positives.push_back(features[0][i]);
            x2_positives.push_back(features[1][i]);
        }
        else {
            x1_negatives.push_back(features[0][i]);
            x2_negatives.push_back(features[1][i]);
        }
    }

    int x1_lower_bound, x1_upper_bound, x2_lower_bound, x2_upper_bound;

    x1_lower_bound = (int) *std::min_element(features[0].begin(), features[0].end()) - 1;
    x1_upper_bound = (int) *std::max_element(features[0].begin(), features[0].end()) + 1;
    x2_lower_bound = (int) *std::min_element(features[1].begin(), features[1].end()) - 1;
    x2_upper_bound = (int) *std::max_element(features[1].begin(), features[1].end()) + 1;

    plt::figure();

    plt::xlim(x1_lower_bound, x1_upper_bound);
    plt::ylim(x2_lower_bound, x2_upper_bound);

    plt::scatter(x1_positives, x2_positives, {{"c", "blue"}});
    plt::scatter(x1_negatives, x2_negatives, {{"c", "red"}});

    std::vector<double> w = model.getW();
    std::vector<double> x = {(double) x1_lower_bound, (double) x1_upper_bound};

    double b = model.getB();

    std::cout << w[0] << " " << w[1] << " " << b << "\n";

    plotLine(x, w, b);

    plotLine(x, w, model.getMinSupport(), "--");
    plotLine(x, w, model.getMaxSupport(), "--");

    plt::show();
}

int main() {
    bool plot;
    int T;
    float lambda;
    float learning_rate_coefficient;
    std::string file_name;

    std::cout << "Provide the name of a file with training data: ";
    std::cin >> file_name;
    std::cout << "Do you want to plot the data and resulting classifier? (Only possible with exactly 2 features) [y/n]: ";
    std::string plot_response;
    std::cin >> plot_response;
    plot = plot_response == "y";
    std::cout << "Choose the learning rate coefficient: ";
    std::cin >> learning_rate_coefficient;
    std::cout << "Choose the lambda parameter (how much should the model minimize the margin): ";
    std::cin >> lambda;
    std::cout << "Provide the number of learning steps: ";
    std::cin >> T;

    rapidcsv::Document doc("../" + file_name);
    std::vector<int> target = doc.GetColumn<int>("y");
    size_t n_cols = doc.GetColumnCount();
    size_t n_features = n_cols - 1;
    size_t n_rows = doc.GetRowCount();

    std::vector<std::vector<double>> features = std::vector<std::vector<double>>(n_features);

    for (size_t i = 0; i < n_features; i++) {
        features[i] = doc.GetColumn<double>(i);
    }

    SVM model(plot);
    model.fit(features, target, T, lambda, learning_rate_coefficient);

    plotSVM(model, features, target);

    return 0;
}