//
// Created by piotr on 18/06/2023.
//
#include "../include/matplotlibcpp.h"
#include <iostream>
#include "svm.h"
#include "../include/rapidcsv.h"

namespace plt = matplotlibcpp;

void plotLine(std::vector<double> x, std::vector<double> w, double intercept, std::string linestyle="-") {
    std::vector<double> x0 = {0, -intercept / w[1]};
    std::vector<double> y = std::vector<double>(x.size());
    y[0] = -(w[0] / w[1]) * (x[0] - x0[0]) - x0[1];
    y[1] = -(w[0] / w[1]) * (x[1] - x0[0]) - x0[1];
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

    plt::figure();

    plt::xlim(0, 25);
    plt::ylim(0, 25);

    plt::scatter(x1_positives, x2_positives, {{"c", "blue"}});
    plt::scatter(x1_negatives, x2_negatives, {{"c", "red"}});

    std::vector<double> w = model.getW();
    std::vector<double> x = {-1.0, 26.0};

    double b = model.getB();

    std::cout << w[0] << " " << w[1] << " " << b << "\n";

    plotLine(x, w, b);

    plt::show();
}

int main() {
    rapidcsv::Document doc("../data.csv");
    std::vector<int> target = doc.GetColumn<int>("y");
    size_t n_cols = doc.GetColumnCount();
    size_t n_features = n_cols - 1;
    size_t n_rows = doc.GetRowCount();

    std::vector<std::vector<double>> features = std::vector<std::vector<double>>(n_features);

    for (size_t i = 0; i < n_features; i++) {
        features[i] = doc.GetColumn<double>(i);
    }

    SVM model(true);
    model.fit(features, target, 1000, 0.5);

    plotSVM(model, features, target);

    return 0;
}