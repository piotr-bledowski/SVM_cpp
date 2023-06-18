//
// Created by piotr on 18/06/2023.
//

#ifndef SVM_CPP_SVM_H
#define SVM_CPP_SVM_H

#endif //SVM_CPP_SVM_H

#include <vector>

class SVM {
private:
    std::vector<double> w;
    double b;
    double lambda;
public:
    SVM() {

    }

    void fit(std::vector<std::vector<double>> X, std::vector<int> y) {

    }

    int predict(std::vector<double> x) {

    }

    double objective(std::vector<std::vector<double>> X, std::vector<int> y) {
        int n = X.size();
        std::vector<double> losses;
        for (int i = 0; i < n; i++) {
            losses.push_back(hingeLoss(X[i], y[i]));
        }

        return 1.0/n * std::accumulate(losses.begin(), losses.end(), 0.0);
    }

    double hingeLoss(std::vector<double> x, int y) {
        double loss = 1 - y*(std::inner_product(w.begin(), w.end(), x.begin(), 0.0) - b);
        if (loss > 0)
            return loss;
        return 0;
    }
};
