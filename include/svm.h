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
};
