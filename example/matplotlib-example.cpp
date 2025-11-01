//
// Created by ml on 11/1/25.
//

#include <matplotlibcpp.h>
#include <vector>
#include <cmath>

namespace plt = matplotlibcpp;

int main() {
    std::vector<double> x, y;
    for (double i = 0; i < 2 * M_PI; i += 0.1) {
        x.push_back(i);
        y.push_back(std::sin(i));
    }

    plt::plot(x, y);
    plt::title("Sine Wave");
    plt::xlabel("x");
    plt::ylabel("sin(x)");
    plt::show();
}
