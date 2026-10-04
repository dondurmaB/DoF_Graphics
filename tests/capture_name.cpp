#include "CaptureName.h"
#include <iostream>
int main() {
    for (double lens : {50., 85., 35.5})
        for (double focus : {1.6, 2.5, 15.})
            for (double fstop : {1.2, 1.4, 22.})
                for (bool sharp : {false, true}) std::cout << captureTag(focus, fstop, lens, sharp) << '\n';
}
