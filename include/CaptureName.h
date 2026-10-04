#pragma once
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

// Python twin: tools/raytraced_reference/render_dof.py capture_tag(). The
// executable contract test exercises both, including sharp and nondefault lens.
inline std::string captureTag(double focus, double fstop, double lens, bool sharp) {
    std::ostringstream out;
    out << std::setprecision(6) << "focus" << focus << "m_f" << fstop;
    if (std::abs(lens - 50.0) > 0.01) out << "_" << lens << "mm";
    if (sharp) out << "_sharp";
    return out.str();
}
