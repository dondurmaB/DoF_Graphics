#pragma once

#include <array>
#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

namespace dof_scene {

struct HudVertex { glm::vec2 position; glm::vec4 color; };

// A small built-in bitmap alphabet keeps the native demo dependency-free.
inline const std::unordered_map<char, std::array<unsigned char, 7>>& alphabet() {
    static const std::unordered_map<char, std::array<unsigned char, 7>> glyphs = {
        {'A',{14,17,17,31,17,17,17}}, {'B',{30,17,17,30,17,17,30}},
        {'C',{14,17,16,16,16,17,14}}, {'D',{30,17,17,17,17,17,30}},
        {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
        {'G',{14,17,16,23,17,17,15}}, {'H',{17,17,17,31,17,17,17}},
        {'I',{14,4,4,4,4,4,14}}, {'J',{7,2,2,2,18,18,12}},
        {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
        {'M',{17,27,21,21,17,17,17}}, {'N',{17,25,25,21,19,19,17}},
        {'O',{14,17,17,17,17,17,14}}, {'P',{30,17,17,30,16,16,16}},
        {'Q',{14,17,17,17,21,18,13}}, {'R',{30,17,17,30,20,18,17}},
        {'S',{15,16,16,14,1,1,30}}, {'T',{31,4,4,4,4,4,4}},
        {'U',{17,17,17,17,17,17,14}}, {'V',{17,17,17,17,17,10,4}},
        {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
        {'Y',{17,17,10,4,4,4,4}}, {'Z',{31,1,2,4,8,16,31}},
        {'0',{14,17,19,21,25,17,14}}, {'1',{4,12,4,4,4,4,14}},
        {'2',{14,17,1,2,4,8,31}}, {'3',{30,1,1,14,1,1,30}},
        {'4',{2,6,10,18,31,2,2}}, {'5',{31,16,16,30,1,1,30}},
        {'6',{14,16,16,30,17,17,14}}, {'7',{31,1,2,4,8,8,8}},
        {'8',{14,17,17,14,17,17,14}}, {'9',{14,17,17,15,1,1,14}},
        {'.',{0,0,0,0,0,12,12}}, {':',{0,12,12,0,12,12,0}},
        {'-',{0,0,0,31,0,0,0}}, {'+',{0,4,4,31,4,4,0}},
        {'/',{1,1,2,4,8,16,16}}, {'[',{14,8,8,8,8,8,14}},
        {']',{14,2,2,2,2,2,14}}, {'(',{2,4,8,8,8,4,2}},
        {')',{8,4,2,2,2,4,8}}, {'=',{0,0,31,0,31,0,0}}
    };
    return glyphs;
}

struct Hud {
    std::vector<HudVertex> vertices;
    void rect(float x, float y, float w, float h, const glm::vec4& color) {
        vertices.insert(vertices.end(), {{{x,y},color},{{x+w,y},color},{{x+w,y+h},color},
                                        {{x,y},color},{{x+w,y+h},color},{{x,y+h},color}});
    }
    void text(float x, float y, const std::string& value, float size, const glm::vec4& color) {
        for (unsigned char raw : value) {
            auto it = alphabet().find(static_cast<char>(std::toupper(raw)));
            if (it != alphabet().end()) {
                for (int row = 0; row < 7; ++row)
                    for (int col = 0; col < 5; ++col)
                        if (it->second[row] & (1 << (4-col))) rect(x+col*size, y+row*size, size, size, color);
            }
            x += 6*size;
        }
    }
};

} // namespace dof_scene
