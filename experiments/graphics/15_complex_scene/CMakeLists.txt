cmake_minimum_required(VERSION 3.24)

project(DepthResearch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)

FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG 2945a967c5303b2c8c14174117c45f3302591150 # v2.0.0rc13
)

FetchContent_MakeAvailable(glfw glm tinyobjloader)

find_library(MACOS_OPENGL_FRAMEWORK OpenGL REQUIRED)

add_library(glad STATIC
    external/glad/src/glad.c
)

target_include_directories(glad
    PUBLIC
        external/glad/include
)

# CPU OBJ loading and one-time GPU upload stay in one small module.
add_library(scene_mesh STATIC src/Mesh.cpp)
target_include_directories(scene_mesh PUBLIC include)
target_link_libraries(scene_mesh PUBLIC glad glm::glm PRIVATE tinyobjloader)

add_executable(DepthResearch src/main.cpp)

target_link_libraries(DepthResearch
    PRIVATE
        scene_mesh
        glfw
        glm::glm
        ${MACOS_OPENGL_FRAMEWORK}
)

include(CTest)
if(BUILD_TESTING)
    add_executable(mesh_loading_tests tests/mesh_loading.cpp)
    target_link_libraries(mesh_loading_tests PRIVATE scene_mesh)
    add_test(NAME mesh_loading COMMAND mesh_loading_tests)
endif()

target_compile_definitions(DepthResearch
    PRIVATE
        PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_command(TARGET DepthResearch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:DepthResearch>/shaders
    COMMENT "Copying shader files next to the executable"
)
