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

FetchContent_MakeAvailable(glfw glm)

find_library(MACOS_OPENGL_FRAMEWORK OpenGL REQUIRED)

add_library(glad STATIC
    external/glad/src/glad.c
)

target_include_directories(glad
    PUBLIC
        external/glad/include
)

add_executable(DepthResearch
    src/main.cpp
)

target_link_libraries(DepthResearch
    PRIVATE
        glad
        glfw
        glm::glm
        ${MACOS_OPENGL_FRAMEWORK}
)

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

# A separate interactive scene keeps the numbered teaching snapshots intact.
find_library(MACOS_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(MACOS_IMAGEIO_FRAMEWORK ImageIO REQUIRED)
add_executable(DoFScene MACOSX_BUNDLE
    src/dof_scene/main.cpp
    src/dof_scene/scene.cpp
)
target_include_directories(DoFScene PRIVATE include)
target_link_libraries(DoFScene PRIVATE
    glfw glm::glm ${MACOS_OPENGL_FRAMEWORK}
    ${MACOS_COREGRAPHICS_FRAMEWORK} ${MACOS_IMAGEIO_FRAMEWORK}
)
target_compile_definitions(DoFScene PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(DoFScene PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(DoFScene PROPERTIES
    MACOSX_BUNDLE_BUNDLE_NAME "Cafe - Depth of Field"
    MACOSX_BUNDLE_GUI_IDENTIFIER "local.dof.cafe"
)
set_property(TARGET DoFScene APPEND PROPERTY LINK_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/scene.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/scene.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/depth.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/depth.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/screen.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene/screen.frag
)
add_custom_command(TARGET DoFScene POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dof_scene
        $<TARGET_FILE_DIR:DoFScene>/shaders/dof_scene
)

include(CTest)
if(BUILD_TESTING)
    add_executable(DoFSceneTests tests/dof_scene_tests.cpp src/dof_scene/scene.cpp)
    target_include_directories(DoFSceneTests PRIVATE include)
    target_link_libraries(DoFSceneTests PRIVATE glm::glm)
    target_compile_options(DoFSceneTests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME dof_scene_math_and_mesh COMMAND DoFSceneTests)
endif()
