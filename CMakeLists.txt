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

# Dear ImGui: immediate-mode UI for the live parameter panel (Experiment 17).
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9
)

FetchContent_MakeAvailable(glfw glm tinyobjloader imgui)

find_library(MACOS_OPENGL_FRAMEWORK OpenGL REQUIRED)

add_library(glad STATIC
    external/glad/src/glad.c
)

target_include_directories(glad
    PUBLIC
        external/glad/include
)

# ImGui ships sources rather than a CMake target, so build the core + GLFW/OpenGL3 backends here.
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
# ImGui's OpenGL3 backend loads its own GL entry points, so it does not use the project's minimal glad stub.
target_link_libraries(imgui_lib PUBLIC glfw ${MACOS_OPENGL_FRAMEWORK})

# The shared scene format. Deliberately free of glad, glm and OpenGL so the
# geometry builder can be tested without a GL context, and so it stays a
# line-for-line twin of tools/scene/scene_loader.py.
add_library(scene_file STATIC src/SceneFile.cpp)
target_include_directories(scene_file PUBLIC include)

# CPU OBJ loading and one-time GPU upload stay in one small module.
add_library(scene_mesh STATIC src/Mesh.cpp)
target_include_directories(scene_mesh PUBLIC include)
target_link_libraries(scene_mesh PUBLIC glad glm::glm PRIVATE tinyobjloader)

add_executable(DepthResearch src/main.cpp)

target_link_libraries(DepthResearch
    PRIVATE
        scene_mesh
        scene_file
        imgui_lib
        glfw
        glm::glm
        ${MACOS_OPENGL_FRAMEWORK}
)

include(CTest)
if(BUILD_TESTING)
    add_executable(mesh_loading_tests tests/mesh_loading.cpp)
    target_link_libraries(mesh_loading_tests PRIVATE scene_mesh)
    add_test(NAME mesh_loading COMMAND mesh_loading_tests)
    add_executable(physical_camera_tests tests/physical_camera.cpp)
    target_include_directories(physical_camera_tests PRIVATE include)
    target_link_libraries(physical_camera_tests PRIVATE glm::glm)
    add_test(NAME physical_camera COMMAND physical_camera_tests)
    # Reads scene/alley.scene from the source tree and checks the geometry it
    # builds against numbers produced by the Python builder. Needs no GL context.
    add_executable(capture_name_dump tests/capture_name.cpp)
    target_include_directories(capture_name_dump PRIVATE include)
    add_executable(scene_dump tests/scene_dump.cpp)
    target_link_libraries(scene_dump PRIVATE scene_file)
    add_executable(scene_file_tests tests/scene_file.cpp)
    target_link_libraries(scene_file_tests PRIVATE scene_file)
    target_compile_definitions(scene_file_tests
        PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME scene_file COMMAND scene_file_tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME comparison_failures COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_comparison.py)
        add_test(NAME capture_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_capture_contract.py $<TARGET_FILE:capture_name_dump>)
        add_test(NAME loader_equivalence COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_loader_equivalence.py $<TARGET_FILE:scene_dump>)
        add_test(NAME reference_config COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reference_config.py)
        # The Python half of the same cross-language check: it also verifies that
        # the expected numbers written into tests/scene_file.cpp still match.
        add_test(NAME scene_file_python COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scene_file.py)
    endif()
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
