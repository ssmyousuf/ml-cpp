project(example_project)
cmake_minimum_required(VERSION 3.10)
add_executable(matplotlib-example  matplotlib-example.cpp)
target_include_directories(matplotlib-example PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/deps/matplotlib-cpp)
target_link_libraries(matplotlib-example PRIVATE ${Python3_LIBRARIES})

add_executable(simple_linear_regression simple_linear_regression.cpp)
target_include_directories(simple_linear_regression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/deps/matplotlib-cpp)
target_link_libraries(simple_linear_regression PRIVATE ${Python3_LIBRARIES})

add_executable(linear_reg_example linear_regression_example.cpp)
target_include_directories(linear_reg_example PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/deps/matplotlib-cpp ${CMAKE_SOURCE_DIR}/src/utilities)
target_link_libraries(linear_reg_example PRIVATE ${Python3_LIBRARIES} Eigen3::Eigen regression utilities)