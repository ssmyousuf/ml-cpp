add_library(regression SHARED LinearRegression.cpp LinearRegression.h)
target_include_directories(regression PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(regression PUBLIC Eigen3::Eigen)