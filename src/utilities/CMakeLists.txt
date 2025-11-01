add_library(utilities SHARED utils.cpp utils.h metrics.cpp metrics.h)
target_include_directories(utilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utilities PUBLIC Eigen3::Eigen)