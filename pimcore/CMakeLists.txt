cmake_minimum_required(VERSION 3.16)
project(pimcore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(pimcore_lib
  src/types.cpp
  src/config.cpp
  src/timing.cpp
  src/substrates/lpddr5x_pim.cpp
  src/substrates/hbm_pim.cpp
  src/substrates/gddr6_aim.cpp
  src/bank.cpp
  src/channel.cpp
  src/trace.cpp
  src/kernels.cpp
  src/stats.cpp
  src/sim.cpp
  src/address.cpp
  src/controller.cpp
  src/traffic.cpp
  src/device.cpp
  src/syscore.cpp
  src/serving.cpp
)
target_include_directories(pimcore_lib PUBLIC include)

add_executable(pimcore src/main.cpp)
target_link_libraries(pimcore pimcore_lib)

add_executable(pimcore_coexec src/coexec_main.cpp)
target_link_libraries(pimcore_coexec pimcore_lib)

add_executable(pimcore_sys src/sys_main.cpp)
target_link_libraries(pimcore_sys pimcore_lib)

add_executable(pimcore_serve src/serve_main.cpp)
target_link_libraries(pimcore_serve pimcore_lib)

add_executable(pimcore_tracegen src/tracegen_main.cpp)
target_link_libraries(pimcore_tracegen pimcore_lib)

enable_testing()
add_executable(pimcore_tests tests/test_pimcore.cpp)
target_link_libraries(pimcore_tests pimcore_lib)
add_test(NAME pimcore_tests COMMAND pimcore_tests)
