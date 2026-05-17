cmake_minimum_required(VERSION 3.22.1)

project(gilbeot_litertlm_lora LANGUAGES CXX)

add_library(gilbeot_litertlm_lora SHARED gilbeot_litertlm_lora_shim.cpp)

target_compile_features(gilbeot_litertlm_lora PRIVATE cxx_std_17)

target_link_libraries(gilbeot_litertlm_lora PRIVATE android log dl)
