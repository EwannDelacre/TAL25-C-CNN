cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tflm_classifier)

target_sources(app PRIVATE src/main.c src/infer.cpp src/model_data.cc)
