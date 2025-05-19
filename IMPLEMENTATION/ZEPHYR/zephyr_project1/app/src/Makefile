# Compilateurs
CC = gcc
CXX = g++

# Options de compilation
CFLAGS = -Wall -Wextra -std=c11
CXXFLAGS = -Wall -Wextra -std=c++17

# Spécifie où chercher les headers
INCLUDES = -I. \
           -I/home/ewann/git/tflite-micro/ \
           -I/home/ewann/git/tflite-micro/tensorflow \
           -I/home/ewann/git/tflite-micro/tensorflow/lite \
           -I/home/ewann/git/tflite-micro/tensorflow/lite/micro \
           -I/home/ewann/git/tflite-micro/tensorflow/lite/kernels \
           -I/home/ewann/git/tflite-micro/tensorflow/lite/c \
           -I/home/ewann/git/gemmlowp

# Liste des fichiers sources
SRCS_C = main.c
SRCS_CPP = infer.cpp model_data.cc

# Objets générés
OBJS = $(SRCS_C:.c=.o) $(SRCS_CPP:.cpp=.o)

# Fichiers TensorFlow Lite Micro nécessaires au linkage
TFLM_SRCS = \
    /home/ewann/git/tflite-micro/tensorflow/lite/micro/micro_error_reporter.cc \
    /home/ewann/git/tflite-micro/tensorflow/lite/micro/micro_allocator.cc \
    /home/ewann/git/tflite-micro/tensorflow/lite/micro/micro_interpreter.cc \
    /home/ewann/git/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.cc \
    /home/ewann/git/tflite-micro/tensorflow/lite/micro/kernels/kernel_util.cc

# Cible principale
all: infer_demo

infer_demo: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(TFLM_SRCS)

# Règles de compilation
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Nettoyage
clean:
	rm -f *.o infer_demo
