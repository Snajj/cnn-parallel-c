main.o: main.c
	gcc -c -g -pg -fopenmp main.c -o main.o

cnn_layer.o: cnn_layer.c cnn_layer.h
	gcc -c -g -pg -fopenmp cnn_layer.c -o cnn_layer.o

all: main.o cnn_layer.o
	gcc -g -pg -fopenmp main.o cnn_layer.o -o App -lm

clean:
	rm -f main.o cnn_layer.o App Output_kernel1.txt Output_kernel2.txt Output_kernel3.txt
