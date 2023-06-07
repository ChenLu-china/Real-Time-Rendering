How to run a .cu function in a .cpp file
run command flow this: 

g++ -c main.cpp 

nvcc - c softmax.cu 

g++ -o run main.o softmax.o -L /usr/include/cuda/lib64 -lcudart 

