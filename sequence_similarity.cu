#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <limits>
#include <vector>
#include <time.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void testKernel(char* a, int* neig, int* itr, int* seqSize){
	int num_str = 64 * 32;
	int txt_len = 51;
	int seqidx = blockIdx.x * blockDim.x + threadIdx.x + *itr*num_str;
	for (int i = 0; i < *seqSize; i++){		
		int matchChar = 0;
		for (int j = 0; j < txt_len; j++){
			int idx = i*txt_len + j;
			int tgtidx = seqidx*txt_len + j;
			if (a[tgtidx] == a[idx]) matchChar++;
		}
		if ((float)matchChar / txt_len >= 0.97) neig[seqidx]++;
	}
}

void reportCudaDevice(){
	int deviceCount;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	cudaGetDeviceProperties(&deviceProp,0);
	cout << deviceProp.name << endl;
	cout << deviceProp.maxThreadsPerBlock << endl;
}


int main(int argc, char* argv[]){
	clock_t t1, t2, tc, ti; //For measure computing time.
	t1 = clock();
	unsigned const int num_blocks = 64;
	unsigned const int num_threads = 32;
	int txt_len = 51;
	const int num_str = num_blocks * num_threads;

	dim3 grid(num_blocks, 1, 1);
	dim3 threads(num_threads, 1, 1);

	vector<string> ID;
	vector<string> seq;
	ifstream fp;
	string line;
	float cutoff = 0.97;
	fp.open(argv[1], ios::in);
	if (!fp){
		std::cout << "fail to open " << argv[1] << std::endl;
		exit(EXIT_FAILURE);
	}
	while (getline(fp, line)){
		if (line.at(0) == '>'){
			ID.push_back(line.substr(1));
		}
		else{
			seq.push_back(line);
		}
	}
	fp.close();
	txt_len = seq.at(0).length();
	std::cout << "seq number = " << seq.size() << std::endl;	
	std::cout << "seq length = " << txt_len << std::endl;

	char *a;
	int *neig;
	char *d_a;
	int *d_neig;
	a = (char *)malloc(seq.size()*txt_len*sizeof(char));
	neig = (int *)malloc(seq.size()*sizeof(int));

	for (int i = 0; i < seq.size(); i++){
		neig[i] = 0;
		for (int j = 0; j < txt_len; j++){
			int idx = i*txt_len + j;
			a[idx] = seq.at(i).at(j);
		}
	}

	gpuErrchk(cudaMalloc((void**)&d_a, seq.size()*txt_len*sizeof(char)));
	gpuErrchk(cudaMemcpy(d_a, a, seq.size()*txt_len*sizeof(char), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_neig, seq.size()*sizeof(int)));
	gpuErrchk(cudaMemcpy(d_neig, neig, seq.size()*sizeof(int), cudaMemcpyHostToDevice));
	int *d_num_str;
	int *d_txt_len;
	int *d_num_threads;
	float *d_cutoff;
	gpuErrchk(cudaMalloc((void**)&d_num_str, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_txt_len, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_num_threads, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_cutoff, sizeof(float)));
	gpuErrchk(cudaMemcpy(d_cutoff, &cutoff, sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_txt_len, &txt_len, sizeof(int), cudaMemcpyHostToDevice));
	int itr = 0;
	int *d_itr;
	cudaMalloc((void**)&d_itr, sizeof(int));
	int iterMax = seq.size() / num_str +1;
	int seqSize = seq.size();
	int *d_seqSize;
	cudaMalloc( (void**)&d_seqSize, sizeof(int));
	cudaMemcpy( d_seqSize, &seqSize, sizeof(int), cudaMemcpyHostToDevice);
	tc = clock();	
	for (itr = 0; itr <= iterMax; itr++){
		cudaMemcpy(d_itr, &itr, sizeof(int), cudaMemcpyHostToDevice);
		testKernel << <grid, threads >> >(d_a, d_neig, d_itr, d_seqSize);
		cudaDeviceSynchronize();
		ti = clock();
		float diff((float)ti - (float)tc);
		float seconds = diff / CLOCKS_PER_SEC;
		cout << "itr=" << itr << ",num=" << itr*num_str << "/" << seq.size() << ",time=" << seconds << endl;
	}

	cudaMemcpy(d_itr, &itr, sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	printf("Cuda status: %s\n", cudaGetErrorString(cudaGetLastError()));
	gpuErrchk(cudaMemcpy(neig, d_neig, seq.size()*sizeof(int), cudaMemcpyDeviceToHost));
	ofstream myfile;
	myfile.open("dbg.csv");

	for (int i = 0; i < seq.size(); i++){
		myfile << ID.at(i) << ',' << neig[i] << endl;
	}
	myfile.close();
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_neig));
	t2 = clock();
	float diff((float)t2 - (float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	cout << "time=" << seconds << endl;
	return EXIT_SUCCESS;
}
