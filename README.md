# CUDA Sequence Similarity Analysis

This project implements a CUDA-based parallel computing program to analyze the similarity between biological sequences. It is optimized for high-performance computation using GPU parallelization, enabling the efficient processing of large-scale sequence datasets.

## Features

- **Fast Sequence Similarity Calculation**: Compares biological sequences in parallel using CUDA kernels, achieving high speed for large datasets.
- **Configurable Similarity Threshold**: Allows setting a similarity cutoff to determine significant matches.
- **Output of Neighbor Counts**: Outputs the number of sequences similar to each sequence in the dataset.
- **Error Handling**: Includes robust CUDA error checking to ensure reliable execution.

## Prerequisites

- **CUDA Toolkit**: Ensure CUDA is installed on your system.
- **C++ Compiler**: A compiler like `g++` capable of compiling CUDA code.
- **GPU Hardware**: A CUDA-capable GPU.

## Input Format

The program expects a FASTA-like input file containing sequences:

```
>Sequence1
ACTGACTGACTG...
>Sequence2
ACTGACTGACTG...
...
```

Each sequence should be on a single line, preceded by a header line starting with `>`.

## How It Works

1. **Load Sequence Data**: Reads the input file and stores sequences in memory.
2. **Memory Allocation**: Allocates host and device memory for sequences and results.
3. **Parallel Comparison**: Uses CUDA threads to compare each sequence against all others, counting the number of similar sequences.
4. **Results Output**: Writes the neighbor counts for each sequence to a CSV file.

## Compilation

Compile the program using the following command:

```bash
g++ -o sequence_similarity sequence_similarity.cu -lcudart -lcuda
```

Ensure that the CUDA libraries are correctly linked and available in your system path.

## Usage

Run the compiled program with an input file:

```bash
./sequence_similarity input.fasta
```

### Output

The program generates a file named `dbg.csv` containing two columns:
- **Sequence ID**: The identifier of each sequence.
- **Neighbor Count**: The number of sequences similar to the given sequence based on the threshold.

## Parameters

- **`num_blocks`**: Number of CUDA blocks (default: 64).
- **`num_threads`**: Number of threads per block (default: 32).
- **`cutoff`**: Similarity threshold for determining neighbors (default: 0.97).

## Example Output

Input:
```
>Sequence1
ACTGACTGACTG
>Sequence2
ACTGACTGACGG
>Sequence3
ACTGACTGACTT
```

Output (`dbg.csv`):
```
Sequence1,2
Sequence2,1
Sequence3,1
```

## Troubleshooting

- **CUDA Errors**: If any CUDA-related error occurs, it will be printed to the console with the file and line number for debugging.
- **Memory Issues**: Ensure sufficient GPU memory for large datasets.

## License

This project is open-source and available under the MIT License.

