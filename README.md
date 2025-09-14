# MP1
CS521 Fall 2025, Machine Project 1

### Compiling Part 1 (CPU) on Apple devices:
Notes: 
1) Please pass the -O3 flag (in your cmake) only for testing the last optimization as also suggested in the mp1 handout.
2) You should see "The CXX compiler identification is Clang" during cmake .. command.
3) If you are running on Apple silicon (ARM 64) CPU, refer to below objdump command for inspecting NEON vector instuctions. 

```sh
brew install cmake llvm libomp
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
cd mp1
mkdir build && cd build
cmake ..
$CXX -march=native -ffast-math -fopenmp ../cpu/gemm_cpu.cpp -L/opt/homebrew/opt/llvm/lib -o mp1_cpu
objdump -d ./mp1_cpu | grep -E "ld1|st1|fmla|fadd|fmul|v[0-9]"
```

### Running Part 2 (GPU) on Google Colab:
In Colab's terminal
```sh
git clone https://github.com/CS521-ml-compilers/mp1.git
cd mp1
mkdir build && cd build
cmake ..
make
```
