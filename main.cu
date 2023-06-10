#include <iostream>
#include <math.h>
#include <stdint.h>
#include <chrono>

// compile command
// nvcc .\main.cu -Xcompiler "/EHsc /O2" -O3 -o main      
// nvcc .\main.cu -Xcompiler "/EHsc /O2" -O3 -I "C:\Users\camku\Documents\vscode\cuda\gol_cpp\SDL2\include" -L "C:\Users\camku\Documents\vscode\cuda\gol_cpp\SDL2\lib" -lmingw32 -lSDL2main -lSDL2 -o main

#define width 1024
#define height 1024

#define rwidth (width+2)
#define read_offset rwidth+1

#define use_2x2_compute

uint8_t *sim_host;
uint8_t *sim_device_0;
uint8_t *sim_device_1;

cudaPitchedPtr cudaPtr_0;
cudaPitchedPtr cudaPtr_1;

#define get_sim_position(s, x, y, b) ((s[(x+1)*rwidth + y+1]))
#define set_sim_position(s, x, y, b, v) s[(x+1)*rwidth + y+1] = (s[(x+1)*rwidth + (y+1)] & ~ (1 << b)) | (v << b)

void check_cuda_call_impl(const cudaError_t err, const char* fileName, const int lineNumber) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\n", fileName, lineNumber, cudaGetErrorString(err));
		exit(1);
	}
}

#define CHECK_LAST_CUDA_CALL() check_cuda_call_impl(cudaGetLastError(), __FILE__, __LINE__)

void fill_sim_rand(float prob = 0.5) {

    for ( int i = 0; i < width+2; i++ ) {

        if ( i == 0 || i == (width+1) ) { 
            for ( int j = 0; j < height+2; j++ ) {
                // sim_host[(rwidth*i)+j] = 0xff;
                sim_host[(rwidth*i)+j] = 0x00;
            }
        }

        else {

            for ( int j = 0; j < height+2; j++ ) {

                sim_host[(width*i) + j] = (float(rand())/RAND_MAX)>prob;
                // sim_host[(rwidth*i) + j] = 0;

                if ( j == 0 || j == height+1 ) { 
                    // sim_host[(rwidth*i) + j] = 0xff; 
                    sim_host[(rwidth*i) + j] = 0x00; 
                }
                
            }

        }
    }

}

// bit_idx is the bit index that the kernel will read from to compute the next iteration, which is stored at index bit_idx+bit_dir 
__global__ void cuda_GOL_kernel(int sim_width, int sim_height, int bit_idx, int bit_dir, cudaPitchedPtr sim)
{

    #define _get_sim_position(s, x, y, b) ((((uint8_t*)s.ptr)[x*rwidth + y + read_offset]>>b)&1)
    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    int row_0 = ( row - 1 );
    int col_0 = ( col - 1 );

    int row_1 = ( row + 1 );
    int col_1 = ( col + 1 );

    //find all active adjacent cells
    int n_adjacent_cells = 0;

    n_adjacent_cells += _get_sim_position(sim, row_0, col_0, bit_idx);
    n_adjacent_cells += _get_sim_position(sim, row_0, col,   bit_idx);
    n_adjacent_cells += _get_sim_position(sim, row_0, col_1, bit_idx);

    n_adjacent_cells += _get_sim_position(sim, row, col_0, bit_idx);
    n_adjacent_cells += _get_sim_position(sim, row, col_1, bit_idx);

    n_adjacent_cells += _get_sim_position(sim, row_1, col_0, bit_idx);
    n_adjacent_cells += _get_sim_position(sim, row_1, col,   bit_idx);
    n_adjacent_cells += _get_sim_position(sim, row_1, col_1, bit_idx);

    // calculate state of next cell
    int lives = 0;

    int raw_cell = ((uint8_t*)sim.ptr)[(row_1)*rwidth + col_1];
    int cell_sts = (raw_cell>>bit_idx)&1;

    lives += ( (n_adjacent_cells >= 2) * (n_adjacent_cells <= 3) ) * cell_sts;
    lives -= ( n_adjacent_cells == 3 ) * ( cell_sts - 1 );

    // set value for next iteration
    ((uint8_t*)sim.ptr)[(row_1)*rwidth + col_1] = (raw_cell & ~(1<<(bit_idx+bit_dir))) | lives<<(bit_idx+bit_dir);

}

__global__ void cuda_GOL_kernel_2x2(int sim_width, int sim_height, int bit_idx, int bit_dir, cudaPitchedPtr sim)
{

    #define _get_sim_position(s, x, y, b) ((((uint8_t*)s.ptr)[x*rwidth + y + read_offset]>>b)&1)
    #define _get_sim_position_raw(s, x, y) (((uint8_t*)s.ptr)[x*rwidth + y + read_offset])
    
    uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	uint32_t col = (blockIdx.y * blockDim.y + threadIdx.y)*2;

    // cell layout
    // 0 1
    // 2 3

    // read layout

    // a b c d
    // e 0 1 f
    // g 2 3 h
    // i j k l

    // adjacency formula

    // c0 = a b c e 1 g 2 3
    // c1 = b c d 0 f 2 3 h
    // c2 = e 0 1 g 3 i j k
    // c3 = 0 1 f 2 h j k l

    int cell_0_adj = 0;
    int cell_1_adj = 0;
    int cell_2_adj = 0;
    int cell_3_adj = 0;

    uint8_t cell_0_val_raw =  _get_sim_position_raw(sim, row,   col);
    uint8_t cell_1_val_raw =  _get_sim_position_raw(sim, row,   col+1);
    uint8_t cell_2_val_raw =  _get_sim_position_raw(sim, row+1, col);
    uint8_t cell_3_val_raw =  _get_sim_position_raw(sim, row+1, col+1);

    uint8_t cell_0_val = (cell_0_val_raw>>bit_idx)&1;
    uint8_t cell_1_val = (cell_1_val_raw>>bit_idx)&1;
    uint8_t cell_2_val = (cell_2_val_raw>>bit_idx)&1;
    uint8_t cell_3_val = (cell_3_val_raw>>bit_idx)&1;
    
    uint8_t cell_tmp;

    // ======================================================================
    // total adjacency values

    // row 0

    // a
    cell_tmp = _get_sim_position(sim, row-1, col-1, bit_idx);
    cell_0_adj += cell_tmp;

    // b
    cell_tmp = _get_sim_position(sim, row-1, col, bit_idx);
    cell_0_adj += cell_tmp;
    cell_1_adj += cell_tmp;

    // c
    cell_tmp = _get_sim_position(sim, row-1, col+1, bit_idx);
    cell_0_adj += cell_tmp;
    cell_1_adj += cell_tmp;

    // d
    cell_tmp = _get_sim_position(sim, row-1, col+2, bit_idx);
    cell_1_adj += cell_tmp;

    // ===================================
    // row 1

    // e
    cell_tmp = _get_sim_position(sim, row, col-1, bit_idx);
    cell_0_adj += cell_tmp;
    cell_2_adj += cell_tmp;
    
    // c0
    cell_1_adj += cell_0_val;
    cell_2_adj += cell_0_val;
    cell_3_adj += cell_0_val;

    // c1
    cell_0_adj += cell_0_val;
    cell_2_adj += cell_0_val;
    cell_3_adj += cell_0_val;

    // f
    cell_tmp = _get_sim_position(sim, row, col+2, bit_idx);
    cell_1_adj += cell_tmp;
    cell_3_adj += cell_tmp;

    // ===================================
    // row 2

    // g
    cell_tmp = _get_sim_position(sim, row, col-1, bit_idx);
    cell_0_adj += cell_tmp;
    cell_2_adj += cell_tmp;

    // c2
    cell_0_adj += cell_tmp;
    cell_1_adj += cell_tmp;
    cell_3_adj += cell_tmp;

    // c3
    cell_0_adj += cell_tmp;
    cell_1_adj += cell_tmp;
    cell_2_adj += cell_tmp;

    // h
    cell_tmp = _get_sim_position(sim, row, col+2, bit_idx);
    cell_1_adj += cell_tmp;
    cell_3_adj += cell_tmp;

    // ===================================
    // row 3

    // i
    cell_tmp = _get_sim_position(sim, row-1, col-1, bit_idx);
    cell_2_adj += cell_tmp;

    // j
    cell_tmp = _get_sim_position(sim, row-1, col, bit_idx);
    cell_2_adj += cell_tmp;
    cell_3_adj += cell_tmp;

    // k
    cell_tmp = _get_sim_position(sim, row-1, col+1, bit_idx);
    cell_2_adj += cell_tmp;
    cell_3_adj += cell_tmp;

    // l
    cell_tmp = _get_sim_position(sim, row-1, col+2, bit_idx);
    cell_3_adj += cell_tmp;

    // ======================================================================
    // set values for next iteration
    
    int lives = 0;

    // cell 0
    lives = (( (cell_0_adj >= 2) * (cell_0_adj <= 3) ) * cell_0_val) - (( cell_0_adj == 3 ) * ( cell_0_val - 1 ));
    ((uint8_t*)sim.ptr)[(row)*rwidth + col] = (cell_0_val_raw & ~(1<<(bit_idx+bit_dir))) | lives<<(bit_idx+bit_dir);

    // cell 1
    lives = (( (cell_1_adj >= 2) * (cell_1_adj <= 3) ) * cell_1_val) - (( cell_1_adj == 3 ) * ( cell_1_val - 1 ));
    ((uint8_t*)sim.ptr)[(row)*rwidth + col + 1] = (cell_1_val_raw & ~(1<<(bit_idx+bit_dir))) | lives<<(bit_idx+bit_dir);

    // cell 2
    lives = (( (cell_2_adj >= 2) * (cell_2_adj <= 3) ) * cell_2_val) - (( cell_2_adj == 3 ) * ( cell_2_val - 1 ));
    ((uint8_t*)sim.ptr)[(row+1)*rwidth + col] = (cell_2_val_raw & ~(1<<(bit_idx+bit_dir))) | lives<<(bit_idx+bit_dir);

    // cell 3
    lives = (( (cell_3_adj >= 2) * (cell_3_adj <= 3) ) * cell_3_val) - (( cell_3_adj == 3 ) * ( cell_3_val - 1 ));
    ((uint8_t*)sim.ptr)[(row+1)*rwidth + col + 1] = (cell_3_val_raw & ~(1<<(bit_idx+bit_dir))) | lives<<(bit_idx+bit_dir);

}

__global__ void cuda_GOL_kernel_intermediate(int sim_width, int sim_height, int bit_idx_src, int bit_idx_dest, cudaPitchedPtr sim, cudaPitchedPtr next_sim)
{

    #define _get_sim_position(s, x, y, b) ((((uint8_t*)s.ptr)[x*rwidth + y + read_offset]>>b)&1)
    

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    // OLD BAD 9 ACCESSES

    int row_0 = ( row - 1 );
    int col_0 = ( col - 1 );

    int row_1 = ( row + 1 );
    int col_1 = ( col + 1 );

    //find all active adjacent cells
    int n_adjacent_cells = 0;

    n_adjacent_cells += _get_sim_position(sim, row_0, col_0, bit_idx_src);
    n_adjacent_cells += _get_sim_position(sim, row_0, col,   bit_idx_src);
    n_adjacent_cells += _get_sim_position(sim, row_0, col_1, bit_idx_src);

    n_adjacent_cells += _get_sim_position(sim, row, col_0, bit_idx_src);
    n_adjacent_cells += _get_sim_position(sim, row, col_1, bit_idx_src);

    n_adjacent_cells += _get_sim_position(sim, row_1, col_0, bit_idx_src);
    n_adjacent_cells += _get_sim_position(sim, row_1, col,   bit_idx_src);
    n_adjacent_cells += _get_sim_position(sim, row_1, col_1, bit_idx_src);

    // calculate state of next cell
    int lives = 0;

    int raw_cell = ((uint8_t*)sim.ptr)[(row_1)*rwidth + col_1];
    int cell_sts = (raw_cell>>bit_idx_src)&1;

    lives += ( (n_adjacent_cells >= 2) * (n_adjacent_cells <= 3) ) * cell_sts;
    lives -= ( n_adjacent_cells == 3 ) * ( cell_sts - 1 );

    // set value for next iteration
    ((uint8_t*)next_sim.ptr)[(row_1)*rwidth + col_1] = lives<<(bit_idx_dest);

}

int main(void)
{

    srand(time(NULL));

    printf("allocating host memory (%lli MB)\n", (width+2)*(height+2)*sizeof(uint8_t)/(1000000));
    sim_host = (uint8_t*)malloc((width+2)*(height+2)*sizeof(uint8_t));

    printf("allocating device memory (%lli MB)", 2*(width+2)*(height+2)*sizeof(uint8_t)/(1000000));

    cudaMalloc(&sim_device_0, (width+2)*(height+2)*sizeof(uint8_t));
    CHECK_LAST_CUDA_CALL();

    cudaMalloc(&sim_device_1, (width+2)*(height+2)*sizeof(uint8_t));
    CHECK_LAST_CUDA_CALL();

    cudaPtr_0.ptr = sim_device_0;
    cudaPtr_1.ptr = sim_device_1;
    
    printf("allocation success\n");

    printf("generating random simulation start\n");
    fill_sim_rand();

    printf("loading device buffer 0 with preset data from host\n\n");
    cudaMemcpy(sim_device_0, sim_host, (width+2)*(height+2), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_CALL();
    
    constexpr int BLOCK_SIZE = 8;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blockCount(int(ceil(float(width)/BLOCK_SIZE)), int(ceil(float(height)/BLOCK_SIZE)));

    dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blockCount2(int(ceil(float(width/2)/BLOCK_SIZE)), int(ceil(float(height/2)/BLOCK_SIZE)));

    printf("device information:\n");
    printf("block size: %i\n", BLOCK_SIZE);
    printf("block dim: %d %d\n", blockDim.x, blockDim.y);
    printf("block count: %d %d %d\n\n", blockCount.x, blockCount.y, blockCount.z);
    
    printf("compiling cuda kernel 0\n");
    cuda_GOL_kernel<<<blockCount, blockDim>>>(width, height, 0, 1, cudaPtr_0);
    CHECK_LAST_CUDA_CALL();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_CALL();

    printf("compiling cuda kernel 1\n");
    cuda_GOL_kernel_intermediate<<<blockCount, blockDim>>>(width, height, 7, 0, cudaPtr_0, cudaPtr_1);
    CHECK_LAST_CUDA_CALL();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_CALL();

    printf("compiling cuda kernel 2\n");

    cuda_GOL_kernel_2x2<<<blockCount2, blockDim2>>>(width, height, 0, 1, cudaPtr_0);
    CHECK_LAST_CUDA_CALL();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_CALL();


    printf("compilation worked\n");

    printf("resetting device memory with preset data from host\n\n");
    cudaMemcpy(sim_device_0, sim_host, (width+2)*(height+2), cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_CALL();

    long long microseconds_taken = 0;
    long long total_time = 0;
    long long total_cells = 0;

    printf("running time test...\n");

    for ( int i = 0; i < 50; i++ ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // ----------------------------------------------------------------------------------------------------
        // first main computation

        // compute the first 7 iterations
        
        #ifdef use_2x2_compute
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel_2x2<<<blockCount2, blockDim2>>>(width, height, j, 1, cudaPtr_0); }
        #else
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel<<<blockCount, blockDim>>>(width, height, j, 1, cudaPtr_0); }
        #endif

        // copy data to host while the kernel is running
        // cudaMemcpy(sim_host, sim_device_1, (width+2)*(height+2), cudaMemcpyDeviceToHost);

        // wait for kernels to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // intermediate transfer / computation

        // old - manually copy data from one simulation buffer to the next
        // cudaMemcpy(sim_device_1, sim_device_0, (width+2)*(height+2), cudaMemcpyDeviceToDevice);
        
        // new - run a seperate kernel that moves the data from one sim buffer to the next

        // this reads bit 7 from sim 0 and writes the next iteration to bit 0 of sim 1
        cuda_GOL_kernel_intermediate<<<blockCount, blockDim>>>(width, height, 7, 0, cudaPtr_0, cudaPtr_1);

        // wait for transfer kernel to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // second main computation

        // compute the next 7 iterations
        #ifdef use_2x2_compute
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel_2x2<<<blockCount2, blockDim2>>>(width, height, j, 1, cudaPtr_0); }
        #else
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel<<<blockCount, blockDim>>>(width, height, j, 1, cudaPtr_0); }
        #endif

        // copy data to host while the kernel is running
        cudaMemcpy(sim_host+rwidth, ((uint8_t*)sim_device_0)+rwidth, (width+2)*(height), cudaMemcpyDeviceToHost);

        // wait for kernels to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // second intermediate transfer / computation

        // old - manually copy data from one simulation buffer to the next
        // cudaMemcpy(sim_device_0, sim_device_1, (width+2)*(height+2), cudaMemcpyDeviceToDevice);
        
        // new - run a seperate kernel that moves the data from one sim buffer to the next

        // this reads bit 7 from sim 0 and writes the next iteration to bit 0 of sim 1
        cuda_GOL_kernel_intermediate<<<blockCount, blockDim>>>(width, height, 7, 0, cudaPtr_1, cudaPtr_0);

        // wait for transfer kernel to finish
        cudaDeviceSynchronize();

        auto end_time_b = std::chrono::high_resolution_clock::now();

        microseconds_taken = (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_b - start_time).count())/1000;
        
        total_time += microseconds_taken*(i>0);
        total_cells += width*height*16*(i>0);

    }

    printf("average gcells/s: %.2f\n\n", (float(total_cells)/(float(total_time))/1000));

    microseconds_taken = 0;
    total_time = 0;
    total_cells = 0;

    while(true) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // ----------------------------------------------------------------------------------------------------
        // first main computation

        // compute the first 7 iterations
        
        #ifdef use_2x2_compute
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel_2x2<<<blockCount2, blockDim2>>>(width, height, j, 1, cudaPtr_0); }
        #else
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel<<<blockCount, blockDim>>>(width, height, j, 1, cudaPtr_0); }
        #endif

        // copy data to host while the kernel is running
        // cudaMemcpy(sim_host, sim_device_1, (width+2)*(height+2), cudaMemcpyDeviceToHost);

        // wait for kernels to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // intermediate transfer / computation

        // old - manually copy data from one simulation buffer to the next
        // cudaMemcpy(sim_device_1, sim_device_0, (width+2)*(height+2), cudaMemcpyDeviceToDevice);
        
        // new - run a seperate kernel that moves the data from one sim buffer to the next

        // this reads bit 7 from sim 0 and writes the next iteration to bit 0 of sim 1
        cuda_GOL_kernel_intermediate<<<blockCount, blockDim>>>(width, height, 7, 0, cudaPtr_0, cudaPtr_1);

        // wait for transfer kernel to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // second main computation

        // compute the next 7 iterations
        #ifdef use_2x2_compute
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel_2x2<<<blockCount2, blockDim2>>>(width, height, j, 1, cudaPtr_0); }
        #else
        for ( int j = 0; j < 7; j++ ) { cuda_GOL_kernel<<<blockCount, blockDim>>>(width, height, j, 1, cudaPtr_0); }
        #endif
        
        // copy data to host while the kernel is running
        cudaMemcpy(sim_host+rwidth, ((uint8_t*)sim_device_0)+rwidth, (width+2)*(height), cudaMemcpyDeviceToHost);

        // wait for kernels to finish
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------------------------------
        // second intermediate transfer / computation

        // old - manually copy data from one simulation buffer to the next
        // cudaMemcpy(sim_device_0, sim_device_1, (width+2)*(height+2), cudaMemcpyDeviceToDevice);
        
        // new - run a seperate kernel that moves the data from one sim buffer to the next

        // this reads bit 7 from sim 0 and writes the next iteration to bit 0 of sim 1
        cuda_GOL_kernel_intermediate<<<blockCount, blockDim>>>(width, height, 7, 0, cudaPtr_1, cudaPtr_0);

        // wait for transfer kernel to finish
        cudaDeviceSynchronize();

        auto end_time_b = std::chrono::high_resolution_clock::now();

        microseconds_taken = (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_b - start_time).count())/1000;
        
        total_time += microseconds_taken;
        total_cells += width*height*16;

        printf("\rgcells/s: %.2f", (float(total_cells)/(float(total_time))/1000));
    }

    // Free memory
    // cudaFree();
    
    return 0;
}

