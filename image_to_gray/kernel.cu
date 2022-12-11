
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <chrono>

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

struct Pixel
{
    unsigned char r, g, b, a;
};

__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
}

int main(int argc, char** argv)
{
    // Start measuring time
    auto begin = chrono::high_resolution_clock::now();

    // Check argument count
    if (argc < 2)
    {
        cout << "Usage: 02_ImageToGray <filename>";
        return -1;
    }

    // Open image
    int width, height, componentCount;
    cout << "Loading png file...";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        cout << endl << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    cout << " DONE" << endl;

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        cout << "Width and/or Height is not dividable by 32!";
        return -1;
    }

    // Copy data to the gpu
    cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    cout << " DONE" << endl;

    // Process image on gpu
    cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    ConvertImageToGrayGpu <<<gridSize, blockSize>>> (ptrImageDataGpu);
    auto err = cudaGetLastError();
    cout << " DONE" << endl;

    // Copy data from the gpu
    cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    cout << " DONE" << endl;

    // Build output filename
    string fileNameOut = argv[1];
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_gray.png";

    // Write image back to disk
    cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
    cout << " DONE" << endl;

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
 
    // Stop measuring time and calculate the elapsed time
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
}