
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <stdio.h>
#include <ctime>

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

    // Start measuring time
    float elapsed1 = 0;
    cudaEvent_t start1, stop1;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, 0);

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
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_gray1.png";

    // Stop measuring time and calculate the elapsed time
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    cudaEventElapsedTime(&elapsed1, start1, stop1);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    float time_1 = elapsed1;

    cout << "The elapsed time in gpu: " << time_1 << "ms" << endl;

    // Write image back to disk
    cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
    cout << " DONE" << endl;

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);

    // Multiple GPUs

    // Check argument count
    if (argc < 2)
    {
        cout << "Usage: 02_ImageToGray <filename>";
        return -1;
    }

    // Open image
    int width2, height2, componentCount2;
    cout << "Loading png file...";
    unsigned char* imageData2 = stbi_load(argv[1], &width2, &height2, &componentCount2, 4);
    if (!imageData2)
    {
        cout << endl << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    cout << " DONE" << endl;

    // Validate image sizes
    if (width2 % 32 || height2 % 32)
    {
        // NOTE: Leaked memory of "imageData"
        cout << "Width and/or Height is not dividable by 32!";
        return -1;
    }

    // Making RGB layers

    // Create two CUDA streams.
    cudaStream_t stream1; cudaStreamCreate(&stream1);
    cudaStream_t stream2; cudaStreamCreate(&stream2);
    cudaStream_t stream3; cudaStreamCreate(&stream3);
    cudaStream_t stream4; cudaStreamCreate(&stream4);

    // Start measuring time
    float elapsed2 = 0;
    cudaEvent_t start2, stop2;

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);

    // Copy data to the gpu
    cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu2 = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu2, width2 * height2 * 4) == cudaSuccess);
    assert(cudaMemcpyAsync(ptrImageDataGpu2, imageData2, width2 * height2 * 4, cudaMemcpyHostToDevice, stream1) == cudaSuccess);
    assert(cudaMemcpyAsync(ptrImageDataGpu2, imageData2, width2 * height2 * 4, cudaMemcpyHostToDevice, stream2) == cudaSuccess);
    assert(cudaMemcpyAsync(ptrImageDataGpu2, imageData2, width2 * height2 * 4, cudaMemcpyHostToDevice, stream3) == cudaSuccess);
    assert(cudaMemcpyAsync(ptrImageDataGpu2, imageData2, width2 * height2 * 4, cudaMemcpyHostToDevice, stream4) == cudaSuccess);
    cout << " DONE" << endl;

    // Process image on gpu
    cout << "Running CUDA Kernel...";
    dim3 blockSize2(32, 32);
    dim3 gridSize2(width2 / blockSize.x, height2 / blockSize.y);
    ConvertImageToGrayGpu <<<gridSize2, blockSize2, 0 ,stream4>>> (ptrImageDataGpu);
    ConvertImageToGrayGpu <<<gridSize2, blockSize2, 0, stream3>>> (ptrImageDataGpu);
    ConvertImageToGrayGpu <<<gridSize2, blockSize2, 0, stream2>>> (ptrImageDataGpu);
    ConvertImageToGrayGpu <<<gridSize2, blockSize2, 0, stream1>>> (ptrImageDataGpu);
    auto error = cudaGetLastError();
    cout << " DONE" << endl;

    // Copy data from the gpu
    cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData2, ptrImageDataGpu2, width2 * height2 * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    cout << " DONE" << endl;

    // Build output filename
    string fileNameOut2 = argv[1];
    fileNameOut2 = fileNameOut2.substr(0, fileNameOut2.find_last_of('.')) + "_gray2.png";

    // Stop measuring time and calculate the elapsed time
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed2, start2, stop2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    float time_2 = elapsed2;

    cout << "The elapsed time in gpu: " << time_2 << "ms" << endl;

    // Write image back to disk
    cout << "Writing png to disk...";
    stbi_write_png(fileNameOut2.c_str(), width2, height2, 4, imageData2, 4 * width2);
    cout << " DONE" << endl;

    // Free memory
    cudaFree(ptrImageDataGpu2);
    stbi_image_free(imageData2);

    float time_d = time_2 - time_1;
    cout << "Time difference: " << time_d << "ms" << endl;
}