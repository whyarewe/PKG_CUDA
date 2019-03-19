#include "CUDAUtils.h"
#include "CUDAEssentials.h"

using namespace CUDAHelpers;

__global__ void kernel(RGB *img, int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height) {
		return;
	}

	RGB *pixelColour = img + (idy*width + idx);
	pixelColour->R = (uint8_t)((1 - (float)idy / (float)width) * 255);
	pixelColour->G = (uint8_t)(((float)idx / (float)height) * 255);
}

void CUDAEssentials::generateImage(RGB *image, int width, int height)
{
	RGB *gc_img = nullptr;
	VALID(cudaMalloc(&gc_img, width * height * sizeof(RGB)));
	kernel <<<dim3(width / 16 + 1, height / 32 + 1), dim3(16, 32)>>> (gc_img, width, height);
	VALID(cudaMemcpy(image, gc_img, width * height * sizeof(RGB), cudaMemcpyDeviceToHost));
}

void CUDAEssentials::writeP6_PPM(const char * filename, RGB * pixelData, int width, int height)
{
	FILE* plik;
	errno_t error;
	error = fopen_s(&plik, filename, "wb");

	if (!error)
	{
		fprintf(plik, "P6\n%d %d\n255\n", width, height);
		fwrite(pixelData, sizeof(RGB), width*height, plik);
		fclose(plik);
	}
	else
	{
		fprintf(stderr, "File open error\n");
	}
}
