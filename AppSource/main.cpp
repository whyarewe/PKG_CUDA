#include "CUDASystemInformation.h"

typedef struct {
	uint8_t R;
	uint8_t G;
	uint8_t B;
} RGB;


int main()
{
	CUDAHelpers::CUDASystemInformation systemInformations;
	systemInformations.displaySystemDevicesProperites();

	int width = 500;
	int height = 500;
	RGB *img = new RGB[width*height];

	//generateImage(img, width, height);
	//writeP6_PPM("output.ppm", img, width, height);

	return 0;
}