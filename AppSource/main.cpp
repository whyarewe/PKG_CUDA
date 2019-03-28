#include "CUDASystemInformation.h"
#include "Engine.h"

int main()
{
	CUDAHelpers::CUDASystemInformation system_information;
	system_information.displaySystemDevicesProperties();

	CoreUtils::Engine engine;
	engine.run();

	return 0;
}
