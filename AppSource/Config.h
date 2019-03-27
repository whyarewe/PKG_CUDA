#pragma once

#include <string>

namespace Config
{
	const std::string project_name = "PKG_CUDA";

	namespace FullHDResolution
	{
		static unsigned const width = 1920;
		static unsigned const height = 1080;
	};

	namespace StandardResolution {
		static unsigned const width = 600;
		static unsigned const height = 600;
	};

	namespace StandardWindowSetting
	{
		static unsigned const frame_rate_limit = 60;
		static unsigned const anti_aliasing_level = 8;
	}
}
