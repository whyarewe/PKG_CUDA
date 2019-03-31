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

	namespace StandardResolution
	{
		static unsigned const width = 600;
		static unsigned const height = 600;
	};

	namespace StandardWindowSetting
	{
		static unsigned const frame_rate_limit = 60;
		static unsigned const anti_aliasing_level = 8;
	}

	namespace GUI_Config
	{
		static unsigned short const system_font_size = 12;
	}

	namespace Entity
	{
		static unsigned short const default_entity_radius = 1;
		static unsigned short const minimal_entity_radius = 1;
		static unsigned short const maximal_entity_radius = 21;
	}
}
