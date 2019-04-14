#pragma once
#include <string>
#include "CUDAPropagation.h"

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
		static unsigned const width = 800;
		static unsigned const height = 800;
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

	namespace Engine_Config
	{
		static auto device = CUDAHelpers::CUDAPropagation::Device::GPU;
		static auto method = CUDAHelpers::CUDAPropagation::Method::Laplace;
	}

	namespace FTCS_Config
	{
		static float dx = 1.f;
		static float dt = 1.f / 60.f;
		static float alpha = 0.f;
	}

	namespace DHE_Config
	{
		static float dx = 1.f;
		static float dy = 1.f;
		static float dt = 1.f / 60.f;
		static float K = 23700000.f;
		static float sh = 900.f;
		static float density = 4700.f;
	}
}
