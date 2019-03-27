#pragma once
#include <memory>
#include <Windows.h>

#include "IWindow.h"
#include "Entity.h"

namespace CoreUtils
{
	class Engine
	{
	private:
		Entity::EntityContainer swarm_;
		std::unique_ptr<IWindow> window_;		
	public:
		auto run()->void;
		static auto getExePath()->std::string;
		Engine();
		~Engine();
	};
}

