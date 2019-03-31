#pragma once
#include "IWindow.h"
#include "IEntityManager.h"
#include "ILevelManager.h"

namespace CoreUtils
{
	class IEventHandler
	{
	public:
		IEventHandler() = default;
		virtual ~IEventHandler() = default;
		IEventHandler(const IEventHandler&) = delete;
		IEventHandler(const IEventHandler&&) = delete;
		IEventHandler& operator=(const IEventHandler&) = delete;
		IEventHandler& operator=(const IEventHandler&&) = delete;

		virtual auto intercept(IWindow&, IEntityManager&, ILevelManager&) -> void = 0;
	};
}