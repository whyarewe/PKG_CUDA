#pragma once
#include "IEventHandler.h"
#include "IWindow.h"
#include "IEntityManager.h"

namespace CoreUtils
{
	class ILevelManager;

	class EventHandler :
		public IEventHandler
	{
	private:
		auto handleInterrupts(IWindow&, IEntityManager&, ILevelManager&)-> EventHandler&;
		auto handleControls(IWindow&, IEntityManager&, ILevelManager&)->EventHandler&;

	public:
		EventHandler();
		~EventHandler();
		auto intercept(IWindow&, IEntityManager&, ILevelManager&) -> void override;
	};
}
