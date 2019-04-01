#pragma once
#include "IWindow.h"
#include "IEventHandler.h"
#include "ILevelManager.h"
#include "IEntityManager.h"

namespace CoreUtils
{
	class EventHandler :
		public IEventHandler
	{
	private:
		auto handleInterrupts(IWindow&, IEntityManager&, ILevelManager&, bool*) -> EventHandler&;
		auto handleControls(IWindow&, IEntityManager&, ILevelManager&) -> EventHandler&;

	public:
		EventHandler();
		~EventHandler();
		auto intercept(IWindow&, IEntityManager&, ILevelManager&, bool*) -> void override;
	};
}
