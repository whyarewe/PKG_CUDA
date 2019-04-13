#pragma once
#include "IWindow.h"
#include "IEventHandler.h"
#include "ILevelManager.h"
#include "IEntityManager.h"
#include "Engine.h"

namespace CoreUtils
{
	class EventHandler :
		public IEventHandler
	{
	private:
		auto EventHandler::handleInterrupts(Engine& engine, bool* debug) -> EventHandler&;

		auto handleControls(IWindow&, IEntityManager&, ILevelManager&) -> EventHandler&;
	public:
		EventHandler();
		~EventHandler();
		auto intercept(Engine&, bool*) -> void override;
	};
}
