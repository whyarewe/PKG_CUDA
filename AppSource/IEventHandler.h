#pragma once

namespace CoreUtils
{
	class Engine;

	class IEventHandler
	{
	public:
		IEventHandler() = default;
		virtual ~IEventHandler() = default;
		IEventHandler(const IEventHandler&) = delete;
		IEventHandler(const IEventHandler&&) = delete;
		IEventHandler& operator=(const IEventHandler&) = delete;
		IEventHandler& operator=(const IEventHandler&&) = delete;

		virtual auto intercept(Engine&, bool*) -> void = 0;
	};
}
