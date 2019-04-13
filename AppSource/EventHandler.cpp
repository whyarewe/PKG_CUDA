#include "EventHandler.h"
#include "ILevelManager.h"
#include "LevelManager.h"
#include "Engine.h"
#include "EntityManager.h"
#include <thread>

CoreUtils::EventHandler::EventHandler() = default;

CoreUtils::EventHandler::~EventHandler() = default;

auto CoreUtils::EventHandler::intercept(Engine& engine, bool* output) -> void
{
	handleInterrupts(engine, output)
		.handleControls(*engine.window_, *engine.entity_manager_, *engine.level_manager_);
}

auto CoreUtils::EventHandler::handleInterrupts(Engine& engine, bool* debug) -> EventHandler&
{
	sf::Event event{};
	while (engine.window_->pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			engine.window_->close();
		}
		else if (event.type == sf::Event::MouseButtonPressed)
		{
			if (event.mouseButton.button == sf::Mouse::Left)
			{
				const auto mouse_position = engine.window_->getMousePosition();

				if (engine.window_->isWithinWindow(mouse_position))
				{
					engine.entity_manager_->spawn(mouse_position, engine.level_manager_->getXAxisLength(),
					                              engine.level_manager_->getYAxisLength());
					*debug = true;
				}
			}
		}
		else if (event.type == sf::Event::EventType::KeyPressed)
		{
			switch (event.key.code)
			{
			case sf::Keyboard::Up:
				engine.entity_manager_->increaseRadius();
				break;
			case sf::Keyboard::Down:
				engine.entity_manager_->decreaseRadius();
				break;
			case sf::Keyboard::BackSpace:
				engine.entity_manager_->killFirst();
				break;
			case sf::Keyboard::F11:
				switch (engine.window_->getStyle())
				{
				case WindowStyles::Resizable:
				case WindowStyles::NonResizable:
					engine.window_->setStyle(WindowStyles::FullScreen);
					engine.reload();
					break;
				case WindowStyles::FullScreen:
					engine.window_->setStyle(WindowStyles::NonResizable);
					engine.reload();
					break;
				}
				break;
			case sf::Keyboard::I:
				engine.window_->toggleControls();
				break;
			case sf::Keyboard::Unknown: break;
			case sf::Keyboard::A: break;
			case sf::Keyboard::B: break;
			case sf::Keyboard::C: break;
			case sf::Keyboard::D: break;
			case sf::Keyboard::E: break;
			case sf::Keyboard::F: break;
			case sf::Keyboard::G: break;
			case sf::Keyboard::H: break;
			case sf::Keyboard::J: break;
			case sf::Keyboard::K: break;
			case sf::Keyboard::L: break;
			case sf::Keyboard::M: break;
			case sf::Keyboard::N: break;
			case sf::Keyboard::O: break;
			case sf::Keyboard::P: break;
			case sf::Keyboard::Q: break;
			case sf::Keyboard::R: break;
			case sf::Keyboard::S: break;
			case sf::Keyboard::T: break;
			case sf::Keyboard::U: break;
			case sf::Keyboard::V: break;
			case sf::Keyboard::W: break;
			case sf::Keyboard::X: break;
			case sf::Keyboard::Y: break;
			case sf::Keyboard::Z: break;
			case sf::Keyboard::Num0: break;
			case sf::Keyboard::Num1: break;
			case sf::Keyboard::Num2: break;
			case sf::Keyboard::Num3: break;
			case sf::Keyboard::Num4: break;
			case sf::Keyboard::Num5: break;
			case sf::Keyboard::Num6: break;
			case sf::Keyboard::Num7: break;
			case sf::Keyboard::Num8: break;
			case sf::Keyboard::Num9: break;
			case sf::Keyboard::Escape:
				engine.window_->reloadWindow();
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(50ms);
				engine.window_->setActive(true);
				engine.window_->close();
				break;
			case sf::Keyboard::LControl: break;
			case sf::Keyboard::LShift: break;
			case sf::Keyboard::LAlt: break;
			case sf::Keyboard::LSystem: break;
			case sf::Keyboard::RControl: break;
			case sf::Keyboard::RShift: break;
			case sf::Keyboard::RAlt: break;
			case sf::Keyboard::RSystem: break;
			case sf::Keyboard::Menu: break;
			case sf::Keyboard::LBracket: break;
			case sf::Keyboard::RBracket: break;
			case sf::Keyboard::Semicolon: break;
			case sf::Keyboard::Comma: break;
			case sf::Keyboard::Period: break;
			case sf::Keyboard::Quote: break;
			case sf::Keyboard::Slash: break;
			case sf::Keyboard::Backslash: break;
			case sf::Keyboard::Tilde: break;
			case sf::Keyboard::Equal: break;
			case sf::Keyboard::Hyphen: break;
			case sf::Keyboard::Space: break;
			case sf::Keyboard::Enter: break;
			case sf::Keyboard::Tab: break;
			case sf::Keyboard::PageUp: break;
			case sf::Keyboard::PageDown: break;
			case sf::Keyboard::End: break;
			case sf::Keyboard::Home: break;
			case sf::Keyboard::Insert: break;
			case sf::Keyboard::Delete: break;
			case sf::Keyboard::Add: break;
			case sf::Keyboard::Subtract: break;
			case sf::Keyboard::Multiply: break;
			case sf::Keyboard::Divide: break;
			case sf::Keyboard::Left: break;
			case sf::Keyboard::Right: break;
			case sf::Keyboard::Numpad0: break;
			case sf::Keyboard::Numpad1: break;
			case sf::Keyboard::Numpad2: break;
			case sf::Keyboard::Numpad3: break;
			case sf::Keyboard::Numpad4: break;
			case sf::Keyboard::Numpad5: break;
			case sf::Keyboard::Numpad6: break;
			case sf::Keyboard::Numpad7: break;
			case sf::Keyboard::Numpad8: break;
			case sf::Keyboard::Numpad9: break;
			case sf::Keyboard::F1: break;
			case sf::Keyboard::F2: break;
			case sf::Keyboard::F3: break;
			case sf::Keyboard::F4: break;
			case sf::Keyboard::F5: break;
			case sf::Keyboard::F6: break;
			case sf::Keyboard::F7: break;
			case sf::Keyboard::F8: break;
			case sf::Keyboard::F9: break;
			case sf::Keyboard::F10: break;
			case sf::Keyboard::F12: break;
			case sf::Keyboard::F13: break;
			case sf::Keyboard::F14: break;
			case sf::Keyboard::F15: break;
			case sf::Keyboard::Pause: break;
			case sf::Keyboard::KeyCount: break;
			default:
				break;
			}
		}
		engine.window_->updateInterface();
	}

	return *this;
}

auto CoreUtils::EventHandler::handleControls(IWindow& window, IEntityManager& entity_manager,
                                             ILevelManager& level_manager) -> EventHandler&
{
	if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
	{
		const auto mouse_position = window.getMousePosition();
		if (window.isWithinWindow(mouse_position))
		{
			entity_manager.spawnTemporary(mouse_position, level_manager.getXAxisLength(),
			                              level_manager.getYAxisLength());
		}
	}
	return *this;
}
