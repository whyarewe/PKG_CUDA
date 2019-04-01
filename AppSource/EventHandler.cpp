#include "EventHandler.h"
#include "ILevelManager.h"

CoreUtils::EventHandler::EventHandler() = default;

CoreUtils::EventHandler::~EventHandler() = default;

auto CoreUtils::EventHandler::intercept(IWindow& window, IEntityManager& entity_manager, ILevelManager& level_manager,
                                        bool* output) -> void
{
	handleInterrupts(window, entity_manager, level_manager, output)
		.handleControls(window, entity_manager, level_manager);
}

auto CoreUtils::EventHandler::handleInterrupts(IWindow& window, IEntityManager& entity_manager,
                                               ILevelManager& level_manager, bool* output) -> EventHandler&
{
	sf::Event event{};
	while (window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			window.close();
		}
		else if (event.type == sf::Event::MouseButtonPressed)
		{
			if (event.mouseButton.button == sf::Mouse::Left)
			{
				const auto mouse_position = window.getMousePosition();

				if (window.isWithinWindow(mouse_position))
				{
					entity_manager.spawn(mouse_position, level_manager.getXAxisLength(),
					                     level_manager.getYAxisLength());
					*output = true;
				}
			}
		}
		else if (event.type == sf::Event::EventType::KeyPressed)
		{
			switch (event.key.code)
			{
			case sf::Keyboard::Up:
				entity_manager.increaseRadius();
				break;
			case sf::Keyboard::Down:
				entity_manager.decreaseRadius();
				break;
			case sf::Keyboard::BackSpace:
				entity_manager.killFirst();
				break;
			case sf::Keyboard::F11:
				switch (window.getStyle())
				{
				case WindowStyles::Resizable:
				case WindowStyles::NonResizable:
					window.setStyle(WindowStyles::FullScreen);
					window.reloadWindow();
					break;
				case WindowStyles::FullScreen:
					window.setStyle(WindowStyles::NonResizable);
					window.reloadWindow();
					break;
				}
				break;
			case sf::Keyboard::I:
				window.toggleControls();
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
			case sf::Keyboard::Escape: break;
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
		window.updateInterface();
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
