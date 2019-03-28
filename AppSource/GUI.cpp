#include "GUI.h"
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>

CoreUtils::GUI::GUI(sf::RenderWindow& window, const sf::Font& font) :
	radius_text_(std::make_unique<sf::Text>()),
	heaters_count_text_(std::make_unique<sf::Text>()),
	info_text_(std::make_unique<sf::Text>()),
	key_bindings_text_(std::make_unique<sf::Text>())
{
	setFontConfiguration(font);

	radius_text_->move(static_cast<float>(window.getSize().x - 110), 30.f);
	heaters_count_text_->move(static_cast<float>(window.getSize().y - 110), 10.f);
	key_bindings_text_->move(5.f, 10.f);

	std::string info_string("Press 'i' to toggle controls\n");
	info_text_->setString(info_string.c_str());
	std::string key_bindings_string("LMB : Spawn heater\n"
		"RMB : Draw heaters\n"
		"Backspace : Delete heater\n"
		"Key Up : Increase radius\n"
		"Key Down : Decrase radius\n"
		"F11 : Fullscreen mode");
	key_bindings_text_->setString(key_bindings_string.c_str());

	std::string heaters_count_string("Heater Count  : " + std::to_string(heaters_count_));
	std::string radius_string("Heater Radius : " + std::to_string(radius_));

	heaters_count_text_->setString(heaters_count_string.c_str());
	radius_text_->setString(radius_string.c_str());
}

auto CoreUtils::GUI::setFontConfiguration(const sf::Font& font) const -> void
{
	radius_text_->setFont(font);
	heaters_count_text_->setFont(font);
	info_text_->setFont(font);
	key_bindings_text_->setFont(font);

	radius_text_->setCharacterSize(Config::GUI_CONFIG::system_font_size);
	heaters_count_text_->setCharacterSize(Config::GUI_CONFIG::system_font_size);
	info_text_->setCharacterSize(Config::GUI_CONFIG::system_font_size);
	key_bindings_text_->setCharacterSize(Config::GUI_CONFIG::system_font_size);

	radius_text_->setFillColor(sf::Color::White);
	heaters_count_text_->setFillColor(sf::Color::White);
	info_text_->setFillColor(sf::Color::White);
	key_bindings_text_->setFillColor(sf::Color::White);
}

auto CoreUtils::GUI::setRadius(uint16_t radius) -> void
{
	radius_ = radius;
	update();
}

auto CoreUtils::GUI::setHeatersCount(uint16_t heaters_count) -> void
{
	heaters_count_ = heaters_count;
	update();
}

auto CoreUtils::GUI::setShowControls(bool show_controls) -> void
{
	show_controls_ = show_controls;
	update();
}

auto CoreUtils::GUI::update() -> void
{
	std::string heaters_count_string("Heater Count  : " + std::to_string(heaters_count_));
	std::string radius_string("Heater Radius : " + std::to_string(radius_));

	heaters_count_text_->setString(heaters_count_string.c_str());
	radius_text_->setString(radius_string.c_str());
}

auto CoreUtils::GUI::display(sf::RenderWindow& window) -> void
{
	if (!show_controls_)
	{
		window.draw(*info_text_);
	}
	else
	{
		window.draw(*key_bindings_text_);
	}

	window.draw(*heaters_count_text_);
	window.draw(*radius_text_);
}

