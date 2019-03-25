#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <Windows.h>

#include "CUDASystemInformation.h"
#include "CUDALaplacePropagation.h"
#include "Entity.h"


namespace
{
	double R(float x)
	{
		return 255.0605 + (0.02909945 - 255.0605) / std::pow((1 + std::pow((2 * x / 68872.05), 2.133224)), 13205500);
	}

	double G(float x)
	{
		return 10 + 6.109578*x*1.2 - 0.2057529*x*x + 0.002335796*x*x*x - 0.00001016682*x*x*x*x + 1.514604e-8*x*x*x*x*x;
	}

	double B(float x)
	{
		return 10 + 3.000859*x - 0.09096409*x*x + 0.0006806883*x*x*x * 2 - 0.000001399089*x*x*x*x;
	}


	void laplace(std::vector<float> &vec, int xAxisBound, int yAxisBound, Entity::EntityContainer swarm)
	{
		bool isUnderEntity = false;

		for (int i = 1; i < yAxisBound - 1; ++i)
		{
			for (int j = 1; j < xAxisBound - 1; j++)
			{
				for (const auto& entity : swarm) {
					uint32_t leftBorder = entity.getCoordinates().getX() - entity.getRadius();
					uint32_t rightBorder = entity.getCoordinates().getX() + entity.getRadius();
					uint32_t topBorder = entity.getCoordinates().getY() - entity.getRadius();
					uint32_t bottomBorder = entity.getCoordinates().getY() + entity.getRadius();

					if (i >= topBorder && i <= bottomBorder && j >= rightBorder && j <= leftBorder)
					{
						isUnderEntity = true;
					}
				}

				if (!isUnderEntity)
				{
					vec[i*xAxisBound + j] = (0.25f * (vec[i*xAxisBound + j - 1] + vec[i*xAxisBound + j + 1]
						+ vec[i*xAxisBound + j + yAxisBound] + vec[i*xAxisBound + j - yAxisBound]));
				}
			}
		}
	}

	sf::Image constructImageFromVector(const std::vector<float> &vec, int xAxisBound, int yAxisBound)
	{
		sf::Image board;
		board.create(xAxisBound, yAxisBound, sf::Color::Black);
		for (int i = 1; i < yAxisBound - 1; ++i)
		{
			for (int j = 1; j < xAxisBound - 1; j++)
			{
				sf::Color pixelColor = sf::Color(
					R(vec[i*xAxisBound + j]),
					G(vec[i*xAxisBound + j]),
					B(vec[i*xAxisBound + j])
				);

				if (vec[i*xAxisBound + j] > 125.f) {
					pixelColor = sf::Color::White;
				}
				else if (vec[i*xAxisBound + j] < 0.1f) {
					pixelColor = sf::Color(
						(vec[i*xAxisBound + j]),
						(vec[i*xAxisBound + j]),
						(vec[i*xAxisBound + j])
					);
				}
				board.setPixel(j, i, pixelColor);
			}
		}
		return board;
	}

	std::string getexepath()
	{
		char result[MAX_PATH];
		return std::string(result, GetModuleFileName(NULL, result, MAX_PATH));
	}
}

int main()
{
	CUDAHelpers::CUDASystemInformation systemInformations;
	systemInformations.displaySystemDevicesProperites();
	int xAxisBound = 600;
	int yAxisBound = 600;

	sf::RenderWindow mainWindow(sf::VideoMode(xAxisBound, yAxisBound), "PKG_CUDA", sf::Style::Titlebar | sf::Style::Close);
	mainWindow.setFramerateLimit(60);

	Entity::EntityContainer swarm;
	uint32_t entityRadius = 1;

	std::vector<float> model;
	model.reserve(xAxisBound * yAxisBound);

	sf::Image board;
	board.create(xAxisBound, yAxisBound, sf::Color::Black);

	sf::Texture texture;
	texture.loadFromImage(board);

	sf::Sprite sprite;
	sprite.setTexture(texture, true);
	std::string fontPath = getexepath();
	fontPath.resize(fontPath.size() - 12);
	fontPath += "FontFile.ttf";
	sf::Font font;
	if (!font.loadFromFile(fontPath))
	{
		std::exit(0);
	}

	sf::Text heaters;
	sf::Text radius;

	heaters.setFont(font);
	radius.setFont(font);	

	radius.setCharacterSize(20);
	heaters.setCharacterSize(20);
	
	radius.setFillColor(sf::Color::White);
	heaters.setFillColor(sf::Color::White);
	
	radius.move(xAxisBound - 180, 30);
	heaters.move(xAxisBound - 180, 10);
	
	std::string heatersCount("Heater Count  : " + std::to_string(swarm.size()));
	std::string heaterRadius("Heater Radius : " + std::to_string(entityRadius));
	heaters.setString(heatersCount.c_str());
	radius.setString(heaterRadius.c_str());

	while (mainWindow.isOpen())
	{
		sf::Event event;

		while (mainWindow.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				mainWindow.close();
			}
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Left) 
				{
					sf::Vector2i mousePositon = sf::Mouse::getPosition(mainWindow);

					if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
						uint32_t x = mousePositon.x;
						uint32_t y = mousePositon.y;

						if (x > 1 && x < xAxisBound && y > 1 && y < yAxisBound)
						{
							swarm.push_back(Entity(x, y, entityRadius));
							std::string heatersCount("Heater Count  : " + std::to_string(swarm.size()));
							heaters.setString(heatersCount.c_str());
						}
					}
				}
			}
			else if (event.type == sf::Event::EventType::KeyPressed) {
			    if (event.key.code == sf::Keyboard::Up)
				{
					entityRadius = entityRadius > 20 ? entityRadius : entityRadius += 2;
					std::string heaterRadius("Heater Radius : " + std::to_string(entityRadius));
					radius.setString(heaterRadius.c_str());
				}
				else if (event.key.code == sf::Keyboard::Down)
				{
					entityRadius = entityRadius == 1 ? entityRadius : entityRadius -= 2;
					std::string heaterRadius("Heater Radius : " + std::to_string(entityRadius));
					radius.setString(heaterRadius.c_str());
				}
				else if (event.key.code == sf::Keyboard::BackSpace)
				{
					swarm.erase(swarm.begin());
					std::string heatersCount("Heater Count  : " + std::to_string(swarm.size()));
					heaters.setString(heatersCount.c_str());
				}
			}
		}
		
		for (const auto& entity : swarm) {
			uint32_t leftBorder = entity.getCoordinates().getX() - entity.getRadius();
			uint32_t rightBorder = entity.getCoordinates().getX() + entity.getRadius();
			uint32_t topBorder = entity.getCoordinates().getY() - entity.getRadius();
			uint32_t bottomBorder = entity.getCoordinates().getY() + entity.getRadius();

			leftBorder = leftBorder <= 1 ? 1 : leftBorder;
			rightBorder = rightBorder >= xAxisBound ? rightBorder - 1 : rightBorder;
			topBorder = topBorder <= 1 ? 1 : topBorder;
			bottomBorder = bottomBorder >= yAxisBound ? yAxisBound - 1 : bottomBorder;

			for (int i = topBorder; i < bottomBorder; ++i) {
				for (int j = leftBorder; j < rightBorder; ++j)
				{
					model[i * xAxisBound + j] = 255.f;
				}
			}
		}

		laplace(model, xAxisBound, yAxisBound, swarm);
		//CUDALaplacePropagation::propagate(model, xAxisBound, yAxisBound, heater.x, heater.y);	//GPU
		board = constructImageFromVector(model, xAxisBound, yAxisBound);
		texture.loadFromImage(board);
		sprite.setTexture(texture, true);

		mainWindow.clear();
		mainWindow.draw(sprite);
		mainWindow.draw(radius);
		mainWindow.draw(heaters);
		mainWindow.display();
	}

	return 0;
}