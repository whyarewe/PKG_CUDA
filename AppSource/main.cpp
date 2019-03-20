#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vector>

#include "CUDASystemInformation.h"
#include "CUDAEssentials.h"
#include "CUDALaplacePropagation.h"

namespace
{
	struct Entity
	{
		int x;
		int y;
	};

	void laplace(std::vector<float> &vec, int xAxisBound, int yAxisBound, Entity heater)
	{
		for (int i = 1; i < yAxisBound - 1; ++i)
		{
			for (int j = 1; j < xAxisBound - 1; j++)
			{
				if (j != heater.x || i != heater.y)
				{
					vec[i*xAxisBound + j] = (0.25f * (vec[i*xAxisBound + j - 1] + vec[i*xAxisBound + j + 1]
						+ vec[i*xAxisBound + j + yAxisBound] + vec[i*xAxisBound + j - yAxisBound]));
				}
			}
		}
	}

	sf::Image constructImageFromVector(const std::vector<float> &vec, int xAxisBound, int yAxisBound, Entity heater)
	{
		sf::Image board;
		board.create(xAxisBound, yAxisBound, sf::Color::Black);
		for (int i = 1; i < yAxisBound - 1; ++i)
		{
			for (int j = 1; j < xAxisBound - 1; j++)
			{
				sf::Color pixelColor = sf::Color(static_cast<int>(vec[i*xAxisBound + j]) % 255, 0, 0);
				board.setPixel(j, i, pixelColor);
			}
		}
		return board;
	}
}

int main()
{
	CUDAHelpers::CUDASystemInformation systemInformations;
	systemInformations.displaySystemDevicesProperites();
	int xAxisBound = 609;
	int yAxisBound = 609;

	sf::RenderWindow mainWindow(sf::VideoMode(xAxisBound, yAxisBound), "PKG_CUDA", sf::Style::Titlebar | sf::Style::Close);
	mainWindow.setFramerateLimit(60);

	Entity heater;

	std::vector<float> model;
	model.reserve(xAxisBound * yAxisBound);

	sf::Image board;
	board.create(xAxisBound, yAxisBound, sf::Color::Black);

	sf::Texture texture;
	texture.loadFromImage(board);

	sf::Sprite sprite;
	sprite.setTexture(texture, true);

	while (mainWindow.isOpen())
	{
		sf::Event event;

		while (mainWindow.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				mainWindow.close();
			}
		}

		sf::Vector2i mousePositon = sf::Mouse::getPosition(mainWindow);
		
		if (mousePositon.x > 1 && mousePositon.x < xAxisBound && mousePositon.y > 1 && mousePositon.y < yAxisBound)
		{
			heater.x = mousePositon.x;
			heater.y = mousePositon.y;
			model[heater.y * xAxisBound + heater.x] = 255.f;

			//laplace(model, xAxisBound, yAxisBound, heater);		//CPU
			CUDALaplacePropagation::propagate(model, xAxisBound, yAxisBound, heater.x, heater.y);	//GPU

			board = constructImageFromVector(model, xAxisBound, yAxisBound, heater);
			texture.loadFromImage(board);
			sprite.setTexture(texture, true);
		}
		
		mainWindow.clear();
		mainWindow.draw(sprite);
		mainWindow.display();
	}
	
	return 0;
}