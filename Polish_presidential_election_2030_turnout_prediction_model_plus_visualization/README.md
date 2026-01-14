Polish Presidential Election 2030 Turnout Prediction Model + Folium Visualization
__________________________________________________________________________________________________________________
License: MIT
Kernel: Python 3.13.9
All modules used in this projects are listed in requirements.txt
__________________________________________________________________________________________________________________
Project Goals:

The data used for this project include a number of demographic and economic indicators available on the Polish Central Statistical Office platform.

This model does not take into account indicators that cannot be included in the data framework, such as the influence of the media or even election campaigns.

Naturally, these factors have a large impact on turnout results, but this project focuses on what demographic indicators, and more specifically their change, influence citizens to be more willing to vote in the presidential elections.
__________________________________________________________________________________________________________________
Data Source:

Election data:
- Data from the presidential elections 2000 - 2025

Indicators:
- GDP per capita
- Average gross salary
- Demographic dependency ratio
- Population 70 plus
- Population density
- Total population
- Unemployment
- Urbanization rate

Other:
- Geo spacial data
- Territory codes
- County names

All data comes from the Polish Central Statistical Office (GUS)
__________________________________________________________________________________________________________________
Methodology:

The project used, among others, linear extrapolation and the ARIMA model to supplement some data that are unavailable for some years.

Some data was also divided between counties, which have changed their borders over the years, adapting them to the current county boundaries, so that today's realities could be reflected as closely as possible and could finally be presented on an interactive visualization of the map of Poland.
__________________________________________________________________________________________________________________
Results:

As a result of the prediction, the most effective XGBoost model achieved an MAE error close to 5.22, training on electoral data and indicators from 2000-2020 and predicting the turnout for 2025, which was then compared with historical data from 2025. 

Using the same model and parameters, voter turnout for 2030 was predicted for two election rounds, which was later visualized on an interactive map of Poland.

Taking into account the fact that factors such as demographic and economic indicators are not the only factors influencing the mobilization of the population to participate in elections, and that factors such as the media, election campaign, competition of political parties or even the election promises of the candidates themselves have a huge impact on the mobilization of the electorate, achieving such a result by the predictive model is undoubtedly a success.

Thanks to this project, it is possible to determine what economic and demographic factors and their changes most mobilize the electorate to participate in the elections.

Of course, due to the significant, in some places, lack of data for some units and the frequent use of various models to supplement them, this project is not one hundred percent based on historical data, and to a small extent on data predicted by statistical models based on data that was present in other years, nevertheless, this project is an interesting object for reflection on the impact on the mobilization of people to participate in elections and can be used for a deeper analysis of patterns and society itself, in this case the inhabitants of Poland, over the years.

__________________________________________________________________________________________________________________
Project Structure:

├── Data
│   ├── Raw          
│   └── Processed      
├── Notebooks
├── Visualization         
├── README.md
├── LICENSE.txt
└── requirements.txt
__________________________________________________________________________________________________________________
Installation & Usage:

1. Cloning the repository.
2. Dependency installation (pip install -r requirements.txt).
3. Launch a notebook or script.
4. Downloading interactive map (HTML format)
