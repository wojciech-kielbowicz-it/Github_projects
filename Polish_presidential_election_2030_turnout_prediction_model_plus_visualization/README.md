Polish Presidential Election 2030 Turnout Prediction Model + Folium Visualization
__________________________________________________________________________________________________________________
License: MIT
Kernel: Python 3.13.9
All modules used in this projects are listed in REQUIREMENTS.txt
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


__________________________________________________________________________________________________________________
Project Structure:

├── Data
│   ├── Raw          
│   └── Processed      
├── Notebooks          
├── README.md
├── LICENSE.txt
└── requirements.txt
__________________________________________________________________________________________________________________
Installation & Usage:

1. Cloning the repository.
2. Dependency installation (pip install -r requirements.txt).
3. Launch a notebook or script.
