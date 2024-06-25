# Portfolio Manager

This tool uses the [SKFolio](https://github.com/skfolio/skfolio) library to optimize a portfolio comprising of various stocks. The data for the stocks is fetched using the [yfinance](https://pypi.org/project/yfinance/) library. 

There is a single crucial file important for the analysis:
1. `src/skfolio_analysis.ipynb`

This is the main piece of code where the optimization takes place. 

**Important Note**: the plots given in `src/skfolio_testing.ipynb` won't show up on the github since they are in HTML format. Run the code locally in order to view the plots. 

**Side note**: It may be required for you to change the values in the `.env` file for the location of the portfolios and the data sources. 