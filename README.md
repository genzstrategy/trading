# Gen Z Strategy's Trading Models
This workspace contains various Jupyter notebooks for creating day trading models.

## Directory Structure

### Bitcoin
- `btc/`
  - `causality/`: Contains notebooks for causal analysis of Bitcoin data.
    - [`btc-causal-tigramite.ipynb`](btc/causality/btc-causal-tigramite.ipynb): Causal analysis using Tigramite.
  - `correlation/`: Contains notebooks for correlation analysis of Bitcoin data.
    - [`btc-correlation-analysis.ipynb`](btc/correlation/btc-correlation-analysis.ipynb): General correlation analysis.
    - [`silly-btc-correlation-analysis.ipynb`](btc/correlation/silly-btc-correlation-analysis.ipynb): A more playful take on correlation analysis.
  - `data/`: Contains notebooks for data collection and amalgamation.
    - [`amalgamator.ipynb`](btc/data/amalgamator.ipynb): Notebook for amalgamating data from various sources.
    - [`coin-metrics.ipynb`](btc/data/coin-metrics.ipynb): Notebook for collecting data from Coin Metrics.
    - [`yahoo-finance.ipynb`](btc/data/yahoo-finance.ipynb): Notebook for collecting data from Yahoo Finance.
  - `dudley/`: Contains notebooks for the Dudley deep learning model (Temporal Fusion Transformer).
    - [`dudley-high.ipynb`](btc/dudley/dudley-high.ipynb): Dudley deep learning model for BTC-USD daily high.
    - [`dudley-low.ipynb`](btc/dudley/dudley-low.ipynb): Dudley deep learning model for BTC-USD daily low.
- `eth/`
  - `cointegration/`: Contains notebooks for cointegration analysis of Ethereum data.
    - [`btc-coint.ipynb`](eth/cointegration/btc-coint.ipynb): Cointegration analysis of Bitcoin and Ethereum.
  - `data/`: Contains notebooks for data collection and amalgamation.
    - [`amalgamator.ipynb`](eth/data/amalgamator.ipynb): Notebook for amalgamating data from various sources.
## License

This project is licensed under the terms of the MIT License.

## Contributing

Please read the CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests to me.
