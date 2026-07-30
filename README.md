# Risk-Scope
RiskScope is a Bloomberg Terminal-inspired platform for market risk analysts, providing real-time market data, portfolio analytics, risk metrics, and interactive financial visualizations in one place.

# 📟 RiskScope – Bloomberg Terminal for Market Risk Analysts

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo_Finance-purple?style=for-the-badge)

### A Bloomberg Terminal-inspired platform for **Market Risk Analysts**, providing institutional-grade portfolio analytics, risk metrics, option pricing, Monte Carlo simulations, and financial visualizations—all in a modern Streamlit interface.

</div>

---

# 📸 Application Preview
ons.png" width="90%">

<img src="images/riskmetrics.png" width="90%">



---

# 📖 Overview

RiskScope is an advanced **Market Risk Analytics Platform** inspired by professional terminals such as **Bloomberg** and **Refinitiv Workspace**.

The application enables traders, risk analysts, finance students, and portfolio managers to analyze financial assets using institutional risk models.

Unlike traditional stock dashboards that only display prices and charts, RiskScope focuses on **risk measurement**, helping users understand:

- How much can I lose?
- What is the probability of that loss?
- How risky is my portfolio?
- How sensitive are my options?
- How would my portfolio behave during market crashes?

The project combines financial engineering with modern data visualization to create an intuitive Bloomberg-like experience.

---

# 🚀 Features

## 📈 Market Risk Terminal

Analyze any financial instrument using Yahoo Finance.

Features include:

- Live Market Prices
- Historical Price Charts
- Daily Returns
- Cumulative Returns
- Annualized Returns
- Annualized Volatility
- Rolling Volatility
- Drawdown Analysis
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Beta
- Alpha
- Skewness
- Kurtosis

---

## ⚠ Value at Risk (VaR)

Supports multiple VaR models.

### Historical VaR

Calculates loss using historical returns.

### Parametric VaR

Variance-Covariance Method

Confidence Levels

- 90%
- 95%
- 99%

---

## 📉 Expected Shortfall (CVaR)

Measures the average loss beyond VaR.

Also known as:

- Conditional VaR
- Tail Risk

---

## 🎲 Monte Carlo Simulation

Simulate thousands of future price paths using Geometric Brownian Motion.

Includes:

- Thousands of simulations
- Expected Future Price
- Confidence Intervals
- Future Return Distribution
- Simulated VaR

---

## 📊 Historical Stress Testing

Analyze how assets perform during historical market crashes.

Examples

- COVID Crash (2020)
- Global Financial Crisis (2008)
- Interest Rate Shock
- Custom Stress Scenarios

---

## 💼 Portfolio Analytics

Build custom portfolios consisting of

- Stocks
- ETFs
- Commodities
- Bonds
- Forex
- Cryptocurrency

Portfolio Metrics

- Portfolio Return
- Portfolio Volatility
- Portfolio VaR
- Portfolio CVaR
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Correlation Matrix
- Diversification Analysis
- Risk Contribution
- Monte Carlo Portfolio Simulation

---

## 🧮 Options Lab

Institutional Black-Scholes Pricing Engine

Calculates

- Call Price
- Put Price

Greeks

- Delta
- Gamma
- Vega
- Theta
- Rho

Interactive Payoff Diagrams included.

---

# 📊 Risk Metrics Included

| Market Risk | Performance | Distribution |
|-------------|------------|--------------|
| Historical VaR | Sharpe Ratio | Mean |
| Parametric VaR | Sortino Ratio | Median |
| Expected Shortfall | Calmar Ratio | Standard Deviation |
| Monte Carlo VaR | Alpha | Skewness |
| Stress Testing | Beta | Kurtosis |
| Max Drawdown | CAGR | Downside Deviation |

---

# 🛠 Technology Stack

### Frontend

- Streamlit

### Backend

- Python

### Data

- Yahoo Finance API

### Libraries

- Pandas
- NumPy
- SciPy
- Plotly
- Matplotlib
- yFinance

---

# 📂 Project Structure

```
RiskScope/

│── app.py
│── calculations.py
│── requirements.txt
│── README.md

├── images/
│     home.png
│     terminal.png
│     portfolio.png
│     options.png
│     montecarlo.png
│     riskmetrics.png
```

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/RiskScope.git
```

Move into project

```bash
cd RiskScope
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# 📌 Supported Assets

Examples

### Stocks

```
AAPL
MSFT
TSLA
RELIANCE.NS
```

### ETFs

```
SPY
QQQ
```

### Indices

```
^GSPC
^NSEI
^IXIC
```

### Commodities

```
GC=F
SI=F
CL=F
```

### Forex

```
EURUSD=X
USDINR=X
```

### Crypto

```
BTC-USD
ETH-USD
```

---

# 🎯 Future Roadmap

- Efficient Frontier
- Portfolio Optimization
- GARCH Volatility Models
- Modified Cornish-Fisher VaR
- VaR Backtesting
- Credit Risk Module
- Fixed Income Analytics
- Option Strategy Builder
- Live News Feed
- Economic Calendar
- AI Risk Assistant
- Portfolio Recommendation Engine

---

# 🎓 Educational Purpose

RiskScope was built to bridge the gap between classroom finance and real-world market risk analytics.

It demonstrates concepts commonly used by

- Bloomberg Terminal
- JP Morgan
- Goldman Sachs
- Morgan Stanley
- BlackRock
- Citi
- Barclays

making it an ideal portfolio project for aspiring

- Market Risk Analysts
- Quantitative Analysts
- Financial Data Scientists
- Portfolio Managers
- Risk Engineers

---

# 👨‍💻 Author

## Adithya Machavarapu

B.Tech Computer Science (Data Analytics)

Market Risk | Quantitative Finance | Financial Data Science

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

# ⭐ Support

If you found this project useful,

⭐ Star the repository

🍴 Fork the repository

🚀 Share it with fellow finance enthusiasts

---

## ⚠ Disclaimer

RiskScope is an educational and research project inspired by professional financial terminals such as Bloomberg Terminal. All market data is sourced from Yahoo Finance and may be delayed. This application is **not intended to provide financial or investment advice**.
