from dataclasses import dataclass, field
from babel.numbers import format_currency
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import os
from icecream import ic

from dotenv import load_dotenv
load_dotenv()

from skfolio import Population, RiskMeasure, Portfolio
from skfolio.optimization import (
    MeanRisk,
    ObjectiveFunction,
)

def fv(amount: float, r: float, t: float):
    return (amount * ((1 + r)**t - 1) * (1 + r)) / r


def print_sip_info(monthly_amount, annual_rate, months):
    fmt = lambda amt: format_currency(amt, 'INR', locale='en_IN')

    monthly_rate = (1+annual_rate)**(1/12) - 1
    future_amount = fv(monthly_amount, monthly_rate, months)

    print('amount invested:', fmt(monthly_amount*months))
    print('final amount:', fmt(future_amount))
    print('amount gained:', fmt(future_amount - monthly_amount*months))
    print(f'CAGR: {round((future_amount - monthly_amount*months) / (monthly_amount*months) * 100, 2)}%')

def get_returns(ticker: str, index=False):
    DATA_PATH = os.getenv('DATA_PATH')
    filepath = DATA_PATH + '/' + ticker + ".parquet"

    def process_df(df):
        df["Returns"] = df["Adj Close"].pct_change()
        df = df.dropna()[["Returns"]]
        df = df.rename(columns={"Returns": ticker}).reset_index().dropna()
        return df

    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        last = (dt.datetime.now() - dt.timedelta(3)).date()

        if df.Date.iloc[-1] >= pd.Timestamp(last):
            return df

        df1 = yf.download(
            ticker if index else ticker + ".NS",
            start=df.Date.iloc[-1].date().isoformat(),
            progress=False,
        )
        if df1.empty:
            return df
        df1 = process_df(df1)
        return pd.concat([df, df1])

    df = yf.download(ticker if index else ticker + ".NS", period="max", progress=False)

    df = process_df(df)
    df.to_parquet(filepath)
    return df


def get_multiple_returns(tickers: list[str]):
    df = get_returns(tickers[0])
    for t in tickers[1:]:
        print(f'Fetching data for {t}:', end=' ')
        tmp = get_returns(t)
        print(f'{tmp.shape[0]} rows')
        df = pd.merge(df, tmp, on="Date")
    df.index = df["Date"]
    return df.drop(columns=["Date"]).dropna()


def sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float):
    return ((w.T @ mu - rf) / (w.T @ sigma @ w) ** 0.5)[0][0]


def weights_to_pd(w: np.ndarray, tickers: list[str]):
    df = pd.DataFrame(w)
    df.index = tickers
    return df.rename(columns={0: "w"})


@dataclass
class PortfolioOptimizer:
    tickers: list[str]
    rf: float
    tangency: bool = False
    df: pd.DataFrame = field(default=None)

    def __post_init__(self) -> None:
        if self.df is None:
            ic("Fetching ticker data")
            self.df = get_multiple_returns(self.tickers)

        self.exp_ret = (((1 + self.df.mean()) ** 252) - 1).to_numpy()
        self.exp_ret = self.exp_ret.reshape(len(self.exp_ret), 1)

        self.cov_mat = (self.df.cov() * 252).to_numpy()

    def portfolio_fitness(self, w) -> float:
        if abs(w.sum() - 1) > 1e-2:
            return -(1e9 + 7)
        if self.tangency:
            return sharpe(w, self.exp_ret, self.cov_mat, self.rf)
        else:
            return (-w.T @ self.cov_mat @ w).flatten()[0]

    def optimize_portfolio(self):
        def fitness(w):
            return self.portfolio_fitness(w)

        ga = GeneticAlgorithm(fitness=fitness)
        ic("Running GA")
        self.w = ga.run(len(self.tickers))
        return self.w

    def returns(self):
        return (self.w.T @ self.exp_ret).flatten()[0]

    def risk(self):
        return (self.w.T @ self.cov_mat @ self.w).flatten()[0]

    def sharpe(self):
        return sharpe(self.w, self.exp_ret, self.cov_mat, self.rf)

    def portfolio_to_excel(self, writer, sheetname, imagename):
        # writer = pd.ExcelWriter(filename, engine="xlsxwriter")

        df = weights_to_pd(self.w, self.tickers)
        df.to_excel(writer, index=True, sheet_name=sheetname, float_format="%.4f")

        worksheet = writer.sheets[sheetname]
        worksheet.insert_image("H1", imagename)

        df_overall = pd.DataFrame(
            {
                "Returns": [self.returns()],
                "Risk": [self.risk()],
                "Sharpe Ratio": [self.sharpe()],
            }
        )

        worksheet.write_string(df.shape[0] + 2, 0, "Annualized Metrics")
        df_overall.to_excel(
            writer,
            sheet_name=sheetname,
            startrow=df.shape[0] + 3,
            index=False,
            float_format="%.4f",
        )
        worksheet.autofit()
        # writer.close()


@dataclass
class GeneticAlgorithm:
    fitness: callable
    iterations = 2000
    mutation_variation = 0.3
    offsprings = 200
    top_select = offsprings // 2

    def init_w(self, n: int, random=True):
        if random:
            w = np.random.uniform(0, 1, (n, 1))
            w /= w.sum()
        else:
            w = np.zeros((n, 1))
            w[0][0] = 1
        return w

    def run(self, N: int):
        solutions = np.array([self.init_w(N) for _ in range(self.offsprings)])

        for i in range(self.iterations):
            fitness_values = np.array([self.fitness(w) for w in solutions])
            ranked_indices = np.argsort(fitness_values)[::-1]
            solutions = solutions[ranked_indices.astype(int)]

            best = solutions[: self.top_select]

            selected_indices = np.random.choice(self.top_select, size=self.offsprings)
            mutated = best[selected_indices] * np.random.uniform(
                1 - self.mutation_variation / 2,
                1 + self.mutation_variation / 2,
                (self.offsprings, N, 1),
            )
            mutated /= mutated.sum(axis=1, keepdims=True)

            solutions = mutated

        solutions = solutions.reshape(
            solutions.shape[0],
            solutions.shape[1],
        )
        solutions = pd.DataFrame(solutions).mean().to_numpy()
        solutions = solutions.reshape(solutions.shape[0], 1)
        return solutions


def rolling_window_portfolio(X, division=5):
    N = X.shape[0]
    k = N // division

    l, r = 0, k
    pred = []
    for i in range(division):

        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            risk_measure=RiskMeasure.VARIANCE,
        )
        pred.append(model.fit_predict(X.iloc[l:r]))
        l += k
        r += k

    population = Population(pred)
    composition = population.composition()
    weights = composition.T.mean().to_numpy()

    mean_portfolio = Portfolio(X=X, weights=weights, name="Rolling Window")
    return mean_portfolio
