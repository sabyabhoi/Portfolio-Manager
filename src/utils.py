from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
from icecream import ic


def get_returns(ticker: str, index=False):
    if index:
        df = yf.download(ticker, period="max", progress=False)
    else:
        df = yf.download(ticker + ".NS", period="max", progress=False)
    df["Returns"] = df["Adj Close"].pct_change()
    df = df.dropna()[["Returns"]]
    return df.rename(columns={"Returns": ticker}).reset_index().dropna()


def get_multiple_returns(tickers: list[str]):
    df = get_returns(tickers[0])
    for t in tickers[1:]:
        df = pd.merge(df, get_returns(t), on="Date")
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
