from utils import PortfolioOptimizer
from icecream import ic

if __name__ == "__main__":
    companies = ["SBIN", "CUMMINSIND", "HCLTECH"]
    po = PortfolioOptimizer(tickers=companies, rf=0.0721, tangency=False)

    po.optimize_portfolio()
    ic(po.returns())
    ic(po.risk())
    ic(po.sharpe())
