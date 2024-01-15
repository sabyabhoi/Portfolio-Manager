# %%
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import utils
import matplotlib.pyplot as plt

sns.set_theme()

# %%
tickers = input("Enter list of comma separated tickers: ").split(",")

# %%
ret = utils.get_multiple_returns(tickers)
# sns.heatmap(ret.corr(), vmin=-1, vmax=1, annot=True)
# plt.savefig("./res/heatmap.png")

# %%
plt.ylabel("Returns")
plt.xticks(rotation=30)
sns.lineplot((1 + ret).cumprod())


# %%
def plot_against_nifty(ret: pd.DataFrame, w: np.ndarray, filename):
    fig = plt.figure()
    plt.title("Portfolio returns vs NIFTY 50")
    df = pd.merge(
        (1 + ret @ w).cumprod(),
        utils.get_returns("^NSEI", index=True),
        on="Date",
        how="inner",
    ).rename(columns={0: "Portfolio", "^NSEI": "NIFTY50"})
    df["NIFTY50"] = (1 + df["NIFTY50"]).cumprod()
    df.index = df["Date"]
    df = df.drop(columns=["Date"])
    lplot = sns.lineplot(df)
    plt.xticks(rotation=45)
    plt.savefig(filename)


# %% [markdown]
# # Tangency Portfolio
#

# %%
po_tangency = utils.PortfolioOptimizer(tickers=tickers, rf=0.0721, tangency=True)
w_tangency = po_tangency.optimize_portfolio()

plot_against_nifty(ret, w_tangency, "./res/tangency.png")

# %%
writer = pd.ExcelWriter("./res/report.xlsx", engine="xlsxwriter")

# %%
ret.to_excel(writer, "Returns data", float_format="%.4f")
worksheet = writer.sheets["Returns data"]
# worksheet.insert_image("H1", "./res/heatmap.png")
worksheet.autofit()

# %%
po_tangency.portfolio_to_excel(writer, "Tangency Portfolio", "./res/tangency.png")

# %% [markdown]
# # Minimum Variance Portfolio
#

# %%
po = utils.PortfolioOptimizer(tickers=tickers, rf=0.0721, tangency=False)
w_min_var = po.optimize_portfolio()

df_w_min_var = utils.weights_to_pd(w_min_var, tickers)
df_w_min_var

# %%
plot_against_nifty(ret, w_min_var, "./res/min_var.png")

# %% [markdown]
# ## Export to Excel
#

# %%
po.portfolio_to_excel(writer, "Minimum Variance Portfolio", "./res/min_var.png")

# %% [markdown]
# # Equal Weight Portfolio
#

# %%
w_equ = np.ones((len(tickers), 1)) / len(tickers)
utils.weights_to_pd(w_equ, tickers)

# %%
plot_against_nifty(ret, w_equ, "./res/equal.png")

# %%
po_equ = utils.PortfolioOptimizer(tickers=tickers, rf=0.0721, tangency=False)
po_equ.w = w_equ

# %%
po.sharpe()

# %%
po_equ.portfolio_to_excel(writer, "Equal Weight Portfolio", "./res/equal.png")
writer.close()
