import pandas as pd # type: ignore
import numpy as np # type: ignore

# Load the datasetdf = pd.read_csv("E:/DSV/Most Profitable Movies of All Time - Top 500 Movies.csv")
df = pd.read_csv("E:/DAV/Most Profitable Movies of All Time - Top 500 Movies.csv")
df.columns = df.columns.str.strip()  # Strip column names


# --------------------------------------
# ✅ UNIT I – NumPy Operations
# --------------------------------------

budget_array = df["budget  (millions)"].to_numpy()
worldwide_gross_array = df["worldwide gross"].to_numpy()
domestic_gross_array = df["domestic gross (m)"].to_numpy()

print("------ Unit I: NumPy Operations ------")
print("Fixed Type Arrays (Budget Slice):", budget_array[10:15])
print("Reshaped Array:\n", budget_array[:6].reshape(2, 3))
print("Concatenation:\n", np.concatenate((budget_array[:3], domestic_gross_array[:3])))
print("Splitting:\n", np.split(budget_array[:6], 3))
print("Universal Function (Log):", np.round(np.log(worldwide_gross_array[:5]), 2))
print("Aggregation (Sum):", np.sum(budget_array))
print("Aggregation (Mean):", np.mean(budget_array))
print("Broadcasting Example (Add 10):", budget_array[:5] + 10)
print("Boolean Mask (budget > 100):", budget_array[budget_array > 100])
print("Fancy Indexing:", budget_array[[0, 2, 4]])
print("Sorted Budgets:", np.sort(budget_array[:10]))
print("Argsorted Budgets:", np.argsort(budget_array[:10]))

# --------------------------------------
# ✅ UNIT II – Pandas Basics
# --------------------------------------

budget_series = pd.Series(df["budget  (millions)"])
gross_series = pd.Series(df["worldwide gross"])

print("\n------ Unit II: Pandas Operations ------")
print("Series Head:\n", budget_series.head())
print("DataFrame Slice:\n", df.loc[0:4, ["title", "budget  (millions)", "worldwide gross"]])
print("UFunc Example (sqrt):\n", np.sqrt(df["budget  (millions)"]).head())
print("Index Alignment:\n", gross_series - df["domestic gross (m)"])
print("Missing Values:\n", df.isnull().sum())
print("Fill NAs:\n", df.fillna({"budget source": "Unknown", "force label": "None"}).head(1))
print("Hierarchical Indexing:\n", df.set_index(["decade", "year"]).head(1))
print("Grouped by Decade:\n", df.set_index(["decade", "year"]).groupby(level="decade").mean(numeric_only=True))

# --------------------------------------
# ✅ UNIT III – Combining, Grouping, Pivoting
# --------------------------------------

top_half = df.iloc[:250]
bottom_half = df.iloc[250:]

print("\n------ Unit III: Combining & Pivoting ------")

# Concatenation
concatenated_df = pd.concat([top_half, bottom_half])
print("Concatenated DF Head:\n", concatenated_df.head(2))

# Append (deprecated but shown here)
appended_df = pd.concat([top_half, bottom_half])

print("Appended DF Head:\n", appended_df.head(2))

# Merge
df_year_budget = df[["year", "budget  (millions)"]]
df_year_title = df[["year", "title"]]
merged_df = pd.merge(df_year_budget, df_year_title, on="year", how="inner")
print("Merged DF Head:\n", merged_df.head(2))

# Grouping
grouped = df.groupby("decade").agg({
    "budget  (millions)": "mean",
    "worldwide gross": "sum"
})
print("Grouped DF:\n", grouped)

# Pivot Table
pivot_table = pd.pivot_table(df, values="worldwide gross", index="decade", columns="horror", aggfunc="mean")
print("Pivot Table:\n", pivot_table)
