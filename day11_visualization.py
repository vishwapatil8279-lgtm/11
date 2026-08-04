import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the CSV Files

customers = pd.read_csv("Capstone Customers.csv")
orders = pd.read_csv("Capstone Orders.csv")

customers.head()
orders.head()

#Check Missing Values

print(customers.isnull().sum())

print(orders.isnull().sum())

#Remove Duplicate Rows

customers.drop_duplicates(inplace=True)

orders.drop_duplicates(inplace=True)

print(customers.duplicated().sum())

print(orders.duplicated().sum())

#Standardize Text

customers["Region"] = customers["Region"].str.strip().str.title()

customers["CustomerName"] = customers["CustomerName"].str.strip().str.title()

orders["Category"] = orders["Category"].str.strip().str.title()

print(customers["Region"].unique())

print(orders["Category"].unique())

#Handle Missing Values Correctly

orders["Sales"] = orders["Sales"].fillna(orders["Sales"].median())

orders["Profit"] = orders["Profit"].fillna(orders["Profit"].median())

orders["Quantity"] = orders["Quantity"].fillna(orders["Quantity"].median())

orders["Discount"] = orders["Discount"].fillna(orders["Discount"].median())

customers["Region"] = customers["Region"].fillna("Unknown")

orders["Category"] = orders["Category"].fillna("Unknown")

print(customers.isnull().sum())

print(orders.isnull().sum())

#Convert OrderDate

orders["OrderDate"] = pd.to_datetime(
    orders["OrderDate"],
    dayfirst=True,
    errors="coerce"
)

print(orders.dtypes)

#Merge Both Tables

df = pd.merge(
    customers,
    orders,
    on="CustomerID"
)

df.head()

#Validate Data

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.head())

#Create a Month Column

df["Month"] = df["OrderDate"].dt.strftime("%b")
print(df[["OrderDate","Month"]].head())

#1 A trend over time Monthly sales trend

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(12,6))

monthly_sales.plot(
    marker='o',
    color='blue',
    label='Sales'
)

plt.title("Monthly Sales Trend")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.savefig("line_chart.png")

plt.show()

print("Insight: Highest monthly sales =", monthly_sales.max())

#2 Bar chart — a metric compared across categorie

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,6))

category_sales.plot(
    kind="bar",
    color="green",
    label="Sales"
)

plt.title("Sales by Category")

plt.xlabel("Category")

plt.ylabel("Sales")

plt.legend()

plt.savefig("Sales by Category.png")

plt.show()

print("Insight: Highest sales category =", category_sales.idxmax())

#3 Grouped bar chart — two dimensions together (e.g. sales by region and category)

pivot = df.pivot_table(
    values="Sales",
    index="Region",
    columns="Category",
    aggfunc="sum"
)

pivot.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Sales by Region and Category")

plt.xlabel("Region")

plt.ylabel("Sales")

plt.legend(title="Category")

plt.savefig(" sales by region and category.png")

plt.show()

print("Insight: Compare category sales across each region.")

#4 Histogram — distribution of a numeric column

plt.figure(figsize=(8,6))

plt.hist(
    df["Sales"],
    bins=20,
    color="orange",
    edgecolor="black",
    label="Sales"
)

plt.title("Distribution of Sales")

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.legend()

plt.savefig("histogram.png")

plt.show()

print("Insight: Maximum sales =", df["Sales"].max())

#5 Scatter plot — relationship between two numeric columns

plt.figure(figsize=(8,6))

plt.scatter(
    df["Sales"],
    df["Profit"],
    color="red",
    alpha=0.6,
    label="Orders"
)

plt.title("Sales vs Profit")

plt.xlabel("Sales")

plt.ylabel("Profit")

plt.legend()

plt.savefig("scatter_plot.png")

plt.show()

print("Insight: Highest profit =", df["Profit"].max())

# 6 Box plot — spread and outliers in a numeric column

plt.figure(figsize=(8,6))

sns.boxplot(
    y=df["Profit"],
    color="skyblue"
)

plt.title("Profit Box Plot")

plt.ylabel("Profit")

plt.savefig("box_plot.png")

plt.show()

print("Insight: Box plot shows outliers in Profit.")

#7 Correlation heatmap — all numeric columns

numeric = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")

plt.show()

print("Insight: Highest correlation =", numeric.corr().max().max())

#8 One chart of your choice, justified in a comment
#Pie chart shows the percentage contribution of each category to total sales.

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,8))

plt.pie(
    category_sales,
    labels=category_sales.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sales Share by Category")

plt.savefig("pie_chart.png")

plt.show()

print("Insight: Largest sales contribution comes from", category_sales.idxmax())
