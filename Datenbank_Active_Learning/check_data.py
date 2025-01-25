import sqlite3
import pandas as pd

con = sqlite3.connect('rotten_tomatoes.db')

cur = con.cursor()

df = pd.read_sql_query("SELECT * FROM reviews", con)

print(df)