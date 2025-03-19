import pandas as pd
from sqlalchemy import create_engine

servername = "bisqldwhd1"
dbname = "MCH"
engine = create_engine(
    f"mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

df = pd.read_csv("data_after2.csv")
df = df.head(1)
df.to_csv("data_after2.csv", encoding="utf-8", index=False)
df.to_sql("temp_yaakov_t", engine, if_exists="replace", index=False)


exit()
import random
import time

start_time = time.perf_counter()
random_uniform_list = [random.randint(1, 100) for _ in range(100_000_000)]
end_time = time.perf_counter()
print(f"{end_time - start_time:.6f} seconds")


start_time = time.perf_counter()
random_uniform_list[500_000], random_uniform_list[-1] = (
    random_uniform_list[-1],
    random_uniform_list[500_000],
)
random_uniform_list.pop()
end_time = time.perf_counter()

"""
start_time = time.perf_counter()
random_uniform_list.pop(500_000)
end_time = time.perf_counter()
"""

print(f"{end_time - start_time:.6f} seconds")
print(len(random_uniform_list))

""" pop
20.544785 seconds
0.080672 seconds
"""

""" swap and pop
22.350239 seconds
0.000003 seconds
"""
