import pandas as pd
from sqlalchemy import create_engine


def get_data():
    servername = "bisqldwhd1"
    dbname = "MCH"
    engine = create_engine(
        f"mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
    )

    return pd.read_sql(
        "SELECT top 250000 * FROM [MCH].[SH\hm24].[yaakov_temp_table]", engine
    )
