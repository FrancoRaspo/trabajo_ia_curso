import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

def get_sqlserver_engine():
    host   = os.environ["SQLSERVER_HOST"]
    instance =  os.environ["SQLSERVER_INSTANCE"]
    database = os.environ["SQLSERVER_DB"]
    user     = os.environ["SQLSERVER_USER"]
    password = os.environ["SQLSERVER_PASSWORD"]


    odbc_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={host}\\{instance};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=15;"
    )

    connection_url = URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": odbc_str}
    )

    return create_engine(connection_url, pool_pre_ping=True)

@contextmanager
def get_sqlserver_session():
    engine  = get_sqlserver_engine()   # se crea aquí, no al importar
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def get_sqlserver_connstring() -> str:
    host = os.environ["SQLSERVER_HOST"]
    instance = os.environ["SQLSERVER_INSTANCE"]
    database = os.environ["SQLSERVER_DB"]
    user = os.environ["SQLSERVER_USER"]
    password = os.environ["SQLSERVER_PASSWORD"]

    return (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={host}\\{instance};"
        f"DATABASE={database};"
        f"UID={user};PWD={password};"
        f"TrustServerCertificate=yes;"
    )