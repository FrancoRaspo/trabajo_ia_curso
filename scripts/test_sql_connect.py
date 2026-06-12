# test_conexion.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

import pyodbc

host     = "192.168.253.10"
instance = "sql2019"
database = os.environ["SQLSERVER_DB"]
user     = os.environ["SQLSERVER_USER"]
password = os.environ["SQLSERVER_PASSWORD"]

print(f"Conectando a: {host}\\{instance} / {database}")
print(f"Usuario: {user}\n")

odbc_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={host}\\{instance};"
    f"DATABASE={database};"
    f"UID={user};"
    f"PWD={password};"
    f"TrustServerCertificate=yes;"
    f"Connection Timeout=15;"
)

try:
    conn = pyodbc.connect(odbc_str)
    cur  = conn.cursor()

    cur.execute("SELECT @@VERSION")
    version = cur.fetchone()[0]
    print(f"✅ Conexión exitosa")
    print(f"   {version[:60]}\n")

    cur.execute("SELECT @@SERVERNAME")
    print(f"   Servidor: {cur.fetchone()[0]}")

    cur.execute("SELECT DB_NAME()")
    print(f"   Base activa: {cur.fetchone()[0]}")

    conn.close()
    print("\n→ Conexión OK. Podés correr el pipeline.")

except Exception as e:
    print(f"❌ Error: {e}")