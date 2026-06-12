import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

def get_postgres_engine():
    user     = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host     = os.environ["POSTGRES_HOST"]
    port     = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ["POSTGRES_DB"]

    conn_str = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"

    return create_engine(
        conn_str,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

@contextmanager
def get_postgres_session():
    engine  = get_postgres_engine()   # se crea aquí, no al importar
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def get_postgres_connstring() -> str:
    user     = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host     = os.environ["POSTGRES_HOST"]
    port     = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ["POSTGRES_DB"]
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"