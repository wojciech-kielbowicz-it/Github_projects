from collections.abc import Iterable
import csv
from io import StringIO
import os
from sqlalchemy import create_engine, Connection
from dotenv import load_dotenv


def get_db_engine():
    
    load_dotenv()
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    engine = create_engine(connection_string)

    return engine

def psql_insert_copy[T: {any}](
        table: T, 
        connect: Connection, 
        keys: list[str], 
        data_iter: Iterable[tuple[T,...]]
) -> int|None :

    s_buf = StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    dbapi_conn = connect.connection
    with dbapi_conn.cursor() as cur:
        schema_prefix = f"{table.schema}." if table.schema else ""
        sql = f"COPY {schema_prefix}{table.name} ({', '.join(keys)}) FROM STDIN WITH CSV"
        cur.copy_expert(sql=sql, file=s_buf)
