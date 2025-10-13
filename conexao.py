import psycopg2
import pandas as pd
import traceback  # 📌 Import para exibir erros detalhados

def insert_table(db_config, tabela, df):
    conn = None
    cursor = None


    try:
        # Conectar ao banco
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Obter colunas e placeholders
        colunas = ", ".join(df.columns)
        valores = ", ".join(["%s"] * len(df.columns))

        # Criar query parametrizada
        insert_query = f"INSERT INTO {tabela} ({colunas}) VALUES ({valores})"

        # Converter DataFrame para lista de tuplas
        data_tuples = [tuple(row) for row in df.itertuples(index=False, name=None)]

        # Executar inserção em lote
        cursor.executemany(insert_query, data_tuples)
        conn.commit()


    except Exception as e:
        print(f"❌ Erro ao inserir dados: {e}")
        traceback.print_exc()  # 📌 Isso exibe o erro completo
        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def sql_generico(sql_query, db_config):
    conn = None
    try:
        # Configura a conexão
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

            # Executa a consulta SQL
        cur.execute(sql_query)

            # Converte o resultado em um DataFrame
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

            # Confirma as operações no banco de dados
        conn.commit()
            
            # Fecha o cursor
        cur.close()
        return df
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    finally:
            # Fecha a conexão se ela foi aberta
        if conn is not None:
            conn.close()