from dotenv import load_dotenv
import os

load_dotenv()
senha = os.getenv("DB_PASS")


# Configurações de conexão
db_config = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': senha,
    'port': '5432'
}

tbl_ipca = 'tbl_ipca'
tbl_inmet = 'tbl_inmet'
source_inmet = 'H:\Meu Drive\DADOS\SOURCE\INMET'
tbl_ons = 'tbl_ons'
source_ons = 'H:\Meu Drive\DADOS\SOURCE\ONS'
estacoes = 'H:\Meu Drive\DADOS\SOURCE\estacoes.csv'
pib_mensal = 'H:\Meu Drive\DISSERTAÇÃO\pib_mensal.xlsx'
ipca_excel = 'H:\Meu Drive\DISSERTAÇÃO\ipca_historico.xlsx'
tbl_pib = 'tbl_pib'
falhas = 'H:\\Meu Drive\\mpes2023abrm\\falhas.txt'
estacoes_cadastrar = ['A301','A729','A808','A338','A420','A623','A435','A757','A941',
                      'A449','A454','S108','S105','S117','S107','S111','S104','S110','A635','A767',
                      'S704','S706','S708','S710','S712','S713','S715']

estados_brasil = {
    1: "AC",
    2: "AL",
    3: "AP",
    4: "AM",
    5: "BA",
    6: "CE",
    7: "DF",
    8: "ES",
    9: "GO",
    10: "MA",
    11: "MT",
    12: "MS",
    13: "MG",
    14: "PA",
    15: "PB",
    16: "PR",
    17: "PE",
    18: "PI",
    19: "RJ",
    20: "RN",
    21: "RS",
    22: "RO",
    23: "RR",
    24: "SC",
    25: "SP",
    26: "SE",
    27: "TO"  
}

submercados ={
    1: "N", 2:"NE", 3:"SE", 4:"S",0:0
}


estados_submercado ={
    1: 3, 2: 2, 3: 1,4: 1,5: 2,6:2,7: 3,8: 3,
    9: 3,10: 1,
    11: 3,12: 3,
    13: 3,
    14: 1,
    15: 2,
    16: 4,
    17: 2,
    18: 2,
    19: 3,
    20: 2,
    21: 4,
    22: 3,
    23: 0,
    24: 4,
    25: 3,
    26: 2,
    27: 1   
}

area_carga_submercado = ['SECO','N','NE','S']

area_carga_geoeletrica = ['BASE','MA','DF','GO','MS','AM','AP','RR','PI','TON','AC','RS','SP','MT','TOCO','BAOE',
                          'PBRN','PR','ES','MG','SC','CE','RO','PA','RJ','ALPE']

"""Submercado: SECO (Sudeste/C. Oeste), N (Norte), NE (Nordeste), S (Sul)
• Área Geoelétrica: BASE (Bahia/Sergipe), MA (Maranhão), DF (Distrito Federal), GO (Goiás), MS (Mato Grosso do Sul), AM (Amazonas),
AP (Amapá), RR (Roraima), PI (Piauí), TON (Tocantins Norte), AC (Acre), RS (Rio Grande do Sul), SP (São Paulo), MT (Mato Grosso),
TOCO (Tocantins), BAOE (Bahia Oeste), PBRN (Paraíba/Rio Grande do Norte), PR (Paraná), ES (Espírito Santo), MG (Minas Gerais),
SC (Santa Catarina), CE (Ceará), RO (Rondônia), PA (Pará), RJ (Rio de Janeiro), ALPE (Alagoas/Pernambuco)"""