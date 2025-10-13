#BIBLIOTECAS


from conexao import sql_generico
from config import db_config


from pmdarima import auto_arima
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import time

from pycaret.regression import *
from pycaret.regression import RegressionExperiment,save_model, load_model, predict_model
from pycaret.regression import save_model, load_model, predict_model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#BASES DE DADOS
df_tempmedia =  []#pd.read_excel("H:\\Meu Drive\\DISSERTAÇÃO\\dados\\tempo_med_uf_diario.xlsx")
df_populacao = [] #pd.read_excel("H:\Meu Drive\DISSERTAÇÃO\dados\populacao_2000_2024.xlsx")
df_modelo = pd.read_excel("C:\\Users\\AnaBeatriz\\Documents\\PESSOAL\\mpes2023abrm\\dados\\df_modelo_carga.xlsx") # sql_generico('select * from mview_dados_modelo', db_config) H:\Meu Drive\mpes2023abrm\script\df_modelo_carga.xlsx
df_ons = sql_generico('select * from tbl_ons', db_config)
df_ipca = sql_generico('select * from tbl_ipca', db_config)
df_pib = sql_generico('select * from tbl_pib', db_config)
df_feriados = sql_generico('select * from tbl_feriados', db_config)

def print_elapsed_time(start_time, step_name):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{step_name}: {elapsed_time:.2f} segundos")
    return end_time 

def db_modelo_carga(df_tempmedia,df_populacao,df_ons,df_ipca,df_pib,df_feriados):

#UNIÃO DE TABELAS
    #agrupar temperatura media por submercado
    print("AGRUPAMENTO TEMPERATURA POR SUBMERCADO")

    df_tempmedia['ano'] = df_tempmedia['dataref'].dt.year
    df_tempmedia = df_tempmedia.merge(df_populacao,on=['uf', 'ano'], how='left')
    df_tempmedia = df_tempmedia.groupby(['submercado', 'dataref']).agg(
        temp_eq = ('temp_med', lambda x: (x * df_tempmedia.loc[x.index, 'populacao']).sum() / df_tempmedia.loc[x.index, 'populacao'].sum()),
    ).reset_index()

    #junção temp + carga
    print("UNIAO TEMPERATURA E CARGA ")

    df_ons['dataref'] = pd.to_datetime(df_ons['dataref'])
    df_tempmedia['dataref'] = pd.to_datetime(df_tempmedia['dataref'])
    df_modelo = df_ons.merge(df_tempmedia,on=['submercado', 'dataref'], how='left')

    #df_inmetdiario = df_inmetdiario[df_inmetdiario['temperatura_media'] != 0]
 
    #TRATAMENTOS DF_MODELO
    df_modelo['dataref'] = pd.to_datetime(df_modelo['dataref'], errors='coerce')
    df_modelo['ano'] = df_modelo['dataref'].dt.year
    df_modelo['mes'] = df_modelo['dataref'].dt.month

    #TRATAMENTOS DF_IPCA
    df_ipca = df_ipca.pivot_table(index=['ano', 'mes'], columns='categoria', values='ipca_taxa').reset_index()
    df_ipca = df_ipca.rename(columns={'saude': 'ipca_saude','transportes': 'ipca_transportes'})

    #UNIR DF_MODELO E DF_IPCA e PIB
    print("INSERINDO DADOS IPCA E PIB")

    df_modelo = df_modelo.merge(df_ipca,on=['ano', 'mes'], how='left')
    df_modelo = df_modelo.merge(df_pib, on=['ano', 'mes'], how='left')

    #FLAG FERIADOS
    print("INSERINDO FLAG FERIADOS")
    df_feriados['data'] = pd.to_datetime(df_feriados['feriados'], errors='coerce')
    feriados = set(df_feriados['data'])
    df_modelo['feriado'] = df_modelo['dataref'].apply(lambda x: 1 if x in feriados else 0)
    df_modelo['dia_semana'] = df_modelo.apply(lambda x: x['dataref'].weekday(), axis=1)  

    #LAG CARGA, IPCA E TEMPERATURA
    print("INSERINDO LAG CARGA, TEMPERATURA E IPCA")
    df_modelo = df_modelo.sort_values(by=['submercado', 'dataref'])
    df_modelo['carga_lag1'] = df_modelo.groupby('submercado')['carga'].shift(1)
    df_modelo['carga_lag7'] = df_modelo.groupby('submercado')['carga'].shift(7)
    df_modelo['ipca_saude_lag1'] = df_modelo.groupby('submercado')['ipca_saude'].shift(1)
    df_modelo['ipca_transportes_lag1'] = df_modelo.groupby('submercado')['ipca_transportes'].shift(1)
    df_modelo['temp_eq_lag1'] = df_modelo.groupby('submercado')['temp_eq'].shift(1)
    df_modelo['temp_eq_lag7'] = df_modelo.groupby('submercado')['temp_eq'].shift(7)

    #EXCLUSÃO COLUNAS e valores
    print("FINALIZANDO UNIAO TABELAS")
    df_modelo = df_modelo.drop(columns=['mes','ano'])
    df_modelo = df_modelo.dropna(subset=['carga','temp_eq'])
    

    return df_modelo




def obter_estacao(data):
    mes = data.month
    if mes in [12, 1, 2]:
        return '0'
    elif mes in [3, 4, 5]:
        return '1'
    elif mes in [6, 7, 8]:
        return '2'
    elif mes in [9, 10, 11]:
        return '3'

def treinamento_ML(df_modelo,n_folds,target_m,submercado,setup_m):

    # ------------------
    # SETUP EXPERIMENTO
    # ------------------
    exp = RegressionExperiment()

    if setup_m == 'timeseries':
        exp.setup(
            data=df_modelo,
            target=target_m,
            fold_strategy='timeseries',
            fold=n_folds,
            data_split_shuffle=False,
            fold_shuffle=False,
            date_features=['dataref'],
            categorical_features=['dia_semana','estacao_ano'],
            normalize=True,
            remove_outliers=True,
            session_id=42,
            feature_selection=False,
            numeric_imputation='median',
            verbose=False
        )

    elif setup_m == 'kfold':
        exp.setup(
            data=df_modelo.drop(columns=['dataref']),
            target=target_m,
            fold_strategy='kfold',
            fold=n_folds, 
            session_id=42,
            normalize=True,
            categorical_features=['dia_semana','estacao_ano'],
            remove_outliers=True,
            numeric_imputation='median',
            feature_selection=False,
            verbose=False
        )

    # ------------------
    # MELHOR MODELO BASE
    # ------------------
    best_model = exp.compare_models(sort='MAPE')
    
    pd.Series(best_model.get_params()).to_excel(f'{submercado}_{setup_m}_best_model_params.xlsx')

    # MÉTRICAS BASE TREINO
    resultados_treino = exp.pull()
    resultados_treino['base'] = 'treino'

    #SALVAR MELHOR MODELO
    modelo_nome = type(best_model).__name__
    print(modelo_nome)
    exp.save_model(best_model, modelo_nome+submercado+setup_m)

    # PREVISÃO NA BASE TESTE (MODELO BASE)
    predictions_base = exp.predict_model(best_model)

    # MÉTRICAS BASE TESTE (MODELO BASE)
    metricas_teste_base = exp.pull()
    metricas_teste_base['base'] = 'teste'

    
    metricas_resumo = pd.concat([resultados_treino, metricas_teste_base], ignore_index=True)
    metricas_resumo['submercado'] = submercado
    metricas_resumo['validacao'] = setup_m
    colunas = ['submercado','validacao','base','Model','MAE','MSE','RMSE',	'R2','RMSLE','MAPE']
    metricas_resumo = metricas_resumo.loc[:, colunas]

    # ------------------
    # GRÁFICOS DE ANÁLISE VISUAL
    # ------------------

    # VALORES REAIS VS PREVISTOS (TESTE - MODELO BASE)
    plt.figure(figsize=(10,6))
    plt.plot(predictions_base['carga'].values, label='Real', linewidth=2)
    plt.plot(predictions_base['prediction_label'].values, label='Previsto', linestyle='--')
    plt.title(submercado +' - Carga Real vs Prevista (Teste)')
    plt.xlabel('Índice')
    plt.ylabel('Carga')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{submercado}_{setup_m}_teste_pycaret.png', dpi=300, bbox_inches='tight')  # <-- ADIÇÃO
    plt.show()

    # ------------------
    # GRÁFICO DO TREINO  # <-- ADIÇÃO
    # ------------------
    predictions_train = exp.predict_model(best_model, data=exp.get_config('X_train'))  # <-- ADIÇÃO
    predictions_train[target_m] = exp.get_config('y_train').values                    # <-- ADIÇÃO

    plt.figure(figsize=(10,6))
    plt.plot(predictions_train[target_m].values, label='Real', linewidth=2)
    plt.plot(predictions_train['prediction_label'].values, label='Previsto', linestyle='--')
    plt.title(submercado +' - Carga Real vs Prevista (Treino)')
    plt.xlabel('Índice')
    plt.ylabel('Carga')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{submercado}_{setup_m}_treino_pycaret.png', dpi=300, bbox_inches='tight')  # <-- ADIÇÃO
    plt.show()

    return metricas_resumo


def treinamento_ST(df_modelo,nome_modelo,target_m,submercado,setup_m):

    modelo = 'na'


    # ------------------
    # DADOS
    # ------------------
    df_modelo = df_modelo.sort_values('dataref').reset_index(drop=True)
    print('df_modelo total:', len(df_modelo))

    # ------------------
    # REMOÇÃO DE OUTLIERS (método IQR)
    # ------------------
    Q1 = df_modelo[target_m].quantile(0.25)
    Q3 = df_modelo[target_m].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_modelo = df_modelo[(df_modelo[target_m] >= lower) & (df_modelo[target_m] <= upper)]
    print('df_modelo após remoção de outliers:', len(df_modelo))

    df_modelo = df_modelo.replace([np.inf, -np.inf], np.nan).dropna()
    print('df_modelo limpo:', len(df_modelo))

    # DIVISÃO TREINO E TESTE (80% treino, 20% teste)
    split_index = int(len(df_modelo) * 0.8)
    df_train = df_modelo.iloc[:split_index]
    df_test = df_modelo.iloc[split_index:]

    # Variáveis exógenas e target
    exog_train = df_train.drop(columns=['dataref', target_m])
    exog_test = df_test.drop(columns=['dataref', target_m])
    y_train = df_train[target_m]
    y_test = df_test[target_m]

    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')

    exog_train = exog_train.apply(pd.to_numeric, errors='coerce')
    exog_test = exog_test.apply(pd.to_numeric, errors='coerce')

    exog_train = exog_train.dropna()
    exog_test = exog_test.dropna()
    y_train = y_train.loc[exog_train.index]
    y_test = y_test.loc[exog_test.index]
    print('Tamanho treino:', len(y_train))
    print('Tamanho teste:', len(y_test))

    # ------------------
    # Seleção automática de parâmetros SARIMA
    # ------------------
    stepwise_model = auto_arima(
        y_train,
        exogenous=exog_train,
        seasonal=True,
        m=7 if setup_m == 'timeseries' else 1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print(f"Parâmetros selecionados: order={stepwise_model.order}, seasonal_order={stepwise_model.seasonal_order}")

    # ------------------
    # Ajuste do modelo SARIMAX
    # ------------------
    sarimax_model = sm.tsa.statespace.SARIMAX(
        y_train,
        exog=exog_train,
        order=stepwise_model.order,
        seasonal_order=stepwise_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # ------------------
    # Previsão no treino (in-sample)
    # ------------------
    y_train_pred = sarimax_model.predict(start=0, end=len(y_train)-1, exog=exog_train)
    y_train_pred.index = y_train.index

    # ------------------
    # Previsão no teste (out-of-sample)
    # ------------------
    y_pred = sarimax_model.predict(
        start=len(y_train),
        end=len(y_train) + len(y_test) - 1,
        exog=exog_test
    )
    y_pred.index = y_test.index

    # ------------------
    # Cálculo das métricas
    # ------------------
    def safe_rmsle(y_true, y_pred):
        y_true_clip = np.maximum(y_true, 1e-9)
        y_pred_clip = np.maximum(y_pred, 1e-9)
        return np.sqrt(mean_squared_log_error(y_true_clip, y_pred_clip))

    metrics_train = {
        'Model': 'SARIMA TREINO',
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'R2': r2_score(y_train, y_train_pred),
        'RMSLE': safe_rmsle(y_train, y_train_pred),
        'MAPE': mean_absolute_percentage_error(y_train, y_train_pred)
    }

    metrics_test = {
        'Model': 'SARIMA TESTE',
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'RMSLE': safe_rmsle(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    }

    df_metrics = pd.DataFrame([metrics_train, metrics_test])

    df_metrics.to_excel('arima'+submercado+setup_m+'.xlsx',engine= 'openpyxl')

    print(df_metrics)

    # ------------------
    # Plot das previsões vs valores reais (TREINO)
    # ------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_train.index, y_train, label='Real', linewidth=2)
    plt.plot(y_train_pred.index, y_train_pred, label='Previsto', linestyle='--')
    plt.title(submercado + ' - Carga Real vs Prevista (Treino)')
    plt.xlabel('Data')
    plt.ylabel(target_m)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{submercado}_treino_real_previsto_{setup_m}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------
    # Plot das previsões vs valores reais (TESTE)
    # ------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test, label='Real', linewidth=2)
    plt.plot(y_pred.index, y_pred, label='Previsto', linestyle='--')
    plt.title(submercado + ' - Carga Real vs Prevista (Teste)')
    plt.xlabel('Data')
    plt.ylabel(target_m)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{submercado}_teste_real_previsto_{setup_m}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------
    # Plot combinado (Treino + Teste)
    # ------------------
    plt.figure(figsize=(14,6))
    plt.plot(y_train.index, y_train, label='Treino Real', linewidth=1.8)
    plt.plot(y_train_pred.index, y_train_pred, label='Treino Previsto', linestyle='--')
    plt.plot(y_test.index, y_test, label='Teste Real', linewidth=1.8)
    plt.plot(y_pred.index, y_pred, label='Teste Previsto', linestyle='--')
    plt.legend()
    plt.title(submercado + ' - Valores Reais vs Previstos (Treino e Teste)')
    plt.xlabel('Data')
    plt.ylabel(target_m)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{submercado}_completo_real_previsto_{setup_m}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------
    # Plot das previsões vs valores reais
    # ------------------
    plt.figure(figsize=(14,6))
    plt.plot(y_train.index, y_train, label='Treino Real')
    plt.plot(y_train_pred.index, y_train_pred, label='Treino Previsto', linestyle='--')
    plt.plot(y_test.index, y_test, label='Teste Real')
    plt.plot(y_test.index, y_pred, label='Teste Previsto', linestyle='--')
    plt.legend()
    plt.title(submercado + ' - Valores Reais vs Previstos')
    plt.xlabel('Data')
    plt.ylabel(target_m)

    plt.savefig(f'{submercado}_grafico_sarima.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Salvar modelo treinado
    with open(f'{nome_modelo}.pkl', 'wb') as f:
        pickle.dump(sarimax_model, f)


    if modelo == 'prophet':
        print("Treinando modelo Prophet...")

        df_prophet = df_modelo[['dataref', target_m]].rename(columns={'dataref': 'ds', target_m: 'y'})

        # Treino e teste
        split_index = int(len(df_prophet) * 0.8)
        df_train = df_prophet.iloc[:split_index]
        df_test = df_prophet.iloc[split_index:]

        # Inicializa e ajusta o modelo
        model_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model_prophet.fit(df_train)

        # Previsão
        future = model_prophet.make_future_dataframe(periods=len(df_test), freq='D')
        forecast = model_prophet.predict(future)

        # Filtra as previsões para o conjunto de teste
        y_pred = forecast['yhat'].iloc[-len(df_test):].values
        y_test = df_test['y'].values

        # Métricas
        metrics_prophet = {
            'Model': 'Prophet',
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'RMSLE': safe_rmsle(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred)
        }

        df_metrics = pd.DataFrame([metrics_prophet])
        df_metrics.to_excel(f'prophet_{submercado}_{setup_m}.xlsx', engine='openpyxl')

        # Plot
        plt.figure(figsize=(14,6))
        plt.plot(df_train['ds'], df_train['y'], label='Treino')
        plt.plot(df_test['ds'], df_test['y'], label='Teste Real')
        plt.plot(df_test['ds'], y_pred, label='Prophet Previsto', linestyle='--')
        plt.title(f'{submercado} - Prophet')
        plt.legend()
        plt.savefig(f'{submercado}_grafico_prophet.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Salvar modelo Prophet
        with open(f'{nome_modelo}_prophet.pkl', 'wb') as f:
            pickle.dump(model_prophet, f)

def treinamento_ST2(df_modelo,nome_modelo,target_m,submercado,setup_m):
    # ------------------
    # DADOS
    # ------------------
    df_modelo = df_modelo.sort_values('dataref').reset_index(drop=True)
    print('df_modelo',len(df_modelo))
    df_modelo = df_modelo.replace([np.inf, -np.inf], np.nan).dropna()
    print('df_modelo_limpo',len(df_modelo))

    #DIVISÃO TREINO E TESTE
    split_index = int(len(df_modelo) * 0.8)
    df_train = df_modelo.iloc[:split_index]
    df_test = df_modelo.iloc[split_index:]

    
    # ------------------
    # Modelo SARIMAX
    # ------------------
    try:
        exog_train = df_train.drop(columns=['dataref', target_m])
        print('exog_train',len(exog_train))
        exog_test = df_test.drop(columns=['dataref', target_m])
        print('exog_test',len(exog_test))
        y_train = df_train[target_m]
        print('y_train',len(y_train))
        y_test = df_test[target_m]
        print('y_test',len(y_test))

        #Seleção automática dos parâmetros
        stepwise_model = auto_arima(
                        y_train,
                        exogenous=exog_train,
                        seasonal=True,
                        m=7 if setup_m == 'timeseries' else 1,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=False,
                        stepwise=True
                    )
        
        sarimax_model = sm.tsa.statespace.SARIMAX(
                        y_train,
                        exog=exog_train,
                        order=stepwise_model.order,
                        seasonal_order=stepwise_model.seasonal_order
                    ).fit(disp=False)

        #print(sarimax_model.summary())

        y_pred = sarimax_model.predict( start=len(y_train),
                                        end=len(y_train) + len(y_test) - 1,
                                        exog=exog_test
                                    )
        y_pred.index = y_test.index

        mae = mean_absolute_error(y_test, y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        
        rmse = np.sqrt(mse)
        
        r2 = r2_score(y_test, y_pred)

        y_pred_log = np.maximum(y_pred, 1e-9)
        y_test_log = np.maximum(y_test, 1e-9)
        rmsle = np.sqrt(mean_squared_log_error(y_test_log, y_pred_log))

        mape = mean_absolute_percentage_error(y_test, y_pred)

        df_metrics = pd.DataFrame([{
            'Model': 'SARIMA TESTE',
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'RMSLE': rmsle,
            'MAPE': mape
        }])

        print(df_metrics)
    except Exception as e:
        print(f"Erro no SARIMAX: {str(e)}")

def treinamento_MLP(df_modelo,nome_modelo,target_m,submercado,setup_m):
    df_modelo = df_modelo.sort_values('dataref').reset_index(drop=True)

    #DIVISÃO TREINO E TESTE
    split_index = int(len(df_modelo) * 0.8)
    df_train = df_modelo.iloc[:split_index]
    df_test = df_modelo.iloc[split_index:]

    exp = RegressionExperiment()
    
    if setup_m == 'timeseries':
        exp.setup(
            data=df_train.drop(columns=['dataref']),
            target=target_m,
            fold_strategy='timeseries',
            fold=6,
            data_split_shuffle=False,
            fold_shuffle=False,
            normalize=True,
            remove_outliers=True,
            session_id=42,
            feature_selection=False,
            numeric_imputation='mean',
            verbose=False
        )

    if setup_m == 'kfold':

        exp.setup(
            data=df_train.drop(columns=['dataref']),
            target=target_m,
            fold_strategy='kfold',
            fold=6, 
            session_id=42,
            normalize=True,
            remove_outliers=True,
            numeric_imputation='mean',
            feature_selection=False
        )

    # ------------------
    # Modelo MLP
    # ------------------
    rnn = exp.create_model('mlp', hidden_layer_sizes=(100,50), max_iter=200)
    rnn = exp.tune_model(rnn, optimize='MAE', search_library='optuna', n_iter=20)
    resultados_rnn_treino = exp.pull()
    print("ML TUNEL MODEL TREINO",resultados_rnn_treino)
    try:
        start_time = time.time()
        df_pred_mlp = exp.predict_model(rnn, data=df_test.copy())
        tt_mlp = time.time() - start_time

        y_true_mlp = df_pred_mlp[target_m]
        y_pred_mlp = df_pred_mlp['prediction_label']

        mae_mlp = mean_absolute_error(y_true_mlp, y_pred_mlp)
        mse_mlp = mean_squared_error(y_true_mlp, y_pred_mlp)
        rmse_mlp = np.sqrt(mse_mlp)
        r2_mlp = r2_score(y_true_mlp, y_pred_mlp)
        rmsle_mlp = np.sqrt(mean_squared_log_error(np.maximum(y_true_mlp, 1e-9), np.maximum(y_pred_mlp, 1e-9)))
        mape_mlp = mean_absolute_percentage_error(y_true_mlp, y_pred_mlp)

        df_mlp_metrics = pd.DataFrame([{
            'Model': 'MLP TESTE',
            'MAE': mae_mlp,
            'MSE': mse_mlp,
            'RMSE': rmse_mlp,
            'R2': r2_mlp,
            'RMSLE': rmsle_mlp,
            'MAPE': mape_mlp
        }])

        print(df_mlp_metrics)

    except Exception as e:
        print(f"Erro no cálculo de métricas para MLP: {str(e)}")



    # Criar DataFrame com métricas do MLP
    df_mlp = pd.DataFrame([{
        'Model': 'MLP',
        'fold_strategy': '-',
        'MAE': mae_mlp,
        'MSE': mse_mlp,
        'RMSE': rmse_mlp,
        'R2': r2_mlp,
        'RMSLE': rmsle_mlp,
        'MAPE': mape_mlp / 100 #,
        #'TT (Sec)': tt_mlp
    }])


def previsao_target_ML(df_prev,nome_modelo,tipo):

    print("INICIANDO PREVISÃO")
    df_prev.columns = df_prev.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
    df_prev = df_prev.drop(columns=['carga'])
    df_prev['ordem'] = range(len(df_prev))


    modelo = load_model(nome_modelo)

    previsoes = predict_model(modelo, data=df_prev)
    
    print("Colunas disponíveis no DataFrame de previsões:", previsoes.columns.tolist())

    return previsoes[['dataref', 'prediction_label']].rename(columns={'prediction_label': 'carga_prevista'})

if __name__ == "__main__":
    from prophet import Prophet
    df_modelo['dataref'] = pd.to_datetime(df_modelo['dataref'])
    df_modelo = df_modelo.sort_values('dataref').reset_index(drop=True)
    df_modelo['dia_semana'] = df_modelo['dia_semana'].astype('category')
    df_modelo['estacao_ano'] = df_modelo['dataref'].apply(obter_estacao)
    df_modelo['estacao_ano'] = df_modelo['estacao_ano'].astype('category')

    print(df_modelo.columns)

    matplotlib.use('Agg')
    start_time = time.time()
    print_elapsed_time(start_time, "INICIO")

    ################################
    
    submercados = ['SE','S','NE','N']
    tipo = 'ST'
    n_folds = 6
    treino = 'SIM'
    correlacao = 'NAO'
    previsao_c ='NAO'


##################################
# CORRELAÇÕES DE PEARSON, SPEARMAN E DIFERENÇA

    if correlacao == 'SIM':
        df_c_pearson = []
        df_c_spearman = []

        for s in submercados:
            # ------------------------------
            # FILTRO DE DADOS
            # ------------------------------
            df_aux = df_modelo[
                (df_modelo['carga_lag7'].notna()) &
                (df_modelo['dataref'] > '2000-01-07') &
                (df_modelo['dataref'] < '2024-07-01') &
                (df_modelo['submercado'] == s)
            ]
            print(df_aux.dtypes)

            df_aux = df_aux.drop(columns=['dataref'])

            colunas_numericas = df_aux.select_dtypes(include=['float64', 'int64']).columns

            # ------------------------------
            # PEARSON
            # ------------------------------
            corr_pearson = df_aux[colunas_numericas].corr(method='pearson')
            df_p = corr_pearson.iloc[:, :1].rename(columns={'carga': 'carga_' + s})
            df_c_pearson.append(df_p)

            # ------------------------------
            # SPEARMAN
            # ------------------------------
            corr_spearman = df_aux[colunas_numericas].corr(method='spearman')
            df_s = corr_spearman.iloc[:, :1].rename(columns={'carga': 'carga_' + s})
            df_c_spearman.append(df_s)

        # ------------------------------
        # AGREGAÇÃO
        # ------------------------------
        df_corr_pearson = pd.concat(df_c_pearson, axis=1)
        df_corr_spearman = pd.concat(df_c_spearman, axis=1)

        # ------------------------------
        # DIFERENÇA (PEARSON - SPEARMAN)
        # ------------------------------
        df_corr_diff = df_corr_pearson - df_corr_spearman

        # ------------------------------
        # EXPORTAÇÃO
        # ------------------------------
        df_corr_pearson.to_excel('corr_pearson.xlsx', engine='openpyxl')
        df_corr_spearman.to_excel('corr_spearman.xlsx', engine='openpyxl')
        df_corr_diff.to_excel('corr_diff.xlsx', engine='openpyxl')

        # ------------------------------
        # VISUALIZAÇÃO
        # ------------------------------
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr_pearson, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlação de Pearson')
        plt.savefig('corr_pearson.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr_spearman, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlação de Spearman')
        plt.savefig('corr_spearman.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr_diff, annot=True, cmap='vlag', center=0)
        plt.title('Diferença (Pearson - Spearman)')
        plt.savefig('corr_diferenca.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ------------------------------
        # PRINTS NO CONSOLE
        # ------------------------------
        print("\n--- Correlação de Pearson ---")
        print(df_corr_pearson)

        print("\n--- Correlação de Spearman ---")
        print(df_corr_spearman)

        print("\n--- Diferença (Pearson - Spearman) ---")
        print(df_corr_diff)


###########################################
    #LOOP TREINAMENTO SUBMERCADOS

    for s in submercados:
        ###########################################
        #PREPARAR DATASET POR SUBMERCADO
        
        
        df_treino = df_modelo[
                        (df_modelo['carga_lag7'].notna()) &
                        (df_modelo['dataref'] > '2000-01-07') &
                        (df_modelo['dataref'] < '2024-07-01') &
                        (df_modelo['submercado'] == s)
                        ]
        
        df_treino = df_treino.drop(columns=[
                                    'ipca_transportes_lag1',
                                    'ipca_saude_lag1',
                                    'ipca_saude',
                                    'ipca_transportes',
                                    'submercado'
                                ])
    
        #########################################
        #TREINAMENTO MODELO
        if treino == 'SIM':
            if tipo == 'ML':
                df_kfold = treinamento_ML(df_treino, n_folds,'carga',s,'kfold')
                df_times = treinamento_ML(df_treino, n_folds,'carga',s,'timeseries')
                print(df_times)
                df = pd.concat([df_kfold, df_times], ignore_index=True)
                df.to_excel('resultados_comparemodels_'+s+'.xlsx', engine='openpyxl', index=False)

            if tipo == 'ST':
                treinamento_ST(df_treino, 'SARIMAX_'+s,'carga',s,'na')


    
    print_elapsed_time(start_time, "Fim")


#           [
#             ['N','HuberRegressorNtimeseries'],
#             ['NE','HuberRegressorNEtimeseries'],
#             ['S','ExtraTreesRegressorSkfold'],
#             ['SE','PassiveAggressiveRegressorSEtimeseries']
#         ]

# [
#             ['N','HuberRegressorNtimeseries'], ['N','ExtraTreesRegressorNkfold'],
#             ['NE','HuberRegressorNEtimeseries'], ['NE','ExtraTreesRegressorNEkfold'],
#             ['S','LassoStimeseries'], ['S','ExtraTreesRegressorSkfold'],
#             ['SE','PassiveAggressiveRegressorSEtimeseries'], ['SE','LGBMRegressorSEkfold']
#         ]


    def previsao_ML():
        modelos_elegiveis = [
             ['N','HuberRegressorNtimeseries'],
             ['NE','HuberRegressorNEtimeseries'],
             ['S','ExtraTreesRegressorSkfold'],
             ['SE','PassiveAggressiveRegressorSEtimeseries']
         ]

        for m in modelos_elegiveis:
            sigla = m[0]
            nome_modelo = m[1]
            modelo = load_model(nome_modelo)

            df_prev = pd.read_excel("H:\\Meu Drive\\DISSERTAÇÃO\\dados\\aux_prev.xlsx", sheet_name='input_'+sigla)
            df_prev['estacao_ano'] = df_prev['dataref'].apply(obter_estacao)
            df_prev = df_prev.drop(columns=[
                'submercado', 'ipca_transportes_lag1', 'ipca_saude_lag1',
                'ipca_saude', 'ipca_transportes', 'previsao', 'diff'
            ])
            df_prev['dia_semana'] = df_prev['dia_semana'].astype('category')
            df_prev['estacao_ano'] = df_prev['estacao_ano'].astype('category')

            # Remove dataref apenas se for modelo kfold
            if nome_modelo.endswith('kfold'):
                df_prev_model = df_prev.drop(columns=['dataref'])
            else:
                df_prev_model = df_prev.copy()

            # Previsão
            resultados = predict_model(modelo, data=df_prev_model)
            resultados['dataref'] = df_prev['dataref']
            resultados['carga_real'] = df_prev['carga']

            resultados.to_excel(sigla+nome_modelo+'.xlsx', engine='openpyxl')

            # # Gráfico Real vs Previsto
            # plt.figure(figsize=(10,6))
            # plt.plot(resultados['dataref'], resultados['carga_real'], label='Carga Real')
            # plt.plot(resultados['dataref'], resultados['prediction_label'], label='Previsto', linestyle='--')
            # plt.title(f'{sigla} - Carga Real vs Prevista ({nome_modelo})')
            # plt.xticks(rotation=45)
            # plt.xlabel('Data')
            # plt.ylabel('Carga')
            # plt.legend()
            # plt.tight_layout()
            # plt.grid(True)
            # plt.savefig(f'{sigla}_{nome_modelo}_externa_real_prevista.png', dpi=300)
            # plt.close()

            # # -------- 6.2.1: GRANULARIDADE TEMPORAL --------

            # # Diária já está implícita

            # # Semanal
            # resultados['semana'] = resultados['dataref'].dt.to_period('W').apply(lambda r: r.start_time)
            # semanal = resultados.groupby('semana')[['carga_real', 'prediction_label']].mean()
            # semanal['MAPE'] = np.abs((semanal['carga_real'] - semanal['prediction_label']) / semanal['carga_real'])
            # semanal.to_excel(f'{sigla}_{nome_modelo}_granularidade_semanal.xlsx')

            # # Mensal
            # resultados['mes'] = resultados['dataref'].dt.to_period('M').apply(lambda r: r.start_time)
            # mensal = resultados.groupby('mes')[['carga_real', 'prediction_label']].mean()
            # mensal['MAPE'] = np.abs((mensal['carga_real'] - mensal['prediction_label']) / mensal['carga_real'])
            # mensal.to_excel(f'{sigla}_{nome_modelo}_granularidade_mensal.xlsx')

            # # Gráfico semanal – carga real vs prevista
            # plt.figure(figsize=(10, 5))
            # plt.plot(semanal.index, semanal['carga_real'], label='Carga Real', linewidth=2)
            # plt.plot(semanal.index, semanal['prediction_label'], label='Carga Prevista', linestyle='--')
            # plt.title(f'{sigla} - Previsão Semanal ({nome_modelo})')
            # plt.xlabel('Semana')
            # plt.ylabel('Carga Média (MWmed)')
            # plt.xticks(rotation=45)
            # plt.grid(True)
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f'{sigla}_{nome_modelo}_previsao_semanal.png', dpi=300)
            # plt.show()

            # # -------- 6.2.2: MAPE POR DIA DA SEMANA --------
            # resultados['dia_semana'] = df_prev['dia_semana']

            # mape_dia = resultados.groupby('dia_semana').apply(
            #     lambda x: np.mean(np.abs((x['carga_real'] - x['prediction_label']) / x['carga_real']))
            # ).reset_index(name='MAPE')

            # mape_dia.to_excel(f'{sigla}_{nome_modelo}_mape_dia_semana.xlsx', index=False)

            # # Gráfico MAPE por dia da semana
            # plt.figure(figsize=(8,5))
            # ordem = ['segunda-feira','terça-feira','quarta-feira','quinta-feira','sexta-feira','sábado','domingo']
            # mape_dia['dia_semana'] = pd.Categorical(mape_dia['dia_semana'], categories=ordem, ordered=True)
            # mape_dia = mape_dia.sort_values('dia_semana')
            # plt.bar(mape_dia['dia_semana'], mape_dia['MAPE'])
            # plt.xticks(rotation=45)
            # plt.ylabel('MAPE')
            # plt.title(f'{sigla} - MAPE por Dia da Semana ({nome_modelo})')
            # plt.tight_layout()
            # plt.savefig(f'{sigla}_{nome_modelo}_mape_dia_semana.png', dpi=300)
            # plt.close()


    def previsao_ST():
        modelos_st = [
        ['N', 'SARIMAX_N'], ['NE', 'SARIMAX_NE'],
        ['S', 'SARIMAX_S'], ['SE', 'SARIMAX_SE']
        ]

        for sigla, nome_modelo in modelos_st:
            # Carregar modelo SARIMAX
            with open(f'{nome_modelo}.pkl', 'rb') as f:
                modelo = pickle.load(f)

            # Carregar input
            df_prev = pd.read_excel("H:\\Meu Drive\\DISSERTAÇÃO\\dados\\aux_prev.xlsx", sheet_name='input_' + sigla)
            df_prev['estacao_ano'] = df_prev['dataref'].apply(obter_estacao)

            # Preparar variáveis
            y_real = df_prev['carga']
            exog = df_prev.drop(columns=[
                'dataref', 'submercado', 'carga', 'ipca_transportes_lag1',
                'ipca_saude_lag1', 'ipca_saude', 'ipca_transportes',
                'previsao', 'diff'
            ], errors='ignore')
            exog = exog.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Previsão
            y_prev = modelo.predict(start=0, end=len(y_real) - 1, exog=exog)

            resultados = pd.DataFrame({
                'dataref': df_prev['dataref'],
                'carga_real': y_real.values,
                'carga_prevista': y_prev.values,
                'dia_semana': df_prev['dia_semana'].astype(int)  # Assume dias como 0–6
            })

            # Gráfico real vs previsto
            plt.figure(figsize=(10, 6))
            plt.plot(resultados['dataref'], resultados['carga_real'], label='Carga Real')
            plt.plot(resultados['dataref'], resultados['carga_prevista'], label='Prevista', linestyle='--')
            plt.title(f'{sigla} - Carga Real vs Prevista (SARIMAX)')
            plt.xlabel('Data')
            plt.ylabel('Carga')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{sigla}_{nome_modelo}_externa_real_prevista.png', dpi=300)
            plt.close()

            # # Granularidade Semanal
            # resultados['semana'] = resultados['dataref'].dt.to_period('W').apply(lambda r: r.start_time)
            # semanal = resultados.groupby('semana')[['carga_real', 'carga_prevista']].mean()
            # semanal['MAPE'] = np.abs((semanal['carga_real'] - semanal['carga_prevista']) / semanal['carga_real'])
            # semanal.to_excel(f'{sigla}_{nome_modelo}_granularidade_semanal.xlsx')

            # plt.figure(figsize=(10, 5))
            # plt.plot(semanal.index, semanal['carga_real'], label='Carga Real', linewidth=2)
            # plt.plot(semanal.index, semanal['carga_prevista'], label='Prevista', linestyle='--')
            # plt.title(f'{sigla} - Previsão Semanal (SARIMAX)')
            # plt.xlabel('Semana')
            # plt.ylabel('Carga Média (MWmed)')
            # plt.xticks(rotation=45)
            # plt.grid(True)
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f'{sigla}_{nome_modelo}_previsao_semanal.png', dpi=300)
            # plt.close()

            # # Granularidade Mensal
            # resultados['mes'] = resultados['dataref'].dt.to_period('M').apply(lambda r: r.start_time)
            # mensal = resultados.groupby('mes')[['carga_real', 'carga_prevista']].mean()
            # mensal['MAPE'] = np.abs((mensal['carga_real'] - mensal['carga_prevista']) / mensal['carga_real'])
            # mensal.to_excel(f'{sigla}_{nome_modelo}_granularidade_mensal.xlsx')

            # # MAPE por Dia da Semana (0–6)
            # mape_dia = resultados.groupby('dia_semana').apply(
            #     lambda x: np.mean(np.abs((x['carga_real'] - x['carga_prevista']) / x['carga_real']))
            # ).reset_index(name='MAPE')
            # mape_dia.to_excel(f'{sigla}_{nome_modelo}_mape_dia_semana.xlsx', index=False)

            # # Gráfico MAPE por dia da semana (0–6)
            # plt.figure(figsize=(8, 5))
            # plt.bar(mape_dia['dia_semana'], mape_dia['MAPE'])
            # plt.xticks(ticks=range(7), labels=[
            #     'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'
            # ])
            # plt.ylabel('MAPE')
            # plt.title(f'{sigla} - MAPE por Dia da Semana (SARIMAX)')
            # plt.tight_layout()
            # plt.savefig(f'{sigla}_{nome_modelo}_mape_dia_semana.png', dpi=300)
            # plt.close()


    if previsao_c == 'SIM' and tipo == 'ML':
        previsao_ML()
    
    if previsao_c == 'SIM' and tipo == 'ST':
        previsao_ST()
