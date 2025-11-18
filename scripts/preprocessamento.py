# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome:
# RA:
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento
import pandas as pd
import numpy as np
import os

def preprocess_user_info(file_path: str):
    """
    Carrega, limpa e pré-processa o arquivo de informações demográficas (users_info.txt).
    """

    try:
        df = pd.read_csv(
            file_path,
            na_values='-',
            skipfooter=10,
            engine='python'
        )
        print("Arquivo de informações do usuário carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return None

    print("\n--- Iniciando Imputação de Valores Ausentes ---")

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                # --- CORREÇÃO DO WARNING AQUI ---
                df[col] = df[col].fillna(mean_val)
                print(f"Coluna numérica '{col}': Valores NaN preenchidos com a média ({mean_val:.2f})")
            else:
                mode_val = df[col].mode()[0]
                # --- CORREÇÃO DO WARNING AQUI ---
                df[col] = df[col].fillna(mode_val)
                print(f"Coluna qualitativa '{col}': Valores NaN preenchidos com a moda ('{mode_val}')")

    print("\nImputação concluída.")

    print("\n--- Iniciando Codificação e Limpeza Final ---")

    # Mapeamento binário para 'Does physical activity regularly?'
    if 'Does physical activity regularly?' in df.columns:
        df['activity_regularly'] = df['Does physical activity regularly?'].map({'Yes': 1, 'No': 0})
        df.drop(columns=['Does physical activity regularly?'], inplace=True)
        print("Coluna 'Does physical activity regularly?' mapeada para 0 e 1.")

    # One-Hot Encoding para 'Gender' e 'Protocol'
    categorical_cols = ['Gender', 'Protocol']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"Colunas {categorical_cols} transformadas com One-Hot Encoding.")

    # **------ NOVA ESTRATÉGIA APLICADA AQUI ------**
    # Remove as colunas de condição do experimento, pois elas vazam a resposta
    # e não são características dos dados fisiológicos do participante.
    cols_to_drop = ['Stress Inducement', 'Aerobic Exercise', 'Anaerobic Exercise']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')  # errors='ignore' evita erro se a coluna não existir
    print(f"Colunas de condição do experimento ({cols_to_drop}) removidas.")
    # **-------------------------------------------**

    print("\nPré-processamento dos dados demográficos concluído.")

    return df

def preprocess_sensor_file(df_sensor, sensor_name: str):
    """
    Aplica um pipeline de pré-processamento robusto, incluindo o cálculo
    da magnitude do acelerômetro e tratamento de outliers com IQR.
    """
    if df_sensor.empty: return None
    processed_df = df_sensor.copy()

    # --- 1. Tratamento de Formato e Conversão Numérica ---
    if sensor_name == 'ACC':
        processed_df.columns = ['X', 'Y', 'Z']
        for col in ['X', 'Y', 'Z']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    elif sensor_name == 'IBI':
        processed_df.columns = ['Timestamp', 'Interval']
        processed_df['Interval'] = pd.to_numeric(processed_df['Interval'], errors='coerce')
    else:  # HR, EDA, TEMP, BVP
        processed_df.columns = ['value']
        processed_df['value'] = pd.to_numeric(processed_df['value'], errors='coerce')

    # --- 2. Imputação de Valores Ausentes (NaN) ---
    processed_df = processed_df.infer_objects(copy=False)
    processed_df.interpolate(method='linear', limit_direction='both', inplace=True)
    processed_df.dropna(inplace=True)
    if processed_df.empty: return None

    # --- 3. Lógica Específica para Acelerômetro (Cálculo da Magnitude) ---
    if sensor_name == 'ACC':
        # Calcula a magnitude do vetor e cria uma nova coluna 'value'
        magnitude = np.sqrt(processed_df['X'] ** 2 + processed_df['Y'] ** 2 + processed_df['Z'] ** 2)
        # Cria um novo DataFrame apenas com a magnitude para padronizar a saída
        processed_df = pd.DataFrame({'value': magnitude})
        target_columns = ['value']  # Agora ACC também tem uma coluna 'value'
    elif sensor_name == 'IBI':
        target_columns = ['Interval']
    else:
        target_columns = ['value']

    # --- 4. Tratamento de Outliers (Híbrido e IQR) ---
    # Camada 1: Clipping Absoluto (Não se aplica mais ao ACC diretamente)
    hard_limits = {'HR': (40, 220), 'TEMP': (20, 42), 'EDA': (0.01, 30), 'IBI': (0.27, 2.0), 'BVP': (-150, 150)}
    if sensor_name in hard_limits:
        lower_limit, upper_limit = hard_limits[sensor_name]
        col_to_clip = 'value' if sensor_name != 'IBI' else 'Interval'
        processed_df[col_to_clip] = processed_df[col_to_clip].clip(lower=lower_limit, upper=upper_limit)

    # Camada 2: Clipping Estatístico com Método IQR (Agora se aplica a todos, incluindo a magnitude do ACC)
    for col in target_columns:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)

    return processed_df


print("Função 'preprocess_sensor_file' atualizada com sucesso.")
print("Agora ela calcula a magnitude do acelerômetro (ACC) e a trata como um sinal único.")