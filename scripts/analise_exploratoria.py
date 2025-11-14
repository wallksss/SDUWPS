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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_signal_distribution_for_class(
        dataset,
        target_class: str,
        sensor_name: str,
        base_dir: str = 'dataset/wearables',
        cols: int = 5,
        save_fig: bool = False
):
    """
    Plota a distribuição de um sinal de sensor para todas as amostras de uma classe específica,
    tratando corretamente sensores de uma ou múltiplas colunas (ACC, IBI).
    """
    filtered_samples = dataset[dataset['Label'] == target_class]
    num_samples = len(filtered_samples)

    if num_samples == 0:
        print(f"Nenhuma amostra encontrada para a classe '{target_class}'.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 5), squeeze=False)
    fig.suptitle(f'Distribuição do Sinal {sensor_name.replace(".csv", "")} na Classe {target_class}',
                 fontsize=24, y=0.99)

    global_min, global_max = float('inf'), float('-inf')
    print(f"Calculando escala global para {num_samples} amostras do sensor {sensor_name}...")
    for user_id in filtered_samples['Id']:
        file_path = os.path.join(base_dir, user_id, sensor_name)
        if not os.path.exists(file_path): continue
        try:
            if sensor_name == 'ACC.csv':
                df = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                if not df.empty:
                    global_min = min(global_min, df.min().min())
                    global_max = max(global_max, df.max().max())
            elif sensor_name == 'IBI.csv':
                df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                intervals = pd.to_numeric(df['Interval'], errors='coerce').dropna()
                if not intervals.empty:
                    global_min = min(global_min, intervals.min())
                    global_max = max(global_max, intervals.max())
            else:
                df = pd.read_csv(file_path, header=None, skiprows=1)
                if not df.empty:
                    global_min = min(global_min, df.iloc[:, 0].min())
                    global_max = max(global_max, df.iloc[:, 0].max())
        except (pd.errors.EmptyDataError, IndexError, ValueError):
            continue

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        print(f"Não foi possível determinar a escala para {sensor_name}. Verifique os arquivos.")
        plt.close(fig)
        return

    margin = (global_max - global_min) * 0.05 if (global_max - global_min) > 0 else 0.1
    global_min -= margin
    global_max += margin

    print("Escala global definida. Plotando gráficos...")
    for idx, user_id in enumerate(filtered_samples['Id']):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        file_path = os.path.join(base_dir, user_id, sensor_name)
        ax.set_title(user_id, fontsize=12)
        try:
            if sensor_name == 'ACC.csv':
                df = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                if df.empty: raise ValueError("Arquivo vazio após pular linha")
                df.plot(ax=ax, linewidth=1)
                ax.legend(fontsize='small')

            elif sensor_name == 'IBI.csv':
                df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                df['Interval'] = pd.to_numeric(df['Interval'], errors='coerce')
                df.dropna(subset=['Interval'], inplace=True)
                if df.empty: raise ValueError("Nenhum dado numérico no IBI")

                # Plota a coluna 'Interval' usando o timestamp como eixo X para uma visualização mais correta
                ax.plot(df['Timestamp'], df['Interval'], linewidth=1)
                ax.set_xlabel('Timestamp (s)', fontsize=7)

            else:  # HR, EDA, TEMP, BVP
                df = pd.read_csv(file_path, header=None, skiprows=1)
                if df.empty: raise ValueError("Arquivo vazio após pular linha")
                df.iloc[:, 0].plot(ax=ax, linewidth=1, legend=False)

            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(global_min, global_max)
        except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, ValueError):
            ax.text(0.5, 0.5, f'{user_id}\n(Arquivo ausente/inválido)',
                    ha='center', va='center', fontsize=10, color='red')
            ax.axis('off')

    for j in range(num_samples, rows * cols):
        row, col = j // cols, j % cols
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_fig:
        output_dir = 'figs/analise_atributos'
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir,
                                   f"{sensor_name.replace('.csv', '')}_{target_class.lower()}_distribution.png")
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"Figura salva em '{output_name}'")

    plt.show()
    print(f"\n{num_samples} gráficos da classe '{target_class}' para o sensor '{sensor_name}' foram plotados.")


def plot_boxplot_comparison_for_class(
        dataset,
        target_class: str,
        sensor_name: str,
        base_dir: str = 'dataset/wearables',
        save_fig: bool = False
):
    """
    Gera um boxplot para cada amostra de uma classe específica, permitindo a
    comparação de distribuições e a identificação de outliers.

    Args:
        dataset (pd.DataFrame): DataFrame contendo as colunas 'Id' e 'Label'.
        target_class (str): A classe a ser visualizada (ex: 'AEROBIC', 'STRESS').
        sensor_name (str): O nome do arquivo do sensor (ex: 'TEMP.csv', 'HR.csv').
        base_dir (str): O diretório base onde as pastas dos usuários estão localizadas.
        save_fig (bool): Se True, salva a figura em um arquivo PNG.
    """

    # 1. Filtra as amostras da classe desejada
    filtered_samples = dataset[dataset['Label'] == target_class]
    num_samples = len(filtered_samples)

    if num_samples == 0:
        print(f"Nenhuma amostra encontrada para a classe '{target_class}'.")
        return

    # 2. Coleta os dados de cada amostra em uma lista
    all_sensor_data = []
    sample_ids = []

    print(f"Coletando dados do sensor '{sensor_name}' para {num_samples} amostras da classe '{target_class}'...")
    for user_id in filtered_samples['Id']:
        file_path = os.path.join(base_dir, user_id, sensor_name)
        try:
            # Lógica de leitura adaptada para cada tipo de sensor
            if sensor_name == 'ACC.csv':
                # Para ACC, podemos analisar a magnitude do vetor de aceleração
                df = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                # Calcula a magnitude: sqrt(X^2 + Y^2 + Z^2)
                magnitude = np.sqrt(df['X'] ** 2 + df['Y'] ** 2 + df['Z'] ** 2).dropna()
                if not magnitude.empty:
                    all_sensor_data.append(magnitude)
                    sample_ids.append(user_id)
            elif sensor_name == 'IBI.csv':
                df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                intervals = pd.to_numeric(df['Interval'], errors='coerce').dropna()
                if not intervals.empty:
                    all_sensor_data.append(intervals)
                    sample_ids.append(user_id)
            else:  # HR, EDA, TEMP, BVP
                df = pd.read_csv(file_path, header=None, skiprows=1)
                signal_data = df.iloc[:, 0].dropna()
                if not signal_data.empty:
                    all_sensor_data.append(signal_data)
                    sample_ids.append(user_id)
        except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, ValueError):
            print(f"Aviso: Não foi possível ler {file_path}. Ignorando.")
            continue

    if not all_sensor_data:
        print(f"Nenhum dado válido encontrado para o sensor '{sensor_name}' na classe '{target_class}'.")
        return

    # 3. Geração do Gráfico de Box Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(20, 10))

    # Cria o boxplot. 'patch_artist=True' permite colorir as caixas.
    box = plt.boxplot(all_sensor_data, labels=sample_ids, vert=True, patch_artist=True)

    # Colore as caixas para melhor visualização
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_sensor_data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Formatação do gráfico
    sensor_title = sensor_name.replace('.csv', '') if sensor_name != 'ACC.csv' else 'Magnitude do Acelerômetro'
    plt.title(f'Distribuição e Outliers por Amostra - Sensor: {sensor_title}, Classe: {target_class}', fontsize=20)
    plt.ylabel(f'Valor do Sinal ({sensor_title})')
    plt.xlabel('ID da Amostra')
    plt.xticks(rotation=60, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 4. Salva a figura, se solicitado
    if save_fig:
        output_dir = 'figs/boxplots'
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, f"{sensor_name.replace('.csv', '')}_{target_class.lower()}_boxplots.png")
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"Figura salva em '{output_name}'")

    plt.show()
    print(f"\n{len(all_sensor_data)} box plots gerados para a classe '{target_class}'.")