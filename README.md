# SDUWPS: Detecção de Estresse e Esforço Físico com Sinais Fisiológicos

![Linguagem](https://img.shields.io/badge/Python-3.9-blue.svg)
![Licença](https://img.shields.io/badge/license-MIT-green.svg)

Repositório do projeto final da disciplina de Aprendizado de Máquina da Universidade Federal de São Carlos (UFSCar). O objetivo deste trabalho é desenvolver e avaliar modelos de Machine Learning capazes de classificar o estado fisiológico de um indivíduo (estresse, esforço aeróbico e anaeróbico) a partir de dados coletados por sensores vestíveis.

## 📝 Sobre o Projeto

O projeto explora um conjunto de dados multivariados de séries temporais para a classificação de três estados fisiológicos distintos: `STRESS`, `AEROBIC` e `ANAEROBIC`. Utilizando sinais como frequência cardíaca (HR), atividade eletrodérmica (EDA), temperatura da pele, aceleração e níveis de oxigenação (SpO2), o desafio consiste em aplicar um pipeline completo de aprendizado de máquina, desde o pré-processamento dos dados até a comparação de desempenho de diferentes algoritmos.

Este trabalho foi desenvolvido com foco na competição do [Kaggle](<URL_DA_COMPETICAO_AQUI>) proposta pela disciplina.

## 📊 Dataset

Os dados utilizados foram coletados em sessões experimentais controladas, onde os participantes foram submetidos a atividades que induziam estresse e esforço físico. As principais variáveis disponíveis são:

*   **Frequência Cardíaca (heart rate)**
*   **Atividade Eletrodérmica (EDA)**
*   **Temperatura Corporal (skin temperature)**
*   **Aceleração Tri-axial (accelerometer)**
*   **Níveis de Oxigenação (SpO2)**

**Importante:** Conforme as regras do projeto, a base de dados não está incluída neste repositório. A implementação é capaz de reproduzir todos os passos a partir da base original, que deve ser obtida separadamente.

## 🤖 Modelos e Metodologia

O pipeline de desenvolvimento seguiu as seguintes etapas:

1.  **Análise Exploratória e Pré-processamento:** Limpeza, normalização, segmentação dos sinais (janelamento) e extração de características (features) estatísticas e de domínio de frequência.
2.  **Modelagem e Avaliação:** Foram implementados e comparados os seguintes modelos, conforme exigido pela disciplina:
    *   k-Vizinhos Mais Próximos (k-NN)
    *   Naïve Bayes
    *   Regressão Logística
    *   Máquinas de Vetores de Suporte (SVM)
    *   Redes Neurais Artificiais (MLP)
3.  **Análise Avançada (Bônus):** [Opcional: Descreva aqui se você usou CNNs, LSTMs ou Transformers] Foram exploradas arquiteturas de Deep Learning, como Redes Neurais Convolucionais (1D-CNN) e Transformers, para aprendizado automático de características a partir dos dados brutos.

## 📁 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir a reprodutibilidade dos resultados:
