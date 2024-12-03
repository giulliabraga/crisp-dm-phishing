# Aplicação Prática da Metodologia CRISP-DM ao dataset Phishing Websites

O objetivo deste projeto foi a aplicação do método CRISP-DM ao dataset [[PhishingWebsites UCI Machine Learning dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)]. 

# README: Estrutura do Projeto

Este documento descreve a estrutura do [repositório GitHub](https://github.com/giulliabraga/crisp-dm-phishing), detalhando a função de cada diretório e arquivo na hierarquia apresentada.

---


## Diretórios Principais

1. `data`
Contém os arquivos de dados utilizados no projeto.
    - **`phishing-dataset.arff`**: Conjunto de dados original.
    - **`feature_description.docx`**: Descrição das features do dataset original.
    - **`phishing_dataset_intro_paper.pdf`**: Artigo introdutório estudado.

2. `metrics`
Armazena métricas de desempenho dos 10 modelos avaliados com cross-validation, em formato CSV.

3. `modules`
Contém os módulos Python com funções reutilizáveis para as diversas etapas do projeto.
    - **`lvq_classifier.py`**: Implementação do modelo LVQ.
    - **`best_pipelines.py`**: Pipelines de seletores de features + classificadores com hiperparâmetros otimizados.
    - **`cross_validation.py`**: Implementação do processo de cross-validation.
    - **`model_selector.py`**: Funções para seleção de modelos.
    - **`preproc.py`**: Funções e métodos para a etapa de pré-processamento.
    - **`eda.py`**: Funções e métodos para a etapa de EDA.
    - **`optimizer.py`**: Algoritmos de otimização de hiperparâmetros.
    - **`results_visualization.py`**: Geração de gráficos e visualizações de métricas pós validação-cruzada.
    - **`statistical_methods.py`**: Métodos estatísticos auxiliares para análise.
    - **`utils.py`**: Funções utilitárias.
    - **`xai.py`**: Funções e métodos para a etapa de XAI.

1. `notebooks`
Reúne notebooks Jupyter para experimentação e documentação interativa.
    
    4.1. **`eda_phishing.ipynb`**: Análise exploratória dos dados (EDA).
    
    4.2. **`preprocessing.ipynb`**: Pré-processamento dos dados.

    4.3. **`optimization.ipynb`**: Otimização de hiperparâmetros dos modelos.
    
    4.4. **`cross_validation.ipynb`**: Execução e análise detalhada de cross-validation.
    
    4.5. **`cv_results.ipynb`**: Análise dos resultados obtidos com cross-validation.
    
    4.6. **`stress_testing.ipynb`**: Testes de estresse dos modelos.

1. `outputs`
Destinado aos resultados gerados pelo pipeline, como gráficos, relatórios e modelos salvos.

---