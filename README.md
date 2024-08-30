# Desenvolvimento de um Modelo de Aprendizado de Máquina para Predição de Infecções por *Trypanosoma cruzi* em *Didelphis aurita*: Abordagem de Dados Desbalanceados e Interpretação Avançada

Este projeto aborda o desenvolvimento de um modelo de aprendizado de máquina para a previsão de infecções por *Trypanosoma cruzi* em *Didelphis aurita*, utilizando um conjunto de dados pré-existente e confiável. Para garantir um modelo robusto e interpretável, foram incorporadas várias etapas no processo de modelagem, incluindo pré-processamento de dados, tratamento de desbalanceamento de classes, seleção de características e otimização de hiperparâmetros.

O tratamento do desbalanceamento de classes foi realizado por meio da técnica de oversampling com SMOTE [1], garantindo um conjunto de dados equilibrado. Para a construção do modelo preditivo, foi utilizado o `GradientBoostingClassifier`, conhecido por sua capacidade de produzir previsões precisas e robustas. A otimização dos hiperparâmetros do modelo foi feita utilizando a técnica de grid search, enquanto a validação cruzada estratificada assegurou a capacidade de generalização do modelo para novos dados.

Para aprimorar a interpretabilidade dos resultados, foram realizadas análises SHAP [2], que forneceram uma compreensão detalhada da importância de cada variável nas previsões do modelo, complementadas por uma aplicação de RFE (Eliminação Recursiva de Características) para a seleção das características mais relevantes. Esta abordagem combina métodos estatísticos e de aprendizado de máquina avançados para melhorar tanto a precisão quanto a interpretabilidade do modelo, contribuindo significativamente para a identificação eficaz de variáveis em cenários com dados complexos e desbalanceados.

## Bibliotecas Utilizadas

- **Pandas**: A biblioteca panda é usada para carregar e manipular o conjunto de dados. Neste estudo, pandas permite a leitura de dados a partir de um arquivo Excel e a execução de operações de pré-processamento, como a remoção de valores ausentes. Além disso, pandas é usada para transformar os dados, separando as variáveis explicativas (features) da variável de interesse (target). Essa separação é essencial para garantir que os dados estejam prontos para o modelo de aprendizado de máquina [3].

- **Matplotlib**: Matplotlib é uma biblioteca de visualização de dados em Python, utilizada para criar gráficos e visualizações que ajudam a entender o desempenho do modelo. No contexto deste estudo, Matplotlib é empregada para gerar curvas de aprendizado, curvas ROC (Receiver Operating Characteristic), e matrizes de confusão, permitindo uma análise visual detalhada da performance do modelo [4].

- **Scikit-plot (scikitplot)**: Scikit-plot é uma extensão da biblioteca scikit-learn que simplifica a criação de gráficos de desempenho para modelos de aprendizado de máquina. A biblioteca é utilizada para gerar visualizações como a curva ROC, a curva de precisão-revocação e a matriz de confusão, que são essenciais para avaliar a eficácia do modelo e identificar possíveis melhorias.

- **Pickle**: Pickle é uma biblioteca padrão do Python usada para serializar (salvar) e desserializar (carregar) objetos Python, como modelos de aprendizado de máquina. Neste estudo, pickle é utilizado para salvar o modelo final treinado, permitindo que ele seja reutilizado posteriormente sem a necessidade de ser treinado novamente, o que é útil para implantação em produção ou para análises futuras.

- **Scikit-learn**: Scikit-learn é uma das bibliotecas mais populares para aprendizado de máquina em Python, oferecendo uma ampla gama de algoritmos para classificação, regressão e clustering, além de ferramentas para seleção de modelos, validação cruzada e pré-processamento de dados. Neste estudo, scikit-learn é usada para dividir o conjunto de dados em treino e teste (`train_test_split`), realizar validação cruzada (`cross_val_score` e `StratifiedKFold`), construir o modelo de aprendizado de máquina (`GradientBoostingClassifier`), e otimizar os hiperparâmetros do modelo (`GridSearchCV`) [5].

- **Imbalanced-learn (imblearn)**: Imbalanced-learn é uma biblioteca que fornece técnicas para lidar com conjuntos de dados desbalanceados, como o oversampling da classe minoritária. Neste estudo, é utilizada para aplicar o algoritmo SMOTE (Synthetic Minority Over-sampling Technique), que gera amostras sintéticas da classe minoritária para equilibrar o conjunto de dados. O uso do SMOTE é crucial para evitar que o modelo seja tendencioso para a classe majoritária, melhorando a sua capacidade de generalização [6].

- **NumPy**: NumPy é uma biblioteca fundamental para computação científica em Python, oferecendo suporte para arrays de grandes dimensões e uma coleção abrangente de funções matemáticas para operações com arrays. Em aprendizado de máquina, NumPy é usada para manipular eficientemente arrays numéricos e realizar cálculos necessários para a validação e avaliação do modelo [7, 8].

- **SHAP (SHapley Additive exPlanations)**: SHAP é uma biblioteca para interpretação de modelos de aprendizado de máquina, fornecendo uma maneira de explicar a contribuição de cada característica para as previsões do modelo com base na teoria dos valores de Shapley. Neste estudo, SHAP é utilizada para criar gráficos que explicam a importância das variáveis de entrada no modelo, ajudando a interpretar o comportamento do modelo e entender como diferentes variáveis influenciam as previsões [2, 9].

- **OS (Operating System)**: OS é uma biblioteca padrão do Python que fornece uma interface para interagir com o sistema operacional. Ela é utilizada para manipular arquivos e diretórios, como verificar a existência de diretórios e criar diretórios para armazenar resultados e gráficos gerados durante o processo de modelagem [10].

## Etapas do Processo de Modelagem

1. **Pré-processamento de Dados**: O pré-processamento de dados é uma etapa fundamental que prepara o conjunto de dados para a modelagem. Essa etapa assegura que os dados estejam limpos, formatados corretamente e que todas as variáveis relevantes sejam incluídas no processo de modelagem [11].

    - **Carregamento dos Dados**: Utilizando pandas, o conjunto de dados é carregado a partir de um arquivo Excel. O arquivo contém várias variáveis que são importantes para a modelagem, incluindo características demográficas e ambientais que podem ser usadas para prever a presença de infecção em animais.

    - **Limpeza de Dados**: Após o carregamento, os dados são limpos para remover quaisquer valores ausentes ou inconsistentes. Isso é essencial para garantir que o modelo seja treinado com dados completos e de alta qualidade, evitando que informações incompletas introduzam viés ou erros no modelo.

    - **Separação das Variáveis**: O conjunto de dados é então dividido em variáveis explicativas (features) e a variável de interesse (target). As features representam os dados de entrada que o modelo utilizará para fazer previsões, como características demográficas ou ambientais, enquanto o target é a variável que o modelo está tentando prever, como a presença ou ausência de uma infecção.

2. **Divisão em Conjuntos de Treino e Teste**: Com as variáveis devidamente preparadas, o conjunto de dados é dividido em conjuntos de treino e teste usando `train_test_split` da biblioteca scikit-learn. Nesta etapa, 67% dos dados são utilizados para treinamento e 33% para teste (`test_size=0.33`). Essa divisão é essencial para avaliar a capacidade de generalização do modelo, permitindo que ele seja treinado em um subconjunto dos dados e testado em outro subconjunto que ele não viu antes [12].

3. **Tratamento de Desbalanceamento com SMOTE**: Após a divisão dos dados, o conjunto de treino pode apresentar um desbalanceamento significativo entre as classes. Para lidar com este problema, é aplicado o SMOTE (Synthetic Minority Over-sampling Technique) da biblioteca imblearn. O SMOTE gera amostras sintéticas da classe minoritária, equilibrando o número de exemplos em cada classe. Essa técnica é essencial para evitar que o modelo se torne tendencioso para a classe majoritária e para melhorar sua capacidade de generalização em situações de desbalanceamento [1].

4. **Seleção de Características com RFE**: A seleção de características é realizada utilizando o método RFE (Recursive Feature Elimination). Este método avalia iterativamente a importância de cada característica no modelo e elimina as menos relevantes, melhorando a interpretabilidade e a eficiência do modelo [13].

5. **Construção e Otimização do Modelo com Gradient Boosting**: O modelo de aprendizado de máquina utilizado é o `GradientBoostingClassifier` da biblioteca scikit-learn. Este é um método de ensemble que combina vários modelos fracos (árvores de decisão) para formar um modelo robusto [14–16, 16, 17].

6. **Otimização de Hiperparâmetros com Grid Search**: Para otimizar o modelo, o `GridSearchCV` é utilizado para realizar uma busca exaustiva nos hiperparâmetros. A seguir estão os hiperparâmetros ajustados e seus respectivos valores testados:

    - `smote__k_neighbors`: Número de vizinhos considerados para criar amostras sintéticas com SMOTE. Os valores testados são [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Este parâmetro influencia a diversidade das amostras sintéticas geradas e, portanto, a capacidade do modelo de generalizar a partir de exemplos minoritários.

    - `gbm__learning_rate`: Taxa de aprendizado do modelo Gradient Boosting, com valores variando de [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]. Este parâmetro controla o impacto de cada árvore adicionada ao modelo. Taxas mais baixas geralmente resultam em melhor generalização, enquanto taxas mais altas podem acelerar o processo de treinamento.

    - `gbm__subsample`: Proporção de amostras usadas para cada árvore, com valores testados de [0.4, 0.5, 0.6, 0.8, 1]. Este parâmetro introduz aleatoriedade, o que pode reduzir o overfitting e melhorar a robustez do modelo.

    - `gbm__n_estimators`: Número de árvores no ensemble, com valores testados de [50, 100, 200, 300, 400, 600]. Este parâmetro controla a complexidade do modelo, onde mais árvores podem capturar padrões mais complexos, mas também aumentar o risco de overfitting.

    - `gbm__max_depth`: Profundidade máxima das árvores de decisão, com valores variando de [3, 4, 5, 6, 7, 8, 9, 10]. Árvores mais profundas podem capturar interações mais complexas entre as características, mas também podem aumentar o risco de overfitting.

    - `gbm__min_samples_split`: Número mínimo de amostras necessárias para dividir um nó interno, com valores testados de [2, 3, 4, 5]. Este parâmetro ajuda a regularizar o modelo, prevenindo divisões que não adicionam informações suficientes.

    - `gbm__min_samples_leaf`: Número mínimo de amostras necessárias em um nó folha, com valores variando de [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. Este parâmetro ajuda a controlar a complexidade da árvore e a regularizar o modelo, prevenindo que nós folhas sejam formados a partir de muitas poucas amostras.

7. **Iterações do Modelo e Avaliação**: O processo de treinamento e otimização do modelo é repetido por 10 iterações. A cada iteração:

    - **Otimização e Avaliação**: Em cada iteração, os hiperparâmetros são ajustados com base nos resultados do `GridSearchCV`, e o desempenho do modelo é avaliado. As métricas de avaliação incluem AUPRC (Área sob a Curva de Precisão-Recall) e ROC AUC, que são calculadas para assegurar que o modelo não apenas tenha um bom desempenho em termos gerais, mas também seja capaz de diferenciar corretamente entre as classes, especialmente em casos de classes desbalanceadas [18].

    - **Depuração e Registro de Performance**: Em cada iteração, os parâmetros utilizados, os resultados das métricas de desempenho e quaisquer ajustes feitos nos hiperparâmetros são registrados para depuração. Isso permite uma análise detalhada de cada etapa do processo de modelagem e facilita a identificação de possíveis melhorias.

8. **Visualizações e Interpretação dos Resultados**: Ao final das 10 iterações, diversas visualizações são geradas para interpretar e entender o modelo final. Essas visualizações incluem:

    - **Curvas de Aprendizado**: Geradas para mostrar como o desempenho do modelo melhora com o aumento da quantidade de dados de treinamento [19, 20].

    - **Curvas ROC e Precisão-Revocação**: Utilizadas para avaliar a capacidade do modelo de discriminar entre classes positivas e negativas, especialmente em situações de desbalanceamento.

    - **Matrizes de Confusão**: Usadas para mostrar a taxa de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos, permitindo uma avaliação detalhada da performance do modelo [21].

Além disso, são geradas explicações baseadas em SHAP (SHapley Additive exPlanations) para identificar a importância de cada característica nas previsões do modelo. Gráficos de abelhas (beeswarm plots) e gráficos de barras são criados para visualizar como cada característica influencia as decisões do modelo, ajudando a interpretar o comportamento do modelo e garantindo sua transparência.

## Conclusão e Salvamento do Modelo

Após a avaliação completa e interpretação dos resultados, o melhor modelo é salvo usando a biblioteca `pickle` (padrão do Python) para uso futuro. As características selecionadas e as métricas de desempenho são documentadas, permitindo que o modelo seja replicado ou ajustado conforme necessário. Essa prática assegura a reprodutibilidade dos resultados e facilita o uso do modelo em aplicações futuras.

## Bibliografia

1. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP (2002) SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research* 16:321–357

2. Shapley LS (1953) A value for n-person games

3. McKinney W, Team PD (2015) Pandas-Powerful python data analysis toolkit. *Pandas—Powerful Python Data Analysis Toolkit* 1625:

4. Tosi S (2009) *Matplotlib for Python developers*. Packt Publishing Ltd

5. Pedregosa F, Varoquaux G, Gramfort A, et al (2011) Scikit-learn: Machine learning in Python. *The Journal of machine Learning research* 12:2825–2830

6. Lemaître G, Nogueira F, Aridas CK (2017) Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of machine learning research* 18:1–5

7. Oliphant TE (2006) *Guide to numpy*. Trelgol Publishing USA

8. Bressert E (2012) *SciPy and NumPy: an overview for developers*

9. Lundberg SM, Lee S-I (2017) A unified approach to interpreting model predictions. *Advances in neural information processing systems* 30:

10. Python W (2021) Python. Python releases for windows 24:

11. García S, Luengo J, Herrera F (2015) *Data Preprocessing in Data Mining*. Springer International Publishing, Cham

12. Rajer-Kanduč K, Zupan J, Majcen N (2003) Separation of data on the training and test set for modelling: a case study for modelling of five colour properties of a white pigment. *Chemometrics and intelligent laboratory systems* 65:221–229

13. Guyon I, Weston J, Barnhill S, Vapnik V (2002) [No title found]. *Machine Learning* 46:389–422.

14. Natekin A, Knoll A (2013) Gradient boosting machines, a tutorial. *Frontiers in neurorobotics* 7:21

15. Friedman JH (2001) Greedy function approximation: a gradient boosting machine. *Annals of statistics* 1189–1232

16. Learn S (2023) Gradient boosting classifier. Available at: https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

17. Dietterich TG (2000) Ensemble Methods in Machine Learning. In: Multiple Classifier Systems. Springer Berlin Heidelberg, Berlin, Heidelberg, pp 1–15

18. McDermott MBA, Hansen LH, Zhang H, et al (2024) A Closer Look at AUROC and AUPRC under Class Imbalance

19. Viering T, Loog M (2022) The shape of learning curves: a review. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 45:7799–7819

20. Perlich C (2010) Learning Curves in Machine Learning.

21. Ting KM (2011) Confusion matrix. *Encyclopedia of machine learning* 209–209
