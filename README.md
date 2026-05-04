# NutriDataLab

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

> 🚀 Sistema completo de Machine Learning para análise nutricional, previsão de sucesso em dietas e segmentação de pacientes.

---

## 📸 Preview

<p align="center">
  <img src="outputs/clusters_pca.png" width="600"/>
</p>

<p align="center">
  <img src="outputs/model_comparison_f1.png" width="600"/>
</p>

<p align="center">
  <img src="outputs/feature_importance_random_forest.png" width="600"/>
</p>

---

## 🚀 Visão Geral

O **NutriPredict AI** é um pipeline completo de Ciência de Dados aplicado à nutrição, simulando um ambiente real de análise preditiva.

O sistema é capaz de:

- 👥 Segmentar pacientes com clustering  
- ✅ Prever sucesso de dietas (classificação)  
- 📉 Estimar perda de peso (regressão)  

---

## 🧠 Problema

Nutricionistas enfrentam dificuldades em:

- Prever resultados de dietas  
- Identificar padrões de comportamento  
- Personalizar estratégias  

👉 Este projeto resolve isso utilizando **Machine Learning + Dados**

---

## 🗂️ Estrutura do Projeto


NutriDataLab/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data/
│ ├── features/
│ ├── clustering/
│ ├── models/
│ └── visualization/
│
├── outputs/
├── main.py
└── README.md


---

## ⚙️ Tecnologias

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔄 Pipeline

### 📥 1. Dados
- Geração de dados sintéticos
- Múltiplas fontes (pacientes, dietas, resultados)

---

### 🔗 2. Integração
- Merge de datasets
- Dataset consolidado

---

### 🧪 3. Feature Engineering
- `bmi_category`
- `lifestyle_score`
- `age_group`
- `adherence_x_motivation`

---

### 🧹 4. Pré-processamento
- Tratamento de valores nulos  
- Encoding de variáveis categóricas  
- Normalização (StandardScaler)  
- Remoção de outliers  
- Prevenção de data leakage  

---

### 👥 5. Clustering
- K-Means  
- Escolha automática do melhor K (Silhouette Score)  
- Visualização com PCA  

---

### 🤖 6. Modelagem

#### 🔵 Classificação (diet_success)
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

#### 🟢 Regressão (weight_loss_kg)
- Ridge Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

---

### 📊 7. Avaliação

#### Classificação:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Cross-validation  

#### Regressão:
- RMSE  
- MAE  
- R²  
- Cross-validation  

---

## 📈 Resultados

- ✔ Alta performance na classificação  
- ✔ Boa precisão na regressão  
- ✔ Segmentação eficiente de pacientes  

---

## ⚠️ Observações

- Dados utilizados são **sintéticos**
- O projeto inclui tratamento para evitar **data leakage**
- Resultados em produção real podem variar

---

## ▶️ Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/NutriDataLab.git
cd NutriDataLab
2. Instalar dependências
pip install -r requirements.txt
3. Executar o projeto
python main.py
📁 Outputs Gerados
📊 Gráficos de análise exploratória
🤖 Comparação de modelos
🧠 Importância de features
👥 Visualização de clusters
🚀 Melhorias Futuras
API com FastAPI
Dashboard interativo (Streamlit)
Deploy em nuvem
Uso de dados reais
Otimização com GridSearchCV
👨‍💻 Autor

João Gabriel

⭐ Destaque

Projeto desenvolvido com foco em portfólio profissional em Data Science, cobrindo todas as etapas de um pipeline real de Machine Learning.