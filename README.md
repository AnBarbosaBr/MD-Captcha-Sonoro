# MD-Captcha-Sonoro

## Treino
https://ufabcedubr-my.sharepoint.com/:u:/g/personal/pedro_faustini_ufabc_edu_br/EelTnusYbRhClO3YhHwI4ekBrrwviMa0df_O3aC_xx-ziw?e=EqgnxL

## Validação
https://ufabcedubr-my.sharepoint.com/:u:/g/personal/pedro_faustini_ufabc_edu_br/EfPA3YkWnK1CuiCOXKBPqCoBxv9HYMULDQKbDS-pNtMjBQ?e=RErceO

## Rodando a(s) Análise(s)
### projeto1.py
A fim de atualizar a análise, basta alterar as variáveis:
**train_path** e **test_path**, apontando para os diretórios em que se encontram os dados de treino e validação.  
As variáveis **train_pickle** e **test_pickle** podem ser definidas como *None*, ou então como um caminho para um arquivo do tipo *pickle*, como *treinamento.pickle*. O comando *load_data* tentará ler os dados primeiro desses arquivos, se eles existirem. Se não existirem *load_data* irá salvar os dados lidos dos diretórios no arquivo indicado. Se o valor das variáveis for *None*, o código executa normalmente

```python
# Não salva os data frames com os dados lidos dos diretórios de treino e validação
train_pickle = None
test_pickle = None

# Lê/Salva os data frames com os dados lidos
train_pickle = os.path.join(".","train.pickle")
test_pickle = os.path.join(".","test.pickle")
```

Após realizar uma análise com a função process_data, a mesma irá imprimir algumas 
métricas baseadas nos dados de validação, e retornará uma tupla com os valores preditos para o conjunto de treino e para o conjunto de validação, respectivamente. Note que essas predições estarãm divididas por letra, e processadas pelo *label_encoder*.

Se desejar obter a classificação em linguagem humana, é necessário usar a 
LabelEncoder, e fazer a transformação inversa:
```python
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(["a", "b", "c", "d", "h", "m", "n", "x", "6", "7","?"])
labels_human = label_encoder.inverse_transform(labels_predicted)
```

Para obter o arquivo original, é necessário efetuar o pré-processamento dos dados com *preprocess(df)* e concatenar as predições. Isso pode ser feito dessa forma:
```python
preprocessed_df = preprocess(df)
preprocessed_df.insert(loc  = 0, column = 'pred_label', value = labels_human)
```

