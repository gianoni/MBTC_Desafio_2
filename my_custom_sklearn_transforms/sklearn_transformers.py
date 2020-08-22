from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# CUSTOM CODE - TALOPES
# Cria a classe para ajuste da coluna NOTA_GO com a média (MEAN) das notas do próprio aluno nas outras disciplinas

# All sklearn Transforms must have the `transform` and `fit` methods
class AjusteNotaGO(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform( self, X, y=None):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #Ajusta a NOTA DE GO com a média da linha
        data['NOTA_GO'].fillna(round(data[['NOTA_DE', 'NOTA_MF', 'NOTA_EM']].mean(axis=1)), inplace = True)
        #Agora preenche os nulos com zero
        data.fillna(value=0, inplace=True)
        # Retornamos um novo dataframe
        return data
    
# CUSTOM CODE - TALOPES
# CRIA A CLASSE DAS TRANSFORMACOES
from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class Custom_Autobots(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
    
    # Metodo para remover os outliers com base em IQR, mas permite ajustar o RANGE para realizar um corte diferente para cada disciplina
    def remove_outlier(df_in, col_name, q1_p, q2_p):
        q1 = df_in[col_name].quantile(q1_p)
        q3 = df_in[col_name].quantile(q2_p)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out

    def transform( self, X, y):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data2 = y.copy()

        # ALTERAR OS VALORES 0 DAS NOTAS PELA MEDIA
        data.loc[ (data.NOTA_DE == 0), 'NOTA_DE' ] = X['NOTA_DE'].mean()
        data.loc[ (data.NOTA_EM == 0), 'NOTA_EM' ] = X['NOTA_EM'].mean()
        data.loc[ (data.NOTA_MF == 0), 'NOTA_MF' ] = X['NOTA_MF'].mean()
        data.loc[ (data.NOTA_GO == 0), 'NOTA_GO' ] = X['NOTA_GO'].mean()
                
        # Remove os outliers
        data = Custom_Autobots.remove_outlier(data,"NOTA_DE", 0.4, 0.75)
        data = Custom_Autobots.remove_outlier(data,"NOTA_EM", 0.4, 0.75)
        data = Custom_Autobots.remove_outlier(data,"NOTA_MF", 0.25, 0.7)
        data = Custom_Autobots.remove_outlier(data,"NOTA_GO", 0.35, 0.65)
        
        data2 = Custom_Autobots.remove_outlier(data2,"NOTA_DE", 0.4, 0.75)
        data2 = Custom_Autobots.remove_outlier(data2,"NOTA_EM", 0.4, 0.75)
        data2 = Custom_Autobots.remove_outlier(data2,"NOTA_MF", 0.25, 0.7)
        data2 = Custom_Autobots.remove_outlier(data2,"NOTA_GO", 0.35, 0.65)


        # Retornamos um novo dataframe lindão
        return data, data2
    
    class SmoteResample(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        return X_resampled, y_resampled
