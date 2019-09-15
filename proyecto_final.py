# TRABAJO FINAL
################################################################
################################################################

# Importando librerías
import pandas as pd
import numpy as np
import seaborn as sns

# Importando data
data = pd.read_csv('DataModelo.csv', sep=';')



# 1 - ANALIZANDO LA DATA
#####################################################

data.shape
data.describe()
data.head(5)

# Nos dimos cuenta que existe un dato inconsistente, porque todo estudiante de la UNI
# debe haber postulado como mínimo 1 vez
data[data.Postulaciones < 1]['Postulaciones']
# Eliminamos el dato erroneo
data.drop(357, inplace = True)


# Analizando los missing
##############################
np.sum(data.isnull())

# Analizando missings
np.sum(data.isnull())
np.mean(data.isnull())
# Conclusión: nos dimos cuenta de que no existen missings


# Analizando Variables numéricas
##############################

print(np.min(data.Edad))
print(np.max(data.Edad))

print(np.min(data.Semestres))
print(np.max(data.Semestres))

print(np.min(data.Postulaciones))
print(np.max(data.Postulaciones))

print(np.min(data.TiempoLlegada))
print(np.max(data.TiempoLlegada))

print(np.min(data.NumServicios))
print(np.max(data.NumServicios))

print(np.min(data.NumEquipos))
print(np.max(data.NumEquipos))

print(np.min(data.SatisProm))
print(np.max(data.SatisProm))

print(np.min(data.NumActividades))
print(np.max(data.NumActividades))

# Conclusión: No se observa valores extraños


# Analizando correlación entre variables
##############################

correlaciones = pd.DataFrame(np.corrcoef(data, rowvar = False))
correlaciones.columns = data.columns
correlaciones.index = data.columns
print(correlaciones)




# 2 - DEPURANDO LA DATA
#####################################################

# Eliminando columnas debido al analisis de correlaciones
columns_delete = ['ServiciosRecreo', 'ServiciosLujo', 'SatisBiblioFacu', 'SatisLaboratorios',
                  'SatisBiblioUNI', 'SatisMedico', 'SatisComedor', 'SatisVerdes', 'SatisDeportivos',
                  'Licuadora', 'Lavadora', 'Microondas', 'Ordinario']
data = data.drop(columns_delete, axis = 1)




# 3 - VISUALIZANDO INFORMACIÓN
#####################################################

#1: Buen rendimiento // 0: Mal rendimiento
#En base al anàlisis de correlaciòn, se analiza la var. Privada 1: Si // 0: No
Cuadro1 = data.groupby(['Privada'])['Rendimiento'].sum()/data.Privada.value_counts()
Cuadro1.index = ("No privado", "Sí privado")
Cuadro1.name = ("Buen rendimiento")

sns.set(style="whitegrid")
g1 = sns.catplot(x="Privada", y="Rendimiento",data=data,
                height=6, kind="bar", palette="muted")
g1.despine(left=True)
g1.set_ylabels("% Buen rendimiento")

# Conclusión: Se encontró, que de todas las variables, existe una mayor probabilidad
# de que los alumnos que estudiaron en un colegio Privado tengan un mejor rendimiento
# que los que no lo hicieron.


# Se analiza la 2da variable Indice de apoyo familiar 5: Mayor // 1: Menor
Cuadro2 = data.groupby(['IndiceApoyoFam'])['Rendimiento'].sum()/data.IndiceApoyoFam.value_counts()
Cuadro2.name = ("% Buen rendimiento")

g2 = sns.catplot(x="IndiceApoyoFam", y="Rendimiento",data=data,
                height=6, kind="bar", palette="muted")
g2.despine(left=True)
g2.set_ylabels("% Buen rendimiento")

# Conclusión: Se encontró, que de todas las variables, a medida que los alumnos tienen
# mayor apoyo familiar, mejora su probabilidad de tener un mejor rendimiento.




# 4 - ENTRENAMIENTO
#####################################################

# Separar variables de manera matricial
X = data.drop(['Rendimiento'], axis = 1)
y = data.loc[:, ['Rendimiento']]

# Importando función propia para obtener dataframe con resumen de indicadores de cada módelo
from modelos import resumen_table
resumen = resumen_table(X, y)
resumen

# Conclusión: Por ahora el módelo logit es el que presenta menos diferencia de los aciertos,
# y el xgb_cl es el que también tiene buenos indicadores.




# 5 - ENTRENAMIENTO CON LAS VARIABLES MÁS IMPORTANTES
#####################################################

# Separando en train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


# Calcular las variables más importantes con el Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier # librería
rf_cl = RandomForestClassifier(max_depth = 10, random_state = 0) # Objeto
rf_cl.fit(X_train, y_train) # Entrenamiento
# Importancia de las variables
importancia = pd.DataFrame(rf_cl.feature_importances_)
importancia.index = X_train.columns


# Entrenamiento con las principales 30 variables más importantes
nombres_imp_30 = importancia.sort_values(by=[0], ascending = False)[:30].index
X1 = data.loc[:, nombres_imp_30]
resumen_imp30 = resumen_table(X1, y)
resumen_imp30


# Entrenamiento con las principales 15 variables más importantes
nombres_imp_15 = importancia.sort_values(by=[0], ascending = False)[:15].index
X2 = data.loc[:, nombres_imp_15]
resumen_imp15 = resumen_table(X2, y)
resumen_imp15


# Entrenamiento con las principales 10 variables más importantes
nombres_imp_10 = importancia.sort_values(by=[0], ascending = False)[:10].index
X3 = data.loc[:, nombres_imp_10]
resumen_impo10 = resumen_table(X3, y)
resumen_impo10

# Conclusión: Se observa que con el XGBoost Classifier y 15 variables se obtiene
# mejores resultados

# Las variables del modelo con mejores indicadores son:
print(nombres_imp_15) 


# Calcular las variables más importantes con el XGBoost Classifier
# Tomamos como base las 30 variables más importantes del Random Forest Classifier
from sklearn.model_selection import train_test_split
X1train, X1test, y1train, y1test = train_test_split(X1, y, test_size = 0.2, random_state = 0)
# XGBoost Clasificacion
import xgboost as xgb
xgb_cl = xgb.XGBClassifier(objective = 'binary:logistic', max_depth = 10,
                           n_estimators = 10, seed = 4) # Objeto
xgb_cl.fit(X1train, y1train) # Entrenamiento
# Importancia de las variables
importancia_xgb = pd.DataFrame(xgb_cl.feature_importances_)
importancia_xgb.index = X1train.columns


# Entrenamiento con las 15 variables XGBoost más importantes
nombresxgb_imp_15 = importancia_xgb.sort_values(by=[0], ascending = False)[:15].index
X4 = data.loc[:, nombresxgb_imp_15]
resumen_imp15xgb = resumen_table(X4, y)
resumen_imp15xgb


# Entrenamiento con las 10 variables XGBoost más importantes
nombresxgb_imp_10 = importancia_xgb.sort_values(by=[0], ascending = False)[:10].index
X5 = data.loc[:, nombresxgb_imp_10]
resumen_impo10xgb = resumen_table(X5, y)
resumen_impo10xgb




# 6 - CONCLUSIONES
#####################################################

# Tomando 15 y 10 variables más importantes de XGBoost Clasiffier no logramos
# mejorar nuestro modelo. La mejor performance la tenemos con XGBoost Classifier,
# utilizando 15 mejores variables obtenidas del Random Forest Classifier:
print(nombres_imp_15.values)
resumen_imp15xgb['xgb_cl']
