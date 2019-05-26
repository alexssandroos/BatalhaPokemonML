import pandas as pd
import numpy as np

#utils 
def battle(first, second):
  dict_battle = {
    'First_pokemon': first, 
    'Second_pokemon': second, 
    'Winner': 0
  }
  type(dict_battle)
  return pd.DataFrame([dict_battle])

def prever_batalha(modelo_treinado, primeiro_pokemon, segundo_pokemon):
  df = battle(primeiro_pokemon, segundo_pokemon)
  data, combat = apply_pipeline(combats, df, pokemon, tipo_vantagens)
  pred = modelo_treinado.predict(data.drop(columns=['Winner']))
  prob = modelo_treinado.predict_proba(data.drop(columns=['Winner']))
  return pred, prob, data, combat

tipo_vantagens = {
  'Grass': ['Ground', 'Rock', 'Water'], 
  'Fire': ['Bug', 'Steel', 'Grass', 'Ice'], 
  'Water': ['Ground', 'Rock', 'Fire'], 
  'Bug': ['Grass', 'Psychic', 'Dark'], 
  'Normal' : [''], 
  'Poison': ['Grass', 'Fairy'], 
  'Electric': ['Flying', 'Water'],
  'Ground': ['Poison', 'Rock', 'Steel', 'Fire', 'Electric'], 
  'Fairy': ['Fighting', 'Dragon', 'Dark'], 
  'Fighting' : ['Normal', 'Rock', 'Steel', 'Ice', 'Dark'], 
  'Psychic': ['Fighting', 'Poison'], 
  'Rock': ['Flying', 'Bug', 'Fire', 'Ice'], 
  'Ghost': ['Ghost', 'Psychic'], 
  'Ice': ['Flying', 'Ground', 'Grass', 'Dragon'],
  'Dragon': ['Dragon'], 
  'Dark': ['Ghost', 'Psychic'], 
  'Steel': ['Rock', 'Ice', 'Fairy'], 
  'Flying' : ['Fighting', 'Bug', 'Grass']}

#https://www.eurogamer.net/articles/2018-12-21-pokemon-go-type-chart-effectiveness-weaknesses
def has_type_advantage(type1, type2):
  return tipo_vantagens[type1].count(type2)

def normalization(data_df):
    stats=["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed", "Legendary"]
    stats_df=pokemon[stats].T.to_dict("list")
    one=data_df.First_pokemon.map(stats_df)
    two=data_df.Second_pokemon.map(stats_df)
    temp_list=[]
    for i in range(len(one)):
        temp_list.append(np.array(one[i])-np.array(two[i]))
    new_test = pd.DataFrame(temp_list, columns=stats)
    return new_test

def apply_pipeline(all_combats, df_combats, df_pokemon, type_strong_against):
  df_combats['first_type1'] = df_combats['First_pokemon'].replace(df_pokemon['Type 1'])
  df_combats['second_type1'] = df_combats['Second_pokemon'].replace(df_pokemon['Type 1'])
  df_combats['has_type_advantage_first'] = df_combats.apply(lambda x: has_type_advantage(x[3], x[4]), axis=1)
  df_combats['has_type_advantage_second'] = df_combats.apply(lambda x: has_type_advantage(x[4], x[3]), axis=1)
  
  df_combats.Winner[df_combats.Winner == df_combats.First_pokemon] = 0
  df_combats.Winner[df_combats.Winner == df_combats.Second_pokemon] = 1

  boolean_map = {False: 0, True: 1}
  df_pokemon['Legendary'] = df_pokemon['Legendary'].map(boolean_map)

  win_rates = (all_combats.groupby('First_pokemon')['Winner'].sum()\
  +all_combats.groupby('Second_pokemon')['Winner'].sum())
  df_combats['win_rate_first'] = df_combats['First_pokemon'].map(win_rates)
  df_combats['win_rate_second'] = df_combats['Second_pokemon'].map(win_rates)     
  df_combats['win_rate_biggest'] = df_combats['win_rate_first']>df_combats['win_rate_second']
  df_combats['win_rate_biggest'] = df_combats['win_rate_biggest'].map(boolean_map)

  data=normalization(combats_final)
  data = pd.concat([data,combats_final.Winner], axis=1)
  data = pd.concat([data,combats_final.win_rate_first], axis=1)
  data = pd.concat([data,combats_final.win_rate_second], axis=1)
  data = pd.concat([data,combats_final.has_type_advantage_first], axis=1)
  data = pd.concat([data,combats_final.has_type_advantage_second], axis=1)
  data = data.replace([np.inf, -np.inf], 0)
  data = data.replace(np.nan, 0)

  return data, combats_final



#import dataset
pokemon=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/data/pokemon.csv',index_col=0)
combats=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/data/combats.csv')

data, train_combats = apply_pipeline(combats, combats, pokemon, tipo_vantagens)  
x_label=data.drop("Winner",axis=1) #inputs
y_label=data["Winner"] #outputs


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
x_label= scaler.fit_transform(x_label)
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size=0.20, random_state=100)


clf = RandomForestClassifier(n_estimators=200)
model = clf.fit(x_train, y_train) #training
pred = model.predict(x_test) #predicting on validation set
print('Accuracy of ', accuracy_score(pred, y_test)*100)


data_valid=pd.read_csv('./data/tests.csv') 
data_valid['Winner'] = 0

pred_ , proba_ , data_, combat_= prever_batalha(model, 4, 10)  

#replace ids with name
new_test_data=data_valid[["First_pokemon","Second_pokemon"]].replace(pokemon.Name)
new_test_data['Winner'] = 0

#normalizing test data
final_data, combats_new =apply_pipeline(a, pokemon, tipo_vantagens)  
pred=model.predict(final_data.drop(columns=['Winner']))#final predictions
pred_proba = model.predict_proba(final_data.drop(columns=['Winner']))
data_valid["Winner"]=pred[:10000]
combats_name = data_valid[["First_pokemon","Second_pokemon"]].replace(pokemon.Name)
combats_name[63:64]

