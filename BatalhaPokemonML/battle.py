import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')


#utils 
def battle(first, second):
  dict_battle = {
    'First_pokemon': first, 
    'Second_pokemon': second, 
    'Winner': 0
  }
  type(dict_battle)
  return pd.DataFrame([dict_battle])

def prever_batalha(modelo_treinado, primeiro_pokemon, segundo_pokemon, pokemons_df, \
  combats=combats, pokemon=pokemon, tipo_vantagens=tipo_vantagens):
  df_ = battle(primeiro_pokemon, segundo_pokemon)
  data_batalha, combat_batalha = apply_pipeline(combats, df_, pokemon, tipo_vantagens)
  pred_batalha = modelo_treinado.predict(data_batalha.drop(columns=['Winner'])/100)
  prob_batalha = modelo_treinado.predict_proba(data_batalha.drop(columns=['Winner']))
  status_batalha = " vence " if pred_batalha[0] == 1 else " perde "
  num_proba_batalha = prob_batalha[0][1] if pred_batalha[0] == 1 else prob_batalha[0][0]
  print(pokemons_df.iloc[primeiro_pokemon]['Name'], ' ao desafiar ', \
    pokemons_df.iloc[segundo_pokemon]['Name'], status_batalha, "com ", num_proba_batalha*100, '% de precisao.' )
  return pred_batalha, prob_batalha, data_batalha, combat_batalha

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

def has_secondary_type(type):
  return 1 if pd.isnull(type) else 0

def featureImportances(rfInstance, X_train_columns):
   feature_importance = pd.DataFrame(rfInstance.feature_importances_,\
      index = X_train_columns, columns=['importance'])\
         .sort_values('importance', ascending=False)
   return feature_importance  

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
  df_combats['first_speed'] = df_combats['First_pokemon'].replace(df_pokemon['Speed'])  
  df_combats['second_speed'] = df_combats['Second_pokemon'].replace(df_pokemon['Speed'])  
  df_combats['first_type2'] = df_combats['First_pokemon'].replace(df_pokemon['Type 2'])
  df_combats['second_type2'] = df_combats['Second_pokemon'].replace(df_pokemon['Type 2'])
  df_combats['has_secondary_type_first'] = df_combats.apply(lambda x: has_secondary_type(x[5]), axis=1)
  df_combats['has_secondary_type_second'] = df_combats.apply(lambda x: has_secondary_type(x[6]), axis=1)
  df_combats['has_type_advantage_first'] = df_combats.apply(lambda x: has_type_advantage(x[3], x[4]), axis=1)
  df_combats['has_type_advantage_second'] = df_combats.apply(lambda x: has_type_advantage(x[4], x[3]), axis=1)
  df_combats['first_atk_power'] = df_combats['First_pokemon'].replace(df_pokemon['Attack'])\
      + df_combats['First_pokemon'].replace(df_pokemon['Sp. Atk'])\
      + df_combats['First_pokemon'].replace(df_pokemon['Speed'])
  df_combats['second_atk_power'] = df_combats['Second_pokemon'].replace(df_pokemon['Attack'])\
      + df_combats['Second_pokemon'].replace(df_pokemon['Sp. Atk'])\
      + df_combats['Second_pokemon'].replace(df_pokemon['Speed'])
  
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
  df_combats['first_more_fast'] = df_combats['first_speed']>df_combats['second_speed']
  df_combats['first_more_fast'] = df_combats['first_more_fast'].map(boolean_map)

  data=normalization(df_combats)
  data = pd.concat([data,df_combats.Winner], axis=1)
  data = pd.concat([data,df_combats.win_rate_first], axis=1)
  data = pd.concat([data,df_combats.win_rate_second], axis=1)
  data = pd.concat([data,df_combats.has_type_advantage_first], axis=1)
  data = pd.concat([data,df_combats.has_type_advantage_second], axis=1)
  data = pd.concat([data,df_combats.has_secondary_type_first], axis=1)
  data = pd.concat([data,df_combats.has_secondary_type_second], axis=1)
  data = pd.concat([data,df_combats.first_atk_power], axis=1)
  data = pd.concat([data,df_combats.second_atk_power], axis=1)
  data = data.replace([np.inf, -np.inf], 0)
  data = data.replace(np.nan, 0)
  # bool_cols = ["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed", "Legendary"]
  # for col in bool_cols:
  #   data[col] = data[col].map(boolean_map)
    
  return data, df_combats



pokemon=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/BatalhaPokemonML/data/pokemon.csv',index_col=0)
combats=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/BatalhaPokemonML/data/combats.csv')

data, train_combats = apply_pipeline(combats, combats, pokemon, tipo_vantagens)  
x_label=data.drop("Winner",axis=1) 
y_label=data["Winner"] 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
x_label= scaler.fit_transform(x_label)
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size=0.25, random_state=100)


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1 )
model = clf.fit(x_train, y_train) 
pred = model.predict(x_test)
print('Accuracy of ', accuracy_score(pred, y_test)*100)

featureImportances(model,data.drop("Winner",axis=1).columns)

pred_ , proba_ , data_, combat_= prever_batalha(model, 9, 30, pokemon) 