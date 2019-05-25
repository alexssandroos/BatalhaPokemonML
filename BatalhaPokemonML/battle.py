import pandas as pd
import numpy as np

#utils 
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

def inverted_dataframe(df):
  list_first = list(df['First_pokemon'])
  df['First_pokemon'], df['Second_pokemon'] = df['Second_pokemon'],list_first 
  df['Winner'] = (df['Winner']+-1)*-1
  df['has_type_advantage'] = (df['has_type_advantage']+-1)*-1
  return df

def normalization(data_df):
    stats=["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Legendary"]
    stats_df=pokemon[stats].T.to_dict("list")
    one=data_df.First_pokemon.map(stats_df)
    two=data_df.Second_pokemon.map(stats_df)
    temp_list=[]
    for i in range(len(one)):
        temp_list.append(np.array(one[i])-np.array(two[i]))
    new_test = pd.DataFrame(temp_list, columns=stats)
    for c in stats:
        description=new_test[c].describe()
        new_test[c]=(new_test[c]-description['min'])/(description['max']-description['min'])
    return new_test

def apply_pipeline(df_combats, df_pokemon, type_strong_against):
  df_combats['first_type1'] = df_combats['First_pokemon'].replace(df_pokemon['Type 1'])
  df_combats['second_type1'] = df_combats['Second_pokemon'].replace(df_pokemon['Type 1'])
  df_combats['has_type_advantage'] = df_combats.apply(lambda x: has_type_advantage(x[3], x[4]), axis=1)
  
  df_combats.Winner[df_combats.Winner == df_combats.First_pokemon] = 0
  df_combats.Winner[df_combats.Winner == df_combats.Second_pokemon] = 1

  pokemon_legendary_map = {False: 0, True: 1}
  df_pokemon['Legendary'] = df_pokemon['Legendary'].map(pokemon_legendary_map)

  combats_invers = df_combats.copy().pipe(inverted_dataframe)
  combats_final = df_combats.append(combats_invers,\
     ignore_index=True)

  win_rate_First = combats_final.groupby('First_pokemon')['Winner'].sum()\
  /combats_final.groupby('First_pokemon')['Winner'].count()
  win_rate_Second = combats_final.groupby('Second_pokemon')['Winner'].sum()\
  /combats_final.groupby('Second_pokemon')['Winner'].count()
  combats_final['win_rate_first'] = combats_final['First_pokemon'].map(win_rate_First)
  combats_final['win_rate_second'] = combats_final['Second_pokemon'].map(win_rate_Second)     
  combats_final['win_rate_battle'] = combats_final['win_rate_first']/combats_final['win_rate_second']

  data=normalization(combats_final)
  data = pd.concat([data,combats_final.Winner], axis=1)
  data = pd.concat([data,combats_final.win_rate_battle], axis=1)
  data = pd.concat([data,combats_final.has_type_advantage], axis=1)
  data = data.replace([np.inf, -np.inf], 0)

  return data



#import dataset
pokemon=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/data/pokemon.csv',index_col=0)
combats=pd.read_csv('/home/alexssandroos/Público/BatalhaPokemonML/data/combats.csv')

data = apply_pipeline(combats, pokemon, tipo_vantagens)  
x_label=data.drop("Winner",axis=1) #inputs
y_label=data["Winner"] #outputs


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scaler = MinMaxScaler()
x_label= scaler.fit_transform(x_label)
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size=0.25, random_state=42)


clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(x_train, y_train) #training
pred = model.predict(x_test) #predicting on validation set
print('Accuracy of ', accuracy_score(pred, y_test)*100)



#importing test dataset 

data_valid=pd.read_csv('./data/tests.csv') 
data_valid['Winner'] = 0


#replace ids with name
new_test_data=data_valid[["First_pokemon","Second_pokemon"]].replace(pokemon.Name)
new_test_data['Winner'] = 0

#normalizing test data
final_data=apply_pipeline(data_valid, pokemon, tipo_vantagens)  
pred=model.predict(final_data.drop(columns=['Winner'))#final predictions
data_valid["Winner"]=[data_valid["First_pokemon"][i] if pred[i]==0 else data_valid["Second_pokemon"][i] for i in range(len(pred))]

combats_name = data_valid[cols].replace(pokemon.Name)
combats_name[63:64]

