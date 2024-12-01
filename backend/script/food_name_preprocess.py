import json
import pandas as pd

file_path = "../data/food_db/fooddb.json"
with open(file_path, 'r') as file:
    data = json.load(file)

food_descriptions = [item['description'] for item in data['SRLegacyFoods']]

food_descriptions_df = pd.DataFrame(food_descriptions, columns=['Description'])

output_path = "../data/food_db/food_descriptions.csv"
food_descriptions_df.to_csv(output_path, index=False, quoting=1)

print(f"Processed food descriptions saved to {output_path}")
