
import json

# Open the dataset JSON
with open("dataSets/ingredient_and_instructions.json", "r") as dataSet:
    data = json.load(dataSet)

recipie = "" # The individual recipie
recipie_list = [] # Where we will store the recipie for training

for meal in data:
    # Get the recipie name and its instructions
    meal_info = data[meal]
    recipie += meal.replace("-", " ").title() + "\n\n"
    recipe_instructions = meal_info["instructions"]

    # Get the ingredients
    ingredients = meal_info["ingredient_sections"]

    for i in ingredients:
        # Retrieve ingredients and their details
        item = ""           # Dough
        qty = ""            # ½
        unit = ""           # cup
        ingredient = ""     # unsalted butter
        primary_unit = { "quantity": None, "display": None }

        # Some ingredients arent named, but are still listed as "1 cup" or similar
        # Additionally, if there is an ingredient name, it may already have a colon
        # in front of it, so we need to check for that before formatting
        if i["name"]:
            if i["name"][-1] != ":": # If there isnt a colon, add one
                item = i["name"] + ": \n"
            else: # Otherwise, just add the name
                item = i["name"] + " \n"
        else:
            item = ""

        recipie += item # Add the ingredient name to the recipie

        # Loop over each ingredient
        j = 0
        while j < len(i["ingredients"]):
            # i["ingredients"][0] may not exist, use try/except to prevent
            # any ListIndexErrors from accessing an index that doesn't exist
            try: primary_unit = i["ingredients"][j]["primary_unit"]
            except: primary_unit = { "quantity": None, "display": None } # Dummy object

            # Some recipies don't list an ingredient at all
            try: ingredient += i["ingredients"][j]["name"]
            except: ingredient += ""

            # If the try/except blocks above are successful, we can set the values
            if primary_unit["quantity"]: qty = primary_unit["quantity"] + " "
            if primary_unit["display"]: unit = primary_unit["display"] + " "

            # Construct the ingredient string
            recipie += qty + unit  + ingredient + "\n"
            ingredient = ""
            j += 1

        recipie += "\n"

    # Newline before adding instructions
    recipie += "\n"

    # Loop over the instructions and format them
    step = 1
    for text in recipe_instructions:
        instruction = text["display_text"]
        recipie += str(step) + ". " + instruction + "\n"
        step += 1

    # Add the recipie to the list
    recipie_list.append(recipie)

    # Free memory for the next recipie
    recipie = ""
    recipe_instructions = []
    ingredients = []
    item = ""
    qty = ""
    unit = ""
    ingredient = ""

# For testing purposes, we can output a random recipie
import random
print(recipie_list[random.randint(0, len(recipie_list) - 1)])

#print(recipie_list[0])

#CSV File 1 Record per row
#Record: 1 string with ingredients + recipe


# [[recipe1], [recipe2], [recipe3]]

import csv

with open("recipe_fixed.csv", 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    for recipie in recipie_list:
        csvwriter.writerow([recipie])

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

tokenizer_file = "aitextgen.tokenizer.json"
config = GPT2ConfigCPU()
ai = aitextgen(tokenizer_file=tokenizer_file, to_gpu=True, config=config, tf_gpt2="355M")

#Ai training disabled to prevent accidental overrides.


file_name = "recipe_fixed.csv"
ai.save("aitextgen.model")
train_tokenizer(file_name)
data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)
ai.train(data, batch_size=32, num_steps=64000, generate_every=1000, save_every=5000)
ai.generate(prompt="steak")

#Save the model
ai.save("aitextgen.model")