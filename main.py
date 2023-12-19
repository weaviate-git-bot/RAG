from fastapi import FastAPI, HTTPException
import weaviate
import pandas as pd

import foodBERT
import helper
import ast
import os

app = FastAPI()

client = weaviate.Client("http://localhost:8080")
# client.timeout_config = (3, 200)


def add_ingredients(data, batch_size=512, debug_mode=False):
    """ upload embeddings to Weaviate

    :param data: wine data in panda dataframe object
    :type data: panda dataframe object (2 columns: 'Ingredient' and 'Embedding_Content')
    :param batch_size: number of data objects to put in one batch, defaults to 512
    :type batch_size: int, optional
    :param debug_mode: set to True if you want to display upload errors, defaults to False
    :type debug_mode: bool, optional
    """

    no_items_in_batch = 0

    for index, row in data.iterrows():
        ingredient_object = {
            "ingredient": index,
        }

        ingredient_uuid = helper.generate_uuid('IngredientVocabulary', index)
        ingredient_emb = ast.literal_eval('[' + row["Embedding_Content"].strip() + ']')
        client.batch.add_data_object(ingredient_object, "IngredientVocabulary", ingredient_uuid, ingredient_emb)
        no_items_in_batch += 1

        if no_items_in_batch >= batch_size:
            results = client.batch.create_objects()

            if debug_mode:
                for result in results:
                    if result['result'] != {}:
                        helper.log(result['result'])

                message = str(index) + ' / ' + str(data.shape[0]) + ' items imported'
                helper.log(message)

            no_items_in_batch = 0

    client.batch.create_objects()


@app.get("/schema")
async def create_schema():
    schema = {
        "classes": [
            {
                "class": "IngredientVocabulary",
                "vectorizer": "none",
                "properties": [
                    {
                        "name": "ingredient",
                        "dataType": ["text"]
                    }
                ]
            }
        ]
    }
    client.schema.create(schema)
    if os.path.exists('output_file.csv'):
        df = pd.read_csv('output_file.csv', index_col=0)
        add_ingredients(df.head(4500), batch_size=99, debug_mode=True)
        return {"message": "Schema created with embeddings"}
    return {"message": "Schema created"}


@app.delete("/schema")
async def delete_schema():
    """
    Delete all schemas.
    Returns:
    """
    try:
        client.schema.delete_all()
        return {"message": "Schema deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingredient")
async def add_ingredient(ingredient: str):
    """
        Add an ingredient to Weaviate.
        Args:
            ingredient (str): The name of the ingredient.
        Returns:
        """
    try:
        ingredient_object = {
            "ingredient": ingredient,
        }

        ingredient_uuid = helper.generate_uuid('IngredientVocabulary', ingredient)
        # TODO - create the embedding for the single ingredient
        ingredient_emb = ''  # ingredient embedding should be the vector as a list
        client.data_object.create(data_object=ingredient_object, class_name="IngredientVocabulary",
                                  uuid=ingredient_uuid,
                                  vector=ingredient_emb)

        return {"message": "Object added to Weaviate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_ingredient(source_ingredient: str, k: int = 100):
    """
        Search for ingredients in Weaviate based on the provided query.
        Args:
            source_ingredient (str): The vector representation of the ingredient.
            k (int): number of ingredients
        Returns:
            dict: The result of the search, including ingredient information and distance.
        """
    try:
        #TODO - Create the embedding for the source ingredient
        # ingredient_emb = foodBERT.get_embedding(source_ingredient)

        ingredient_emb = ast.literal_eval(source_ingredient.strip())
        near_vector = {
            "vector": ingredient_emb
        }

        return (
            client.query
            .get("IngredientVocabulary", ["ingredient"])
            .with_additional(["distance"])
            .with_near_vector(near_vector)
            .with_limit(k)
            .do()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
