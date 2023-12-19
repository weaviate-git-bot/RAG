import torch
from transformers import BertModel, BertTokenizer
import json

# Load used ingredients
with open('foodbert/data/used_ingredients.json', 'r') as f:
    used_ingredients = json.load(f)

# Load tokenizer
tokenizer = BertTokenizer(vocab_file='foodbert/data/bert-base-cased-vocab.txt', do_lower_case=False, max_len=128,
                          never_split=used_ingredients)

# Load model
model = BertModel.from_pretrained(pretrained_model_name_or_path='foodbert/data/mlm_output/checkpoint-final')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Get input embeddings
embeddings = model.get_input_embeddings()


def get_embedding(ingredient):
    # Tokenize and encode the ingredient
    ingredient = ingredient.replace(' ', '_')
    input_ids = torch.tensor(tokenizer.encode(ingredient, add_special_tokens=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    outputs = model(input_ids)

    # Get embeddings for used ingredients
    ingredient_embeddings = []
    ingredient_names = []
    tokens = []

    for token_id in input_ids[0]:
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        tokens.append(token)
        if token in used_ingredients:
            ingredient_embeddings.append(embeddings.weight[token_id])
            ingredient_names.append(token)

    # Stack the embeddings
    ingredient_embeddings = torch.stack(ingredient_embeddings)

    # Print the embeddings with their ingredient names
    # for name, embedding in zip(ingredient_names, ingredient_embeddings):
    #     print(f"{name}: {embedding}")

    embedding_tensor = str(ingredient_embeddings[0])

    start_index = embedding_tensor.find("[")
    end_index = embedding_tensor.find("]")

    embedding = embedding_tensor[start_index:end_index + 1]

    return embedding
