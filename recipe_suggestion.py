# Install the transformers library
!pip install transformers

# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the RecipeGenerator class
class RecipeGenerator:
    def _init_(self):
        # Load pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_recipe(self, ingredients):
        # Encode the ingredients
        input_text = " ".join(ingredients)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate a recipe using the GPT-2 model
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2)

        generated_recipe = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_recipe

# Create an instance of RecipeGenerator
recipe_generator = RecipeGenerator()

# Example usage
ingredients_input = ['tomato', 'pasta', 'cheese','chicken']
output_recipe = recipe_generator.generate_recipe(ingredients_input)

# Print the generated recipe
print(output_recipe)