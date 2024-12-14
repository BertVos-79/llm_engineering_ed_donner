# Importing 'Optional' from 'typing' to annotate variables that can be optional (i.e., have a value or be None).
# Importing 'AutoTokenizer' from 'transformers' to tokenize text using a pre-trained language model.
# Importing 're' for regular expression operations to clean and process text.

from typing import Optional
from transformers import AutoTokenizer
import re


# Specifying the base model name for the pre-trained language model used for tokenization.
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# Setting the minimum number of tokens required for text to be considered useful.
# Ensures the text contains enough content to be meaningful.
MIN_TOKENS = 150

# Setting the maximum number of tokens allowed in processed text to avoid exceeding model input limits.
# Texts longer than this limit will be truncated.
MAX_TOKENS = 160

# Defining the minimum number of characters required in a text before further processing.
# Acts as a pre-filter to exclude overly short content.
MIN_CHARS = 300

# Setting a character ceiling based on the maximum token limit, assuming an average token length of 7 characters.
# This ensures text length remains consistent with token constraints.
CEILING_CHARS = MAX_TOKENS * 7

# Defining the 'Item' class to represent a cleaned and curated product datapoint with a price.
# Includes functionality to process and generate prompts for machine learning tasks.

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    
    # Initializing a tokenizer using the pre-trained language model specified by 'BASE_MODEL'.
    # Allows tokenization of text for processing and prompt generation.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # A fixed string prefix used in generated prompts to include the product's price.
    PREFIX = "Price is $"

    # A predefined question used in prompts to ask about the product's price.
    QUESTION = "How much does this cost to the nearest dollar?"

    # A list of common phrases or patterns to remove during text cleaning.
    REMOVALS = [
        '"Batteries Included?": "No"',
        '"Batteries Included?": "Yes"',
        '"Batteries Required?": "No"',
        '"Batteries Required?": "Yes"',
        "By Manufacturer",
        "Item",
        "Date First",
        "Package",
        ":",
        "Number of",
        "Best Sellers",
        "Number",
        "Product "
    ]

    # Instance variables with type annotations for each product's properties.
    # Type annotation specifies the expected types of variables and function parameters or return values
    # 'title' is the name of the product.
    title: str

    # 'price' is the cost of the product.
    price: float

    # 'category' is the classification of the product (e.g., Electronics, Appliances).
    category: str

    # 'token_count' stores the number of tokens in the generated prompt. Defaults to 0.
    token_count: int = 0

    # 'details' optionally contains additional information about the product.
    details: Optional[str]

    # 'prompt' optionally contains the generated prompt for the product. Defaults to None.
    prompt: Optional[str] = None

    # 'include' is a boolean flag indicating whether the product passes the criteria for inclusion.
    include = False


    # Constructor method for the 'Item' class, with arguments:
    # - data: A dictionary containing information about the product (e.g., title, details, etc.).
    # - price: The price of the product as a float.
    # Initializes an instance with the provided product data and price.
    # Sets the 'title' attribute to the value of the 'title' key from the data dictionary.
    # Assigns the 'price' attribute directly from the input parameter.
    # Calls the 'parse' method to process the provided data and evaluate inclusion criteria.

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)


    # scrub_details() is a method to clean up the 'details' string by removing predefined patterns from the 'REMOVALS' list.
    # Each element in 'REMOVALS' list is removed from the 'details' string using the 'replace' method.
    # Returns the cleaned 'details' string.

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details


        # scrub is a method to clean and process a given text string ('stuff'):
        # 1. Subsitutes unnecessary characters with white space using regular expressions: re.sub(pattern, replacement, string)
        #    and strip() leading and trailing white space.
        # 2. Normalizes commas by reducing redundant instances to a single comma.
        # 3. Splits the text on white spaces into a list of words and in a double for-loop filters out:
        #    - for word in words: words longer than 6 characters.
        #    - for char in word: words containing numbers, as these often form product numbers that are irrelevant for our LLM training purpose.
        # 4. Joins the filtered words back into a single string and returns the cleaned text.

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    # parse is a method to parse a product's data and evaluate its inclusion based on character and token limits.
    # 1. Combines the description, features, and details into a single content string.
    # 2. Cleans the title and content using the 'scrub' and 'scrub_details' methods.
    # 3. Ensures the content meets minimum character and token length requirements.
    # 4. Truncates content to fit within the maximum character and token limits.
    # 5. Generates a prompt using the cleaned and truncated content.
    # 6. Sets the 'include' attribute to True if all conditions are satisfied.

    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        # Combine description into a single string with newlines.
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
    
        # Add features to the content string if they exist.
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
    
        # Add cleaned details to the content string if they exist.
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
    
        # Ensure the content meets the minimum character length requirement.
        if len(contents) > MIN_CHARS:
            # Truncate content to the maximum allowed character length.
            contents = contents[:CEILING_CHARS]
        
            # Clean the title and combined content for tokenization.
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
        
            # Tokenize the cleaned text and evaluate token length.
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                # Truncate tokens to the maximum allowed token count.
                tokens = tokens[:MAX_TOKENS]
                # Decode tokens back into text for prompt generation.
                text = self.tokenizer.decode(tokens)
                # Generate a prompt and set inclusion flag to True.
                self.make_prompt(text)
                self.include = True

                
    # make_prompt is a method to generate a structured prompt suitable for training.
    # 1. Combines the question, processed text, and price prefix into a formatted string.
    # 2. Rounds the price to the nearest dollar and appends it to the prompt.
    # 3. Tokenizes the generated prompt and calculates its token count.
    # This method sets the values of the attributes 'prompt' and 'token_count' of the Item instance.

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        # Construct the prompt by combining the question, processed text, and price.
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
    
        # Tokenize the prompt and calculate the token count.
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    
    # test_prompt method to generate a version of the prompt suitable for testing.
    # 1. Removes the actual price value from the original 'prompt' attribute.
    # 2. Splits the 'prompt' string at the 'PREFIX' (which introduces the price).
    # 3. Retains the part of the prompt before the price and reattaches the 'PREFIX'.
        
    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        # split prompt in 2 at PREFIX
        # retain the first element
        # add PREFIX to it 
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX


    # __repr__() is a special method to provide a string representation of the 'Item' object.
    # 1. Uses an f-string to format the title and price of the item.
    # 2. Encloses the formatted string in angle brackets for clarity.
    # 3. Designed for debugging and developer use, providing a concise summary of the item's key attributes.

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.title} = ${self.price}>"