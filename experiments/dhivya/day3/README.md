# Recipe Chain Utility

A Python utility that uses Gemini prompt chaining to convert recipe ingredients into a categorized shopping list.

## What it does

This utility runs a chained processing flow to transform the input through multiple stages. Step 1 extracts ingredient names and quantities from recipe text. Step 2 organizes the extracted ingredients into categories such as baking and dairy.

## Setup

1. Install required Python packages.
   ```bash
   pip install google-generativeai python-dotenv
   ```
2. Create and configure a `.env` file.
   ```text
   GEMINI_API_KEY=your_api_key_here
   ```
3. Verify dependencies and run the utility from the terminal.

## Usage

```bash
python recipe_chain.py --input "2 cups flour, 1 cup sugar, 3 eggs"
```

## Example

Input:
```text
2 cups flour, 1 cup sugar, 3 eggs
```

Output:
```text
Baking:
- Flour
- Sugar

Dairy:
- Eggs
```

## Known limitations

- The utility may not handle all unusual input formats.
- It assumes required packages and environment variables are correctly configured.
- Error handling is limited for unexpected edge cases.
