# KG Evaluation 
This repository is adapted from [MINE](https://github.com/stair-lab/kg-gen/tree/main/MINE) and aims to provide a more flexible evaluation environment. 

Improvements over existing MINE codebase:
- Allows for customization of judge LLM 
- Better input format facilitates custom input data

## Usage:
First run:

`uv run evaluation.py <the model to generate knowledge graphs> <the model to evaluate the knowledge graphs>`

Which will generate knowledge graphs and evaluate their efficacy. 

Then run:

`uv run results.py results/<the model to generate knowledge graphs>/<the model to evaluate the knowledge graphs>`

Which will display the results in an easy-to-comprehend format. 
