# Text to SQL Model

A machine learning model that converts natural language questions into SQL queries, specifically designed for inventory management systems.

## Features

- **Natural Language to SQL**: Convert plain English questions to SQL queries
- **Pattern Matching**: Fast pattern-based SQL generation for common queries
- **Feedback Learning**: Model learns from user corrections
- **Schema-Aware**: Uses database schema information for better query generation
- **Interactive Console**: User-friendly console interface for testing queries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hongjarren/TexttoSQLREAL.git
cd TexttoSQLREAL
```

2. Install required dependencies:
```bash
pip install torch transformers datasets evaluate scikit-learn pandas numpy
```

## Usage

### Training the Model

```python
from texttosql import TextToSQLModel

# Initialize model
model = TextToSQLModel()
model.initialize_model()

# Load and train on data
datasets = model.load_data()
model.train(datasets, output_dir="./text_to_sql_results", num_epochs=20)
```

### Using the Interactive Console

```bash
python sql_console.py
```

### Generating SQL Queries

```python
# Generate SQL from natural language
question = "Show me all parts created in 2024"
sql_query = model.generate_sql(question)
print(sql_query)
```

## Supported Query Types

- **Basic Selection**: "Show me all items", "List all part numbers"
- **Date Filtering**: "Show parts created in 2024", "Items updated last month"
- **Code Filtering**: "Parts with ER code XYZ789"
- **Status Filtering**: "Active items only", "Items that are orderable"
- **Quantity Filtering**: "Items with price greater than 100"

## Model Architecture

- **Base Model**: Salesforce CodeT5+ (220M parameters)
- **Fine-tuning**: Seq2Seq training on inventory-specific queries
- **Pattern Matching**: Rule-based fallback for common query patterns
- **Post-processing**: Schema-aware SQL correction and validation

## Project Structure

```
TexttoSQLREAL/
├── texttosql.py          # Main model implementation
├── sql_console.py        # Interactive console interface
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── requirements.txt     # Python dependencies
```

## Example Queries

| Natural Language | Generated SQL |
|-----------------|---------------|
| "Show me all parts" | `SELECT * FROM vMTL_SYSTEM_ITEMS` |
| "Parts with ER code XYZ789" | `SELECT PART_NUMBER, DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE ER_CODE = 'XYZ789'` |
| "Items created in February 2024" | `SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-02-01' AND CREATION_DATE <= '2024-02-29'` |
| "Active items only" | `SELECT * FROM vMTL_SYSTEM_ITEMS WHERE INVENTORY_ITEM_STATUS_CODE = 'ACTIVE'` |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
