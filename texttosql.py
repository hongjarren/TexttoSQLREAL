import os
import torch
import pandas as pd
import numpy as np
import re
import logging
import datetime
import json
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate

class TextToSQLModel:
    def __init__(self, model_name="Salesforce/codet5p-220m", max_input_length=128, max_target_length=128, 
                 log_file="sql_queries.log", feedback_file="sql_feedback.json", schema_file="schema.json"):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        self.log_file = log_file
        self._setup_logging()
        
        # Set up feedback collection
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback_data()
        
        # Track performance metrics
        self.performance_metrics = {
            "total_queries": 0,
            "pattern_matched": 0,
            "model_generated": 0,
            "user_corrected": 0,
            "success_rate": 0.0
        }
        
        # Load schema configuration
        self.schema = self._load_schema(schema_file)
    
    def _setup_logging(self):
        """Set up logging configuration"""
        # Create a logger
        self.logger = logging.getLogger("TextToSQL")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(self.log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | Question: %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Set formatters
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"TextToSQL model initialized with {self.model_name}")
    
    def log_query(self, question, sql_query, success=True, method="model"):
        """Log a question and its generated SQL query"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a formatted log entry
        log_entry = f"\n--- Query Log Entry: {timestamp} ---\n"
        log_entry += f"Question: {question}\n"
        log_entry += f"SQL: {sql_query}\n"
        log_entry += f"Status: {'Success' if success else 'Failure'}\n"
        log_entry += f"Method: {method}\n"
        log_entry += "-" * 50
        
        # Append to log file
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry)
        
        # Also log to the logger
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"{question} -> {sql_query}")
        
        return log_entry
    
    def load_data(self, data_path=None, data_list=None):
        """
        Load data either from a CSV file or a list of dictionaries.
        Each item should have 'question' and 'sql' keys.
        """
        if data_path and os.path.exists(data_path):
            # Load from CSV file
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        elif data_list:
            # Use provided list
            data = data_list
        else:
            # Balanced sample data covering various SQL operations
            data = [
                {"question": "Show me all items", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS"},
                {"question": "List all part numbers", "sql": "SELECT PART_NUMBER FROM vMTL_SYSTEM_ITEMS"},
                {"question": "Get all descriptions of items", "sql": "SELECT DESCRIPTION FROM vMTL_SYSTEM_ITEMS"},
                {"question": "Show items that are not orderable", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE ORDERABLE_ON_WEB_FLAG = 'N'"},
                {"question": "List all items with a price greater than 100", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE LIST_PRICE_PER_UNIT > 100"},
                {"question": "Get items with a warranty", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE WARRANTY_VENDOR_ID IS NOT NULL"},
                {"question": "Get all items in a specific organization", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE ORGANIZATION_ID = 1"},
                {"question": "List all items with a certain inventory status", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE INVENTORY_ITEM_STATUS_CODE = 'ACTIVE'"},
                {"question": "Show all items from a specific supplier", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE SUPPLIER = 'SupplierA'"},
                {"question": "Get the planning make-buy code for each item", "sql": "SELECT PART_NUMBER, PLANNING_MAKE_BUY_CODE_NAME FROM vMTL_SYSTEM_ITEMS"},
                {"question": "Find all items with a minimum order quantity of 5", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE MINIMUM_ORDER_QUANTITY >= 5"},
                {"question": "Show the list of items with a cost of sales account", "sql": "SELECT PART_NUMBER, COST_OF_SALES_ACCOUNT FROM vMTL_SYSTEM_ITEMS"},
                {"question": "List all items eligible for internal order", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE INTERNAL_ORDER_ENABLED_FLAG = 'Y'"},
                {"question": "Find all items with fixed lead time", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE FIXED_LEAD_TIME IS NOT NULL"},
                {"question": "Get all items in a specific location", "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE LOCATION_CONTROL_CODE = 'LOC1'"},
                
                {"question": "Show me all items along with their supplier names", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, suppliers.SUPPLIER_NAME FROM vMTL_SYSTEM_ITEMS JOIN suppliers ON vMTL_SYSTEM_ITEMS.SUPPLIER = suppliers.SUPPLIER_ID"},
                
                {"question": "List all items with their current stock levels and their corresponding warehouse locations", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, inventory_levels.STOCK_LEVEL, warehouse.LOCATION_NAME FROM vMTL_SYSTEM_ITEMS JOIN inventory_levels ON vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID = inventory_levels.INVENTORY_ITEM_ID JOIN warehouse ON inventory_levels.WAREHOUSE_ID = warehouse.WAREHOUSE_ID"},
                
                {"question": "Get all items that have a warranty and are from a specific supplier", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, WARRANTY_VENDOR_ID FROM vMTL_SYSTEM_ITEMS WHERE WARRANTY_VENDOR_ID IS NOT NULL AND SUPPLIER = 'SupplierA'"},
                
                {"question": "List all items that have been recently updated and their corresponding inventory status", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, LAST_UPDATE_DATE, INVENTORY_ITEM_STATUS_CODE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE > '2025-01-01'"},
                
                {"question": "Find items with the highest price from each inventory organization", 
                "sql": "SELECT ORGANIZATION_ID, PART_NUMBER, DESCRIPTION, MAX(LIST_PRICE_PER_UNIT) AS MAX_PRICE FROM vMTL_SYSTEM_ITEMS GROUP BY ORGANIZATION_ID"},
                
                {"question": "Get the items with the highest inventory levels by warehouse location", 
                "sql": "SELECT warehouse.LOCATION_NAME, vMTL_SYSTEM_ITEMS.PART_NUMBER, inventory_levels.STOCK_LEVEL FROM inventory_levels JOIN vMTL_SYSTEM_ITEMS ON inventory_levels.INVENTORY_ITEM_ID = vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID JOIN warehouse ON inventory_levels.WAREHOUSE_ID = warehouse.WAREHOUSE_ID WHERE inventory_levels.STOCK_LEVEL = (SELECT MAX(STOCK_LEVEL) FROM inventory_levels WHERE WAREHOUSE_ID = warehouse.WAREHOUSE_ID)"},
                
                {"question": "List all items that are marked as orderable on the web along with their supplier and inventory status", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, SUPPLIER, INVENTORY_ITEM_STATUS_CODE FROM vMTL_SYSTEM_ITEMS WHERE ORDERABLE_ON_WEB_FLAG = 'Y'"},
                
                {"question": "Show all items with their cost and sales accounts", 
                "sql": "SELECT PART_NUMBER, COST_OF_SALES_ACCOUNT, SALES_ACCOUNT FROM vMTL_SYSTEM_ITEMS JOIN sales_accounts ON vMTL_SYSTEM_ITEMS.SALES_ACCOUNT_ID = sales_accounts.SALES_ACCOUNT_ID"},
                
                {"question": "List all items with a planning time fence greater than 30 days", 
                "sql": "SELECT PART_NUMBER, PLANNING_TIME_FENCE_DAYS, DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE PLANNING_TIME_FENCE_DAYS > 30"},
                
                {"question": "Show the total number of items ordered by each buyer along with their total cost", 
                "sql": "SELECT BUYER_NAME, COUNT(PART_NUMBER) AS ITEM_COUNT, SUM(LIST_PRICE_PER_UNIT) AS TOTAL_COST FROM vMTL_SYSTEM_ITEMS GROUP BY BUYER_NAME"},
                
                {"question": "Get items with their corresponding tax code and description", 
                "sql": "SELECT PART_NUMBER, TAX_CODE, DESCRIPTION FROM vMTL_SYSTEM_ITEMS JOIN tax_codes ON vMTL_SYSTEM_ITEMS.TAX_CODE = tax_codes.TAX_CODE_ID"},
                
                {"question": "Find all items with a quantity on hand less than 10 and from a specific catalog", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, inventory_levels.STOCK_LEVEL FROM vMTL_SYSTEM_ITEMS JOIN inventory_levels ON vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID = inventory_levels.INVENTORY_ITEM_ID JOIN item_catalog ON vMTL_SYSTEM_ITEMS.ITEM_CATALOG_GROUP_ID = item_catalog.CATALOG_GROUP_ID WHERE inventory_levels.STOCK_LEVEL < 10 AND item_catalog.CATALOG_NAME = 'CatalogA'"},
                
                {"question": "Show all items with their respective minimum and maximum order quantities", 
                "sql": "SELECT PART_NUMBER, MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY FROM vMTL_SYSTEM_ITEMS"},
                
                {"question": "Get all items where the cost of sales account matches the expense account", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, COST_OF_SALES_ACCOUNT, EXPENSE_ACCOUNT FROM vMTL_SYSTEM_ITEMS WHERE COST_OF_SALES_ACCOUNT = EXPENSE_ACCOUNT"},
                
                {"question": "List all items that have a predefined serial number and are active", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, SERIAL_NUMBER_CONTROL_CODE FROM vMTL_SYSTEM_ITEMS WHERE SERIAL_NUMBER_CONTROL_CODE IS NOT NULL AND INVENTORY_ITEM_STATUS_CODE = 'ACTIVE'"},
                
                {"question": "Show items that are marked for preventive maintenance", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE PREVENTIVE_MAINTENANCE_FLAG = 'Y'"},
                
                {"question": "Get all items with their environmental compliance status", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, ENVIRONMENTAL_COMPLIANCE_STATUS FROM vMTL_SYSTEM_ITEMS"},
                
                {"question": "List all items with their pricing and vendor warranty details", 
                "sql": "SELECT PART_NUMBER, LIST_PRICE_PER_UNIT, WARRANTY_VENDOR_ID, WARRANTY_VENDOR_NAME FROM vMTL_SYSTEM_ITEMS JOIN warranty_vendors ON vMTL_SYSTEM_ITEMS.WARRANTY_VENDOR_ID = warranty_vendors.VENDOR_ID"},
                
                {"question": "Get all items, their descriptions, and their corresponding lead times from the inventory", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, vMTL_SYSTEM_ITEMS.FULL_LEAD_TIME FROM vMTL_SYSTEM_ITEMS"},
                
                {"question": "Show all items, their associated tax codes, and their corresponding sales account", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, vMTL_SYSTEM_ITEMS.TAX_CODE, vMTL_SYSTEM_ITEMS.SALES_ACCOUNT FROM vMTL_SYSTEM_ITEMS"},
                
                {"question": "List all items and their total order quantities per organization", 
                "sql": "SELECT ORGANIZATION_ID, PART_NUMBER, SUM(ORDER_QUANTITY) AS TOTAL_ORDER_QUANTITY FROM vMTL_SYSTEM_ITEMS JOIN orders ON vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID = orders.INVENTORY_ITEM_ID GROUP BY ORGANIZATION_ID, PART_NUMBER"},
                
                {"question": "Show all items with their current inventory status and orderable status", 
                "sql": "SELECT PART_NUMBER, INVENTORY_ITEM_STATUS_CODE, ORDERABLE_ON_WEB_FLAG FROM vMTL_SYSTEM_ITEMS WHERE ORDERABLE_ON_WEB_FLAG = 'Y'"},
                
                {"question": "Get the items with the highest and lowest prices, along with their descriptions", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, LIST_PRICE_PER_UNIT FROM vMTL_SYSTEM_ITEMS WHERE LIST_PRICE_PER_UNIT = (SELECT MAX(LIST_PRICE_PER_UNIT) FROM vMTL_SYSTEM_ITEMS) OR LIST_PRICE_PER_UNIT = (SELECT MIN(LIST_PRICE_PER_UNIT) FROM vMTL_SYSTEM_ITEMS)"},
                
                {"question": "Find all items from a specific buyer with their current inventory levels and supplier names", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, inventory_levels.STOCK_LEVEL, suppliers.SUPPLIER_NAME FROM vMTL_SYSTEM_ITEMS JOIN inventory_levels ON vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID = inventory_levels.INVENTORY_ITEM_ID JOIN suppliers ON vMTL_SYSTEM_ITEMS.SUPPLIER = suppliers.SUPPLIER_ID WHERE vMTL_SYSTEM_ITEMS.BUYER_NAME = 'John Doe'"},
                
                {"question": "Get all items from a certain organization that are currently backordered", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, BACK_ORDERABLE_FLAG FROM vMTL_SYSTEM_ITEMS WHERE ORGANIZATION_ID = 'OrgA' AND BACK_ORDERABLE_FLAG = 'Y'"},
                
                {"question": "Show all items with their associated fixed order quantity and maximum order quantity", 
                "sql": "SELECT PART_NUMBER, FIXED_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY FROM vMTL_SYSTEM_ITEMS WHERE FIXED_ORDER_QUANTITY IS NOT NULL AND MAXIMUM_ORDER_QUANTITY IS NOT NULL"},
                
                {"question": "Get the list of items that are marked for environmental compliance and their safety stock bucket days", 
                "sql": "SELECT PART_NUMBER, ENVIRONMENTAL_COMPLIANCE_STATUS, SAFETY_STOCK_BUCKET_DAYS FROM vMTL_SYSTEM_ITEMS WHERE ENVIRONMENTAL_COMPLIANCE_STATUS = 'Compliant'"},
                
                {"question": "Find all items and their corresponding inventory planning codes and descriptions", 
                "sql": "SELECT PART_NUMBER, INVENTORY_PLANNING_CODE, INVENTORY_PLANNING_CODE_NAME FROM vMTL_SYSTEM_ITEMS"},
                
                {"question": "Show the total count of items ordered by each vendor along with the total cost of orders", 
                "sql": "SELECT suppliers.SUPPLIER_NAME, COUNT(vMTL_SYSTEM_ITEMS.PART_NUMBER) AS ITEM_COUNT, SUM(vMTL_SYSTEM_ITEMS.LIST_PRICE_PER_UNIT) AS TOTAL_COST FROM vMTL_SYSTEM_ITEMS JOIN suppliers ON vMTL_SYSTEM_ITEMS.SUPPLIER = suppliers.SUPPLIER_ID GROUP BY suppliers.SUPPLIER_NAME"},
                
                {"question": "Get all items where the order quantity exceeds a specific threshold and show their planning time fence", 
                "sql": "SELECT PART_NUMBER, ORDER_QUANTITY, PLANNING_TIME_FENCE_DAYS FROM vMTL_SYSTEM_ITEMS WHERE ORDER_QUANTITY > 100 AND PLANNING_TIME_FENCE_DAYS > 30"},
                
                {"question": "Find items that have been ordered with a price higher than their defined maximum price", 
                "sql": "SELECT PART_NUMBER, ORDER_QUANTITY, LIST_PRICE_PER_UNIT FROM vMTL_SYSTEM_ITEMS WHERE ORDER_QUANTITY > 0 AND LIST_PRICE_PER_UNIT > MAXIMUM_ORDER_QUANTITY"},
                
                {"question": "Get all items from a particular catalog and show their buyer's name, price, and planning make-buy code", 
                "sql": "SELECT vMTL_SYSTEM_ITEMS.PART_NUMBER, vMTL_SYSTEM_ITEMS.DESCRIPTION, vMTL_SYSTEM_ITEMS.BUYER_NAME, vMTL_SYSTEM_ITEMS.LIST_PRICE_PER_UNIT, vMTL_SYSTEM_ITEMS.PLANNING_MAKE_BUY_CODE_NAME FROM vMTL_SYSTEM_ITEMS JOIN item_catalog ON vMTL_SYSTEM_ITEMS.ITEM_CATALOG_GROUP_ID = item_catalog.CATALOG_GROUP_ID WHERE item_catalog.CATALOG_NAME = 'CatalogA'"},
                
                {"question": "List all items that have a warranty expiration date and their warranty vendor details", 
                "sql": "SELECT PART_NUMBER, WARRANTY_VENDOR_ID, WARRANTY_EXPIRATION_DATE FROM vMTL_SYSTEM_ITEMS JOIN warranty_vendors ON vMTL_SYSTEM_ITEMS.WARRANTY_VENDOR_ID = warranty_vendors.VENDOR_ID WHERE WARRANTY_EXPIRATION_DATE > '2025-01-01'"},
                
                {"question": "Get all items with their planning time fence, orderable flag, and backorderable flag", 
                "sql": "SELECT PART_NUMBER, PLANNING_TIME_FENCE_DAYS, ORDERABLE_ON_WEB_FLAG, BACK_ORDERABLE_FLAG FROM vMTL_SYSTEM_ITEMS WHERE ORDERABLE_ON_WEB_FLAG = 'Y' AND BACK_ORDERABLE_FLAG = 'Y'"},
                
                {"question": "Show the most expensive item from each organization and its supplier", 
                "sql": "SELECT ORGANIZATION_ID, PART_NUMBER, SUPPLIER, MAX(LIST_PRICE_PER_UNIT) AS MAX_PRICE FROM vMTL_SYSTEM_ITEMS GROUP BY ORGANIZATION_ID, SUPPLIER"},
                
                {"question": "Get the items that are flagged for inspection, along with their hazard class description", 
                "sql": "SELECT PART_NUMBER, DESCRIPTION, HAZARD_CLASS_DESC FROM vMTL_SYSTEM_ITEMS WHERE INSPECTION_REQUIRED_FLAG = 'Y'"},
                
                {"question": "List all items, their corresponding cost of sales account, and the total quantity ordered", 
                "sql": "SELECT PART_NUMBER, COST_OF_SALES_ACCOUNT, SUM(ORDER_QUANTITY) AS TOTAL_ORDERED FROM vMTL_SYSTEM_ITEMS JOIN orders ON vMTL_SYSTEM_ITEMS.INVENTORY_ITEM_ID = orders.INVENTORY_ITEM_ID GROUP BY PART_NUMBER, COST_OF_SALES_ACCOUNT"},
                
            
                {
                    "question": "Which parts were updated in the last 6 months?",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE >= ADD_MONTHS(SYSDATE, -6)"
                },
                {
                    "question": "Get all items along with how many days since they were last updated",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE, TRUNC(SYSDATE - LAST_UPDATE_DATE) AS DAYS_SINCE_LAST_UPDATE FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Show parts that were created more than 5 years ago",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE < ADD_MONTHS(SYSDATE, -60)"
                },
                {
                    "question": "Get the number of parts created each year",
                    "sql": "SELECT EXTRACT(YEAR FROM CREATION_DATE) AS YEAR, COUNT(*) AS TOTAL_PARTS FROM vMTL_SYSTEM_ITEMS GROUP BY EXTRACT(YEAR FROM CREATION_DATE) ORDER BY YEAR"
                },
                {
                    "question": "Find all parts that were created in the same month as today",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE EXTRACT(MONTH FROM CREATION_DATE) = EXTRACT(MONTH FROM SYSDATE)"
                },
                {
                    "question": "List all suppliers and the most recent part they supplied",
                    "sql": "SELECT SUPPLIER, MAX(CREATION_DATE) AS MOST_RECENT_PART_DATE FROM vMTL_SYSTEM_ITEMS GROUP BY SUPPLIER"
                },
                
                {
                    "question": "Show all parts created in the last 30 days",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= TRUNC(SYSDATE - 30)"
                },
                {
                    "question": "Show all parts created in 2024",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "List part numbers added in 2023",
                    "sql": "SELECT PART_NUMBER FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2023-01-01' AND CREATION_DATE <= '2023-12-31'"
                },
                {
                    "question": "Get items last updated in 2022",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE >= '2022-01-01' AND LAST_UPDATE_DATE <= '2022-12-31'"
                },
                {
                    "question": "Find all items created in December 2024",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-12-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "Which parts were updated in Q1 2023?",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE >= '2023-01-01' AND LAST_UPDATE_DATE <= '2023-03-31'"
                },
                {
                    "question": "Get all parts ordered in 2021",
                    "sql": "SELECT PART_NUMBER, ATTRIBUTE11 FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2021-01-01' AND CREATION_DATE <= '2021-12-31'"
                },
                {
                    "question": "List parts modified in 2020",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE >= '2020-01-01' AND LAST_UPDATE_DATE <= '2020-12-31'"
                },
                {
                    "question": "Show everything created in the year 2019",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2019-01-01' AND CREATION_DATE <= '2019-12-31'"
                },
                {
                    "question": "How many parts were added in 2024?",
                    "sql": "SELECT COUNT(*) AS TOTAL_PARTS_2024 FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "Get all parts added in the same year as part 12345",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE EXTRACT(YEAR FROM CREATION_DATE) = (SELECT EXTRACT(YEAR FROM CREATION_DATE) FROM vMTL_SYSTEM_ITEMS WHERE PART_NUMBER = '12345')"
                },
                {
                    "question": "Give me all part numbers created by ER code XYZ789",
                    "sql": "SELECT PART_NUMBER FROM vMTL_SYSTEM_ITEMS WHERE ORGANIZATION_CODE = 'XYZ789'"
                },
                {
                    "question": "Hey, can you show me a list of all the parts we have?",
                    "sql": "SELECT ORGANIZATION_ID, ORGANIZATION_CODE, INVENTORY_ITEM_ID, PART_NUMBER, DESCRIPTION, ITEM_TYPE, ITEM_TYPE_NAME, INVENTORY_ITEM_STATUS_CODE, PLANNER_CODE, PLANNER_NAME, BUYER_NAME, PLANNING_MAKE_BUY_CODE_NAME FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "I need to see just the part numbers, nothing else.",
                    "sql": "SELECT PART_NUMBER FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "I'm looking for any parts that have XYZ789 as their ER code.",
                    "sql": "SELECT PART_NUMBER, ORGANIZATION_ID, INVENTORY_ITEM_ID, DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE ORGANIZATION_CODE = 'XYZ789'"
                },
                {
                    "question": "Get all items with their planning time fence, orderable flag, and backorderable flag",
                    "sql": "SELECT PART_NUMBER, PLANNING_TIME_FENCE_CODE, ORDERABLE_ON_WEB_FLAG, BACK_ORDERABLE_FLAG FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Get all items where the cost of sales account matches the expense account",
                    "sql": "SELECT PART_NUMBER, COST_OF_SALES_ACCOUNT, EXPENSE_ACCOUNT FROM vMTL_SYSTEM_ITEMS WHERE COST_OF_SALES_ACCOUNT = EXPENSE_ACCOUNT"
                },
                {
                    "question": "List all items that have been recently updated and their corresponding inventory status",
                    "sql": "SELECT PART_NUMBER, INVENTORY_ITEM_STATUS_CODE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS ORDER BY LAST_UPDATE_DATE DESC"
                },
                {
                    "question": "Get items with their corresponding tax code and description",
                    "sql": "SELECT PART_NUMBER, TAX_CODE, DESCRIPTION FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Get all items from a certain organization that are currently backordered",
                    "sql": "SELECT PART_NUMBER, ORGANIZATION_ID, BACK_ORDERABLE_FLAG FROM vMTL_SYSTEM_ITEMS WHERE BACK_ORDERABLE_FLAG = 'Y'"
                },
                {
                    "question": "Get the items with the highest and lowest prices, along with their descriptions",
                    "sql": "SELECT PART_NUMBER, DESCRIPTION, ACTUAL_PRICE FROM vMTL_SYSTEM_ITEMS WHERE ACTUAL_PRICE = (SELECT MAX(ACTUAL_PRICE) FROM vMTL_SYSTEM_ITEMS) OR ACTUAL_PRICE = (SELECT MIN(ACTUAL_PRICE) FROM vMTL_SYSTEM_ITEMS)"
                },
                {
                    "question": "Which items have not been updated since 2018?",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE < TO_DATE('2019-01-01', 'YYYY-MM-DD')"
                },
                {
                    "question": "Get the items that are flagged for inspection, along with their hazard class description",
                    "sql": "SELECT PART_NUMBER, INSPECTION_REQUIRED_FLAG, HAZARD_CLASS_DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE INSPECTION_REQUIRED_FLAG = 'Y'"
                },
                {
                    "question": "Get all items, their descriptions, and their corresponding lead times from the inventory",
                    "sql": "SELECT PART_NUMBER, DESCRIPTION, CUM_MANUFACTURING_LEAD_TIME FROM vMTL_SYSTEM_ITEMS"
                },
                 {
                    "question": "Get all items created after January 1, 2020",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE > '2020-01-01'"
                },
                {
                    "question": "List all items added in the year 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE EXTRACT(YEAR FROM CREATION_DATE) = 2021"
                },
                {
                    "question": "Show me all items created before March 15, 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE < '2022-03-15'"
                },
                {
                    "question": "Find all items created between January 1, 2021 and December 31, 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE BETWEEN '2021-01-01' AND '2021-12-31'"
                },
                {
                    "question": "Get all items that were created in February 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE EXTRACT(MONTH FROM CREATION_DATE) = 2 AND EXTRACT(YEAR FROM CREATION_DATE) = 2022"
                },
                {
                    "question": "List all items created on or after February 1, 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2021-02-01'"
                },
                {
                    "question": "Show all items created in the last 30 days",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= CURRENT_DATE - INTERVAL '30 days'"
                },
                {
                    "question": "Get all items created in the first quarter of 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2022-01-01' AND CREATION_DATE < '2022-04-01'"
                },
                {
                    "question": "Find all items created on February 29, 2020",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE = '2020-02-29'"
                },
                {
                    "question": "List all items created in the summer of 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2021-06-01' AND CREATION_DATE < '2021-09-01'"
                },
                {
                    "question": "Get all items created in the last year",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= CURRENT_DATE - INTERVAL '1 year'"
                },
                {
                    "question": "Show me items created on the last day of 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE = '2021-12-31'"
                },
                {
                    "question": "Find all items created in the month of January 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE EXTRACT(MONTH FROM CREATION_DATE) = 1 AND EXTRACT(YEAR FROM CREATION_DATE) = 2022"
                },
                {
                    "question": "Get all items created after the last quarter of 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE > '2021-12-31'"
                },
                {
                    "question":"​Please help to pull all the PTO model for ER R1 for all statuses. Thanks.",
                    "sql": "select ER_code,PART_NUMBER, ITEM_TYPE_NAME,INVENTORY_ITEM_STATUS_CODE from vMTL_SYSTEM_ITEMS where ER_CODE='R1' and ITEM_TYPE_NAME = 'PTO Model'"
                },
                {
                    "question": "Find all items where the make or buy code is set to Make",
                    "sql": "SELECT PART_NUMBER, PLANNING_MAKE_BUY_CODE_NAME FROM vMTL_SYSTEM_ITEMS WHERE PLANNING_MAKE_BUY_CODE_NAME = 'Make'"
                },
                {
                    "question": "Show me all items that have no supplier listed",
                    "sql": "SELECT PART_NUMBER, SUPPLIER FROM vMTL_SYSTEM_ITEMS WHERE SUPPLIER IS NULL"
                },
                {
                    "question": "Which items have a description that contains the word 'bolt'?",
                    "sql": "SELECT PART_NUMBER, DESCRIPTION FROM vMTL_SYSTEM_ITEMS WHERE LOWER(DESCRIPTION) LIKE '%bolt%'"
                },
                {
                    "question": "List the top 10 most recently created items",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS ORDER BY CREATION_DATE DESC FETCH FIRST 10 ROWS ONLY"
                },
                {
                    "question": "Get all items where the category is Mechanical",
                    "sql": "SELECT PART_NUMBER, MAIN_CATEGORY FROM vMTL_SYSTEM_ITEMS WHERE MAIN_CATEGORY = 'Mechanical'"
                },
                {
                    "question": "Find items with a lead time greater than 30 days",
                    "sql": "SELECT PART_NUMBER, CUM_MANUFACTURING_LEAD_TIME FROM vMTL_SYSTEM_ITEMS WHERE CUM_MANUFACTURING_LEAD_TIME > 30"
                },
                {
                    "question": "Show me all active items from organization 202",
                    "sql": "SELECT PART_NUMBER, ORGANIZATION_ID, INVENTORY_ITEM_STATUS_CODE FROM vMTL_SYSTEM_ITEMS WHERE ORGANIZATION_ID = 202 AND INVENTORY_ITEM_STATUS_CODE = 'Active'"
                },
                {
                    "question": "Find all items that are returnable and trackable",
                    "sql": "SELECT PART_NUMBER, RETURNABLE_FLAG, TRACKING_QUANTITY_IND FROM vMTL_SYSTEM_ITEMS WHERE RETURNABLE_FLAG = 'Y' AND TRACKING_QUANTITY_IND = 'Y'"
                },
                {
                    "question": "List all unique suppliers",
                    "sql": "SELECT DISTINCT SUPPLIER FROM vMTL_SYSTEM_ITEMS WHERE SUPPLIER IS NOT NULL"
                },
            
                {
                    "question": "Find items where the inventory item type is 'Finished Good'",
                    "sql": "SELECT PART_NUMBER, ITEM_TYPE_NAME FROM vMTL_SYSTEM_ITEMS WHERE ITEM_TYPE_NAME = 'Finished Good'"
                },
                {
                    "question": "Show items where the serial generation flag is not enabled",
                    "sql": "SELECT PART_NUMBER, SERIAL_NUMBER_GENERATION_FLAG FROM vMTL_SYSTEM_ITEMS WHERE SERIAL_NUMBER_GENERATION_FLAG = 'N'"
                },
                {
                    "question": "List the items with a planner name starting with 'Ali'",
                    "sql": "SELECT PART_NUMBER, PLANNER_NAME FROM vMTL_SYSTEM_ITEMS WHERE PLANNER_NAME LIKE 'Ali%'"
                },
                {
                    "question": "Show me parts created before 2020 but updated after 2023",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE < TO_DATE('2020-01-01', 'YYYY-MM-DD') AND LAST_UPDATE_DATE > TO_DATE('2023-01-01', 'YYYY-MM-DD')"
                },
                {
                    "question": "Give me a count of all parts per organization",
                    "sql": "SELECT ORGANIZATION_ID, COUNT(*) AS PART_COUNT FROM vMTL_SYSTEM_ITEMS GROUP BY ORGANIZATION_ID"
                },
                {
                    "question": "Get all items with both tax code and list price available",
                    "sql": "SELECT PART_NUMBER, TAX_CODE, LIST_PRICE FROM vMTL_SYSTEM_ITEMS WHERE TAX_CODE IS NOT NULL AND LIST_PRICE IS NOT NULL"
                },
                {
                    "question": "Find items where the warranty classification is 'Premium'",
                    "sql": "SELECT PART_NUMBER, WARRANTY_CLASSIFICATION FROM vMTL_SYSTEM_ITEMS WHERE WARRANTY_CLASSIFICATION = 'Premium'"
                },
                {
                    "question": "List all parts that are customer order enabled but not internal order enabled",
                    "sql": "SELECT PART_NUMBER, CUSTOMER_ORDER_ENABLED_FLAG, INTERNAL_ORDER_ENABLED_FLAG FROM vMTL_SYSTEM_ITEMS WHERE CUSTOMER_ORDER_ENABLED_FLAG = 'Y' AND INTERNAL_ORDER_ENABLED_FLAG = 'N'"
                },
                {
                    "question": "Find all items that have been deleted or disabled",
                    "sql": "SELECT PART_NUMBER, INVENTORY_ITEM_STATUS_CODE FROM vMTL_SYSTEM_ITEMS WHERE INVENTORY_ITEM_STATUS_CODE IN ('Disabled', 'Deleted')"
                },
                {
                    "question": "Show me all distinct item types and their names",
                    "sql": "SELECT DISTINCT ITEM_TYPE, ITEM_TYPE_NAME FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Get all items created on or after January 1st, 2024",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01'"
                },
                {
                    "question": "Find items that were updated before June 15, 2023",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE < '2023-06-15'"
                },
                {
                    "question": "Show parts created between February 1st, 2024 and March 1st, 2024",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-02-01' AND CREATION_DATE <= '2024-03-01'"
                },
                {
                    "question": "List all items updated after December 31, 2023",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE > '2023-12-31'"
                },
                {
                    "question": "Which items were created before 2022?",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE < '2022-01-01'"
                },
                {
                    "question": "Find items that were both created and last updated in 2024",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31' AND LAST_UPDATE_DATE >= '2024-01-01' AND LAST_UPDATE_DATE <= '2024-12-31'"
                },
                {
                    "question": "Get parts created exactly on March 15, 2024",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE = '2024-03-15'"
                },
                {
                    "question": "Show me parts that were created after January 1, 2023 and not updated after January 1, 2024",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE > '2023-01-01' AND LAST_UPDATE_DATE <= '2024-01-01'"
                },
                {
                    "question": "List all items that were created and updated on the same day",
                    "sql": "SELECT PART_NUMBER, CREATION_DATE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE = LAST_UPDATE_DATE"
                },
                {
                    "question": "Find items that have not been updated since 2022",
                    "sql": "SELECT PART_NUMBER, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE < '2023-01-01'"
                },
                 {
                    "question": "Show me all items created before March 15, 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE < '2022-03-15'"
                },
                {
                    "question": "Get the items with the highest and lowest prices, along with their descriptions",
                    "sql": "SELECT PART_NUMBER, DESCRIPTION, ACTUAL_PRICE FROM vMTL_SYSTEM_ITEMS ORDER BY ACTUAL_PRICE DESC FETCH FIRST 1 ROWS ONLY UNION ALL SELECT PART_NUMBER, DESCRIPTION, ACTUAL_PRICE FROM vMTL_SYSTEM_ITEMS ORDER BY ACTUAL_PRICE ASC FETCH FIRST 1 ROWS ONLY"
                },
                {
                    "question": "Show the total number of items ordered by each buyer along with their total cost",
                    "sql": "SELECT BUYER_NAME, COUNT(*) AS TOTAL_ITEMS, SUM(ACTUAL_PRICE) AS TOTAL_COST FROM vMTL_SYSTEM_ITEMS GROUP BY BUYER_NAME"
                },
                {
                    "question": "Get all items from a particular catalog and show their buyer's name, price, and planning make-buy code",
                    "sql": "SELECT BUYER_NAME, ACTUAL_PRICE, PLANNING_MAKE_BUY_CODE_NAME FROM vMTL_SYSTEM_ITEMS WHERE CATALOG_GROUP_ID = :catalog_id"
                },
                {
                    "question": "Get all items created in the last year",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "List all items updated after December 31, 2023",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE > '2023-12-31'"
                },
                {
                    "question": "Get all items where the cost of sales account matches the expense account",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE COST_OF_SALES_ACCOUNT = EXPENSE_ACCOUNT"
                },
                {
                    "question": "List all items created in the summer of 2021",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2021-06-01' AND CREATION_DATE <= '2021-08-31'"
                },
                {
                    "question": "Show me all distinct item types and their names",
                    "sql": "SELECT DISTINCT ITEM_TYPE, ITEM_TYPE_NAME FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Get all items where the category is Mechanical",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CATEGORY_NAME = 'Mechanical'"
                },
                {
                    "question": "Get all items created on or after January 1st, 2024",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE > '2024-01-01'"
                },
                {
                    "question": "Get all descriptions of items",
                    "sql": "SELECT DESCRIPTION FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "Find items that were both created and last updated in 2024",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "Show all parts created in 2024",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2024-01-01' AND CREATION_DATE <= '2024-12-31'"
                },
                {
                    "question": "Find all parts that were created in the same month as today",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2025-04-01' AND CREATION_DATE < '2025-05-01'"
                },
                  {
                    "question": "List all items with a price greater than 100",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE LIST_PRICE > 100"
                },
                {
                    "question": "Get the items with the highest inventory levels by warehouse location",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS ORDER BY INVENTORY_QUANTITY DESC"
                },
                {
                    "question": "Find all parts that were created in the same month as today",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2025-04-01' AND CREATION_DATE < '2025-05-01'"
                },
                {
                    "question": "Find all items created in the month of January 2022",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '2022-01-01' AND CREATION_DATE < '2022-02-01'"
                },
                {
                    "question": "List all items that have been recently updated and their corresponding inventory status",
                    "sql": "SELECT PART_NUMBER, INVENTORY_ITEM_STATUS_CODE, LAST_UPDATE_DATE FROM vMTL_SYSTEM_ITEMS WHERE LAST_UPDATE_DATE >= '2024-12-01'"
                },
                {
                    "question": "Find all items with fixed lead time",
                    "sql": "SELECT * FROM vMTL_SYSTEM_ITEMS WHERE FIXED_LEAD_TIME_FLAG = 'Y'"
                },
                {
                    "question": "Get all items with their planning time fence, orderable flag, and backorderable flag",
                    "sql": "SELECT PART_NUMBER, PLANNING_TIME_FENCE_CODE, ORDERABLE_FLAG, BACKORDERABLE_FLAG FROM vMTL_SYSTEM_ITEMS"
                },
                {
                    "question": "List all unique suppliers",
                    "sql": "SELECT DISTINCT SUPPLIER FROM vMTL_SYSTEM_ITEMS WHERE SUPPLIER IS NOT NULL"
                },
                {
                    "question": "List all items, their corresponding cost of sales account, and the total quantity ordered",
                    "sql": "SELECT PART_NUMBER, COST_OF_SALES_ACCOUNT, SUM(QUANTITY_ORDERED) AS TOTAL_ORDERED FROM vMTL_SYSTEM_ITEMS GROUP BY PART_NUMBER, COST_OF_SALES_ACCOUNT"
                }
                ]

        # Load feedback data and append to the existing data
        feedback_data = self.feedback_data.get("feedback", [])
        for feedback in feedback_data:
            if "question" in feedback and "corrected_sql" in feedback:
                data.append({
                    "question": feedback["question"],
                    "sql": feedback["corrected_sql"]
                })

        # Convert to Dataset
        dataset = Dataset.from_list(data)
        
        # Split into train and test
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        
        return train_test
    
    def initialize_model(self):
        """Initialize the tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def preprocess_data(self, examples):
        """Tokenize and prepare the data for training"""
        # Use a clearer instruction prompt with schema information
        schema_info = """
        Tables:
        - parts (part_number, description, price, weight, er_code, supplier_id, engineer_id, creation_date, modification_date, last_order_date, category, status)
        - inventory (part_number, quantity, location, update_date, last_count_date)
        - suppliers (id, supplier_name, contact, contract_date)
        - engineers (id, name, department, hire_date)
        """
        
        inputs = [f"{schema_info}\nConvert this question to SQL: {question}" for question in examples["question"]]
        targets = examples["sql"]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding="max_length"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding token id's of the labels by -100 so they're ignored in the loss
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in model_inputs["labels"]
        ]
        
        return model_inputs
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics"""
        metric = evaluate.load("sacrebleu")
        
        preds, labels = eval_preds
        
        # Replace -100 with the pad_token_id in the labels before decoding
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Make sure there are no negative values in preds as well
        preds = np.where(preds >= 0, preds, self.tokenizer.pad_token_id)
        
        # Decode predictions and references
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # BLEU expects a list of references for each prediction
        formatted_refs = [[ref] for ref in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=formatted_refs)
        
        # Add exact match score
        exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip())
        result["exact_match"] = exact_matches / len(decoded_preds)
        
        return result
    
    def train(self, dataset, output_dir="./results", num_epochs=20, batch_size=8, learning_rate=2e-5):
        """Train the model with improved parameters"""
        # Tokenize datasets
        tokenized_datasets = dataset.map(
            self.preprocess_data, 
            batched=True, 
            remove_columns=dataset["train"].column_names
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments with better parameters
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",  # Set evaluation strategy to epoch
            save_strategy="epoch",  # Set save strategy to epoch to match eval strategy
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            logging_steps=10,
            generation_max_length=self.max_target_length,
            generation_num_beams=5,
            load_best_model_at_end=True,  # Load the best model at the end of training
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False  # Lower loss is better
        )
        
        # Initialize trainer with early stopping
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8, early_stopping_threshold=0.01)]  # Increased patience and set a threshold
        )
        
        # Train model
        trainer.train()
        
        # Save model
        self.model.save_pretrained(os.path.join(output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        return trainer
    
    def _load_feedback_data(self):
        """Load existing feedback data or create new file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse {self.feedback_file}, creating new feedback file")
                return {"examples": [], "corrections": [], "metadata": {"last_updated": None, "last_retrained": None}}
        else:
            return {"examples": [], "corrections": [], "metadata": {"last_updated": None, "last_retrained": None}}
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        self.feedback_data["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
        self.logger.info(f"Feedback data saved to {self.feedback_file}")
    
    def add_correction(self, question, generated_sql, corrected_sql, user_id=None):
        """Add a user correction to the feedback data"""
        correction = {
            "question": question,
            "generated_sql": generated_sql,
            "corrected_sql": corrected_sql,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id
        }
        
        self.feedback_data["corrections"].append(correction)
        self._save_feedback_data()
        
        # Update performance metrics
        self.performance_metrics["user_corrected"] += 1
        
        self.logger.info(f"Added correction for question: {question}")
        
        # Check if we should retrain
        if len(self.feedback_data["corrections"]) % 10 == 0:  # Every 10 corrections
            self.logger.info("Reached 10 new corrections, consider retraining the model")
            return True
        return False
    
    def add_example(self, question, sql, source="user", user_id=None):
        """Add a new example to the training data"""
        example = {
            "question": question,
            "sql": sql,
            "source": source,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id
        }
        
        self.feedback_data["examples"].append(example)
        self._save_feedback_data()
        
        self.logger.info(f"Added new example: {question} -> {sql}")
        return True
    
    def prepare_retraining_data(self):
        """Prepare data for retraining from feedback"""
        # Convert corrections to training examples
        for correction in self.feedback_data["corrections"]:
            # Only add if this correction hasn't been converted to an example yet
            if not any(ex["question"] == correction["question"] and 
                      ex["source"] == "correction" for ex in self.feedback_data["examples"]):
                self.add_example(
                    question=correction["question"],
                    sql=correction["corrected_sql"],
                    source="correction"
                )
        
        # Prepare dataset from all examples
        examples = [{"question": ex["question"], "sql": ex["sql"]} 
                   for ex in self.feedback_data["examples"]]
        
        # Get ALL original training data (both train and test splits)
        original_data = self.load_data(data_list=None)
        original_examples = []
        
        # Include both train and test examples from original data
        for split in ["train", "test"]:
            for item in original_data[split]:
                original_examples.append({"question": item["question"], "sql": item["sql"]})
        
        # Combine datasets with priority to user corrections
        combined_examples = []
        seen_questions = set()
        
        # First add user examples (they take precedence)
        for ex in examples:
            combined_examples.append(ex)
            seen_questions.add(ex["question"])
        
        # Then add original examples that don't conflict
        for ex in original_examples:
            if ex["question"] not in seen_questions:
                combined_examples.append(ex)
        
        # Create dataset
        dataset = Dataset.from_list(combined_examples)
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        
        return train_test
    
    def retrain_from_feedback(self, output_dir="./text_to_sql_results"):
        """Retrain the model using feedback data"""
        if not self.feedback_data["corrections"] and not self.feedback_data["examples"]:
            self.logger.warning("No feedback data available for retraining")
            return False
        
        self.logger.info("Starting model retraining with feedback data")
        
        # Prepare data
        datasets = self.prepare_retraining_data()
        
        # Initialize model if needed
        if self.model is None or self.tokenizer is None:
            self.initialize_model()
        
        # Train model with fewer epochs since we're fine-tuning
        trainer = self.train(datasets, output_dir=output_dir)
        
        # After successful training, update the metadata
        self.feedback_data["metadata"]["last_retrained"] = datetime.datetime.now().isoformat()
        self._save_feedback_data()
        
        # Save model
        self.save_model(os.path.join(output_dir, "final_model"))
        
        self.logger.info(f"Model retrained and saved to {output_dir}/final_model")
        return True
    
    def analyze_performance(self):
        """Analyze model performance based on feedback"""
        if self.performance_metrics["total_queries"] > 0:
            self.performance_metrics["success_rate"] = 1.0 - (
                self.performance_metrics["user_corrected"] / self.performance_metrics["total_queries"]
            )
        
        # Identify common error patterns
        error_patterns = {}
        for correction in self.feedback_data["corrections"]:
            # Simple error categorization
            if "WHERE" not in correction["generated_sql"] and "WHERE" in correction["corrected_sql"]:
                error_patterns.setdefault("missing_where_clause", 0)
                error_patterns["missing_where_clause"] += 1
            elif "=" in correction["corrected_sql"] and "=" not in correction["generated_sql"]:
                error_patterns.setdefault("missing_equality_operator", 0)
                error_patterns["missing_equality_operator"] += 1
            elif "'" in correction["corrected_sql"] and "'" not in correction["generated_sql"]:
                error_patterns.setdefault("missing_quotes", 0)
                error_patterns["missing_quotes"] += 1
        
        return {
            "metrics": self.performance_metrics,
            "error_patterns": error_patterns,
            "feedback_stats": {
                "total_corrections": len(self.feedback_data["corrections"]),
                "total_examples": len(self.feedback_data["examples"]),
                "last_retrained": self.feedback_data["metadata"]["last_retrained"]
            }
        }
    
    def _load_schema(self, schema_file):
        """Load database schema from configuration file"""
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                return json.load(f)
        else:
            # Default schema if file doesn't exist
            return {
                "tables": {
                    "vMTL_SYSTEM_ITEMS": {
                        "columns": [
                            "ORGANIZATION_ID", "ORGANIZATION_CODE", "INVENTORY_ITEM_ID", "PART_NUMBER", "DESCRIPTION","ITEM_TYPE", "ITEM_TYPE_NAME", "INVENTORY_ITEM_STATUS_CODE", "PLANNER_CODE", "PLANNER_NAME",
                "BUYER_NAME", "PLANNING_MAKE_BUY_CODE_NAME", "ACCOUNTING_RULE_ID", "ACCOUNTING_RULE_NAME",
                "ALLOW_ITEM_DESC_UPDATE_FLAG", "ALLOWED_UNITS_LOOKUP_CODE", "ALLOWED_UNITS_LOOKUP_CODE_NAME",
               "ATO_FORECAST_CONTROL", "ATO_FORECAST_CONTROL_NAME", "ATP_COMPONENTS_FLAG", "ATP_COMPONENTS_FLAG_NAME",
                "ATP_FLAG", "ATP_FLAG_NAME", "ATP_RULE_DESCRIPTION", "ATP_RULE_ID", "ATP_RULE_NAME",
               "ATTRIBUTE_CATEGORY", "ATTRIBUTE11", "AUTO_CREATED_CONFIG_FLAG", "AUTO_LOT_ALPHA_PREFIX",
                "AUTO_REDUCE_MPS", "AUTO_REDUCE_MPS_NAME", "AUTO_SERIAL_ALPHA_PREFIX", "BACK_ORDERABLE_FLAG",
              "BASE_ITEM_ID", "BASE_PART_NUMBER", "BASE_WARRANTY_SERVICE_ID", "BOM_ENABLED_FLAG",
              "BOM_ITEM_TYPE", "BOM_ITEM_TYPE_NAME", "BUILD_IN_WIP_FLAG", "BULK_PICKED_FLAG", "BUYER_ID",
             "CATALOG_STATUS_FLAG", "CHECK_SHORTAGES_FLAG", "COMMS_ACTIVATION_REQD_FLAG", "COMMS_NL_TRACKABLE_FLAG",
               "COMPLIANCE_REQUIRED", "CONTAINER_ITEM_FLAG", "CONTAINER_TYPE_CODE", "COST_OF_SALES_ACCOUNT",
                "COST_OF_SALES_ACCOUNT_FLEX", "COSTING_ENABLED_FLAG", "COUPON_EXEMPT_FLAG", "COVERAGE_SCHEDULE_ID",
               "CREATION_DATE", "CUM_MANUFACTURING_LEAD_TIME", "CUMULATIVE_TOTAL_LEAD_TIME", "CUSTOMER_ORDER_ENABLED_FLAG",
              "CUSTOMER_ORDER_FLAG", "CYCLE_COUNT_ENABLED_FLAG", "DEFAULT_INCLUDE_IN_ROLLUP_FLAG", "DEFAULT_LOT_STATUS_ID",
              "DEFAULT_SERIAL_STATUS_ID", "DEFAULT_SHIPPING_ORG", "DEFAULT_SHIPPING_ORG_CODE", "DEFAULT_SHIPPING_ORG_CODE_DESC"
                        ],
                        "primary_key": "ORGANIZATION_ID"
                    }
                },
                "relationships": [
                    {"from": "vMTL_SYSTEM_ITEMS.SUPPLIER", "to": "suppliers.SUPPLIER_ID"},
                    {"from": "vMTL_SYSTEM_ITEMS.BUYER_NAME", "to": "engineers.ENGINEER_ID"},
                    {"from": "inventory.PART_NUMBER", "to": "vMTL_SYSTEM_ITEMS.PART_NUMBER"}
                ],
                "patterns": {
                    "date_columns": [
                        "vMTL_SYSTEM_ITEMS.CREATION_DATE",
                        "vMTL_SYSTEM_ITEMS.LAST_UPDATE_DATE"
                    ],
                    "category_columns": [
                        "vMTL_SYSTEM_ITEMS.ITEM_TYPE",
                        "vMTL_SYSTEM_ITEMS.ITEM_TYPE_NAME"
                    ]
                }
            }
    
    def _get_schema_info(self):
        """Generate schema information for the prompt"""
        schema_info = "Tables:\n"
        for table_name, table_info in self.schema["tables"].items():
            if isinstance(table_info, dict) and "columns" in table_info:
                columns = ", ".join(table_info["columns"])
                schema_info += f"- {table_name} ({columns})\n"
        return schema_info
    
    def generate_sql(self, question):
        """Generate SQL from a natural language question"""
        # Update metrics
        self.performance_metrics["total_queries"] += 1
        
        # Ensure model is on the right device
        self.model.to(self.device)
        
        # First try pattern matching for common query types
        sql_query = self._pattern_match_sql(question)
        if sql_query:
            # Update metrics
            self.performance_metrics["pattern_matched"] += 1
            
            # Log the pattern-matched query
            self.log_query(question, sql_query, success=True, method="pattern")
            return sql_query
        
        # Use dynamic schema info
        schema_info = self._get_schema_info()
        
        input_text = f"{schema_info}\nConvert this question to SQL. Use proper SQL syntax with quotes for string values: {question}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate output with improved parameters
        outputs = self.model.generate(
            **inputs, 
            max_length=self.max_target_length,
            num_beams=5,
            temperature=0.3,  # Lower temperature for more focused outputs
            top_p=0.95,
            early_stopping=True
        )
        
        # Decode and return
        sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process the SQL to fix common errors
        sql_query = self._post_process_sql(sql_query, question)
        
        # Update metrics
        self.performance_metrics["model_generated"] += 1
        
        # Check if we have a correction for this exact question
        for correction in self.feedback_data["corrections"]:
            if correction["question"].lower() == question.lower():
                # Use the corrected SQL instead
                sql_query = correction["corrected_sql"]
                self.logger.info(f"Using previously corrected SQL for question: {question}")
                break
        
        # Log the model-generated query
        success = self._is_valid_sql(sql_query)
        self.log_query(question, sql_query, success=success, method="model")
        
        return sql_query
    
    def _pattern_match_sql(self, question):
        """Use pattern matching for common query types based on schema configuration"""
        question_lower = question.lower()

        # Month mapping
        month_mapping = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12"
        }

        # Check for "after" and month in the question
        for month_name, month_number in month_mapping.items():
            if f"after {month_name}" in question_lower:
                year_pattern = re.search(r'(\d{4})', question_lower)
                if year_pattern:
                    year = year_pattern.group(1)
                    # Set the date to the first day of the next month
                    next_month_number = str(int(month_number) + 1).zfill(2)  # Increment month
                    return f"SELECT * FROM vMTL_SYSTEM_ITEMS WHERE CREATION_DATE >= '{year}-{next_month_number}-01'"

        # Get schema information
        date_columns = self.schema.get("patterns", {}).get("date_columns", [])
        category_columns = self.schema.get("patterns", {}).get("category_columns", [])
        quantity_columns = self.schema.get("patterns", {}).get("quantity_columns", [])

        # If no patterns defined in schema, can't do pattern matching
        if not date_columns and not category_columns and not quantity_columns:
            return None

        # Extract primary date column if available
        date_column = None
        date_table = None
        if date_columns:
            # Use the first date column
            date_parts = date_columns[0].split('.')
            if len(date_parts) == 2:
                date_table, date_column = date_parts

        # Extract primary category column if available
        category_column = None
        category_table = None
        if category_columns:
            # Use the first category column
            category_parts = category_columns[0].split('.')
            if len(category_parts) == 2:
                category_table, category_column = category_parts

        # Extract primary quantity column if available
        quantity_column = None
        quantity_table = None
        if quantity_columns:
            # Use the first quantity column
            quantity_parts = quantity_columns[0].split('.')
            if len(quantity_parts) == 2:
                quantity_table, quantity_column = quantity_parts

        # Pattern for date ranges with years
        if date_column and date_table:
            year_pattern = re.search(r'(?:created|made|from|in) (\d{4})', question_lower)
            if year_pattern:
                year = year_pattern.group(1)

                # Check for category filtering
                if category_column and category_table:
                    category_pattern = re.search(r'(?:in|are in) the (\w+) (?:category|type|status)', question_lower)
                    not_category_pattern = re.search(r'not in the (\w+) (?:category|type|status)', question_lower)

                    if not_category_pattern:
                        category = not_category_pattern.group(1)
                        return f"SELECT * FROM {date_table} WHERE {date_column} >= '{year}-01-01' AND {date_column} <= '{year}-12-31' AND {category_column} != '{category}'"
                    elif category_pattern:
                        category = category_pattern.group(1)
                        return f"SELECT * FROM {date_table} WHERE {date_column} >= '{year}-01-01' AND {date_column} <= '{year}-12-31' AND {category_column} = '{category}'"

                # Just date filtering
                return f"SELECT * FROM {date_table} WHERE {date_column} >= '{year}-01-01' AND {date_column} <= '{year}-12-31'"

        # Pattern for items with a specific code/ID
        for table_name, table_info in self.schema["tables"].items():
            # Ensure table_info is a dictionary
            if isinstance(table_info, dict):
                for column in table_info["columns"]:
                    if "code" in column.lower() or "id" in column.lower():
                        # Fix: Use raw string for regex pattern
                        column_in_question = column.replace("_", " ")
                        pattern = fr'(?:with|has) {re.escape(column_in_question)} (\w+)'
                        code_pattern = re.search(pattern, question_lower, re.IGNORECASE)
                        if code_pattern:
                            code_value = code_pattern.group(1)
                            # Check if we're looking for just IDs/numbers
                            if "number" in question_lower or "id" in question_lower:
                                primary_key = table_info.get("primary_key")
                                if primary_key:
                                    return f"SELECT {primary_key} FROM {table_name} WHERE {column} = '{code_value}'"
                            return f"SELECT * FROM {table_name} WHERE {column} = '{code_value}'"
            else:
                self.logger.warning(f"Expected table_info to be a dictionary for table: {table_name}, but got {type(table_info)}")

        # Pattern for quantity comparisons
        if quantity_column and quantity_table:
            # Less than pattern - Fix: Use raw string
            quantity_in_question = quantity_column.replace("_", " ")
            less_pattern = re.search(fr'{re.escape(quantity_in_question)} less than (\d+)', question_lower)
            if less_pattern:
                quantity = less_pattern.group(1)
                return f"SELECT * FROM {quantity_table} WHERE {quantity_column} < {quantity}"

            # More than pattern - Fix: Use raw string
            more_pattern = re.search(fr'{re.escape(quantity_in_question)} more than (\d+)', question_lower)
            if more_pattern:
                quantity = more_pattern.group(1)
                return f"SELECT * FROM {quantity_table} WHERE {quantity_column} > {quantity}"

            # General quantity patterns
            more_pattern = re.search(r'more than (\d+)', question_lower)
            if more_pattern and quantity_column in question_lower:
                quantity = more_pattern.group(1)
                return f"SELECT * FROM {quantity_table} WHERE {quantity_column} > {quantity}"

        # Pattern for alphabetical listing
        for table_name, table_info in self.schema["tables"].items():
            if table_name.lower() in question_lower and "alphabetical" in question_lower:
                # Find a name-like column for sorting
                sort_column = None
                for col in table_info["columns"]:
                    if "name" in col.lower():
                        sort_column = col
                        break

                if sort_column:
                    return f"SELECT * FROM {table_name} ORDER BY {sort_column}"

        # Pattern for counting by group
        for table_name, table_info in self.schema["tables"].items():
            if "count" in question_lower and table_name.lower() in question_lower:
                for col in table_info["columns"]:
                    col_in_question = col.replace("_", " ")
                    if f"by {col_in_question}" in question_lower:
                        return f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col}"

        # Pattern for latest/max date queries
        latest_pattern = re.search(r'(?:latest|most recent|last|newest) (\w+) (?:date|time)', question.lower())
        if latest_pattern:
            date_field = latest_pattern.group(1)

            # Convert to actual column name based on schema
            date_column = None
            table_name = None

            # Try to find the matching date column in the schema
            for t_name, t_info in self.schema["tables"].items():
                for col in t_info["columns"]:
                    if date_field in col.lower() or col.lower() in date_field:
                        date_column = col
                        table_name = t_name
                        break
                if date_column:
                    break

            # If no exact match, try to find any date column
            if not date_column:
                for date_col in self.schema.get("patterns", {}).get("date_columns", []):
                    if "." in date_col:
                        t_name, col = date_col.split(".")
                        date_column = col
                        table_name = t_name
                        break

            if date_column and table_name:
                # Check if filtering for specific part/item
                item_pattern = re.search(r'for (?:part|item) (\w+)', question.lower())
                if item_pattern:
                    item_id = item_pattern.group(1)
                    primary_key = self.schema["tables"][table_name].get("primary_key")
                    if primary_key:
                        return f"SELECT MAX({date_column}) as latest_{date_column} FROM {table_name} WHERE {primary_key} = '{item_id}'"

                # If asking for each part/item, use GROUP BY
                if "each" in question.lower() or "all" in question.lower():
                    primary_key = self.schema["tables"][table_name].get("primary_key")
                    if primary_key:
                        return f"SELECT {primary_key}, MAX({date_column}) as latest_{date_column} FROM {table_name} GROUP BY {primary_key}"

                # General case
                return f"SELECT MAX({date_column}) as latest_{date_column} FROM {table_name}"

        # Pattern for wildcard searches
        search_term_pattern = re.search(r'(?:containing|contains|with|has) [\'"]?(\w+)[\'"]?', question_lower)
        if search_term_pattern:
            search_term = search_term_pattern.group(1)

            # Try to identify which column to search in
            if "name" in question_lower:
                return f"SELECT * FROM parts WHERE part_name LIKE '%{search_term}%'"
            elif "description" in question_lower:
                return f"SELECT * FROM parts WHERE description LIKE '%{search_term}%'"
            else:
                # Default to searching in multiple text columns
                return f"SELECT * FROM parts WHERE part_name LIKE '%{search_term}%' OR description LIKE '%{search_term}%'"

        # No pattern matched
        return None
    
    def _post_process_sql(self, sql_query, question):
        """Post-process the SQL to fix common errors using schema information"""
        # Get schema information
        date_columns = self.schema.get("patterns", {}).get("date_columns", [])
        
        # Check for date conditions in the question but missing in the query
        if date_columns:
            # Extract the primary date column
            date_parts = date_columns[0].split('.')
            if len(date_parts) == 2:
                date_table, date_column = date_parts
                
                year_pattern = re.search(r'(?:created|made|from|in) (\d{4})', question.lower())
                if year_pattern and date_column not in sql_query:
                    year = year_pattern.group(1)
                    # Add date condition to WHERE clause
                    if "WHERE" in sql_query:
                        sql_query = sql_query.replace("WHERE", f"WHERE {date_column} >= '{year}-01-01' AND {date_column} <= '{year}-12-31' AND")
                    else:
                        table_match = re.search(r'FROM (\w+)', sql_query)
                        if table_match:
                            sql_query = sql_query + f" WHERE {date_column} >= '{year}-01-01' AND {date_column} <= '{year}-12-31'"
        
        # Check for negation in the question but missing in the query
        category_columns = self.schema.get("patterns", {}).get("category_columns", [])
        if category_columns and "not in" in question.lower() and "!=" not in sql_query and "<>" not in sql_query:
            # Extract the primary category column
            category_parts = category_columns[0].split('.')
            if len(category_parts) == 2:
                category_table, category_column = category_parts
                
                category_pattern = re.search(r'not in the (\w+) (?:category|type|status)', question.lower())
                if category_pattern and category_column in sql_query:
                    category = category_pattern.group(1)
                    sql_query = sql_query.replace(f"{category_column} = '{category}'", f"{category_column} != '{category}'")
        
        # Fix case issues with table and column names using schema
        for table_name, table_info in self.schema["tables"].items():
            if isinstance(table_info, dict):  # Ensure table_info is a dictionary
                # Fix table name case
                sql_query = re.sub(r'\b' + re.escape(table_name.upper()) + r'\b', table_name, sql_query)
                sql_query = re.sub(r'\b' + re.escape(table_name.title()) + r'\b', table_name, sql_query)
                
                # Fix column name case
                for column in table_info["columns"]:
                    sql_query = re.sub(r'\b' + re.escape(column.upper()) + r'\b', column, sql_query)
                    sql_query = re.sub(r'\b' + re.escape(column.title()) + r'\b', column, sql_query)
        
        # Handle special formatting cases from schema
        special_formatting = self.schema.get("patterns", {}).get("special_formatting", [])
        for format_rule in special_formatting:
            if isinstance(format_rule, dict) and "from" in format_rule and "to" in format_rule:
                sql_query = re.sub(r'\b' + re.escape(format_rule["from"]) + r'\b', format_rule["to"], sql_query)
        
        # Ensure string values are properly quoted for all columns that might need it
        for table_name, table_info in self.schema["tables"].items():
            if isinstance(table_info, dict):  # Ensure table_info is a dictionary
                for column in table_info["columns"]:
                    # Skip numeric columns
                    if any(term in column.lower() for term in ["price", "quantity", "amount", "count", "number", "id"]):
                        continue
                    
                    # Add quotes to string values
                    pattern = f"{column} = (\\w+)"
                    match = re.search(pattern, sql_query)
                    if match and "'" not in match.group(0):
                        value = match.group(1)
                        sql_query = sql_query.replace(f"{column} = {value}", f"{column} = '{value}'")
        
        # Fix common syntax errors
        if "SELECT" not in sql_query and "select" in sql_query:
            sql_query = sql_query.replace("select", "SELECT")
        
        if "FROM" not in sql_query and "from" in sql_query:
            sql_query = sql_query.replace("from", "FROM")
        
        if "WHERE" not in sql_query and "where" in sql_query:
            sql_query = sql_query.replace("where", "WHERE")
        
        # Check if the query is completely wrong and try to fix based on the question
        if "SELECT" not in sql_query and "FROM" not in sql_query:
            # Try to extract key information from the question
            for table_name, table_info in self.schema["tables"].items():
                if isinstance(table_info, dict):  # Ensure table_info is a dictionary
                    if table_name.lower() in question.lower():
                        # Find a potential ID or code column
                        id_column = None
                        for col in table_info["columns"]:
                            if "id" in col.lower() or "code" in col.lower():
                                id_column = col
                                id_pattern = re.search(f'{col.replace("_", " ")} (\\w+)', question, re.IGNORECASE)
                                if id_pattern:
                                    id_value = id_pattern.group(1)
                                    primary_key = table_info.get("primary_key")
                                    if primary_key and "number" in question.lower():
                                        return f"SELECT {primary_key} FROM {table_name} WHERE {col} = '{id_value}'"
                                    return f"SELECT * FROM {table_name} WHERE {col} = '{id_value}'"
        
        # Check for latest/max date intent but missing GROUP BY
        if "MAX(" in sql_query and "GROUP BY" not in sql_query:
            # Check if the question implies grouping
            if any(term in question.lower() for term in ["each", "every", "per", "by"]):
                # Extract table name
                table_match = re.search(r'FROM (\w+)', sql_query)
                if table_match:
                    table_name = table_match.group(1)
                    # Find primary key for this table
                    primary_key = self.schema["tables"].get(table_name, {}).get("primary_key")
                    if primary_key:
                        # Add GROUP BY clause
                        sql_query = sql_query.rstrip(";")
                        sql_query += f" GROUP BY {primary_key}"
        
        # Simplify SELECT * or excessive column lists
        if "SELECT" in sql_query:
            question_lower = question.lower()
            if "just" in question_lower and "part number" in question_lower:
                sql_query = sql_query.replace(sql_query[6:sql_query.find("FROM")], "PART_NUMBER ")
            elif "show me all" in question_lower or "list all" in question_lower:
                sql_query = sql_query.replace(sql_query[6:sql_query.find("FROM")], "* ")
        
        return sql_query
    
    def _is_valid_sql(self, sql_query):
        """Basic validation of SQL query"""
        # Check for basic SQL structure
        has_select = "SELECT" in sql_query.upper()
        has_from = "FROM" in sql_query.upper()
        
        # Check for common syntax errors
        has_unbalanced_quotes = sql_query.count("'") % 2 != 0
        has_unbalanced_parentheses = sql_query.count("(") != sql_query.count(")")
        
        return has_select and has_from and not has_unbalanced_quotes and not has_unbalanced_parentheses
    
    def evaluate_model(self, test_data):
        """Evaluate the model on test data"""
        results = []
        
        for item in test_data:
            question = item["question"]
            expected_sql = item["sql"]
            generated_sql = self.generate_sql(question)
            
            results.append({
                "question": question,
                "expected_sql": expected_sql,
                "generated_sql": generated_sql,
                "is_correct": expected_sql.strip() == generated_sql.strip()
            })
        
        # Calculate accuracy
        accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
        
        return {
            "accuracy": accuracy,
            "detailed_results": results
        }
    
    def save_model(self, path):
        """Save the model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
    
    def load_most_recent_model(self):
        """Load the most recently trained model"""
        base_dir = "./text_to_sql_results"
        
        # Find all model directories
        model_dirs = []
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.isdir(dir_path) and (dir_name == "final_model" or dir_name.startswith("text_to_sql_results_")):
                # Get directory creation time
                creation_time = os.path.getctime(dir_path)
                model_dirs.append((dir_path, creation_time))
        
        if not model_dirs:
            self.logger.warning("No trained models found. Using initial model.")
            return self.initialize_model()
        
        # Sort by creation time and get most recent
        most_recent_dir = sorted(model_dirs, key=lambda x: x[1], reverse=True)[0][0]
        self.logger.info(f"Loading most recent model from: {most_recent_dir}")
        
        # Load the model and tokenizer
        self.load_model(most_recent_dir)


def main():
    # Example usage
    text_to_sql = TextToSQLModel(model_name="Salesforce/codet5p-220m", log_file="sql_queries.log")
    
    # Initialize model
    text_to_sql.initialize_model()
    
    # Load data
    datasets = text_to_sql.load_data()
    
    # Train model
    trainer = text_to_sql.train(datasets, output_dir="./text_to_sql_results", num_epochs=20)
    
    # Generate SQL from a question
    question = "Give me all part numbers created by ER code XYZ789"
    sql_query = text_to_sql.generate_sql(question)
    print(f"Question: {question}")
    print(f"Generated SQL: {sql_query}")
    
    # Try a w more questions to demonstrate logging
    test_questions = [
        "Hey, can you show me a list of all the parts we have?",
        "I need to see just the part numbers, nothing else.",
        "I'm looking for any parts that have XYZ789 as their ER code."
    ]
    
    for q in test_questions:
        sql = text_to_sql.generate_sql(q)
        print(f"\nQuestion: {q}")
        print(f"Generated SQL: {sql}")
    
    # Evaluate on test data
    eval_results = text_to_sql.evaluate_model(datasets["test"])
    print(f"\nModel accuracy: {eval_results['accuracy']:.2f}")
    
    # Save model
    text_to_sql.save_model("./text_to_sql_model")
    
    print(f"\nAll queries have been logged to {text_to_sql.log_file}")


if __name__ == "__main__":
    main()