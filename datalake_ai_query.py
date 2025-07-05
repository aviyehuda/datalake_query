from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from langchain_community.chat_models import ChatOpenAI
import os
import json
from typing import Dict, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt

class MultiDatasetQuerySystem:
    def __init__(self):
        """Initialize the multi-dataset query system"""
        self.spark = self.create_spark_session()
        self.llm = ChatOpenAI(model='gpt-4')
        self.dataframes_map = {}
        self.setup_dataframes()
        
    def create_spark_session(self):
        """Create and return a Spark session"""
        return SparkSession.builder \
            .appName("Multi-Dataset Query System") \
            .master("local[*]") \
            .getOrCreate()
    
    def setup_dataframes(self):
        """Create sample dataframes and store them in the map with descriptions"""
        
        # Create products dataframe
        products_data = [
            (1, "Laptop", "High-performance laptop", 1200.0, "Electronics"),
            (2, "Smartphone", "Latest smartphone model", 800.0, "Electronics"),
            (3, "Coffee Maker", "Automatic coffee machine", 150.0, "Home Appliances"),
            (4, "Running Shoes", "Comfortable athletic shoes", 120.0, "Sports"),
            (5, "Bluetooth Speaker", "Portable wireless speaker", 80.0, "Electronics"),
            (6, "Yoga Mat", "Non-slip exercise mat", 25.0, "Sports"),
            (7, "Microwave", "Countertop microwave oven", 200.0, "Home Appliances"),
            (8, "Gaming Mouse", "High-precision gaming mouse", 60.0, "Electronics")
        ]
        
        products_schema = StructType([
            StructField("product_id", IntegerType(), False),
            StructField("product_name", StringType(), False),
            StructField("description", StringType(), False),
            StructField("price", DoubleType(), False),
            StructField("category", StringType(), False)
        ])
        
        products_df = self.spark.createDataFrame(products_data, products_schema)
        
        # Create product reviews dataframe
        reviews_data = [
            (1, 1, 5, "Excellent laptop, very fast!", "2024-01-15"),
            (2, 1, 4, "Good performance, but expensive", "2024-01-20"),
            (3, 2, 5, "Amazing smartphone camera", "2024-02-01"),
            (4, 2, 3, "Battery life could be better", "2024-02-10"),
            (5, 3, 4, "Makes great coffee", "2024-01-25"),
            (6, 4, 5, "Very comfortable for running", "2024-02-05"),
            (7, 5, 4, "Good sound quality", "2024-01-30"),
            (8, 6, 5, "Perfect for yoga sessions", "2024-02-15"),
            (9, 7, 3, "Works well but noisy", "2024-02-20"),
            (10, 8, 5, "Best gaming mouse I've used", "2024-01-28")
        ]
        
        reviews_schema = StructType([
            StructField("review_id", IntegerType(), False),
            StructField("product_id", IntegerType(), False),
            StructField("rating", IntegerType(), False),
            StructField("comment", StringType(), False),
            StructField("review_date", StringType(), False)
        ])
        
        reviews_df = self.spark.createDataFrame(reviews_data, reviews_schema)
        
        # Create product sales dataframe
        sales_data = [
            (1, 1, 50, "2024-01-01", "Online"),
            (2, 1, 30, "2024-01-15", "Store"),
            (3, 2, 100, "2024-01-05", "Online"),
            (4, 2, 75, "2024-01-20", "Store"),
            (5, 3, 25, "2024-01-10", "Store"),
            (6, 4, 80, "2024-01-25", "Online"),
            (7, 5, 60, "2024-02-01", "Online"),
            (8, 6, 120, "2024-02-05", "Store"),
            (9, 7, 40, "2024-02-10", "Store"),
            (10, 8, 90, "2024-02-15", "Online"),
            (11, 1, 45, "2024-02-20", "Online"),
            (12, 2, 85, "2024-02-25", "Store")
        ]
        
        sales_schema = StructType([
            StructField("sale_id", IntegerType(), False),
            StructField("product_id", IntegerType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("sale_date", StringType(), False),
            StructField("channel", StringType(), False)
        ])
        
        sales_df = self.spark.createDataFrame(sales_data, sales_schema)
        
        # Store dataframes in the map with descriptions
        self.dataframes_map = {
            "products": {
                "dataframe": products_df,
                "description": "Contains product information including product_id, product_name, description, price, and category. This dataset stores details about various products available in the store."
            },
            "product_reviews": {
                "dataframe": reviews_df,
                "description": "Contains customer reviews for products including review_id, product_id, rating (1-5), comment, and review_date. This dataset stores customer feedback and ratings for products."
            },
            "product_sales": {
                "dataframe": sales_df,
                "description": "Contains sales transaction data including sale_id, product_id, quantity sold, sale_date, and channel (Online/Store). This dataset tracks product sales across different channels."
            }
        }
    
    def get_dataset_info(self) -> str:
        """Get information about all datasets for the AI prompt"""
        dataset_info = []
        
        for name, info in self.dataframes_map.items():
            df = info["dataframe"]
            description = info["description"]
            columns = [field.name for field in df.schema.fields]
            
            dataset_info.append(f"""
Dataset: {name}
Description: {description}
Columns: {', '.join(columns)}
""")
        
        return "\n".join(dataset_info)
    
    def create_ai_prompt(self, user_question: str) -> str:
        """Create a prompt for the AI model"""
        dataset_info = self.get_dataset_info()
        
        prompt = f"""
You are a data analyst assistant. You have access to the following datasets:

{dataset_info}

The user has asked: "{user_question}"

Based on the dataset information above, you need to either:
1. Ask a follow-up question to clarify what the user wants, OR
2. Provide a Spark SQL query to answer their question

IMPORTANT: 
- Do NOT include the actual data in your response
- Only use the dataset names, descriptions, and column information provided
- If you need more information to answer the question, ask a follow-up question
- If you can answer the question, provide ONLY the Spark SQL query

Respond in the following JSON format:
{{
    "type": "followup_question" or "sql_query",
    "content": "your follow-up question or SQL query here"
}}

Examples:
- For follow-up: {{"type": "followup_question", "content": "Which specific product category are you interested in?"}}
- For SQL query: {{"type": "sql_query", "content": "SELECT product_name, AVG(rating) as avg_rating FROM products p JOIN product_reviews pr ON p.product_id = pr.product_id GROUP BY product_name ORDER BY avg_rating DESC"}}
"""
        return prompt
    
    def process_ai_response(self, response: str) -> Tuple[str, str]:
        """Process the AI response and return type and content"""
        try:
            # Try to parse as JSON
            response_data = json.loads(response)
            return response_data.get("type", "error"), response_data.get("content", "Invalid response format")
        except json.JSONDecodeError:
            # If not JSON, try to extract SQL query or follow-up question
            response_lower = response.lower()
            if "select" in response_lower and "from" in response_lower:
                return "sql_query", response
            else:
                return "followup_question", response
    
    def execute_query(self, query: str):
        """Execute a Spark SQL query and return the result"""
        try:
            # Register all dataframes as temporary views
            for name, info in self.dataframes_map.items():
                info["dataframe"].createOrReplaceTempView(name)
            
            # Execute the query
            result = self.spark.sql(query)
            return result
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
    
    def display_result(self, df):
        """Display the query result appropriately"""
        count = df.count()
        
        if count == 0:
            print("No results found.")
            return
        
        if count == 1 and len(df.columns) == 1:
            # Single value result
            value = df.collect()[0][0]
            print(f"Result: {value}")
        else:
            # Multiple rows/columns result
            print("\nQuery Results:")
            df.show()
            
            # Ask if user wants to see a graph
            if count > 1:
                show_graph = input("\nWould you like to see a graph of the results? (y/n): ").strip().lower()
                if show_graph in ['y', 'yes']:
                    try:
                        # Convert to pandas for plotting
                        pandas_df = df.toPandas()
                        
                        # Simple plotting - can be enhanced based on data types
                        if len(pandas_df.columns) >= 2:
                            # Try to create a bar plot with first two columns
                            plt.figure(figsize=(10, 6))
                            pandas_df.plot(kind='bar', x=pandas_df.columns[0], y=pandas_df.columns[1])
                            plt.title("Query Results Visualization")
                            plt.xlabel(pandas_df.columns[0])
                            plt.ylabel(pandas_df.columns[1])
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.show()
                        else:
                            print("Cannot create a meaningful graph with the current result structure.")
                    except Exception as e:
                        print(f"Error creating graph: {str(e)}")
    
    def run(self):
        """Main loop for the query system"""
        print("Welcome to the Multi-Dataset Query System!")
        print("You can ask questions about products, reviews, and sales data.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                # Get user question
                user_question = input("What would you like to know about the data? ").strip()
                
                if user_question.lower() == 'exit':
                    break
                
                if not user_question:
                    continue
                
                # Process the question through AI
                prompt = self.create_ai_prompt(user_question)
                ai_response = self.llm.predict(prompt)
                
                # Process AI response
                response_type, content = self.process_ai_response(ai_response)
                
                if response_type == "followup_question":
                    print(f"\nAI Assistant: {content}")
                    continue
                
                elif response_type == "sql_query":
                    print(f"\nExecuting query: {content}")
                    
                    # Execute the query
                    result_df = self.execute_query(content)
                    
                    # Display results
                    self.display_result(result_df)
                
                else:
                    print("Error: Could not process AI response properly.")
                    print(f"Raw response: {ai_response}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again with a different question.")
    
    def cleanup(self):
        """Clean up resources"""
        if self.spark:
            self.spark.stop()

def main():
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = 'xxxxxxxxxxx-xxxxxxxxxxx-xxxxxxxxxxx-xxxxxxxxxxx-xxxxxxxxxxx-xxxxxxxxxxx'
    
    # Create and run the query system
    query_system = MultiDatasetQuerySystem()
    
    try:
        query_system.run()
    finally:
        query_system.cleanup()

if __name__ == "__main__":
    main() 
