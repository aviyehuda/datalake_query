from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from langchain.chat_models import ChatOpenAI
import os
import json
from datetime import datetime, timedelta
import random

def create_spark_session():
    """Create and return a Spark session with Hive support"""
    return SparkSession.builder \
        .appName("Multi-Dataset Query System") \
        .master("local[*]") \
        .config("spark.sql.warehouse.dir", "spark-warehouse") \
        .enableHiveSupport() \
        .getOrCreate()

def create_dummy_data(spark):
    """Create dummy data for the three tables"""
    
    # Product Description Table
    product_data = [
        (1, "Laptop Pro", "High-performance laptop for professionals", "Electronics", 1299.99),
        (2, "Smartphone X", "Latest smartphone with advanced features", "Electronics", 899.99),
        (3, "Coffee Maker", "Automatic coffee maker for home use", "Home & Kitchen", 199.99),
        (4, "Running Shoes", "Comfortable running shoes for athletes", "Sports", 129.99),
        (5, "Bluetooth Headphones", "Wireless headphones with noise cancellation", "Electronics", 299.99),
        (6, "Yoga Mat", "Premium yoga mat for fitness enthusiasts", "Sports", 49.99),
        (7, "Blender", "High-speed blender for smoothies", "Home & Kitchen", 89.99),
        (8, "Gaming Console", "Next-gen gaming console", "Electronics", 499.99)
    ]
    
    product_schema = StructType([
        StructField("product_id", IntegerType(), False),
        StructField("product_name", StringType(), False),
        StructField("description", StringType(), False),
        StructField("category", StringType(), False),
        StructField("price", DoubleType(), False)
    ])
    
    product_df = spark.createDataFrame(product_data, product_schema)
    
    # Product Reviews Table
    review_data = []
    for i in range(1, 9):  # For each product
        for j in range(random.randint(3, 8)):  # 3-8 reviews per product
            review_data.append((
                len(review_data) + 1,
                i,  # product_id
                f"user_{random.randint(1000, 9999)}",
                random.randint(1, 5),  # rating
                random.choice([
                    "Great product, highly recommend!",
                    "Good quality for the price",
                    "Average product, nothing special",
                    "Excellent value for money",
                    "Could be better",
                    "Love this product!",
                    "Not worth the money",
                    "Perfect for my needs"
                ]),
                datetime.now() - timedelta(days=random.randint(1, 365))
            ))
    
    review_schema = StructType([
        StructField("review_id", IntegerType(), False),
        StructField("product_id", IntegerType(), False),
        StructField("user_id", StringType(), False),
        StructField("rating", IntegerType(), False),
        StructField("review_text", StringType(), False),
        StructField("review_date", DateType(), False)
    ])
    
    review_df = spark.createDataFrame(review_data, review_schema)
    
    # Product Sales Table
    sales_data = []
    for i in range(1, 9):  # For each product
        for j in range(random.randint(5, 15)):  # 5-15 sales per product
            sales_data.append((
                len(sales_data) + 1,
                i,  # product_id
                random.randint(1, 5),  # quantity
                datetime.now() - timedelta(days=random.randint(1, 365)),
                random.choice(["Online", "Store", "Mobile App"]),
                random.choice(["Credit Card", "PayPal", "Cash", "Bank Transfer"])
            ))
    
    sales_schema = StructType([
        StructField("sale_id", IntegerType(), False),
        StructField("product_id", IntegerType(), False),
        StructField("quantity", IntegerType(), False),
        StructField("sale_date", DateType(), False),
        StructField("channel", StringType(), False),
        StructField("payment_method", StringType(), False)
    ])
    
    sales_df = spark.createDataFrame(sales_data, sales_schema)
    
    return product_df, review_df, sales_df

def create_table_dictionary():
    """Create a dictionary with table metadata"""
    return {
        "product_description": {
            "description": "Contains product information including name, description, category, and price",
            "columns": ["product_id", "product_name", "description", "category", "price"]
        },
        "product_reviews": {
            "description": "Contains customer reviews for products including ratings, review text, and dates",
            "columns": ["review_id", "product_id", "user_id", "rating", "review_text", "review_date"]
        },
        "product_sales": {
            "description": "Contains sales transaction data including quantities, dates, sales channels, and payment methods",
            "columns": ["sale_id", "product_id", "quantity", "sale_date", "channel", "payment_method"]
        }
    }

def create_prompt(table_dict):
    """Create the prompt for ChatOpenAI with table information"""
    prompt = """You are a helpful assistant that helps users query a dataset. Here are the available tables and their information:

"""
    
    for table_name, table_info in table_dict.items():
        prompt += f"Table: {table_name}\n"
        prompt += f"Description: {table_info['description']}\n"
        prompt += f"Columns: {', '.join(table_info['columns'])}\n\n"
    
    prompt += """Please analyze the user's question and respond in one of two formats:

1. If you need more information to answer the question, respond with a follow-up question starting with "FOLLOW_UP: "
2. If you can answer the question with the available data, respond with a Spark SQL query starting with "QUERY: "

The tables can be joined using product_id as the common key.

User question: """
    
    return prompt

def get_ai_response(chat_model, prompt, user_question):
    """Get response from ChatOpenAI"""
    full_prompt = prompt + user_question
    response = chat_model.predict(full_prompt)
    return response.strip()

def execute_query(spark, query):
    """Execute Spark SQL query and return result"""
    try:
        result_df = spark.sql(query)
        return result_df
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")

def display_result(result_df):
    """Display the result in appropriate format"""
    count = result_df.count()
    
    if count == 0:
        print("No results found.")
        return False
    
    if count == 1 and len(result_df.columns) == 1:
        # Single value result
        value = result_df.collect()[0][0]
        print(f"Result: {value}")
        return False
    else:
        # Table result
        print(f"\nResults ({count} rows):")
        result_df.show(truncate=False)
        return True

def ask_for_graph():
    """Ask user if they want to see a graph"""
    while True:
        response = input("\nWould you like to see a graph of the results? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def main():
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = 'xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx-xxxxxx'
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        print("Creating dummy data and tables...")
        
        # Create dummy data
        product_df, review_df, sales_df = create_dummy_data(spark)
        
        # Create tables in Hive catalog
        product_df.write.mode("overwrite").saveAsTable("product_description")
        review_df.write.mode("overwrite").saveAsTable("product_reviews")
        sales_df.write.mode("overwrite").saveAsTable("product_sales")
        
        print("Tables created successfully!")
        
        # Create table dictionary
        table_dict = create_table_dictionary()
        print("\nAvailable tables:")
        for table_name, table_info in table_dict.items():
            print(f"- {table_name}: {table_info['description']}")
        
        # Initialize ChatOpenAI
        chat_model = ChatOpenAI(model='gpt-4')
        
        # Create prompt template
        prompt_template = create_prompt(table_dict)
        
        print("\n" + "="*50)
        print("Interactive Query System")
        print("="*50)
        print("Ask questions about the data in natural language!")
        print("Type 'exit' to quit.")
        print("="*50)
        
        while True:
            user_question = input("\nYour question: ").strip()
            
            if user_question.lower() == 'exit':
                break
            
            if not user_question:
                continue
            
            try:
                # Get AI response
                ai_response = get_ai_response(chat_model, prompt_template, user_question)
                
                if ai_response.startswith("FOLLOW_UP: "):
                    # Handle follow-up question
                    follow_up = ai_response[12:]  # Remove "FOLLOW_UP: " prefix
                    print(f"\nAI needs more information: {follow_up}")
                    continue
                
                elif ai_response.startswith("QUERY: "):
                    # Execute the query
                    query = ai_response[7:]  # Remove "QUERY: " prefix
                    print(f"\nExecuting query: {query}")
                    
                    result_df = execute_query(spark, query)
                    has_multiple_rows = display_result(result_df)
                    
                    # Ask for graph if multiple rows
                    if has_multiple_rows:
                        if ask_for_graph():
                            try:
                                # Try to create a simple plot
                                if len(result_df.columns) >= 2:
                                    # Convert to pandas for plotting
                                    pandas_df = result_df.toPandas()
                                    if len(pandas_df) <= 20:  # Limit for plotting
                                        import matplotlib.pyplot as plt
                                        plt.figure(figsize=(10, 6))
                                        pandas_df.plot(kind='bar')
                                        plt.title("Query Results")
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.show()
                                    else:
                                        print("Too many rows to display as a graph effectively.")
                                else:
                                    print("Need at least 2 columns to create a meaningful graph.")
                            except Exception as e:
                                print(f"Could not create graph: {str(e)}")
                
                else:
                    print("Unexpected response format from AI. Please try again.")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try rephrasing your question.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Stop Spark session
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    main() 
