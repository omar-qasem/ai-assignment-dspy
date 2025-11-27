import sqlite3

class SQLiteTool:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_schema(self):
        """Returns the schema of all tables in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Get column information for each table
            # Quote table name to handle spaces, as PRAGMA table_info does not automatically handle it
            cursor.execute(f"PRAGMA table_info('{table}');")
            columns = [f"{col[1]} {col[2]}" for col in cursor.fetchall()]
            schema[table] = ", ".join(columns)
            
        conn.close()
        
        # Add views for lowercase compatibility as per assignment
        views = {
            "orders": "OrderID INTEGER, CustomerID TEXT, EmployeeID INTEGER, OrderDate TEXT, RequiredDate TEXT, ShippedDate TEXT, ShipVia INTEGER, Freight REAL, ShipName TEXT, ShipAddress TEXT, ShipCity TEXT, ShipRegion TEXT, ShipPostalCode TEXT, ShipCountry TEXT",
            "order_items": "OrderID INTEGER, ProductID INTEGER, UnitPrice REAL, Quantity INTEGER, Discount REAL",
            "products": "ProductID INTEGER, ProductName TEXT, SupplierID INTEGER, CategoryID INTEGER, QuantityPerUnit TEXT, UnitPrice REAL, UnitsInStock INTEGER, UnitsOnOrder INTEGER, ReorderLevel INTEGER, Discontinued INTEGER",
            "customers": "CustomerID TEXT, CompanyName TEXT, ContactName TEXT, ContactTitle TEXT, Address TEXT, City TEXT, Region TEXT, PostalCode TEXT, Country TEXT, Phone TEXT, Fax TEXT"
        }
        schema.update(views)
        
        schema_str = "\n".join([f"Table: {table}\nColumns: {cols}" for table, cols in schema.items()])
        return schema_str

    def execute_query(self, query):
        """Executes a SQL query and returns the results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            # Fetch column names
            columns = [description[0] for description in cursor.description]
            # Fetch all rows
            rows = cursor.fetchall()
            
            conn.close()
            return {"columns": columns, "rows": rows, "error": None}
        except sqlite3.Error as e:
            conn.close()
            return {"columns": [], "rows": [], "error": str(e)}

if __name__ == '__main__':
    # Example usage for testing
    tool = SQLiteTool("../../data/northwind.sqlite")
    schema = tool.get_schema()
    print("--- Schema ---")
    print(schema)
    
    print("\n--- Test Query ---")
    test_query = "SELECT CompanyName, Country FROM Customers LIMIT 3;"
    result = tool.execute_query(test_query)
    print(f"Query: {test_query}")
    print(f"Columns: {result['columns']}")
    print(f"Rows: {result['rows']}")
    print(f"Error: {result['error']}")
