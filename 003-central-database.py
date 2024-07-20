import sqlite3

# Sample extracted lease data
lease_data = {
    "Lease_ID": 1,
    "Landlord": "ABC Corp",
    "Tenant": "XYZ Inc",
    "Start_Date": "2023-02-01",
    "End_Date": "2028-01-31",
    "Monthly_Rent": 2500
}

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('lease_management.db')
cursor = conn.cursor()

# Create a table for lease data
cursor.execute('''
CREATE TABLE IF NOT EXISTS leases (
    Lease_ID INTEGER PRIMARY KEY,
    Landlord TEXT,
    Tenant TEXT,
    Start_Date TEXT,
    End_Date TEXT,
    Monthly_Rent INTEGER
)
''')

# Insert sample lease data
cursor.execute('''
INSERT INTO leases (Lease_ID, Landlord, Tenant, Start_Date, End_Date, Monthly_Rent)
VALUES (:Lease_ID, :Landlord, :Tenant, :Start_Date, :End_Date, :Monthly_Rent)
''', lease_data)

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Lease data inserted into the database.")
