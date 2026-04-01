"""Seed a sample SQLite database for the SQL agent demo."""

import os
import sqlite3

DB_PATH = "data/sample.db"


def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Products table ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)
    products = [
        (1, "NVIDIA A100 GPU", "hardware", 10999.00, 45),
        (2, "NVIDIA H100 GPU", "hardware", 32999.00, 20),
        (3, "NVIDIA L40S GPU", "hardware", 7499.00, 80),
        (4, "NeMo Framework License", "software", 0.00, 999),
        (5, "NIM Microservice - LLM", "service", 4500.00, 100),
        (6, "NIM Microservice - Embeddings", "service", 1200.00, 100),
        (7, "DGX Cloud Instance (monthly)", "cloud", 36999.00, 15),
        (8, "Jetson Orin Nano", "hardware", 499.00, 200),
        (9, "CUDA Toolkit License", "software", 0.00, 999),
        (10, "AI Enterprise Suite", "software", 8999.00, 50),
    ]
    cur.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)", products)

    # ── Customers table ─────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            industry TEXT NOT NULL,
            country TEXT NOT NULL,
            tier TEXT NOT NULL
        )
    """)
    customers = [
        (1, "Acme AI Labs", "technology", "US", "enterprise"),
        (2, "MedVision Health", "healthcare", "UK", "enterprise"),
        (3, "AutoDrive Inc", "automotive", "Germany", "enterprise"),
        (4, "DataSpark Analytics", "finance", "Singapore", "startup"),
        (5, "GreenEnergy AI", "energy", "US", "startup"),
        (6, "RetailBot Corp", "retail", "Canada", "mid-market"),
        (7, "EduTech Global", "education", "India", "startup"),
        (8, "Quantum Research Lab", "research", "US", "enterprise"),
    ]
    cur.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    # ── Orders table ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    orders = [
        (1, 1, 2, 8, "2024-11-15", "delivered"),
        (2, 1, 5, 10, "2024-12-01", "delivered"),
        (3, 2, 1, 4, "2025-01-10", "delivered"),
        (4, 3, 2, 16, "2025-01-20", "shipped"),
        (5, 4, 6, 5, "2025-02-01", "delivered"),
        (6, 5, 8, 50, "2025-02-15", "processing"),
        (7, 6, 10, 3, "2025-03-01", "shipped"),
        (8, 7, 4, 20, "2025-03-10", "delivered"),
        (9, 8, 7, 2, "2025-03-15", "processing"),
        (10, 1, 3, 12, "2025-03-20", "shipped"),
    ]
    cur.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"Sample database seeded at {DB_PATH}")
    print("  Tables: products (10 rows), customers (8 rows), orders (10 rows)")


if __name__ == "__main__":
    seed()
