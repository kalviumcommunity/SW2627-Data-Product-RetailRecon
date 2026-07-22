# SQL Filter Master

Welcome. Metrics are defined. Now comes filtering and grouping: `WHERE` clauses that apply business rules, `GROUP BY` that slices data by dimensions, `HAVING` that filters aggregated results, `ORDER BY` that surfaces top performers. These four clauses are the foundation of operational reporting. You must master them to compute KPIs reliably.

Every analyst who could not filter data correctly, who did not understand the difference between `WHERE` and `HAVING`, who could not order results meaningfully had the same problem: they never learned these fundamental SQL clauses deeply. This lesson teaches SQL filtering and aggregation thoroughly.

## The Real Scenario

### The Problem

Someone writes: "Show me revenue by customer where revenue > 1000". Do you filter before grouping or after? If before (`WHERE`), you exclude customers entirely if any transaction is < 1000. If after (`HAVING`), you count all transactions but show only customers whose total exceeds 1000. Results differ wildly. Nobody knows which is right. Questions repeat constantly: "Should I use `WHERE` or `HAVING`?" This lesson answers it definitively.

### The Solution

`WHERE` filters before grouping. `HAVING` filters after. They answer different questions. Use `WHERE` to exclude invalid data (`transaction_date` in this year). Use `HAVING` to filter aggregated metrics (`SUM(amount) > 1000`). Combined properly, queries produce correct results consistently.

## WHERE vs HAVING

### Filtering at Different Stages

### WHERE

Filters rows before grouping. Data-quality check. `WHERE customer_type = 'Enterprise'`. Rows not matching are never included in group.

### HAVING

Filters groups after aggregation. Metric threshold. `HAVING SUM(amount) > 1000`. Groups not matching are excluded.

You just learned `WHERE` and `HAVING` roles. Now apply them to queries.

## Filtering and Aggregation Queries

### WHERE: Filter Data Before Grouping

```sql
-- Show only Enterprise customers
SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
WHERE customer_type = 'Enterprise'
GROUP BY customer_id
ORDER BY total_spent DESC;
```

### HAVING: Filter Groups After Aggregation

```sql
-- Show only customers who spent more than $10,000 total
SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
GROUP BY customer_id
HAVING SUM(amount) > 10000
ORDER BY total_spent DESC;
```

### WHERE + HAVING: Both Together

```sql
-- Enterprise customers who spent > $10k
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_spent
FROM transactions
WHERE customer_type = 'Enterprise'          -- Filter data first
GROUP BY customer_id
HAVING SUM(amount) > 10000                  -- Filter groups second
ORDER BY total_spent DESC;
```

### ORDER BY: Sort Results

```sql
-- Top 10 customers by revenue
SELECT
    customer_id,
    customer_type,
    SUM(amount) AS total_revenue,
    COUNT(*) AS order_count
FROM transactions
WHERE transaction_date >= '2024-01-01'
GROUP BY customer_id, customer_type
HAVING COUNT(*) >= 5
ORDER BY total_revenue DESC
LIMIT 10;
```

You just learned filtering and aggregation fundamentals. Operational reporting is now possible.

## Bonus Resources

- [SQL WHERE Documentation - complete reference for filtering conditions and logical operators](https://www.postgresql.org/docs/current/sql-select.html)
- [GROUP BY Best Practices - avoiding common errors and understanding aggregation semantics](https://use-the-index-luke.com/sql/join/hash-join-vs-sort-merge-join)
- [HAVING Clause Guide - filtering aggregated results and common WHERE vs HAVING mistakes](https://www.postgresql.org/docs/current/sql-select.html#SQL-GROUPBY)
*** Update File: c:\Users\SUPRIYA\OneDrive\Desktop\Data-Product-RetailRecon\SW2627-Data-Product-RetailRecon\README.md
@@
 # Stock Sync
 
 ## Project Description
@@
 Stock Sync is an inventory analytics project that helps inventory managers identify stock shortages, excess inventory, sales trends, and return patterns across regional stores using an interactive dashboard.

## Learning Guides

- [SQL Filtering Guide](SQL_FILTERING_GUIDE.md) - a focused lesson on `WHERE`, `GROUP BY`, `HAVING`, and `ORDER BY` for operational reporting.
 
 ---