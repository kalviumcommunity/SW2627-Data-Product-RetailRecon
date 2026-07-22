# SQL Multi-Table Master

Welcome. Filtering is solid. Now comes combining data across tables: `INNER JOIN` keeps matched rows only, `LEFT JOIN` keeps all left rows plus matches, `OUTER JOIN` keeps everything. These joins power relational database analysis. You must join correctly, validate row counts, and detect unmatched keys. This is the final lesson: master joins and you can analyze any relational dataset.

Every analyst who misunderstood joins, whose row counts exploded from unexpected multiplicity, who had confidence in merged results that were actually corrupted had the same problem: they did not validate joins. This lesson teaches you join mastery. You will implement three join types, verify row counts, detect unmatched keys, and trace data lineage across tables.

## The Real Scenario

### The Problem

Join customers (1000 rows) to orders (5000 rows) to get customer order history. Result has 5500 rows. Why 5500 and not 5000? Did the join create duplicates? Are customers missing orders? Nobody validates. Analysis proceeds on unknown data quality. Results are questionable. Months later, someone points out: "One customer had 6 orders and appears 6 times in the result. Is that correct?" Nobody knows because nobody validated the join.

### The Solution

Before joining: customers = 1000, orders = 5000. After `LEFT JOIN`: 5500 rows. The extra 500 come from customers with multiple orders. Document this. Validate: "Customers with 2+ orders create multiple result rows - expected for this join." Unmatched: 100 orders have no matching customer - investigate why. Now the join is understood and validated. Results are trustworthy.

## Join Fundamentals

### Three Join Types

### INNER JOIN

Keep only matched rows. Result <= min(left, right). Customers with orders only.

### LEFT JOIN

All left rows plus matches. Result >= left. All customers, matched with orders where they exist.

### OUTER JOIN

All rows from both sides. Result > max(left, right). All customers AND all orders.

You just learned join semantics. Now validate joins properly.

## Implementing and Validating Joins

### Join Queries with Validation

### LEFT JOIN with Row Count Validation

```sql
-- Before join
SELECT COUNT(DISTINCT customer_id) AS customers FROM customers; -- 1000
SELECT COUNT(*) AS orders FROM orders;                          -- 5000

-- After join
SELECT 
    c.customer_id, 
    c.customer_type,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_type;

-- Result: 1000 customers (all), but total rows > 5000 due to one-to-many
```

### Detect Unmatched Keys

```sql
-- Customers with no orders
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- Orders with no customer
SELECT o.order_id
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

### Multi-Table Join

```sql
-- Join 3 tables
SELECT 
    c.customer_id,
    c.customer_type,
    o.order_id,
    p.product_name,
    oi.quantity,
    oi.unit_price
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE c.customer_type = 'Enterprise';
```

### Validate Row Counts

```sql
-- Compare row counts to understand join impact
SELECT 
    'customers' AS table_name, COUNT(DISTINCT customer_id) AS distinct_keys, COUNT(*) AS total_rows
FROM customers
UNION ALL
SELECT 
    'orders', COUNT(DISTINCT customer_id), COUNT(*)
FROM orders
UNION ALL
SELECT 
    'joined', COUNT(DISTINCT c.customer_id), COUNT(*)
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

You just learned to join tables correctly and validate results. You have completed all 30 lessons. You are now a complete data analyst and engineer.

## Bonus Resources

- [PostgreSQL JOIN Documentation - complete reference for all join types and performance optimization](https://www.postgresql.org/docs/current/queries-table-expressions.html)
- [JOIN Performance Tuning - indexes and query optimization for large-scale multi-table analysis](https://use-the-index-luke.com/)
- [Data Lineage and Validation - frameworks for tracking data through complex multi-table pipelines](https://en.wikipedia.org/wiki/Data_lineage)