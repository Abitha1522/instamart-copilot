
-- Q1: WHERE and HAVING
-- Find customers whose completed orders above INR 500
-- have a combined spend above INR 8000.

SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Three-table JOIN
-- Calculate completed revenue by city and product category.

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, total_revenue DESC;


-- Q3: Scalar subquery
-- Find products priced above the average catalogue price.

SELECT *
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: LEFT JOIN
-- Find customers who have never placed a completed order.

SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Two Common Table Expressions (CTEs)
-- Find customers whose completed spend is above
-- the average spend per active customer.

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(av.avg_spend, 2) AS average_spend
FROM customer_spend cs
CROSS JOIN average_spend av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6: Window functions
-- Rank customers by completed spend within each city.

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
),
ranked_customers AS (
    SELECT
        customer_id,
        city,
        ROUND(total_spend, 2) AS total_spend,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS rank_num,
        DENSE_RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS dense_rank_num,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC, customer_id
        ) AS row_num
    FROM customer_spend
)
SELECT *
FROM ranked_customers
ORDER BY city, rank_num, customer_id;


-- Q7: Running total
-- Calculate daily completed revenue and cumulative revenue.

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
