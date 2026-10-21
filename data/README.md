# Sample data

`sample_stock_prices.csv` is **synthetic demonstration data**. The companies and tickers are fictional and the prices come from a seeded random-walk model. They are not real market quotes and do not describe any real company.

| Column | Meaning |
| --- | --- |
| Date | Trading date (weekdays, US federal holidays excluded) |
| Ticker | Fictional 4-letter symbol |
| Company | Fictional company name |
| Open, High, Low, Close | Simulated daily prices |
| Adjusted_Close | Close adjusted for simulated quarterly dividends (equal to Close for non-dividend stocks) |
| Volume | Simulated shares traded |

Regenerate the file (same output every time, seed 42) from the project root:

```bash
python scripts/generate_sample_data.py
```
