"""Extract the raw events needed by the old 2025 BiLSTM baseline for the 38 fold-0 signals.

Writes hive-partitioned parquet (one dir per DeviceId, one file per day) to
C:\\Users\\hwyr67g\\dc_work\\cache\\fold0_events\\ with columns DeviceId, Timestamp, EventId, Parameter.
Only the 6 event codes the old model used (1,7,43,44,81,82) and Parameter <= 64.
"""
import duckdb

WORK = r"C:\Users\hwyr67g\dc_work"
OUT = WORK + r"\cache\fold0_events"

con = duckdb.connect()
con.execute("SET memory_limit='8GB'; SET threads=8;")
con.execute(f"SET temp_directory='{WORK}\\tmp';")
con.execute(
    "CREATE VIEW valid AS SELECT DeviceId FROM "
    f"read_csv('{WORK}\\data\\splits\\device_id_valid.csv', header=true, names=['idx','DeviceId'])"
)

for day in (2, 3, 4):
    src = f"{WORK}\\data\\raw\\Train_Dec_{day}_2024.parquet"
    con.execute(f"""
        COPY (
            SELECT e.DeviceId, e.Timestamp, e.EventId::INTEGER AS EventId, e.Parameter::INTEGER AS Parameter,
                   {day} AS day
            FROM '{src}' e
            SEMI JOIN valid v ON v.DeviceId = e.DeviceId
            WHERE e.EventId IN (1,7,43,44,81,82) AND e.Parameter <= 64
            ORDER BY e.DeviceId, e.Timestamp, e.EventId
        ) TO '{OUT}' (FORMAT PARQUET, PARTITION_BY (DeviceId, day),
                      OVERWRITE_OR_IGNORE, FILENAME_PATTERN 'ev_{{i}}')
    """)
    print("day", day, "done", flush=True)

print(con.execute(f"SELECT count(*) n, count(DISTINCT DeviceId) d FROM read_parquet('{OUT}/**/*.parquet', hive_partitioning=1)").fetchall())
