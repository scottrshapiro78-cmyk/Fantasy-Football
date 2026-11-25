"""Fantasy football projection and ranking model."""

# Fallback for nflverse schedules: when the GitHub release download fails (Nov 25 2026 outage returned 404),
# use the same table from the nflverse/nfldata repo (raw.githubusercontent.com), cached to data/raw/schedules_fallback.parquet.
try:
    import nflreadpy as _nfl
    import polars as _pl
    _orig = _nfl.load_schedules

    def _load_schedules(seasons=None, *a, **k):
        try:
            return _orig(seasons, *a, **k)
        except Exception:
            from .config import DATA as _D
            p = _D / "raw" / "schedules_fallback.parquet"
            try:
                df = _pl.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
                                  infer_schema_length=20000, null_values=["NA", ""])
                df.write_parquet(p)
            except Exception:
                df = _pl.read_parquet(p)
            if seasons is True or seasons is None:
                return df
            ss = [seasons] if isinstance(seasons, int) else list(seasons)
            return df.filter(_pl.col("season").is_in(ss))

    _nfl.load_schedules = _load_schedules
except Exception:
    pass
