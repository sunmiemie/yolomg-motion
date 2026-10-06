# Export tracking observations

```powershell
python tools/export_tracking_csv.py --run outputs/tracking --output reports/tracking.csv
```

The command reads compact v2 and legacy tracking runs without loading a model
or decoding video. It writes one row per recorded center frame, including source
frame/time, state, identity, unpadded xyxy coordinates, detector confidence,
appearance score, and observation flags. UTF-8 with a BOM supports Excel on
Windows. Null boxes and confidences remain empty; boolean flags use 0 and 1.

The destination must be new. Validation happens before any output is created,
and the finished CSV is published atomically. Original run artifacts are unchanged.
Tracking states and coverage describe recorded behavior; this export does not
measure detection accuracy or compare predictions with ground truth.
