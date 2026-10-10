# Compare runs without repeating inference

```powershell
python tools/compare_tracking_runs.py --left outputs/original --right outputs/replay
```

The command prints JSON to stdout and never modifies either run. It matches
records by source frame index, reports frames available on only one side, and
counts exact changes to state, identity, boxes, observation flags and detector
proposals. Model and tracker setting differences are reported separately.

Both artifacts must name the same source video path. This check does not hash
the video or establish that its bytes have remained unchanged. Float values
are compared exactly, so small backend differences are visible. A changed box
or identity number is a behavioral difference, not evidence that either run is
more accurate. Use verified annotations for accuracy evaluation.
