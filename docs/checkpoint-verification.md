# Check a local checkpoint

The detector accepts a custom checkpoint path. Check whether a copied file is
identical to this project's pinned published release before running inference:

```powershell
python tools/fetch_weights.py --output D:/models/best.pt --verify-only
```

Verification streams the file in 1 MiB blocks and performs no network access.
Missing files and checksum mismatches exit with code 2. The command does not
create directories or modify the checkpoint.

To download the pinned release to that location, omit `--verify-only`. Existing
files are verified and never overwritten. The revision and expected SHA-256
remain fixed in `tools/fetch_weights.py`; this command does not authenticate an
arbitrary third-party checkpoint or change the detector's trusted-source policy.
