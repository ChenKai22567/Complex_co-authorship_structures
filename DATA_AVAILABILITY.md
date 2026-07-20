# Data Availability

## Original source data

Release `v0.1.0` includes the raw Web of Science Excel exports used by the study. They are stored in the large network-and-community data archive rather than in the Git commit tree. The records are published as used, without anonymization or field-value replacement, and therefore retain author names, addresses, affiliations, and other source metadata.

The workflows cover records through 2025. The core scripts expect WOS Excel exports containing, as applicable:

- `UT (Unique WOS ID)` or a documented equivalent;
- `Author Full Names`;
- `Addresses`;
- `Publication Year`;
- `Author Keywords`;
- `Times Cited, WoS Core`;
- `WoS Categories`, `Web of Science Categories`, or `WC`;
- `Reprint Addresses` or `Reprint Address`.

After extracting the data archive at the repository root, the files are located under:

```text
reproducibility/data/01_data_preparation/wos_1900_2025/
reproducibility/data/01_data_preparation/wos_2000_2025/
```

These large paths are ignored by Git and are supplied through the Release asset. WOS and other third-party records remain subject to the original database, institutional subscription, and applicable reuse terms; this repository does not relicense them.

## Full derived data

Release `v0.1.0` contains the derived networks, author labels, aggregate metrics, model outputs, robustness curves, plotting inputs, author-disambiguation records, validation records, and merge-candidate tables used in the paper. Names, addresses, affiliations, and other original fields are preserved without anonymization.

The two data archives are designed to be extracted at the repository root. SHA-256 values are supplied in `SHA256SUMS.txt` and mirrored in `reproducibility/metadata/release_manifest.csv`.
