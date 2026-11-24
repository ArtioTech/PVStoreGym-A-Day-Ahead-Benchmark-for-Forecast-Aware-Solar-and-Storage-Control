# Data: sources, provenance and licences

PV-BESS-Gym does **not** redistribute Greek market data. The benchmark dataset
is rebuilt on your machine from the publishers' own public files:

```bash
python scripts/auxiliary/build_greek_2025_dataset.py --year 2025
python scripts/data/verify_dataset.py data/processed
```

The first command downloads the raw files into `data/raw/` (about 150 MB) and
writes twelve monthly CSVs plus one `*_metadata.json` per month into
`data/processed/`. The second command fails if any column is a proxy or looks
synthetic. Both directories are git-ignored.

## Columns

| Column | Source | Native resolution | Status |
|---|---|---|---|
| `ghi_wm2`, `temp_air_c`, `wind_mps` | Open-Meteo archive API | 60 min, time-interpolated | real (reanalysis) |
| `ghi_fcst_wm2`, `temp_air_fcst_c`, `wind_fcst_mps` | Open-Meteo historical forecast API | 60 min, time-interpolated | real (archived forecast) |
| `price_dam_eur_mwh` | HEnEx `EL-DAM_Results` (annual archive) | 60 min, then 15 min from 1 Oct 2025 | real |
| `price_id_eur_mwh` | HEnEx `EL-IDA1/2/3_Results` | as above | real, see note 1 |
| `price_imb_eur_mwh` | IPTO `IMBABE` | 15 min | real |
| `afrr_cap_up_eur_mw_h`, `afrr_cap_down_eur_mw_h` | IPTO `ISP1ISPResults`, aFRR sheet | 30 min, then 15 min from Oct 2025 | real, see note 2 |
| `afrr_act_up`, `afrr_act_down` | Seeded regime model (`--seed`, default 2025) | 15 min | **synthetic**, see note 3 |

Prices are held constant over their market time unit; they are never
interpolated. The per-month metadata files record, for every column, the
source, the number of steps that had to be gap-filled, and the assumptions
below.

### Notes and assumptions

1. **Intraday price is an auction price, not a continuous-trading price.**
   The column takes the latest intraday *auction* clearing price available for
   each delivery period (IDA3 over IDA2 over IDA1). The environment's intraday
   layer, with its rolling 60-minute gate closure, behaves like continuous
   (XBID) trading. Auction prices are used as the reference price for those
   trades because the public results archive used here contains auction results
   only.
2. **Unit of the aFRR capacity price.** The ISP1 `Price Up` / `Price Down`
   values are used as EUR per MW per hour of awarded capacity. This should be
   confirmed against the Balancing Market Rulebook before capacity revenues
   are quoted.
3. **aFRR activations are synthetic by default.** IPTO publishes system
   totals of activated aFRR energy and awarded aFRR capacity, and the builder
   can turn their ratio into an activation signal
   (`--afrr-activation system_ratio`, plant activated pro rata). This is
   opt-in because the ratio is not a clean utilisation figure: in 2025 the
   upward ratio behaves well (median 0.06, 44 steps above 1), but the downward
   ratio exceeds 1 in 5,927 of 35,040 steps, so activated downward energy is
   evidently not bounded by the awarded downward band. Choosing the
   activation model is a modelling decision for the paper.
4. **aFRR activation energy price** is not in the dataset. The environment
   settles activated energy at the imbalance price unless the optional
   `afrr_en_up_eur_mwh` / `afrr_en_down_eur_mwh` columns are supplied.
5. **Time stamps of the market files (rev1).** The HEnEx results files
   (`DELIVERY_MTU`, `SORT`) and the IPTO files (IMBABE `STARTDATE`, ISP1
   header) are in CET/CEST wall clock: on 2025-03-30 they skip 02:00 and on
   2025-10-26 they repeat 02:00. The dataset builder stores them as
   Europe/Athens wall clock, i.e. one hour early. The correction is applied
   when the signals are loaded (`market_clock: cet`,
   `pv_bess_gym/data/time_alignment.py`), so the processed files are
   unchanged and protocols v3/v4 stay reproducible. The publication times
   (`PUB_TIME`) are consistent with Athens time and are not moved.
6. **Irradiance time stamps (rev1).** Open-Meteo documents hourly solar
   radiation as the mean of the preceding hour (temperature and wind are
   instantaneous). With `radiation_stamp: interval_centre` the GHI series
   (actual and forecasts) are moved 30 minutes earlier, to the centre of their
   averaging interval.

## Licences

| Source | Terms | Consequence |
|---|---|---|
| Open-Meteo | CC BY 4.0; the free API is for non-commercial use | Attribution required: "Weather data by Open-Meteo.com" |
| HEnEx (enexgroup.gr) | Redistribution terms not confirmed | Not redistributed; downloaded by the user |
| IPTO / ADMIE (admie.gr) | Redistribution terms not confirmed | Not redistributed; downloaded by the user |

Until HEnEx and IPTO confirm in writing that derived 15-minute series may be
republished, only the download-and-build scripts are shipped.

## The `example_data/` folder (v0.4 legacy sample)

`example_data/greek_2025_*.csv` is the dataset used for the SyNERGY MED 2026
conference paper. It is kept unchanged so those results stay reproducible, but
it is **not** suitable for new work:

- `price_dam_eur_mwh` is a smoothed transform of the imbalance price;
- `price_id_eur_mwh` is exactly `0.65 * DAM + 0.35 * imbalance`;
- both aFRR capacity prices are functions of the imbalance-DAM spread;
- weather forecasts are synthetic, and wind speed is in km/h despite the
  column name;
- aFRR activations are synthetic.

Only the imbalance price and the weather actuals in that folder are published
data. `scripts/data/verify_dataset.py example_data` reports each of these.
