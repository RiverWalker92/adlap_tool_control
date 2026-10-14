# Knipdata: uitsluitend stroomresidu, inclusief sluit-RC 30–80%

## Installeren en uitvoeren

Deze versie gebruikt `numpy.trapezoid` als die beschikbaar is, en anders
`numpy.trapz` met dezelfde trapeziumintegratie. Een NumPy-upgrade is niet nodig.

Pak de ZIP bij voorkeur apart uit en installeer met:

```bash
python3 /pad/naar/uitgepakte/update/install_update.py /pad/naar/scissors
```

De installer bewaart vervangen scripts in een backupmap en raakt meetdata
en bestaande resultaten niet aan.


Pak de inhoud van deze ZIP uit in de bestaande analysemap met `code/`, `data/`
en eventueel `outputs/`. Laat de bestaande ROS-logs en Motor DT-replays in
`data/` staan; submappen zijn toegestaan. Vervang de meegeleverde codebestanden.
De ZIP bevat geen meetdata en geen nieuwe experimentele resultaten.

Voer vanuit de analysemap uit met jouw bestaande Python-omgeving:

```bash
python3 run_all.py
```

`python3 runall.py` werkt ook. Het script gebruikt de bekende 14 runs voor
iteraties 0,1,2,3,4,6,8: twee runs per iteratie, drie afzonderlijke knips per run.
De nieuwe uitvoer staat in **`outputs_residual/`**; de oude `outputs/` blijft intact.
De vereiste Python-pakketten staan in `requirements.txt`.

## Wat verandert?

- Alle geregistreerde classificatiemodellen gebruiken stroomresidu:
  gefilterde gemeten stroom minus de vaste gezonde Motor DT-voorspelling.
- Sluit-RC wordt voor M0 en M3 berekend door OLS-regressie van het residu
  tegen de oorspronkelijke tijd, over 30–80% van de aangestuurde sluitverplaatsing.
  Dit is geen selectie op stroomamplitude. Eenheid: mA/s.
- Nieuwe kolommen: `closing_M0_rc30_80` en `closing_M3_rc30_80`.
- Nieuwe modellen omvatten beide RC's afzonderlijk, de gekoppelde RC's,
  en M3 absolute residupiek + RC, residupiek + mean, residupiek + RMS en RC + RMS.
- De oude gemeten-stroommodellen en mengvormen worden niet meer getraind.
  Oude modelnamen zoals `filtered_peak_rc` en `rc30_80` worden bewust niet
  hergebruikt: de nieuwe modellen hebben andere invoersignalen.
- De tien bestaande basisparameters, twee gezamenlijke parameters, originele
  samples, eerste-3%-uitsluiting en aparte knips blijven behouden.
  Fourierberekeningen gebruiken een uniform tijdrooster met het werkelijke
  aantal samples in dat specifieke venster; andere parameters gebruiken
  de originele samples/tijden. Er wordt niet over knips gemiddeld.
- `Mean` is de tijdintegraal gedeeld door de geldige bewegingsduur.
  Active cycle combineert openen en sluiten binnen één cyclus, zonder pauzes.
  Legacy `original_*_mean`-modellen blijven als controle bestaan en gebruiken
  het rekenkundig samplegemiddelde, dus een andere mean-definitie.

## Validatie en tellingen

Hele runs blijven buiten training. StandardScaler en LDA worden binnen iedere
fold gefit. Gelijke klassepriors en automatische covariance shrinkage blijven
behouden. De drie selecties worden apart getraind/gevalideerd: alle knips,
alleen knip 1 en knips 2 en 3 als afzonderlijke observaties.

De primaire uitvoer gebruikt `weight=1`, de gewone LDA-beslisregel.
`weight=3` blijft als expliciet aanvullende binaire analyse beschikbaar.
Ook `predict_cuts.py` gebruikt nu standaard `weight=1`.

`loocv_metrics.csv` bevat nu expliciete `TP`, `FN`, `TN`, `FP` bij twee klassen.
Bij drie klassen zijn deze vier velden leeg; `alarm_TP`, `alarm_FN`,
`alarm_TN`, `alarm_FP` combineren intermediate en blunt als positieve klasse.
De exacte severity-labels staan in `class0/1/2_correct`, `class0/1/2_n`
en de 3×3 `confusion_matrix` (rij = werkelijk; kolom = voorspeld).

Let op: in het afzonderlijke scenario medium-versus-blunt is medium de
referentie en blunt de positieve klasse. Dat is niet healthy-versus-faulty.
`reference_label` en `positive_labels` leggen deze betekenis vast.

## Oude gemeten-stroommodellen eerst beoordelen

Vóór extractie leest `summarize_previous_current_models.py` automatisch het
bestaande `outputs/loocv_metrics.csv`, indien aanwezig. Het schrijft:

- `previous_measured_current_models.csv`: alle oude gemeten-stroommodellen
  en mengvormen, met hun standaard-threshold-resultaten en tellingen.
- `previous_best_binary_models.csv`: vijf modellen per knipselectie voor
  healthy-versus-all, gerangschikt op minder FN en daarna minder FP.

Dit archief bevat oude resultaten op gemeten stroom en staat los van de nieuwe
residu-analyse. Zonder het oude CSV-bestand worden geen oude prestaties verzonnen.

## Belangrijkste nieuwe uitvoer

- `cut_features.csv`, `feature_definitions.csv`: parameters per knip en definities.
- `loocv_metrics.csv`, `loocv_predictions.csv`, `fold_coefficients.csv`:
  alle prestaties, testvoorspellingen en fold-specifieke modellen.
- `model_registry.csv/json`, `fitted_models.json`: geteste invoersets en
  beschrijvende modellen gefit op alle beschikbare observaties.
- `scissors_binary_compact.csv/tex`: zeven geselecteerde regels, met tellingen.
- `healthy_vs_faulty_summary.csv`, `feature_validation_summary.csv`:
  uitgebreidere vergelijkingen van modellen en individuele sluitparameters.
- `scenario_comparison.csv/tex`, `selected_threeclass_summary.csv`:
  severity-vergelijkingen, waaronder all-cuts/M3 sluiten+openen en
  cuts-2,3/M0+M3 openen. De compacte severity-tabel rapporteert exacte labels
  en faults als healthy, zonder BA-kolom.
- `residual_RC_check.csv`, `checks.json`: onafhankelijke broncontroles.
- `figures/`: PNG- en SVG-uitvoer voor individuele profielen, boxplots,
  LDA mean/range, M0+M3 opening mean, gemiste detecties/false alarms,
  verwarringsmatrices, knipvolgorde en parametercorrelaties.

De LaTeX-tabellen gebruiken de bestaande pakketten `booktabs` en `tabularx`.

Tabel- en aslabels noemen niet telkens opnieuw 'current residual'; dit staat
eenmaal in de caption/tekst. Alle nieuwe classificaties gebruiken het residu.
Correlatiematrices sluiten numerieke iteratie-, run- en knip-ID's uit.
De Fouriercomponenten beschrijven de profielvorm, geen fysieke rotatieorders.

LDA-grenzen in figuren zijn beschrijvende full-data fits. De gerapporteerde
classificatietellingen komen uit afzonderlijke run-wise validation fits.
Gebruik die beschrijvende grenzen dus niet als validatieresultaat.

## Getest

De pipeline is gecontroleerd op synthetische ROS/replay-paren met alle
14 runs en 42 knips, inclusief submappen en niet-uniforme timestamps.
Sluit-RC is onafhankelijk nagerekend met least-squares regressie.
De ongewijzigde parameters en de voorspellingen van vier bestaande
residumodellen zijn vergeleken met de geüploade scripts.

De werkelijke ROS-logs en replays waren niet bij deze vijf scripts gevoegd.
Dit pakket bevat daarom geen nieuwe classificatieprestaties voor jouw data.
Die worden berekend zodra je `run_all.py` uitvoert met jouw bestaande `data/`.
