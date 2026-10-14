# Knipdata — één actuele analyse

Gebruik alleen deze analysemap. De oorspronkelijke 14 ROS-logs en 14 DT-replays
zijn ongewijzigd aanwezig; de actuele analyse gebruikt uitsluitend stroomresidu.
42 knips blijven afzonderlijke observaties, met volledige runs buiten training.

## Uitvoeren

```bash
python3 run_all.py
```

Dit voert achtereenvolgens extractie, LDA, figuren/tabellen en broncontroles uit.
Alle actuele resultaten staan in **`outputs/`**; figuren in **`outputs/figures/`**.
De bestanden zijn reeds met de meegeleverde meetdata gegenereerd en gecontroleerd.
NumPy met en zonder `trapezoid` wordt ondersteund via een `trapz`-fallback.

## Mappen

- `code/`: de actuele analysescripts en het script voor nieuwe voorspellingen.
- `data/`: oorspronkelijke meetdata, vaste DT-replays en de bestaande referentie-CSV.
- `outputs/`: uitsluitend de actuele residuanalyse, inclusief sluit-RC 30–80%.
- `archive/`: vorige uitvoer, oude code, eerdere downloads en hun documentatie.

`archive/` wordt niet gebruikt door `run_all.py`. Het is aanwezig om eerdere
resultaten terug te kunnen vinden. Laat deze map ingeklapt tijdens het werken.
Er is één actuele README, één requirements-bestand en één runner in de hoofdmap.

## Parameters en validatie

Tien basisparameters voor M0/M3, bij openen/sluiten/actieve cyclus:
signed maximum, absolute maximum, mean, absolute integral, positive integral,
RMS, SD, range en de eerste twee Fourieramplitudes. Twee gezamenlijke parameters
beschrijven M0–M3 waveform correlation en covariance ellipse area.

RC wordt op het residu van M0 en M3 tegen native tijd berekend over 30–80% van
de commanded closing travel. De oorspronkelijke eerste-3%-uitsluiting en
native sample-aantallen blijven behouden. Alleen Fourierberekeningen gebruiken
interpolatie, op een rooster met het werkelijke aantal samples per venster.
RC is in mA/s; integralen zijn in mA s, maxima/mean/RMS/SD/range in mA.

De oorspronkelijke gefilterde stroom en gezonde DT-voorspelling bepalen het
residu. Er vindt geen DT-hertraining plaats en geen middeling over drie knips.
`Mean` is de tijdintegraal gedeeld door de geldige bewegingsduur; legacy
`original_*_mean`-modellen gebruiken het rekenkundig samplegemiddelde als controle.
Active cycle combineert openen en sluiten binnen dezelfde cyclus, zonder pauzes.

Iedere selectie (alle knips, alleen knip 1, alleen knips 2 en 3) wordt afzonderlijk
getraind en gevalideerd. StandardScaler en LDA worden binnen iedere run-wise fold
gefit, met gelijke priors en automatische shrinkage. Primaire resultaten hebben
`weight=1`; `weight=3` is een afzonderlijke, aanvullende drempelvergelijking.

## Resultaten lezen

- `scissors_binary_compact.csv/tex`: compacte healthy-versus-after-filing tabel.
- `healthy_vs_faulty_summary.csv`: geselecteerde modellen voor de drie knipselecties.
- `selected_threeclass_summary.csv`, `scenario_comparison.csv/tex`: severity-labels
  en gemiste fault alarms voor de verschillende intermediate-definities.
- `loocv_metrics.csv`: alle modellen, met TP/FN/TN/FP en aanvullende percentages.
- `loocv_predictions.csv`: iedere afzonderlijke testvoorspelling.
- `fold_coefficients.csv`: training/test-runs en fold-specifieke modellen.
- `cut_features.csv`, `feature_definitions.csv`: invoerwaarden en definities.
- `fitted_models.json`: beschrijvende full-data modellen voor figuren/nieuwe voorspellingen.
- `residual_RC_check.csv`, `checks.json`: regressie- en broncontroles.

Bij binary healthy-versus-faulty is after filing de positieve klasse. Bij drie
klassen zijn `TP/FN/TN/FP` leeg; gebruik `alarm_TP/FN/TN/FP` voor healthy versus
intermediate+blunt, en de 3×3 `confusion_matrix` voor exacte severity-labels.
Bij medium-versus-blunt is medium de referentie en blunt de positieve klasse.

Full-data LDA-grenzen in de figuren zijn beschrijvend; classificatietellingen
komen uit aparte run-wise cross-validation fits. Correlaties zijn beschrijvend
en sluiten de numerieke run-, knip- en iteratie-ID's uit.

## Eerdere gemeten-stroomvergelijkingen

Deze staan in `archive/old_outputs/loocv_metrics.csv`. Het optionele script
`python3 code/summarize_previous_current_models.py` schrijft een apart overzicht
naar `archive/previous_current_model_scores/`, niet naar de actuele resultaten.

Voor het opnemen van de tabellen in Overleaf zijn `booktabs` en `tabularx` nodig.

## Extra parametercombinaties

Deze uitbreiding vergelijkt alle 68 losse primaire residukenmerken en alle
2.278 combinaties van twee verschillende kenmerken (2.346 kandidaten).
Combinaties mogen verschillende parameters, motoren en bewegingsvensters
gebruiken. Bijvoorbeeld M0 closing RMS + M3 closing absolute maximum.
De legacy sample-mean-controles worden niet opnieuw als extra kenmerken
toegevoegd: de gewone mean blijft de tijdintegraal gedeeld door bewegingsduur.

Uitvoer staat uitsluitend onder `outputs/combinations/`. De bestaande
121-model-resultaten in `outputs/` blijven beschikbaar.

```bash
python3 run_all.py --combinations-only
```

Dit hergebruikt de bestaande features en voert de nieuwe vergelijking,
referentiecontroles en figuurexport uit. `python3 run_all.py` voert beide
analyses uit, inclusief nieuwe extractie en de bestaande 121 modellen.
`python3 run_all.py --skip-combinations` voert alleen de bestaande analyse uit.

### Methode

Dezelfde featurewaarden en individuele knips, dezelfde hele runs buiten
training, dezelfde train-only standardisation, dezelfde equal-prior
`lsqr` LDA en dezelfde automatische Ledoit-Wolf-shrinkage.
Voor snelheid worden de berekeningen voor 1/2 kenmerken gebundeld;
iedere combinatie houdt zijn eigen shrinkage, niet die van een 68D-model.
Er worden geen ruwe of gemeten-stroomfeatures toegevoegd en geen knips gemiddeld.

Bronimplementatie:
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/discriminant_analysis.py
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/covariance/_shrunk_covariance.py

De gebundelde berekening wordt gecontroleerd tegen bestaande testvoorspellingen
en rechtstreeks tegen sklearn voor nieuwe kandidaten, inclusief de gerapporteerde
koplopers. `checks.json` legt deze controles vast.

### Resultaten

- `model_registry.csv`: iedere unieke kandidaat en beide featurekolommen.
- `cv_metrics.csv`: alle run-wise resultaten voor zeven klasse-indelingen en
  drie knipselecties, inclusief TP/FN/TN/FP en exacte drieklassen-tellingen.
- `top_models.csv`: tien kandidaten per combinatie van scenario/selectie.
- `best_models_summary.csv`, `binary_compact.csv`, `severity_compact.csv`:
  compacte tabellen met de gekozen vertegenwoordiger van iedere rangschikking.
- `selected_cv_predictions.csv`: leesbare individuele testbeslissingen voor
  de gerapporteerde kandidaten.
- `cv_decisions.npz`: voorspellingen, scores, cycle/run-ID's en confusion matrices
  voor iedere kandidaat. Open met `numpy.load`; kandidaatvolgorde is die van de registry.
- `full_models.npz`: beschrijvende full-data class scores in originele eenheden.
- `fitted_models.json`: de koplopers in het bestaande `predict_cuts.py`-formaat.
  Bij binary LDA is `lda_score = coefficients @ features + intercept`;
  een positieve score voorspelt de positieve klasse. Dit is een score,
  geen onafhankelijk gevalideerde kans op botheid.
- `fold_models.npz`, `fold_manifest.csv`: fold-specifieke coefficients en de
  expliciete training/test-runs. Arrays zijn geordend zoals de registry.
- `figures/`: PNG/SVG-grenzen van de gekozen kandidaten en aparte withheld confusion matrices.

Bij twee klassen is de laatste klasse positief; bij healthy-versus-after-filing
is dat after filing. Bij medium-versus-blunt is medium de referentie en blunt
positief. Bij drie klassen beschrijven `alarm_*` de combinatie intermediate+blunt
versus healthy, terwijl `class*_correct/n` en `confusion_matrix` exacte labels tonen.

Fault-detection-rangschikking: minder alarm-FN, daarna minder alarm-FP,
daarna hogere balanced accuracy, daarna minder features en een vaste ID-volgorde.
Severity-rangschikking: hogere balanced accuracy, daarna minder gemiste fault alarms.
Deze criteria staan vast voordat de vergelijking wordt uitgevoerd.
De tabel noemt één vertegenwoordiger; gelijke resultaten kunnen meerdere kandidaten hebben.

### Selectiebias en aanvullende selectiecontrole

Een winnaar uit 2.346 kandidaten is op dezelfde kleine dataset gezocht.
Zijn per-candidate CV-resultaat is daarom verkennend: het is geen onafhankelijke
prestatiemeting van de keuze voor die winnaar.

Als aanvullende controle wordt voor healthy-versus-all een nested selectie uitgevoerd:
ieder buitengehouden outer run blijft ook volledig buiten parameterkeuze.
De keuze gebeurt via inner run-wise validation op de resterende runs.
Inner folds die een trainingsklasse leeg zouden maken, worden overgeslagen.

Met slechts twee healthy runs blijft bij een outer healthy holdout één healthy
trainingsrun over. Die kan niet ook voor inner testing worden buitengehouden;
dan bevat inner testing uitsluitend after-filing-runs. Deze beperking staat
expliciet in `nested_selection_folds.csv` (`inner_healthy_test_available`).
De nested resultaten horen bij deze selectieprocedure, niet bij één vast paar.

Bestanden: `nested_predictions.csv`, `nested_metrics.csv`,
`nested_selection_folds.csv`, `candidate_vs_nested.csv`.
Grensfiguren zijn beschrijvend en geen afbeeldingen van deze nested procedure.

Het compacte downloadpakket bevat alle CV-voorspellingen, maar laat de
grote score-arrays per kandidaat en `fold_models.npz` weg. De full-data
coëfficiënten, modeltabellen, fold-manifest en checks blijven meegeleverd.
`run_all.py --combinations-only` genereert ook deze grotere tussenbestanden
opnieuw. Dit verandert de analyse of de meegeleverde tellingen niet.

Voor een andere kandidaat uit de registry:

```bash
python3 code/export_combination_models.py --model F1234 --scenario healthy_all --selection all
```

Dit schrijft `requested_fitted_models.json`. Gebruik bij `predict_cuts.py`
de expliciete `--models`, `--model`, `--scenario` en `--selection`-opties.

### Installeren

Pak de update apart uit en gebruik vanuit de bestaande scissors-map:

```bash
python3 /pad/naar/uitgepakte/update/install_combination_update.py "$PWD"
```

Nieuwe scripts worden in `code/` gezet, de bestaande README aangevuld en reeds
berekende extra resultaten in `outputs/combinations/` geplaatst. Vervangen
bestanden worden bewaard in `archive/`; data en originele resultaten blijven intact.
